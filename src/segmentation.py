"""Module Phân Đoạn Màu Kênh HSV (Color Segmentation).

Thực hiện phân đoạn các vùng ảnh màu có khả năng chứa biển báo giao thông Việt Nam
dựa trên 3 nhóm màu chủ đạo:
- Đỏ (Red): Biển cấm (viền đỏ), biển cảnh báo nguy hiểm (viền đỏ).
- Vàng (Yellow): Nền biển cảnh báo nguy hiểm.
- Xanh dương (Blue): Biển hiệu lệnh, biển chỉ dẫn.

Sử dụng không gian màu HSV vì kênh Hue tách biệt thông tin màu sắc khỏi độ sáng (Value).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Bộ dải ngưỡng HSV chuẩn cho biển báo giao thông Việt Nam
DEFAULT_HSV_RANGES: Dict[str, Tuple[List[int], List[int]]] = {
    "red1": ([0, 50, 50], [15, 255, 255]),
    "red2": ([162, 50, 50], [180, 255, 255]),
    "yellow": ([14, 50, 50], [38, 255, 255]),
    "blue": ([88, 50, 50], [132, 255, 255]),
}


def get_hsv_ranges(set_number: int = 1) -> Dict[str, Tuple[List[int], List[int]]]:
    """Lấy danh sách các dải ngưỡng không gian màu HSV tương ứng.

    Màu đỏ nằm ở 2 đầu dải Hue ([0..15] và [162..180]) trong OpenCV (Hue từ 0 đến 180).

    Args:
        set_number (int): Bộ dải màu (1: Tiêu chuẩn, 2: Độ bão hòa trung bình, 3: Dải hẹp).

    Returns:
        Dict[str, Tuple[List[int], List[int]]]: Từ điển chứa các dải (lower_bound, upper_bound).
    """
    ranges_by_set = {
        1: DEFAULT_HSV_RANGES,
        2: {
            "red1": ([0, 70, 70], [10, 255, 255]),
            "red2": ([170, 70, 70], [180, 255, 255]),
            "yellow": ([20, 70, 70], [30, 255, 255]),
            "blue": ([100, 70, 70], [125, 255, 255]),
        },
        3: {
            "red1": ([0, 100, 100], [5, 255, 255]),
            "red2": ([175, 100, 100], [180, 255, 255]),
            "yellow": ([22, 100, 100], [28, 255, 255]),
            "blue": ([105, 100, 100], [115, 255, 255]),
        },
    }
    return ranges_by_set.get(set_number, DEFAULT_HSV_RANGES)


def generate_combined_mask(
    img_hsv: np.ndarray,
    ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tạo mặt nạ (mask) kết hợp từ các dải màu Đỏ, Xanh dương và Vàng.

    Args:
        img_hsv (np.ndarray): Ảnh đã chuyển đổi sang không gian màu HSV.
        ranges (Optional[Dict]): Từ điển các dải màu. Nếu None, dùng DEFAULT_HSV_RANGES.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (combined_mask, mask_red, mask_blue, mask_yellow)
    """
    hsv_ranges = ranges if ranges is not None else DEFAULT_HSV_RANGES
    lower_r1, upper_r1 = hsv_ranges["red1"]
    lower_r2, upper_r2 = hsv_ranges["red2"]
    lower_y, upper_y = hsv_ranges["yellow"]
    lower_b, upper_b = hsv_ranges["blue"]

    # Đỏ được hợp nhất từ 2 dải do Hue bao quanh mốc 0/180
    mask_red = cv2.bitwise_or(
        cv2.inRange(
            img_hsv, np.array(lower_r1, dtype=np.uint8), np.array(upper_r1, dtype=np.uint8)
        ),
        cv2.inRange(
            img_hsv, np.array(lower_r2, dtype=np.uint8), np.array(upper_r2, dtype=np.uint8)
        ),
    )
    mask_yellow = cv2.inRange(
        img_hsv, np.array(lower_y, dtype=np.uint8), np.array(upper_y, dtype=np.uint8)
    )
    mask_blue = cv2.inRange(
        img_hsv, np.array(lower_b, dtype=np.uint8), np.array(upper_b, dtype=np.uint8)
    )

    # Kết hợp tất cả bằng phép toán OR bitwise
    combined = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_yellow), mask_blue)
    return combined, mask_red, mask_blue, mask_yellow


def build_achromatic_mask(
    img_hsv: np.ndarray, s_max: int = 40, v_white_min: int = 150, v_black_max: int = 80
) -> np.ndarray:
    """Tạo mặt nạ vùng phi sắc (Trắng/Đen) như viền hoặc ký hiệu nội dung biển báo."""
    _, s, v = cv2.split(img_hsv)
    white = (s <= s_max) & (v >= v_white_min)
    black = v <= v_black_max
    mask = np.zeros_like(s, dtype=np.uint8)
    mask[white | black] = 255
    return mask


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Điền đầy các lỗ hổng bên trong đường viền mặt nạ (Hole filling)."""
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment_colors(
    image_bgr: np.ndarray,
    set_number: int = 1,
    include_achromatic: bool = False,
    achromatic_dilate_ksize: int = 9,
    fill_holes: bool = False,
    ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phân đoạn màu ảnh BGR đầu vào theo các dải màu biển báo.

    Args:
        image_bgr (np.ndarray): Ảnh gốc BGR.
        set_number (int): Bộ dải màu HSV. Mặc định 1.
        include_achromatic (bool): Có hợp nhất vùng trắng/đen hay không. Mặc định False.
        achromatic_dilate_ksize (int): Kích thước kernel giãn nở mặt nạ phi sắc. Mặc định 9.
        fill_holes (bool): Lấp đầy lỗ hổng trong mặt nạ hay không. Mặc định False.
        ranges (Optional[Dict]): Bộ ngưỡng HSV tùy chỉnh.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Ảnh sau phân đoạn BGR, mặt nạ nhị phân kết hợp).

    Raises:
        ValueError: Nếu ảnh rỗng.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_ranges = ranges if ranges is not None else get_hsv_ranges(set_number)
    combined_mask, _, _, _ = generate_combined_mask(img_hsv, hsv_ranges)

    if include_achromatic:
        achromatic_mask = build_achromatic_mask(img_hsv)
        if achromatic_dilate_ksize and achromatic_dilate_ksize > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (achromatic_dilate_ksize, achromatic_dilate_ksize)
            )
            achromatic_mask = cv2.dilate(achromatic_mask, kernel)
        combined_mask = cv2.bitwise_or(combined_mask, achromatic_mask)

    if fill_holes:
        combined_mask = fill_mask_holes(combined_mask)

    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=combined_mask)
    return segmented, combined_mask


# Bí danh tương thích ngược
segment_task2 = segment_colors


def build_sign_color_mask(
    image_bgr: np.ndarray,
    set_number: int = 1,
    include_achromatic: bool = False,
    fill_holes: bool = False,
) -> np.ndarray:
    """Tạo mặt nạ nhị phân thể hiện màu biển báo (Hàm tiện ích nhanh)."""
    _, combined_mask = segment_colors(
        image_bgr,
        set_number=set_number,
        include_achromatic=include_achromatic,
        achromatic_dilate_ksize=0,
        fill_holes=fill_holes,
    )
    return combined_mask
