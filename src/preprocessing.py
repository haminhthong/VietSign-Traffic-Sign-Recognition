"""Module Tiền Xử Lý Ảnh (Preprocessing).

Cung cấp các kỹ thuật tiền xử lý ảnh truyền thống:
- Lọc nhiễu muối tiêu (Salt-and-pepper noise) bằng Median Filter
- Tăng cường tương phản cục bộ tự thích nghi (CLAHE) trên kênh L của không gian màu LAB
  nhằm giữ nguyên thông tin màu sắc (kênh A và B) trước khi phân đoạn màu HSV.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def apply_median_filter(image_bgr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Lọc nhiễu hạt (Salt-and-pepper noise) bằng lọc Median Filter.

    Args:
        image_bgr (np.ndarray): Ảnh đầu vào định dạng BGR.
        kernel_size (int): Kích thước cửa sổ lọc median (phải là số nguyên lẻ >= 3). Mặc định 3.

    Returns:
        np.ndarray: Ảnh đã được khử nhiễu.

    Raises:
        ValueError: Nếu ảnh rỗng hoặc kernel_size không hợp lệ.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size phải là số nguyên lẻ và lớn hơn hoặc bằng 3")

    image_bgr = np.ascontiguousarray(image_bgr)
    return cv2.medianBlur(image_bgr, kernel_size)


def compute_dynamic_clip_limit(l_channel: np.ndarray) -> Tuple[float, float]:
    """Tính clipLimit cho CLAHE dựa trên độ lệch chuẩn của kênh L.

    Args:
        l_channel (np.ndarray): Kênh Lightness (L) từ không gian màu LAB.

    Returns:
        Tuple[float, float]: (clip_limit, std_value).
    """
    std = float(np.std(l_channel))

    if std < 50.0:
        clip_limit = 4.0
    elif std < 100.0:
        clip_limit = 2.0
    else:
        clip_limit = 1.0

    return clip_limit, std


def apply_clahe_lab(
    image_bgr: np.ndarray,
    tile_grid_size: Tuple[int, int] = (8, 8),
    clip_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Tăng cường độ tương phản cục bộ (CLAHE) trên kênh L của không gian màu LAB.

    Giúp duy trì màu sắc ở kênh A và B mà không bị biến đổi khi thay đổi độ sáng kênh L.

    Args:
        image_bgr (np.ndarray): Ảnh BGR đầu vào.
        tile_grid_size (Tuple[int, int]): Kích thước ô phân chia ảnh. Mặc định (8, 8).
        clip_limit (Optional[float]): Ngưỡng giới hạn tương phản. Nếu None, tự tính theo std.

    Returns:
        Tuple[np.ndarray, float, float]: (Ảnh BGR đã tăng cường, clip_limit đã dùng, std kênh L).

    Raises:
        ValueError: Nếu ảnh rỗng hoặc tile_grid_size không đúng định dạng.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")
    if len(tile_grid_size) != 2 or any(int(v) <= 0 for v in tile_grid_size):
        raise ValueError("tile_grid_size phải gồm hai số nguyên dương")

    tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    # Chuyển từ BGR sang không gian màu LAB
    image_bgr = np.ascontiguousarray(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    std = float(np.std(lightness))
    used_clip_limit = (
        float(clip_limit) if clip_limit is not None else compute_dynamic_clip_limit(lightness)[0]
    )

    # Khởi tạo thuật toán CLAHE và áp dụng lên kênh L
    clahe = cv2.createCLAHE(clipLimit=used_clip_limit, tileGridSize=tile_grid_size)
    lightness_enhanced = clahe.apply(lightness)

    # Tái hợp các kênh và chuyển về BGR
    lab_enhanced = cv2.merge([lightness_enhanced, channel_a, channel_b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced_bgr, used_clip_limit, std


def preprocess_image(
    image_bgr: np.ndarray,
    kernel_size: int = 3,
    tile_grid_size: Tuple[int, int] = (8, 8),
    clip_limit: Optional[float] = 2.0,
) -> np.ndarray:
    """Quy trình tiền xử lý ảnh hoàn chỉnh: Median Filter + LAB CLAHE.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        kernel_size (int): Kích thước kernel lọc Median. Mặc định 3.
        tile_grid_size (Tuple[int, int]): Kích thước chia lưới CLAHE. Mặc định (8, 8).
        clip_limit (Optional[float]): Ngưỡng giới hạn CLAHE. Mặc định 2.0.

    Returns:
        np.ndarray: Ảnh BGR đã qua tiền xử lý, sẵn sàng cho phân đoạn màu.
    """
    denoised = apply_median_filter(image_bgr, kernel_size=kernel_size)
    enhanced, _, _ = apply_clahe_lab(denoised, tile_grid_size=tile_grid_size, clip_limit=clip_limit)
    return enhanced


# Bí danh tương thích ngược cho codebase cũ
preprocess_task1 = preprocess_image
