"""Module Hợp Nhất Vùng Ứng Viên (Task 2 Union Candidate Box Extraction).

Kết hợp các kỹ thuật trích xuất vùng đề xuất (Candidate Bounding Boxes) từ 3 nguồn:
1. Phân đoạn màu HSV (hsv_boxes)
2. Vùng cực trị đồng nhất MSER (mser_boxes)
3. Biên Canny & Bao lồi Convex Hull (edge_shape_boxes_hull)

Sau đó áp dụng thuật toán NMS (Non-Maximum Suppression) dựa trên chỉ số IoU để loại bỏ trùng lặp.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.segmentation import generate_combined_mask, get_hsv_ranges
from src.utils import box_xywh_to_xyxy, compute_iou


def hsv_boxes(
    img_bgr: np.ndarray,
    set_number: int = 2,
    min_area: float = 300.0,
    max_area_ratio: float = 0.3,
    ar_range: Tuple[float, float] = (0.4, 2.5),
    min_extent: float = 0.25,
    ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Trích xuất bounding boxes ứng viên từ mặt nạ phân đoạn màu HSV.

    Args:
        img_bgr (np.ndarray): Ảnh BGR đầu vào.
        set_number (int): Bộ dải HSV (1, 2, 3). Mặc định 2.
        min_area (float): Diện tích vùng nhỏ nhất (pixels). Mặc định 300.
        max_area_ratio (float): Tỷ lệ diện tích tối đa so với toàn ảnh. Mặc định 0.3.
        ar_range (Tuple[float, float]): Tỷ lệ khung hình (w/h) cho phép (min, max).
        min_extent (float): Tỷ lệ lấp đầy (contour_area / bounding_box_area) tối thiểu.
        ranges (Optional[Dict]): Bộ ngưỡng HSV tùy chỉnh cho set_number.

    Returns:
        List[Tuple[int, int, int, int]]: Danh sách các hộp dạng (x, y, w, h).
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_ranges = ranges if ranges is not None else get_hsv_ranges(set_number)
    mask, _, _, _ = generate_combined_mask(hsv, hsv_ranges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img_bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > (h_img * w_img * max_area_ratio):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / float(h)
        if not (ar_range[0] <= ar <= ar_range[1]):
            continue
        extent = area / float(w * h)
        if extent < min_extent:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def mser_boxes(
    gray: np.ndarray,
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14400,
    ar_range: Tuple[float, float] = (0.6, 1.6),
    min_wh: int = 15,
    min_extent: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """Trích xuất ứng viên bằng thuật toán MSER (Maximally Stable Extremal Regions).

    MSER hoạt động cực tốt trong việc tìm kiếm các vùng ký hiệu văn bản hoặc hình họa
    đồng nhất về độ sáng trên biển báo.

    Args:
        gray (np.ndarray): Ảnh xám (Gray level image uint8).
        delta (int): Khoảng chênh lệch mức xám để so sánh sự ổn định vùng.
        min_area (int): Diện tích nhỏ nhất của vùng.
        max_area (int): Diện tích lớn nhất của vùng.
        ar_range (Tuple[float, float]): Tỷ lệ khung hình w/h.
        min_wh (int): Kích thước chiều rộng và chiều cao tối thiểu (pixel).
        min_extent (float): Tỷ lệ lấp đầy tối thiểu.

    Returns:
        List[Tuple[int, int, int, int]]: Danh sách các hộp dạng (x, y, w, h).
    """
    if gray is None or gray.size == 0:
        return []
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    regions, bboxes = mser.detectRegions(gray)
    h_img, w_img = gray.shape[:2]

    kept: List[Tuple[int, int, int, int]] = []
    for region, (x, y, w, h) in zip(regions, bboxes):
        if w < min_wh or h < min_wh:
            continue
        if w > w_img * 0.5 or h > h_img * 0.5:
            continue
        if h == 0 or w == 0:
            continue
        ar = w / float(h)
        if not (ar_range[0] <= ar <= ar_range[1]):
            continue
        extent = len(region) / float(w * h)
        if extent < min_extent:
            continue
        kept.append((int(x), int(y), int(w), int(h)))

    return kept


def edge_shape_boxes_hull(
    gray: np.ndarray,
    low: int = 50,
    high: int = 150,
    min_area: float = 300.0,
    max_area_ratio: float = 0.3,
    circularity_min: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """Phát hiện ứng viên qua thuật toán dò biên Canny + Bao lồi Convex Hull.

    Giúp bắt các biển báo bị mất màu hoặc bị chói sáng nhờ vào hình dáng hình học.

    Args:
        gray (np.ndarray): Ảnh xám đầu vào.
        low (int): Ngưỡng Canny dưới.
        high (int): Ngưỡng Canny trên.
        min_area (float): Diện tích tối thiểu.
        max_area_ratio (float): Tỷ lệ diện tích tối đa so với ảnh.
        circularity_min (float): Độ tròn tối thiểu $4 \\pi \\text{Area} / \\text{Perimeter}^2$.

    Returns:
        List[Tuple[int, int, int, int]]: Danh sách các hộp dạng (x, y, w, h).
    """
    if gray is None or gray.size == 0:
        return []

    h_img, w_img = gray.shape[:2]
    edges = cv2.Canny(gray, low, high)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < min_area or area > (h_img * w_img * max_area_ratio):
            continue
        x, y, w, h = cv2.boundingRect(hull)
        if h == 0 or w == 0:
            continue
        ar = w / float(h)
        if not (0.6 <= ar <= 1.6):
            continue
        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter**2)
        if circularity < circularity_min:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def merge_boxes_nms(
    box_lists: List[List[Tuple[int, int, int, int]]], iou_thresh: float = 0.4
) -> List[Tuple[int, int, int, int]]:
    """Gộp các danh sách hộp ứng viên và áp dụng Non-Maximum Suppression (NMS).

    Ưu tiên giữ lại các hộp có diện tích lớn hơn khi có sự trùng lặp IoU > iou_thresh.

    Args:
        box_lists (List[List[Tuple[int, int, int, int]]]): Danh sách các nhóm hộp đầu vào.
        iou_thresh (float): Ngưỡng IoU để coi 2 hộp là trùng lặp. Mặc định 0.4.

    Returns:
        List[Tuple[int, int, int, int]]: Danh sách các hộp đã qua lọc NMS.
    """
    all_boxes: List[Tuple[int, int, int, int]] = []
    for boxes in box_lists:
        all_boxes.extend(boxes)

    # Sắp xếp theo diện tích giảm dần
    all_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    merged: List[Tuple[int, int, int, int]] = []
    for box in all_boxes:
        box_xyxy = box_xywh_to_xyxy(box)
        if all(
            compute_iou(box_xyxy, box_xywh_to_xyxy(kept_box)) < iou_thresh for kept_box in merged
        ):
            merged.append(box)

    return merged


def build_union_boxes(
    img_bgr: np.ndarray,
    hsv_set: Union[int, List[int]] = 1,
    mser_delta: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    iou_thresh: float = 0.4,
    hsv_ar_range: Tuple[float, float] = (0.4, 2.5),
    hsv_min_extent: float = 0.25,
    mser_ar_range: Tuple[float, float] = (0.6, 1.6),
    mser_min_extent: float = 0.2,
    hsv_ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, List[Tuple[int, int, int, int]]]]:
    """Hợp nhất ứng viên từ cả 3 nguồn (HSV + MSER + Edge Hull) và lọc bằng NMS.

    Args:
        img_bgr (np.ndarray): Ảnh BGR đầu vào.
        hsv_set (int): Bộ dải HSV.
        mser_delta (int): Delta cho MSER.
        canny_low (int): Ngưỡng Canny dưới.
        canny_high (int): Ngưỡng Canny trên.
        iou_thresh (float): Ngưỡng IoU cho NMS.
        hsv_ar_range (Tuple[float, float]): Dải aspect ratio cho HSV.
        hsv_min_extent (float): Extent tối thiểu cho HSV.
        mser_ar_range (Tuple[float, float]): Dải aspect ratio cho MSER.
        mser_min_extent (float): Extent tối thiểu cho MSER.
        hsv_ranges (Optional[Dict]): Ngưỡng HSV tùy chỉnh áp dụng cho set 1.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Dict[str, List[Tuple[int, int, int, int]]]]:
            (Danh sách hộp hợp nhất, Từ điển chi tiết từng nguồn).
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], {"hsv": [], "mser": [], "edge_hull": []}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    hsv_sets = [hsv_set] if isinstance(hsv_set, int) else list(hsv_set)
    if not hsv_sets:
        raise ValueError("hsv_set phải là một số hoặc danh sách preset không rỗng")

    b_hsv: List[Tuple[int, int, int, int]] = []
    for hsv_index in hsv_sets:
        b_hsv.extend(
            hsv_boxes(
                img_bgr,
                set_number=hsv_index,
                ar_range=hsv_ar_range,
                min_extent=hsv_min_extent,
                ranges=hsv_ranges if hsv_index == 1 else None,
            )
        )
    b_mser = mser_boxes(
        gray_blur, delta=mser_delta, ar_range=mser_ar_range, min_extent=mser_min_extent
    )
    b_edge = edge_shape_boxes_hull(gray_blur, low=canny_low, high=canny_high)

    union_boxes = merge_boxes_nms([b_hsv, b_mser, b_edge], iou_thresh=iou_thresh)
    return union_boxes, {"hsv": b_hsv, "mser": b_mser, "edge_hull": b_edge}
