"""Module Trích Xuất Vùng Đề Xuất Ứng Viên (Candidate Proposal Generation).

Trích xuất các vùng chứa khả năng là biển báo giao thông (Candidate Bounding Boxes)
từ phân đoạn màu sắc HSV kết hợp kiểm tra hình dạng hình học (độ tròn, tam giác, chữ nhật),
sau đó áp dụng thuật toán NMS để loại bỏ trùng lặp.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.segmentation import generate_combined_mask, get_hsv_ranges
from src.utils import box_xywh_to_xyxy, compute_iou


def hsv_boxes(
    img_bgr: np.ndarray,
    set_number: int = 1,
    min_area: float = 150.0,
    max_area_ratio: float = 0.35,
    ar_range: Tuple[float, float] = (0.4, 2.5),
    min_extent: float = 0.20,
    ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Trích xuất bounding boxes ứng viên từ mặt nạ phân đoạn màu HSV.

    Args:
        img_bgr (np.ndarray): Ảnh BGR đầu vào.
        set_number (int): Bộ dải HSV (1: Chuẩn). Mặc định 1.
        min_area (float): Diện tích vùng nhỏ nhất (pixels). Mặc định 150.
        max_area_ratio (float): Tỷ lệ diện tích tối đa so với toàn ảnh. Mặc định 0.35.
        ar_range (Tuple[float, float]): Dải aspect ratio cho phép (w/h).
        min_extent (float): Tỷ lệ lấp đầy (contour_area / bounding_box_area) tối thiểu.
        ranges (Optional[Dict]): Bộ ngưỡng HSV tùy chỉnh.

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


def merge_boxes_nms(
    box_lists: List[List[Tuple[int, int, int, int]]], iou_thresh: float = 0.4
) -> List[Tuple[int, int, int, int]]:
    """Gộp các danh sách hộp ứng viên và áp dụng Non-Maximum Suppression (NMS).

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
    hsv_min_extent: float = 0.20,
    mser_ar_range: Tuple[float, float] = (0.6, 1.6),
    mser_min_extent: float = 0.2,
    hsv_ranges: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, List[Tuple[int, int, int, int]]]]:
    """Trích xuất ứng viên từ phân đoạn màu HSV và lọc bằng NMS.

    Args:
        img_bgr (np.ndarray): Ảnh BGR đầu vào.
        hsv_set (Union[int, List[int]]): Bộ dải HSV.
        mser_delta (int): Tương thích ngược.
        canny_low (int): Tương thích ngược.
        canny_high (int): Tương thích ngược.
        iou_thresh (float): Ngưỡng IoU cho NMS.
        hsv_ar_range (Tuple[float, float]): Dải aspect ratio cho HSV.
        hsv_min_extent (float): Extent tối thiểu cho HSV.
        mser_ar_range (Tuple[float, float]): Tương thích ngược.
        mser_min_extent (float): Tương thích ngược.
        hsv_ranges (Optional[Dict]): Ngưỡng HSV tùy chỉnh.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Dict[str, List[Tuple[int, int, int, int]]]]:
            (Danh sách hộp ứng viên, Từ điển chi tiết).
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], {"hsv": [], "mser": [], "edge_hull": []}

    set_idx = (
        hsv_set[0]
        if isinstance(hsv_set, list) and hsv_set
        else (hsv_set if isinstance(hsv_set, int) else 1)
    )

    b_hsv = hsv_boxes(
        img_bgr,
        set_number=set_idx,
        ar_range=hsv_ar_range,
        min_extent=hsv_min_extent,
        ranges=hsv_ranges,
    )

    union_boxes = merge_boxes_nms([b_hsv], iou_thresh=iou_thresh)
    return union_boxes, {"hsv": b_hsv, "mser": [], "edge_hull": []}
