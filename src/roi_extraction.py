"""Module Trích Xuất Vùng Quan Tâm (ROI Extraction & NMS Filtering).

Thực hiện cắt (crop), nắn thẳng góc nhìn (Perspective/Affine Transform khi có đỉnh đa giác),
chuẩn hóa kích thước ROI ($64 \\times 64$), kiểm tra tỷ lệ khung hình (Aspect Ratio),
và áp dụng thuật toán NMS (Non-Maximum Suppression) để loại bỏ các vùng ứng viên trùng lặp.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import compute_iou


def crop_roi(image_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Cắt vùng ảnh (Crop ROI) theo hình chữ nhật bounding box thẳng.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        x (int): Tọa độ góc trên bên trái X.
        y (int): Tọa độ góc trên bên trái Y.
        w (int): Chiều rộng.
        h (int): Chiều cao.

    Returns:
        np.ndarray: Vùng ảnh đã crop (hoặc ma trận rỗng nếu ngoài phạm vi).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    H, W = image_bgr.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), W), min(int(y + h), H)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return image_bgr[y0:y1, x0:x1]


def _order_triangle_pts(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp các đỉnh tam giác bắt đầu từ đỉnh trên cùng."""
    top_idx = int(np.argmin(pts[:, 1]))
    return np.roll(pts, -top_idx, axis=0)


def _order_rect_pts(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp các đỉnh tứ giác theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_roi_warped(
    image_bgr: np.ndarray, vertices: List[List[float]], size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Cắt và nắn thẳng góc nhìn (Warp Transform) dựa trên các đỉnh tam giác/tứ giác.

    Chuẩn hóa hình ảnh biển báo bị nghiêng góc nhìn trước khi trích xuất HOG.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        vertices (List[List[float]]): Danh sách các đỉnh (3 cho tam giác, 4 cho tứ giác).
        size (Tuple[int, int]): Kích thước (w, h) mong muốn.

    Returns:
        Optional[np.ndarray]: Vùng ảnh đã biến đổi nắn thẳng (hoặc None nếu lỗi).
    """
    pts = np.array(vertices, dtype=np.float32)
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0 or len(pts) not in (3, 4):
        return None

    try:
        if len(pts) == 3:
            pts_ordered = _order_triangle_pts(pts)
            dst = np.array([[w / 2.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32)
            matrix = cv2.getAffineTransform(pts_ordered, dst)
            return cv2.warpAffine(image_bgr, matrix, (w, h))
        else:
            pts_ordered = _order_rect_pts(pts)
            dst = np.array(
                [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(pts_ordered, dst)
            return cv2.warpPerspective(image_bgr, matrix, (w, h))
    except cv2.error:
        return None


def is_valid_roi(
    w: int,
    h: int,
    min_w: int = 10,
    min_h: int = 10,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.0,
) -> Tuple[bool, str]:
    """Kiểm tra tính hợp lệ về kích thước và tỷ lệ khung hình của ROI.

    Args:
        w (int): Chiều rộng.
        h (int): Chiều cao.
        min_w (int): Chiều rộng tối thiểu. Mặc định 10.
        min_h (int): Chiều cao tối thiểu. Mặc định 10.
        min_aspect_ratio (float): Aspect ratio tối thiểu (w/h). Mặc định 0.3.
        max_aspect_ratio (float): Aspect ratio tối đa (w/h). Mặc định 3.0.

    Returns:
        Tuple[bool, str]: (Hợp lệ hay không, Lý do nếu không hợp lệ).
    """
    if w <= 0 or h <= 0:
        return False, "kích thước không hợp lệ (<= 0)"
    if w < min_w or h < min_h:
        return False, f"quá nhỏ ({w}x{h} < {min_w}x{min_h})"
    aspect_ratio = w / float(h)
    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return False, f"tỷ lệ khung hình bất thường ({aspect_ratio:.2f})"
    return True, "ok"


def resize_roi(
    crop: np.ndarray, size: Tuple[int, int] = (64, 64), interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """Thay đổi kích thước ROI về chuẩn $64 \\times 64$ pixels trước khi trích xuất HOG."""
    if crop is None or crop.size == 0:
        raise ValueError("Crop rỗng không thể resize")
    return cv2.resize(crop, size, interpolation=interpolation)


def extract_rois(
    image_bgr: np.ndarray,
    contour_items: List[Dict[str, Any]],
    min_w: int = 10,
    min_h: int = 10,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Cắt và xác minh danh sách các ROI ứng viên.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        contour_items (List[Dict[str, Any]]): Danh sách các vùng ứng viên.
        min_w (int): Chiều rộng tối thiểu.
        min_h (int): Chiều cao tối thiểu.
        min_aspect_ratio (float): Aspect ratio tối thiểu.
        max_aspect_ratio (float): Aspect ratio tối đa.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (ROI hợp lệ có key 'crop', ROI bị loại).
    """
    valid_rois: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for i, item in enumerate(contour_items):
        x, y, w, h = item["bounding_box"]
        ok, reason = is_valid_roi(w, h, min_w, min_h, min_aspect_ratio, max_aspect_ratio)
        if not ok:
            rejected.append({"index": i, "bounding_box": [x, y, w, h], "reason": reason})
            continue

        vertices = item.get("vertices")
        crop = None
        if vertices is not None and len(vertices) in (3, 4):
            crop = crop_roi_warped(image_bgr, vertices, size=(w, h))

        if crop is None or crop.size == 0:
            crop = crop_roi(image_bgr, x, y, w, h)

        if crop is None or crop.size == 0:
            rejected.append({"index": i, "bounding_box": [x, y, w, h], "reason": "crop rỗng"})
            continue

        valid_rois.append({**item, "crop": crop})

    return valid_rois, rejected


def draw_rois(
    image_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ bounding box của các ROI lên ảnh BGR."""
    result = image_bgr.copy()
    for r in rois:
        x, y, w, h = r["bounding_box"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def compute_proposal_quality_score(item: Dict[str, Any]) -> float:
    """Trả về điểm tin cậy của candidate (Giữ tương thích ngược)."""
    return float(item.get("confidence", 1.0))


def merge_candidates(
    primary_items: List[Dict[str, Any]],
    secondary_items: List[Dict[str, Any]],
    iou_dedup_threshold: Optional[float] = 0.45,
) -> List[Dict[str, Any]]:
    """Gộp các ứng viên từ nhiều bộ phát hiện với cơ chế khử trùng lặp IoU."""
    merged: List[Dict[str, Any]] = [dict(it) for it in primary_items]

    if iou_dedup_threshold is None:
        merged.extend([dict(it) for it in secondary_items])
        return merged

    for sec in secondary_items:
        s_it = dict(sec)
        bx = s_it["bounding_box"]
        sbox = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])

        best_iou = 0.0
        best_match_idx = -1
        for idx, base_item in enumerate(merged):
            bb = base_item["bounding_box"]
            bbox = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
            iou = compute_iou(sbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = idx

        if best_iou >= iou_dedup_threshold and best_match_idx >= 0:
            # Nếu trùng lặp cao, giữ lại thông tin đỉnh hoặc thuộc tính bổ trợ
            target = merged[best_match_idx]
            if "vertices" in s_it and "vertices" not in target:
                target["vertices"] = s_it["vertices"]
        else:
            merged.append(s_it)

    return merged


def apply_nms(
    items: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    prioritize_source: bool = False,
) -> List[Dict[str, Any]]:
    """Áp dụng thuật toán NMS chuẩn để loại bỏ các bounding box đè lên nhau.

    Sắp xếp ứng viên theo độ tin cậy mô hình (model_score / confidence) và diện tích.

    Args:
        items (List[Dict[str, Any]]): Danh sách ứng viên.
        iou_threshold (float): Ngưỡng IoU coi là chồng lấp. Mặc định 0.5.
        prioritize_source (bool): Giữ tương thích ngược.

    Returns:
        List[Dict[str, Any]]: Danh sách các ứng viên duy nhất sau NMS.
    """
    if not items:
        return items

    def score_of(it: Dict[str, Any]) -> Tuple[float, int]:
        conf = float(it.get("model_score", it.get("confidence", 0.0)))
        _, _, w, h = it["bounding_box"]
        return (conf, w * h)

    scored = []
    for it in items:
        x, y, w, h = it["bounding_box"]
        scored.append((it, (x, y, x + w, y + h), score_of(it)))

    # Sắp xếp giảm dần theo điểm tin cậy, sau đó là diện tích
    scored.sort(key=lambda t: t[2], reverse=True)

    kept: List[Dict[str, Any]] = []
    while scored:
        current_item, current_box, _ = scored.pop(0)
        kept.append(current_item)
        scored = [
            (it, box, s) for (it, box, s) in scored if compute_iou(current_box, box) < iou_threshold
        ]

    return kept
