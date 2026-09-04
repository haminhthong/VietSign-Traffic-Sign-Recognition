"""Module Trích Xuất Vùng Quan Tâm (Task 4: ROI Extraction & NMS Filtering).

Thực hiện cắt (crop), biến đổi góc nhìn/xoay (Warp Perspective / Affine Transform cho đa giác),
chuẩn hóa kích thước ROI ($64 \\times 64$), kiểm tra tỷ lệ khung hình (Aspect Ratio),
và áp dụng thuật toán NMS (Non-Maximum Suppression) nâng cao để lọc các vùng ứng viên.
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
    """Sắp xếp các đỉnh tam giác bắt đầu từ đỉnh trên cùng (Top vertex)."""
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
    """Cắt và nắn thẳng góc nhìn (Warp Transform) dựa trên các đỉnh tam giác/tứ giác thực tế.

    Giúp chuẩn hóa hình ảnh biển báo bị nghiêng hoặc biến dạng góc nhìn trước khi trích xuất HOG.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        vertices (List[List[float]]): Danh sách các đỉnh (3 cho tam giác, 4 cho tứ giác).
        size (Tuple[int, int]): Kích thước chiều rộng và chiều cao mong muốn (w, h).

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
                [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32
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
        Tuple[bool, str]: (Hợp lệ hay không, Lý do từ chối nếu không hợp lệ).
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
    """Thay đổi kích thước ROI về chuẩn $64 \\times 64$ pixels trước khi trích xuất HOG.

    Args:
        crop (np.ndarray): Ảnh crop BGR.
        size (Tuple[int, int]): Kích thước mới. Mặc định (64, 64).
        interpolation (int): Phương pháp nội suy OpenCV. Mặc định cv2.INTER_AREA cho thu nhỏ.

    Returns:
        np.ndarray: Ảnh đã resize $64 \\times 64$.
    """
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
        contour_items (List[Dict[str, Any]]): Danh sách từ điển các vùng ứng viên.
        min_w (int): Chiều rộng tối thiểu.
        min_h (int): Chiều cao tối thiểu.
        min_aspect_ratio (float): Aspect ratio tối thiểu.
        max_aspect_ratio (float): Aspect ratio tối đa.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (Danh sách ROI hợp lệ có kèm key 'crop', Danh sách bị loại).
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
    """Vẽ bounding box của các ROI lên ảnh BGR.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        rois (List[Dict[str, Any]]): Danh sách các ROI.
        color (Tuple[int, int, int]): Màu khung (BGR). Mặc định xanh lá.
        thickness (int): Độ dày nét vẽ.

    Returns:
        np.ndarray: Ảnh mới đã vẽ khung.
    """
    result = image_bgr.copy()
    for r in rois:
        x, y, w, h = r["bounding_box"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def merge_candidates(
    contour_items: List[Dict[str, Any]],
    circle_items: List[Dict[str, Any]],
    iou_dedup_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Gộp các ứng viên từ phân đoạn màu/MSER/Polygon và Hough Circle.

    Args:
        contour_items (List[Dict[str, Any]]): Danh sách từ phân đoạn contour.
        circle_items (List[Dict[str, Any]]): Danh sách từ Hough Circle.
        iou_dedup_threshold (Optional[float]): Ngưỡng khử trùng lặp IoU.

    Returns:
        List[Dict[str, Any]]: Danh sách ứng viên đã gộp.
    """
    merged = [{**c, "source": c.get("source", "contour")} for c in contour_items]

    if iou_dedup_threshold is None:
        merged.extend({**c, "source": c.get("source", "circle")} for c in circle_items)
        return merged

    contour_boxes = [
        (
            c["bounding_box"][0],
            c["bounding_box"][1],
            c["bounding_box"][0] + c["bounding_box"][2],
            c["bounding_box"][1] + c["bounding_box"][3],
        )
        for c in contour_items
    ]
    for circle in circle_items:
        bx = circle["bounding_box"]
        cbox = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
        is_duplicate = any(compute_iou(cbox, cb) >= iou_dedup_threshold for cb in contour_boxes)
        if not is_duplicate:
            merged.append({**circle, "source": circle.get("source", "circle")})

    return merged


def apply_nms(
    items: List[Dict[str, Any]], iou_threshold: float = 0.5, prioritize_source: bool = True
) -> List[Dict[str, Any]]:
    """Áp dụng thuật toán NMS (Non-Maximum Suppression) để lọc các ứng viên đè lên nhau.

    Sắp xếp các ứng viên theo thứ tự ưu tiên điểm số (Confidence / Source Priority)
    và giữ lại ứng viên tốt nhất khi chỉ số IoU với các ứng viên khác > iou_threshold.

    Args:
        items (List[Dict[str, Any]]): Danh sách ứng viên.
        iou_threshold (float): Ngưỡng IoU coi là chồng lấp. Mặc định 0.5.
        prioritize_source (bool): Ưu tiên nguồn contour trước circle hay không. Mặc định True.

    Returns:
        List[Dict[str, Any]]: Danh sách các ứng viên duy nhất đã qua lọc.
    """
    if not items:
        return items

    def score_of(it: Dict[str, Any]) -> Tuple[Any, ...]:
        conf = float(it.get("confidence", 0.0))
        _, _, w, h = it["bounding_box"]
        area = w * h
        aspect_penalty = abs(1.0 - (w / float(h) if h > 0 else 1.0))
        if prioritize_source:
            source_priority = 1 if it.get("source") == "contour" else 0
            return (source_priority, conf, area)
        return (round(conf - 0.15 * aspect_penalty, 4), area)

    scored = []
    for it in items:
        x, y, w, h = it["bounding_box"]
        scored.append((it, (x, y, x + w, y + h), score_of(it)))

    scored.sort(key=lambda t: t[2], reverse=True)

    kept: List[Dict[str, Any]] = []
    while scored:
        current_item, current_box, _ = scored.pop(0)
        kept.append(current_item)
        scored = [
            (it, box, s) for (it, box, s) in scored if compute_iou(current_box, box) < iou_threshold
        ]

    return kept
