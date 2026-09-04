"""Module Phân Tích Hình Học Đa Giác (Task 3: Polygon Detection).

Phát hiện các biển báo hình tam giác (biển cảnh báo nguy hiểm) và hình chữ nhật/tứ giác (biển chỉ dẫn)
sử dụng thuật toán xấp xỉ đa giác Ramer-Douglas-Peucker (`cv2.approxPolyDP`) và loại bỏ các đỉnh thẳng hàng.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def preprocess_for_polygon(image_bgr: np.ndarray) -> np.ndarray:
    """Tiền xử lý chuyển ảnh BGR sang ảnh xám trước khi dò biên đa giác.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.

    Returns:
        np.ndarray: Ảnh xám uint8.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _polygon_interior_angles(approx: np.ndarray) -> List[float]:
    """Tính các góc trong của đa giác.

    Args:
        approx (np.ndarray): Mảng các đỉnh của đa giác shape (N, 1, 2) hoặc (N, 2).

    Returns:
        List[float]: Danh sách các góc trong tính bằng độ (Degrees).
    """
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    angles: List[float] = []
    for i in range(n):
        p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = p_prev - p_curr, p_next - p_curr
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_angle))))
    return angles


def _merge_collinear_vertices(approx: np.ndarray, angle_thresh_deg: float = 160.0) -> np.ndarray:
    """Gộp các đỉnh gần thẳng hàng (Collinear) để loại bỏ các đỉnh thừa sinh ra do nhiễu biên.

    Args:
        approx (np.ndarray): Đa giác gốc từ approxPolyDP.
        angle_thresh_deg (float): Ngưỡng góc bẹt (độ). Nếu góc trong > threshold thì gộp đỉnh.

    Returns:
        np.ndarray: Đa giác mới đã rút gọn đỉnh.
    """
    pts = approx.reshape(-1, 2)
    n = len(pts)
    if n <= 3:
        return approx

    angles = _polygon_interior_angles(approx)
    keep = [pts[i] for i in range(n) if angles[i] < angle_thresh_deg]
    if len(keep) < 3:
        return approx  # Tránh loại bỏ quá nhiều đỉnh gây hỏng đa giác

    return np.array(keep, dtype=approx.dtype).reshape(-1, 1, 2)


def is_roughly_equilateral(
    approx: np.ndarray, min_angle_deg: float = 12.0, max_side_ratio: float = 5.0
) -> bool:
    """Kiểm tra tam giác có gần đều/gần cân hay không (phù hợp biển cảnh báo nguy hiểm).

    Args:
        approx (np.ndarray): Tam giác 3 đỉnh.
        min_angle_deg (float): Góc trong nhỏ nhất cho phép. Mặc định 12 độ.
        max_side_ratio (float): Tỷ lệ cạnh dài nhất / cạnh ngắn nhất tối đa. Mặc định 5.0.

    Returns:
        bool: True nếu thỏa mãn điều kiện tam giác biển báo.
    """
    pts = approx.reshape(-1, 2).astype(np.float64)
    if len(pts) != 3:
        return False
    angles = _polygon_interior_angles(approx)
    if min(angles) < min_angle_deg:
        return False  # Góc quá nhọn do nhiễu
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
    if max(sides) / (min(sides) + 1e-6) > max_side_ratio:
        return False  # Cạnh quá chênh lệch
    return True


def is_roughly_rectangular(
    approx: np.ndarray,
    min_extent: float = 0.70,
    max_aspect: float = 5.0,
    angle_tol: Optional[float] = None,
) -> bool:
    """Kiểm tra tứ giác có gần hình chữ nhật hay không (phù hợp biển chỉ dẫn/lệnh).

    Args:
        approx (np.ndarray): Tứ giác 4 đỉnh.
        min_extent (float): Tỷ lệ diện tích đa giác / diện tích hình chữ nhật bao quanh tối thiểu.
        max_aspect (float): Tỷ lệ cạnh dài / cạnh ngắn tối đa.
        angle_tol (Optional[float]): Độ lệch góc vuông cho phép (ví dụ +-20 độ so với 90 độ).

    Returns:
        bool: True nếu thỏa mãn điều kiện hình chữ nhật.
    """
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) != 4:
        return False

    contour_area = cv2.contourArea(pts)
    if contour_area <= 0:
        return False

    (_, (w, h), _) = cv2.minAreaRect(pts)
    if w <= 0 or h <= 0:
        return False

    rect_area = w * h
    if (contour_area / rect_area) < min_extent:
        return False

    aspect = max(w, h) / min(w, h)
    if aspect > max_aspect:
        return False

    if angle_tol is not None:
        angles = _polygon_interior_angles(approx.reshape(-1, 1, 2))
        if any(abs(a - 90.0) > angle_tol for a in angles):
            return False

    return True


def _color_plausibility_triangle(
    image_bgr: np.ndarray,
    approx: np.ndarray,
    min_red_ratio: float = 0.15,
    min_yellow_fill: float = 0.35,
    border_thickness: int = 4,
) -> bool:
    """Xác minh sự hợp lý về màu sắc của tam giác biển báo (Viền đỏ + Ruột vàng).

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        approx (np.ndarray): Đa giác 3 đỉnh.
        min_red_ratio (float): Tỷ lệ màu đỏ trên đường viền tối thiểu.
        min_yellow_fill (float): Tỷ lệ màu vàng trong ruột tam giác tối thiểu.
        border_thickness (int): Độ dày viền lấy mẫu.

    Returns:
        bool: True nếu thỏa mãn tiêu chuẩn màu sắc tam giác cảnh báo.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pts = approx.reshape(-1, 1, 2).astype(np.int32)

    border_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.polylines(border_mask, [pts], isClosed=True, color=255, thickness=border_thickness)
    border_px = hsv[border_mask == 255]
    if len(border_px) == 0:
        return False

    h, s, v = border_px[:, 0], border_px[:, 1], border_px[:, 2]
    red = ((h <= 10) | (h >= 170)) & (s >= 60) & (v >= 60)
    if (red.sum() / len(border_px)) < min_red_ratio:
        return False

    fill_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(fill_mask, [pts], 255)
    interior_px = hsv[fill_mask == 255]
    if len(interior_px) == 0:
        return False

    hi, si, vi = interior_px[:, 0], interior_px[:, 1], interior_px[:, 2]
    yellow = (hi >= 15) & (hi <= 40) & (si >= 60) & (vi >= 80)
    yellow_fill_ratio = yellow.sum() / len(interior_px)

    return bool(yellow_fill_ratio >= min_yellow_fill)


def detect_polygons(
    gray: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    target_sides: int = 3,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area: float = 300.0,
    max_area_ratio: float = 0.3,
    approx_eps_ratio: float = 0.03,
    min_angle_deg: float = 12.0,
    max_side_ratio: float = 5.0,
    min_extent: float = 0.70,
    max_aspect: float = 5.0,
    angle_tol: Optional[float] = None,
    min_red_ratio: Optional[float] = None,
    min_yellow_fill: float = 0.35,
) -> List[Dict[str, Any]]:
    """Phát hiện các biển báo dạng đa giác (Tam giác hoặc Chữ nhật/Tứ giác).

    Args:
        gray (np.ndarray): Ảnh xám.
        image_bgr (Optional[np.ndarray]): Ảnh BGR gốc để kiểm định màu.
        target_sides (int): Số cạnh mục tiêu (3 cho Tam giác, 4 cho Tứ giác).
        canny_low (int): Ngưỡng Canny dưới.
        canny_high (int): Ngưỡng Canny trên.
        min_area (float): Diện tích nhỏ nhất.
        max_area_ratio (float): Tỷ lệ diện tích tối đa so với ảnh.
        approx_eps_ratio (float): Hệ số epsilon cho xấp xỉ đa giác approxPolyDP.
        min_angle_deg (float): Góc trong nhỏ nhất cho tam giác.
        max_side_ratio (float): Tỷ lệ cạnh tối đa.
        min_extent (float): Extent tối thiểu cho hình chữ nhật.
        max_aspect (float): Aspect ratio tối đa.
        angle_tol (Optional[float]): Độ lệch góc vuông tối đa cho hình chữ nhật.
        min_red_ratio (Optional[float]): Ngưỡng màu đỏ viền tam giác.
        min_yellow_fill (float): Ngưỡng màu vàng ruột tam giác.

    Returns:
        List[Dict[str, Any]]: Danh sách các kết quả đa giác tìm được.
    """
    if target_sides not in (3, 4):
        raise ValueError("target_sides chỉ hỗ trợ 3 (tam giác) hoặc 4 (tứ giác/hình chữ nhật)")

    if gray is None or gray.size == 0:
        return []

    shape_name = "triangle" if target_sides == 3 else "rectangle"
    h_img, w_img = gray.shape[:2]

    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results: List[Dict[str, Any]] = []
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < min_area or area > (h_img * w_img * max_area_ratio):
            continue

        peri = cv2.arcLength(hull, True)
        if peri == 0:
            continue

        approx = cv2.approxPolyDP(hull, approx_eps_ratio * peri, True)
        approx = _merge_collinear_vertices(approx)
        if len(approx) != target_sides:
            continue

        if target_sides == 3 and not is_roughly_equilateral(approx, min_angle_deg, max_side_ratio):
            continue

        if target_sides == 3 and min_red_ratio is not None and image_bgr is not None:
            if not _color_plausibility_triangle(image_bgr, approx, min_red_ratio, min_yellow_fill):
                continue

        if target_sides == 4 and not is_roughly_rectangular(
            approx, min_extent, max_aspect=max_aspect, angle_tol=angle_tol
        ):
            continue

        x, y, w, h = cv2.boundingRect(hull)
        if w == 0 or h == 0:
            continue

        confidence = float(area / (w * h))

        results.append(
            {
                "x": int(x + w / 2),
                "y": int(y + h / 2),
                "bounding_box": [int(x), int(y), int(w), int(h)],
                "confidence": round(confidence, 4),
                "vertices": approx.reshape(-1, 2).tolist(),
                "source": shape_name,
            }
        )

    return results


def draw_polygons(
    image_bgr: np.ndarray,
    polygons: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ các đa giác tìm thấy lên ảnh BGR.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        polygons (List[Dict[str, Any]]): Danh sách thông tin đa giác.
        color (Tuple[int, int, int]): Màu nét vẽ (BGR). Mặc định vàng.
        thickness (int): Độ dày đường vẽ.

    Returns:
        np.ndarray: Ảnh mới đã vẽ đa giác và điểm tâm.
    """
    result = image_bgr.copy()
    for p in polygons:
        pts = np.array(p["vertices"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
        cv2.circle(result, (p["x"], p["y"]), 2, (0, 0, 255), 3)
    return result
