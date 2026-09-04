"""Module Xác Minh Hình Học: Biển Báo Hình Tròn (Task 3: Hough Circle Detection).

Sử dụng biến đổi Hough Circle (`cv2.HoughCircles`) để phát hiện các biển báo dạng hình tròn
(biển cấm, biển hiệu lệnh) và tính toán điểm tin cậy (Confidence Score) dựa trên biên Canny và tỷ lệ màu viền.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def preprocess_for_hough(image_bgr: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """Tiền xử lý ảnh trước khi chạy biến đổi Hough (Chuyển ảnh xám + Median Blur).

    Args:
        image_bgr (np.ndarray): Ảnh BGR đầu vào.
        blur_ksize (int): Kích thước kernel làm mịn Median Filter. Mặc định 5.

    Returns:
        np.ndarray: Ảnh xám đã làm mịn.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur_ksize)
    return gray


def compute_circle_confidence(
    edges: np.ndarray, x: int, y: int, r: int, n_samples: int = 72
) -> float:
    """Tính độ tin cậy của hình tròn dựa trên số lượng điểm biên khớp trên chu vi.

    Lấy mẫu `n_samples` điểm trên đường tròn bán kính `r` và kiểm tra điểm biên trong ma trận Canny.

    Args:
        edges (np.ndarray): Ma trận ảnh biên Canny nhị phân (0 hoặc 255).
        x (int): Tọa độ tâm X của hình tròn.
        y (int): Tọa độ tâm Y của hình tròn.
        r (int): Bán kính R của hình tròn.
        n_samples (int): Số mẫu lấy kiểm tra trên chu vi. Mặc định 72 điểm (mỗi 5 độ).

    Returns:
        float: Độ khớp biên trong khoảng [0.0, 1.0].
    """
    h, w = edges.shape[:2]
    hits = 0
    for i in range(n_samples):
        theta = 2.0 * np.pi * i / n_samples
        px = int(x + r * np.cos(theta))
        py = int(y + r * np.sin(theta))
        if 0 <= px < w and 0 <= py < h:
            region = edges[max(py - 1, 0) : py + 2, max(px - 1, 0) : px + 2]
            if region.size > 0 and region.max() > 0:
                hits += 1
    return float(hits / n_samples)


def _ring_color_ratio(image_bgr: np.ndarray, x: int, y: int, r: int, ring_width: int = 4) -> float:
    """Tính tỷ lệ pixel viền tròn có màu đặc trưng (Đỏ cho biển cấm, Xanh cho biển hiệu lệnh).

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        x (int): Tâm X.
        y (int): Tâm Y.
        r (int): Bán kính.
        ring_width (int): Độ rộng đường viền nhẫn lấy mẫu. Mặc định 4.

    Returns:
        float: Tỷ lệ màu hợp lệ trên vòng nhẫn viền (0.0 đến 1.0).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, thickness=ring_width)
    ring_px = hsv[mask == 255]
    if len(ring_px) == 0:
        return 0.0

    h, s, v = ring_px[:, 0], ring_px[:, 1], ring_px[:, 2]
    red = ((h <= 10) | (h >= 170)) & (s >= 60) & (v >= 60)
    blue = (h >= 100) & (h <= 130) & (s >= 60) & (v >= 60)
    return float(max(red.sum(), blue.sum()) / len(ring_px))


def detect_circles(
    gray: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    dp: float = 1.2,
    min_dist: float = 30.0,
    param1: float = 100.0,
    param2: float = 30.0,
    min_radius: int = 10,
    max_radius: int = 100,
    min_color_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Phát hiện và xác minh biển báo hình tròn bằng biến đổi Hough Circles.

    Args:
        gray (np.ndarray): Ảnh xám đã qua tiền xử lý.
        image_bgr (Optional[np.ndarray]): Ảnh BGR gốc để kiểm tra màu sắc viền nhẫn.
        dp (float): Tỉ số nghịch đảo độ phân giải tích lũy (Accumulator resolution).
        min_dist (float): Khoảng cách tối thiểu giữa tâm các hình tròn phát hiện được.
        param1 (float): Ngưỡng trên cho bộ dò biên Canny (ngưỡng dưới bằng param1/2).
        param2 (float): Ngưỡng tích lũy cho tâm hình tròn (accumulator threshold).
        min_radius (int): Bán kính nhỏ nhất cần tìm.
        max_radius (int): Bán kính lớn nhất cần tìm.
        min_color_ratio (Optional[float]): Tỷ lệ màu tối thiểu trên viền nhẫn.

    Returns:
        List[Dict[str, Any]]: Danh sách từ điển các hình tròn đủ điều kiện gồm tâm x, y, radius, bounding_box, confidence.
    """
    if gray is None or gray.size == 0:
        return []

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    results: List[Dict[str, Any]] = []
    if circles is None:
        return results

    edges = cv2.Canny(gray, int(param1 // 2), int(param1))
    circles_arr = np.round(circles[0, :]).astype(int)

    for x, y, r in circles_arr:
        color_ratio = None
        if image_bgr is not None:
            color_ratio = _ring_color_ratio(image_bgr, x, y, r)
            if min_color_ratio is not None and color_ratio < min_color_ratio:
                continue

        edge_conf = compute_circle_confidence(edges, x, y, r)
        confidence = (
            edge_conf * (0.5 + 0.5 * min(color_ratio, 1.0))
            if color_ratio is not None
            else edge_conf
        )

        height, width = gray.shape[:2]
        bbox_x, bbox_y = max(int(x - r), 0), max(int(y - r), 0)
        bbox_x2, bbox_y2 = min(int(x + r), width), min(int(y + r), height)
        bbox_w, bbox_h = bbox_x2 - bbox_x, bbox_y2 - bbox_y

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        results.append(
            {
                "x": int(x),
                "y": int(y),
                "radius": int(r),
                "bounding_box": [bbox_x, bbox_y, bbox_w, bbox_h],
                "confidence": round(float(confidence), 4),
                "source": "circle",
            }
        )

    return results


def draw_circles(
    image_bgr: np.ndarray,
    circles: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ các đường tròn phát hiện được lên ảnh BGR.

    Args:
        image_bgr (np.ndarray): Ảnh gốc.
        circles (List[Dict[str, Any]]): Danh sách thông tin hình tròn từ detect_circles.
        color (Tuple[int, int, int]): Màu sắc đường vẽ (BGR). Mặc định tím hồng.
        thickness (int): Độ dày nét vẽ.

    Returns:
        np.ndarray: Ảnh mới đã vẽ hình tròn và chấm tâm.
    """
    result = image_bgr.copy()
    for c in circles:
        cv2.circle(result, (c["x"], c["y"]), c["radius"], color, thickness)
        cv2.circle(result, (c["x"], c["y"]), 2, (0, 0, 255), 3)
    return result
