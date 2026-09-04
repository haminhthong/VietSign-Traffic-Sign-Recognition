"""Module Trích Xuất Đặc Trưng HOG (Task 5: Feature Extraction).

Thực hiện trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ vùng ROI đã chuẩn hóa ($64 \\times 64$).
Vector đặc trưng có độ dài $1.764$ chiều, phục vụ cho mô hình phân loại SVM 2 tầng.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog


def extract_hog_features(
    image_bgr: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    resize_to: Tuple[int, int] = (64, 64),
    visualize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Trích xuất vector đặc trưng HOG và ảnh minh họa độ dốc gradient (khi visualize=True).

    Công thức tính số chiều vector HOG:
    - Ảnh kích thước $64 \\times 64$ với `pixels_per_cell` = (8, 8) $\\rightarrow 8 \\times 8 = 64$ ô (cells).
    - Lưới block `cells_per_block` = (2, 2) $\\rightarrow (8 - 1) \\times (8 - 1) = 49$ khối (blocks).
    - Mỗi block chứa $2 \\times 2 \\times 9 = 36$ giá trị.
    - Tổng số chiều đặc trưng: $49 \\times 36 = 1.764$ chiều.

    Args:
        image_bgr (np.ndarray): Vùng ảnh ROI định dạng BGR.
        orientations (int): Số lượng thùng góc hướng gradient (bins). Mặc định 9.
        pixels_per_cell (Tuple[int, int]): Kích thước ô vuông tính gradient (pixel). Mặc định (8, 8).
        cells_per_block (Tuple[int, int]): Số ô trong một khối chuẩn hóa. Mặc định (2, 2).
        resize_to (Tuple[int, int]): Kích thước ảnh chuẩn hóa trước khi trích xuất. Mặc định (64, 64).
        visualize (bool): Trả về ảnh bản đồ HOG minh họa hay không. Mặc định True.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: (Vector đặc trưng HOG 1.764 chiều, Ảnh HOG map nếu visualize=True).

    Raises:
        ValueError: Nếu ảnh đầu vào rỗng hoặc không hợp lệ.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Không thể trích xuất đặc trưng HOG từ ảnh rỗng")

    # Chuẩn hóa về kích thước cố định 64x64
    img_resized = cv2.resize(image_bgr, resize_to, interpolation=cv2.INTER_AREA)

    # Chuyển về ảnh xám để tính gradient độ sáng
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    result = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )

    if visualize:
        features, hog_image = result
        return features, hog_image
    return result, None
