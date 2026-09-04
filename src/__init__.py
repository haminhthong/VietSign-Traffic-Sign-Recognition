"""VietSign Vision Package - Vietnamese Traffic Sign Recognition with Classical Computer Vision.

Cung cấp gói thư viện nhận dạng biển báo giao thông Việt Nam sử dụng các kỹ thuật
xử lý ảnh truyền thống (HSV, MSER, Canny, Hough, Polygon) và mô hình học máy 2 tầng (HOG + SVM).
"""

__version__ = "1.0.0"
__author__ = "VietSign Vision Team"

from src.classifier import load_model, save_model
from src.data_loader import load_config, load_image, save_image
from src.pipeline import (
    load_pipeline_config,
    load_pipeline_models,
    process_image_to_rois,
    run_pipeline_on_image,
)

__all__ = [
    "run_pipeline_on_image",
    "process_image_to_rois",
    "load_pipeline_config",
    "load_pipeline_models",
    "load_image",
    "save_image",
    "load_config",
    "load_model",
    "save_model",
]
