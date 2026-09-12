"""Module Luồng Phân Loại & Phát Hiện Hoàn Chỉnh (End-to-End Pipeline).

Quy trình nhận dạng biển báo giao thông Việt Nam bằng Classical Computer Vision:
1. Tiền xử lý ảnh: Khử nhiễu Median Filter + Cân bằng tương phản tự thích nghi (CLAHE trên kênh L).
2. Phát hiện ứng viên (Candidate Generation): Phân đoạn màu HSV (Đỏ, Vàng, Xanh) + Lọc hình học (Contour Circularity & Polygons).
3. Chuẩn hóa ROI (ROI Normalization): Cắt vùng, kiểm tra kích thước / tỷ lệ khung hình và chuẩn hóa $64 \\times 64$.
4. Trích xuất đặc trưng HOG (Histogram of Oriented Gradients): Vector 1.764 chiều mô tả cạnh và hướng gradient.
5. Phân loại 2 tầng SVM:
   - Tầng 1: Binary SVM lọc bỏ vùng nền (Sign vs Background).
   - Tầng 2: Multiclass SVM nhận diện chính xác 52 lớp biển báo.
6. Hậu xử lý NMS (Non-Maximum Suppression): Loại bỏ trùng lặp và xuất kết quả.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.classifier import load_model, predict_proba_safe
from src.data_loader import load_config, load_image
from src.feature_extraction import extract_hog_features
from src.preprocessing import preprocess_image
from src.roi_extraction import apply_nms, extract_rois
from src.segmentation import generate_combined_mask, get_hsv_ranges
from src.task2_union import build_union_boxes


def _resolve_project_path(path_value: Union[str, Path], project_root: Path) -> Path:
    """Chuẩn hóa đường dẫn tương đối sang tuyệt đối từ gốc dự án."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def _load_required_model(path_value: Union[str, Path], model_name: str) -> Tuple[Any, Any]:
    """Nạp model bắt buộc và cung cấp thông báo lỗi có ngữ cảnh."""
    model_path = Path(path_value)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {model_name} tại: {model_path}. "
            "Vui lòng huấn luyện mô hình hoặc kiểm tra lại đường dẫn."
        )
    return load_model(model_path)


def load_pipeline_models(params: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Nạp hai model SVM (Binary và Multiclass) để tái sử dụng."""
    clf_cfg = params.get("classifier", params.get("task6", {}))
    model_bin, scaler_bin = _load_required_model(clf_cfg["model_bin_path"], "mô hình Tầng 1")
    model_multi, scaler_multi = _load_required_model(clf_cfg["model_multi_path"], "mô hình Tầng 2")
    return model_bin, scaler_bin, model_multi, scaler_multi


def load_pipeline_config(
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Tải và chuẩn hóa cấu hình từ config.yaml cho toàn bộ pipeline.

    Hỗ trợ cả schema sạch mới (preprocessing, candidate, shape, roi, hog, classifier)
    và schema cũ (task1..task6) để đảm bảo tính tương thích ngược tuyệt đối.
    """
    cfg, project_root, config_path = load_config(project_root)

    # Đọc theo schema mới hoặc fallback schema cũ
    prep_cfg = cfg.get("preprocessing", cfg.get("task1", {}))
    cand_cfg = cfg.get("candidate", cfg.get("task2_union", cfg.get("task2", {})))
    shape_cfg = cfg.get("shape", cfg.get("task3_shape", {}))
    roi_cfg = cfg.get("roi", cfg.get("task4", {}))
    hog_cfg = cfg.get("hog", cfg.get("task5", {}).get("hog_params", cfg.get("task5", {})))
    clf_cfg = cfg.get("classifier", cfg.get("task6", {}))

    # Cấu hình ngưỡng HSV nếu có khai báo trực tiếp
    configured_hsv_ranges = None
    required_hsv_keys = {
        "red_lower1",
        "red_upper1",
        "red_lower2",
        "red_upper2",
        "yellow_lower",
        "yellow_upper",
        "blue_lower",
        "blue_upper",
    }
    if required_hsv_keys.issubset(cand_cfg):
        configured_hsv_ranges = {
            "red1": (list(cand_cfg["red_lower1"]), list(cand_cfg["red_upper1"])),
            "red2": (list(cand_cfg["red_lower2"]), list(cand_cfg["red_upper2"])),
            "yellow": (list(cand_cfg["yellow_lower"]), list(cand_cfg["yellow_upper"])),
            "blue": (list(cand_cfg["blue_lower"]), list(cand_cfg["blue_upper"])),
        }

    default_bin_path = project_root / "outputs" / "models" / "svm_binary.joblib"
    default_multi_path = project_root / "outputs" / "models" / "svm_multiclass.joblib"

    params = {
        "preprocessing": {
            "kernel_size": prep_cfg.get("median_kernel", 3),
            "tile_grid_size": tuple(prep_cfg.get("tile_grid_size", [8, 8])),
            "clip_limit": prep_cfg.get("clahe_clip_limit", 2.0),
        },
        "candidate": {
            "min_area": cand_cfg.get("min_area", 150.0),
            "max_area_ratio": cand_cfg.get("max_area_ratio", 0.35),
            "aspect_ratio_range": tuple(
                cand_cfg.get("aspect_ratio_range", cand_cfg.get("hsv_ar_range", [0.4, 2.5]))
            ),
            "min_extent": cand_cfg.get("min_extent", cand_cfg.get("hsv_min_extent", 0.20)),
            "nms_iou_threshold": cand_cfg.get(
                "nms_iou_threshold", cand_cfg.get("nms_iou_thresh", 0.4)
            ),
            "hsv_ranges": configured_hsv_ranges,
        },
        "shape": {
            "min_circularity": shape_cfg.get("min_circularity", 0.65),
            "triangle_min_angle": shape_cfg.get("triangle_min_angle", 12.0),
            "triangle_max_side_ratio": shape_cfg.get("triangle_max_side_ratio", 4.0),
            "rectangle_min_extent": shape_cfg.get("rectangle_min_extent", 0.65),
            "rectangle_max_aspect": shape_cfg.get("rectangle_max_aspect", 2.5),
        },
        "roi": {
            "min_w": roi_cfg.get("min_w", 12),
            "min_h": roi_cfg.get("min_h", 12),
            "min_aspect_ratio": roi_cfg.get("min_aspect_ratio", 0.4),
            "max_aspect_ratio": roi_cfg.get("max_aspect_ratio", 2.5),
            "resize": tuple(roi_cfg.get("resize", [64, 64])),
            "nms_iou_threshold": roi_cfg.get("nms_iou_threshold", 0.4),
        },
        "hog": {
            "orientations": hog_cfg.get("orientations", 9),
            "pixels_per_cell": tuple(hog_cfg.get("pixels_per_cell", [8, 8])),
            "cells_per_block": tuple(hog_cfg.get("cells_per_block", [2, 2])),
        },
        "classifier": {
            "model_bin_path": str(
                _resolve_project_path(clf_cfg.get("model_bin_path", default_bin_path), project_root)
            ),
            "model_multi_path": str(
                _resolve_project_path(
                    clf_cfg.get("model_multi_path", default_multi_path), project_root
                )
            ),
            "bin_confidence_threshold": clf_cfg.get(
                "bin_confidence_threshold", clf_cfg.get("bin_thr", 0.5)
            ),
            "multi_confidence_threshold": clf_cfg.get(
                "multi_confidence_threshold", clf_cfg.get("multi_thr", 0.3)
            ),
        },
    }

    # Bổ sung các key task cũ để tương thích ngược 100% với mã gọi ngoài
    params["task1"] = params["preprocessing"]
    params["task2"] = {
        "hsv_ranges": configured_hsv_ranges,
        "debug_mask_set_number": 1,
        "union_hsv_sets": False,
    }
    params["task2_union"] = {
        "nms_iou_thresh": params["candidate"]["nms_iou_threshold"],
        "hsv_ar_range": params["candidate"]["aspect_ratio_range"],
        "hsv_min_extent": params["candidate"]["min_extent"],
        "hsv_set": 1,
        "mser_delta": 5,
        "canny_low": 50,
        "canny_high": 150,
    }
    params["task3_shape"] = {
        "circle": {"min_circularity": params["shape"]["min_circularity"]},
        "triangle": {"min_angle_deg": params["shape"]["triangle_min_angle"]},
        "rectangle": {"min_extent": params["shape"]["rectangle_min_extent"]},
    }
    params["task4"] = params["roi"]
    params["task5"] = {"hog_params": params["hog"]}
    params["task6"] = params["classifier"]

    return params, project_root, config_path


def process_image_to_rois(
    image_bgr: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tiền xử lý và trích xuất danh sách ROI ứng viên (Phục vụ cả chế độ --detect-only).

    Quy trình:
    1. Tiền xử lý (Median Filter + LAB CLAHE).
    2. Phân đoạn màu HSV tạo mặt nạ màu biển báo.
    3. Tìm contours và lọc sơ bộ theo diện tích, tỷ lệ khung hình.
    4. Cắt và xác thực ROI, áp dụng NMS.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")

    prep = params.get("preprocessing", params.get("task1", {}))
    enhanced = preprocess_image(
        image_bgr,
        kernel_size=prep.get("kernel_size", 3),
        tile_grid_size=prep.get("tile_grid_size", (8, 8)),
        clip_limit=prep.get("clip_limit", 2.0),
    )

    cand = params.get("candidate", params.get("task2_union", {}))
    hsv_ranges = cand.get("hsv_ranges")
    iou_thresh = cand.get("nms_iou_threshold", cand.get("nms_iou_thresh", 0.4))
    ar_range = cand.get("aspect_ratio_range", cand.get("hsv_ar_range", (0.4, 2.5)))
    min_extent = cand.get("min_extent", cand.get("hsv_min_extent", 0.20))

    union_boxes, union_components = build_union_boxes(
        enhanced,
        iou_thresh=iou_thresh,
        hsv_ar_range=ar_range,
        hsv_min_extent=min_extent,
        hsv_ranges=hsv_ranges,
    )

    candidate_items: List[Dict[str, Any]] = []
    for x, y, w, h in union_boxes:
        candidate_items.append(
            {
                "bounding_box": [x, y, w, h],
                "source": "color_contour",
            }
        )

    # Mặt nạ nhị phân phục vụ debug và hiển thị trực quan
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    ranges = hsv_ranges if hsv_ranges is not None else get_hsv_ranges(1)
    mask_debug, _, _, _ = generate_combined_mask(hsv, ranges)

    roi_cfg = params.get("roi", params.get("task4", {}))
    rois, rejected = extract_rois(
        enhanced,
        candidate_items,
        min_w=roi_cfg.get("min_w", 12),
        min_h=roi_cfg.get("min_h", 12),
        min_aspect_ratio=roi_cfg.get("min_aspect_ratio", 0.4),
        max_aspect_ratio=roi_cfg.get("max_aspect_ratio", 2.5),
    )

    # Áp dụng NMS để làm sạch các vùng ứng viên đè lên nhau
    rois = apply_nms(rois, iou_threshold=roi_cfg.get("nms_iou_threshold", 0.4))

    return enhanced, mask_debug, union_components, rois, rejected


def _binary_sign_proba(
    clf_bin: Any, scaler_bin: Any, feats: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính xác suất vùng ROI là biển báo từ Binary SVM Tầng 1."""
    X = scaler_bin.transform(feats)
    pred = clf_bin.predict(X)
    if hasattr(clf_bin, "predict_proba"):
        try:
            proba = clf_bin.predict_proba(X)
            classes = list(clf_bin.classes_)
            if 1 in classes:
                p_sign = proba[:, classes.index(1)]
            else:
                p_sign = proba.max(axis=1)
            return p_sign, pred
        except Exception:
            pass

    labels, conf = predict_proba_safe(clf_bin, scaler_bin, feats)
    p_sign = np.where(labels == 1, conf, 1.0 - conf)
    return p_sign, labels


def classify_rois(
    rois: List[Dict[str, Any]],
    params: Dict[str, Any],
    clf_bin: Any,
    scaler_bin: Any,
    clf_multi: Any,
    scaler_multi: Any,
) -> List[Dict[str, Any]]:
    """Trích xuất HOG và phân loại 2 tầng SVM cho danh sách ROI ứng viên.

    Tầng 1 (Binary SVM): Lọc bỏ các vùng nền gây báo giả (nhãn 0: background, 1: sign).
    Tầng 2 (Multiclass SVM): Nhận diện cụ thể lớp biển báo (52 lớp).
    """
    if not rois:
        return []

    roi_cfg = params.get("roi", params.get("task4", {}))
    hog_cfg = params.get("hog", params.get("task5", {}).get("hog_params", {}))
    clf_cfg = params.get("classifier", params.get("task6", {}))

    resize_to = roi_cfg.get("resize", (64, 64))
    bin_thr = clf_cfg.get("bin_confidence_threshold", 0.5)
    multi_thr = clf_cfg.get("multi_confidence_threshold", 0.3)

    feats = []
    for roi in rois:
        hog_feat, _ = extract_hog_features(
            roi["crop"],
            resize_to=resize_to,
            visualize=False,
            orientations=hog_cfg.get("orientations", 9),
            pixels_per_cell=hog_cfg.get("pixels_per_cell", (8, 8)),
            cells_per_block=hog_cfg.get("cells_per_block", (2, 2)),
        )
        feats.append(hog_feat)
    feats_arr = np.array(feats)

    # Tầng 1: Lọc nền
    p_sign, _ = _binary_sign_proba(clf_bin, scaler_bin, feats_arr)
    sign_mask = p_sign >= bin_thr
    sign_idx = np.where(sign_mask)[0]

    if len(sign_idx) == 0:
        return []

    # Tầng 2: Nhận diện 52 lớp
    pred_multi, conf_multi = predict_proba_safe(clf_multi, scaler_multi, feats_arr[sign_idx])

    results: List[Dict[str, Any]] = []
    for idx, pred_class, conf in zip(sign_idx, pred_multi, conf_multi):
        if conf < multi_thr:
            continue
        roi_item = rois[idx]
        results.append(
            {
                "bounding_box": roi_item["bounding_box"],
                "predicted_class": int(pred_class),
                "confidence": float(conf),
                "bin_confidence": float(p_sign[idx]),
            }
        )

    final_iou_thr = roi_cfg.get("nms_iou_threshold", 0.4)
    results = apply_nms(results, iou_threshold=final_iou_thr)
    return results


def run_pipeline_on_image(
    image_path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None,
    model_bin: Optional[Any] = None,
    scaler_bin: Optional[Any] = None,
    model_multi: Optional[Any] = None,
    scaler_multi: Optional[Any] = None,
    return_debug: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]],
]:
    """Chạy quy trình nhận dạng biển báo hoàn chỉnh cho một ảnh.

    Args:
        image_path (Union[str, Path]): Đường dẫn tệp ảnh.
        project_root (Optional[Union[str, Path]]): Thư mục gốc dự án.
        model_bin (Optional[Any]): Mô hình Binary SVM nạp sẵn.
        scaler_bin (Optional[Any]): Scaler Binary nạp sẵn.
        model_multi (Optional[Any]): Mô hình Multiclass SVM nạp sẵn.
        scaler_multi (Optional[Any]): Scaler Multiclass nạp sẵn.
        return_debug (bool): Có trả về thông tin debug hay không.

    Returns:
        Tuple: (Ảnh đã tăng cường, Mặt nạ nhị phân, Danh sách biển báo phát hiện được, [Debug Info]).
    """
    params, project_root, _ = load_pipeline_config(project_root)
    clf_cfg = params.get("classifier", params.get("task6", {}))

    if model_bin is None or scaler_bin is None:
        model_bin, scaler_bin = _load_required_model(clf_cfg["model_bin_path"], "mô hình Tầng 1")

    if model_multi is None or scaler_multi is None:
        model_multi, scaler_multi = _load_required_model(
            clf_cfg["model_multi_path"], "mô hình Tầng 2"
        )

    image_bgr = load_image(image_path)
    if image_bgr is None:
        raise ValueError(f"Không thể nạp tệp ảnh tại đường dẫn: {image_path}")

    enhanced, mask_debug, union_components, rois, rejected = process_image_to_rois(
        image_bgr, params
    )
    detections = classify_rois(rois, params, model_bin, scaler_bin, model_multi, scaler_multi)

    if return_debug:
        debug_info = {
            "rois_extracted": len(rois),
            "rois_rejected": len(rejected),
            "final_detections": len(detections),
            "rejected_rois": rejected,
        }
        return enhanced, mask_debug, detections, debug_info

    return enhanced, mask_debug, detections


_VN_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}


def _get_vn_font(size: int = 18) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """Lấy phông chữ hỗ trợ gõ Tiếng Việt Unicode."""
    if size in _VN_FONT_CACHE:
        return _VN_FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = None
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                pass
    if font is None:
        font = ImageFont.load_default()

    if isinstance(font, ImageFont.FreeTypeFont):
        _VN_FONT_CACHE[size] = font

    return font


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Vẽ bounding box và nhãn lớp lên ảnh BGR."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_vn_font(18)
    color_rgb = (color[2], color[1], color[0])

    for det in detections:
        x, y, w, h = det["bounding_box"]
        cls = det["predicted_class"]
        conf = det.get("confidence")
        base = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(cls)
        label = f"{base} {conf:.2f}" if conf is not None else base

        draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=2)
        draw.text((x, max(y - 22, 0)), label, font=font, fill=color_rgb)

    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def draw_candidates(
    image_bgr: np.ndarray,
    candidates: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """Vẽ các hộp ứng viên phục vụ chế độ demo chỉ phát hiện (--detect-only)."""
    result = image_bgr.copy()
    for candidate in candidates:
        x, y, w, h = candidate["bounding_box"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        source = candidate.get("source", "candidate")
        cv2.putText(
            result,
            source,
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return result
