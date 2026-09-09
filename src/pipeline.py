"""Module Luồng Phân Loại & Phát Hiện Tổng Thể (End-to-End Pipeline).

Kết nối toàn bộ hệ thống VietSign Vision:
1. Tiền xử lý ảnh (Median Filter + Dynamic CLAHE - Task 1)
2. Trích xuất ứng viên (HSV + MSER + Canny Hull Union - Task 2)
3. Xác minh hình học (Hough Circle + Tam giác/Tứ giác - Task 3)
4. Lọc NMS & Trích xuất ROI (Task 4)
5. Trích xuất đặc trưng HOG 1.764 chiều (Task 5)
6. Phân loại 2 tầng SVM (Binary Sign/Bg -> Multiclass 52 Lớp - Task 6)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.classifier import load_model, predict_proba_safe
from src.data_loader import load_config, load_image
from src.feature_extraction import extract_hog_features
from src.hough_detection import detect_circles, preprocess_for_hough
from src.polygon_detection import detect_polygons, preprocess_for_polygon
from src.preprocessing import preprocess_task1
from src.roi_extraction import apply_nms, extract_rois, merge_candidates
from src.segmentation import segment_task2
from src.task2_union import build_union_boxes
from src.utils import compute_iou


def _resolve_project_path(path_value: Union[str, Path], project_root: Path) -> Path:
    """Chuẩn hóa đường dẫn tương đối từ gốc dự án sang đường dẫn tuyệt đối."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def _load_required_model(path_value: Union[str, Path], model_name: str) -> Tuple[Any, Any]:
    """Nạp model bắt buộc và cung cấp lỗi có ngữ cảnh khi tệp chưa tồn tại."""
    model_path = Path(path_value)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {model_name} tại: {model_path}. "
            "Vui lòng tải model hoặc chạy 06_task6_svm.ipynb."
        )
    return load_model(model_path)


def load_pipeline_models(params: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Nạp hai model SVM đúng một lần để tái sử dụng khi xử lý nhiều ảnh."""
    model_bin, scaler_bin = _load_required_model(
        params["task6"]["model_bin_path"], "mô hình Tầng 1"
    )
    model_multi, scaler_multi = _load_required_model(
        params["task6"]["model_multi_path"], "mô hình Tầng 2"
    )
    return model_bin, scaler_bin, model_multi, scaler_multi


def load_pipeline_config(
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Tải và chuẩn hóa cấu hình từ tệp config.yaml cho toàn bộ pipeline.

    Args:
        project_root (Optional[Union[str, Path]]): Thư mục gốc dự án. Mặc định tự nhận diện.

    Returns:
        Tuple[Dict[str, Any], Path, Path]: (Tham số cấu hình pipeline, Đường dẫn gốc dự án, Đường dẫn tệp config).
    """
    cfg, project_root, config_path = load_config(project_root)

    t1 = cfg.get("task1", {})
    t2 = cfg.get("task2", {})
    t2_union = cfg.get("task2_union", {})
    t3_shape = cfg.get("task3_shape", {})
    t4 = cfg.get("task4", {})
    t5 = cfg.get("task5", {})
    t6 = cfg.get("task6", {})
    hog_cfg = t5.get("hog_params", {})

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
    if required_hsv_keys.issubset(t2):
        configured_hsv_ranges = {
            "red1": (list(t2["red_lower1"]), list(t2["red_upper1"])),
            "red2": (list(t2["red_lower2"]), list(t2["red_upper2"])),
            "yellow": (list(t2["yellow_lower"]), list(t2["yellow_upper"])),
            "blue": (list(t2["blue_lower"]), list(t2["blue_upper"])),
        }

    default_circle = dict(dp=1.2, min_dist=30, param1=120, param2=40, min_radius=10, max_radius=100)
    default_triangle = dict(
        approx_eps_ratio=0.12, max_side_ratio=5.0, min_red_ratio=None, min_yellow_fill=0.0
    )
    default_rectangle = dict(approx_eps_ratio=0.1, min_extent=0.70, max_aspect=5.0, angle_tol=None)

    default_bin_path = project_root / "outputs" / "models" / "svm_binary.joblib"
    default_multi_path = project_root / "outputs" / "models" / "svm_multiclass.joblib"

    params = {
        "task1": dict(
            kernel_size=t1.get("median_kernel", 3),
            tile_grid_size=tuple(t1.get("tile_grid_size", [8, 8])),
        ),
        "task2": dict(
            debug_mask_set_number=t2.get("hsv_set", 1),
            union_hsv_sets=bool(t2.get("union_hsv_sets", True)),
            include_achromatic=bool(t2.get("include_achromatic", False)),
            achromatic_dilate_ksize=int(t2.get("achromatic_dilate_ksize", 9)),
            fill_holes=bool(t2.get("fill_holes", True)),
            hsv_ranges=configured_hsv_ranges,
        ),
        "task2_union": dict(
            nms_iou_thresh=t2_union.get("nms_iou_thresh", 0.4),
            hsv_ar_range=tuple(t2_union.get("hsv_ar_range", [0.4, 2.5])),
            hsv_min_extent=t2_union.get("hsv_min_extent", 0.25),
            mser_ar_range=tuple(t2_union.get("mser_ar_range", [0.6, 1.6])),
            mser_min_extent=t2_union.get("mser_min_extent", 0.2),
            hsv_set=t2_union.get("hsv_set_used", 1),
            mser_delta=t2_union.get("mser_delta_used", 5),
            canny_low=t2_union.get("canny_low_used", 50),
            canny_high=t2_union.get("canny_high_used", 150),
        ),
        "task3_shape": dict(
            circle={**default_circle, **t3_shape.get("circle", {})},
            triangle={**default_triangle, **t3_shape.get("triangle", {})},
            rectangle={**default_rectangle, **t3_shape.get("rectangle", {})},
        ),
        "task4": dict(
            min_w=t4.get("min_w", 10),
            min_h=t4.get("min_h", 10),
            min_aspect_ratio=t4.get("min_aspect_ratio", 0.4),
            max_aspect_ratio=t4.get("max_aspect_ratio", 2.5),
            resize=tuple(t4.get("resize", [64, 64])),
            nms_iou_threshold=t4.get("nms_iou_threshold", 0.4),
        ),
        "task5": dict(
            hog_params=dict(
                orientations=hog_cfg.get("orientations", 9),
                pixels_per_cell=tuple(hog_cfg.get("pixels_per_cell", [8, 8])),
                cells_per_block=tuple(hog_cfg.get("cells_per_block", [2, 2])),
            ),
        ),
        "task6": dict(
            model_bin_path=str(
                _resolve_project_path(t6.get("model_bin_path", default_bin_path), project_root)
            ),
            model_multi_path=str(
                _resolve_project_path(t6.get("model_multi_path", default_multi_path), project_root)
            ),
            bin_confidence_threshold=t6.get("bin_thr", t6.get("bin_confidence_threshold", 0.20)),
            multi_confidence_threshold=t6.get(
                "multi_thr", t6.get("multi_confidence_threshold", 0.15)
            ),
        ),
    }
    return params, project_root, config_path


def process_image_to_rois(
    image_bgr: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chạy tiền xử lý và trích xuất danh sách ROI ứng viên (Phục vụ cả --detect-only).

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        params (Dict[str, Any]): Tham số cấu hình pipeline.

    Returns:
        Tuple[...]: (enhanced_image, debug_mask, union_components, valid_rois, rejected_rois).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")

    enhanced = preprocess_task1(image_bgr, **params["task1"])
    union_nms_thr = params["task2_union"].get("nms_iou_thresh", 0.4)

    hsv_set = params["task2_union"]["hsv_set"]
    if params["task2"].get("union_hsv_sets"):
        hsv_set = sorted(set([int(hsv_set), 2 if int(hsv_set) != 2 else 1]))

    union_boxes, union_components = build_union_boxes(
        enhanced,
        iou_thresh=union_nms_thr,
        hsv_set=hsv_set,
        mser_delta=params["task2_union"]["mser_delta"],
        canny_low=params["task2_union"]["canny_low"],
        canny_high=params["task2_union"]["canny_high"],
        hsv_ar_range=params["task2_union"]["hsv_ar_range"],
        hsv_min_extent=params["task2_union"]["hsv_min_extent"],
        mser_ar_range=params["task2_union"]["mser_ar_range"],
        mser_min_extent=params["task2_union"]["mser_min_extent"],
        hsv_ranges=params["task2"].get("hsv_ranges"),
    )
    contour_items: List[Dict[str, Any]] = []
    hsv_list = union_components.get("hsv", [])
    mser_list = union_components.get("mser", [])
    edge_list = union_components.get("edge_hull", [])

    for (x, y, w, h) in union_boxes:
        box_xyxy = (x, y, x + w, y + h)
        sources: List[str] = []
        if any(compute_iou(box_xyxy, (b[0], b[1], b[0] + b[2], b[1] + b[3])) >= 0.25 for b in hsv_list):
            sources.append("HSV")
        if any(compute_iou(box_xyxy, (b[0], b[1], b[0] + b[2], b[1] + b[3])) >= 0.25 for b in mser_list):
            sources.append("MSER")
        if any(compute_iou(box_xyxy, (b[0], b[1], b[0] + b[2], b[1] + b[3])) >= 0.25 for b in edge_list):
            sources.append("CANNY_HULL")
        if not sources:
            sources.append("CONTOUR")

        contour_items.append(
            {
                "bounding_box": [x, y, w, h],
                "source": sources[0].lower(),
                "proposal_sources": sources,
            }
        )

    debug_set_number = params["task2"]["debug_mask_set_number"]
    include_achromatic = params["task2"].get("include_achromatic", False)
    dilate_k = params["task2"].get("achromatic_dilate_ksize", 9)
    fill_holes = params["task2"].get("fill_holes", True)
    _, mask_debug = segment_task2(
        enhanced,
        set_number=debug_set_number,
        include_achromatic=include_achromatic,
        achromatic_dilate_ksize=dilate_k,
        fill_holes=fill_holes,
        ranges=(
            params["task2"].get("hsv_ranges")
            if int(debug_set_number) == 1
            else None
        ),
    )
    if params["task2"].get("union_hsv_sets"):
        other = 2 if debug_set_number != 2 else 1
        _, mask2 = segment_task2(
            enhanced, set_number=other, include_achromatic=False, fill_holes=fill_holes
        )
        mask_debug = cv2.bitwise_or(mask_debug, mask2)

    gray_hough = preprocess_for_hough(enhanced)
    circle_items = detect_circles(gray_hough, image_bgr=enhanced, **params["task3_shape"]["circle"])
    for c in circle_items:
        c["proposal_sources"] = ["HOUGH_CIRCLE"]

    gray_poly = preprocess_for_polygon(enhanced)
    triangle_items = detect_polygons(
        gray_poly, image_bgr=enhanced, target_sides=3, **params["task3_shape"]["triangle"]
    )
    for t in triangle_items:
        t["proposal_sources"] = ["POLYGON_TRIANGLE"]

    rectangle_items = detect_polygons(
        gray_poly, image_bgr=enhanced, target_sides=4, **params["task3_shape"]["rectangle"]
    )
    for r in rectangle_items:
        r["proposal_sources"] = ["POLYGON_RECTANGLE"]

    shape_items = circle_items + triangle_items + rectangle_items

    nms_thr = params["task4"].get("nms_iou_threshold", 0.4)
    merged_items = merge_candidates(contour_items, shape_items, iou_dedup_threshold=0.45)
    merged_items = apply_nms(merged_items, iou_threshold=nms_thr, prioritize_source=False)

    rois, rejected = extract_rois(
        enhanced,
        merged_items,
        min_w=params["task4"]["min_w"],
        min_h=params["task4"]["min_h"],
        min_aspect_ratio=params["task4"]["min_aspect_ratio"],
        max_aspect_ratio=params["task4"]["max_aspect_ratio"],
    )
    return enhanced, mask_debug, union_components, rois, rejected


def _binary_sign_proba(
    clf_bin: Any, scaler_bin: Any, feats: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính xác suất vùng ROI là biển báo (Sign proba) từ Binary SVM Tầng 1."""
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
        except (AttributeError, ValueError):
            # Nếu model không hỗ trợ xác suất, dùng fallback thống nhất bên dưới.
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
    """Trích xuất HOG và thực hiện phân loại 2 tầng SVM cho các ROI ứng viên.

    Tầng 1 (Binary SVM): Lọc bỏ các vùng nền (Background, label 0).
    Tầng 2 (Multiclass SVM): Phân loại lớp biển báo (52 lớp, label 0..51).

    Args:
        rois (List[Dict[str, Any]]): Danh sách các ROI hợp lệ.
        params (Dict[str, Any]): Tham số cấu hình.
        clf_bin (Any): Mô hình Binary SVM.
        scaler_bin (Any): Scaler Tầng 1.
        clf_multi (Any): Mô hình Multiclass SVM.
        scaler_multi (Any): Scaler Tầng 2.

    Returns:
        List[Dict[str, Any]]: Danh sách kết quả phát hiện biển báo cuối cùng.
    """
    if not rois:
        return []

    resize_to = params["task4"]["resize"]
    hog_params = params["task5"]["hog_params"]
    bin_thr = params["task6"]["bin_confidence_threshold"]
    multi_thr = params["task6"]["multi_confidence_threshold"]

    feats = []
    for roi in rois:
        hog_feat, _ = extract_hog_features(
            roi["crop"], resize_to=resize_to, visualize=False, **hog_params
        )
        feats.append(hog_feat)
    feats_arr = np.array(feats)

    p_sign, _ = _binary_sign_proba(clf_bin, scaler_bin, feats_arr)
    sign_mask = p_sign >= bin_thr
    sign_idx = np.where(sign_mask)[0]

    if len(sign_idx) == 0:
        return []

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
                "model_score": float(conf),
                "confidence": float(conf),
                "bin_model_score": float(p_sign[idx]),
                "bin_confidence": float(p_sign[idx]),
                "proposal_sources": roi_item.get("proposal_sources", []),
                "proposal_score": roi_item.get("proposal_score"),
            }
        )

    final_thr = params["task4"].get("nms_iou_threshold", 0.4)
    results = apply_nms(results, iou_threshold=final_thr, prioritize_source=False)

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
    """Chạy quy trình nhận dạng biển báo giao thông hoàn chỉnh (End-to-End Pipeline) cho một ảnh.

    Args:
        image_path (Union[str, Path]): Đường dẫn tệp ảnh.
        project_root (Optional[Union[str, Path]]): Thư mục gốc dự án.
        model_bin (Optional[Any]): Mô hình Binary SVM nạp sẵn.
        scaler_bin (Optional[Any]): Scaler Binary nạp sẵn.
        model_multi (Optional[Any]): Mô hình Multiclass SVM nạp sẵn.
        scaler_multi (Optional[Any]): Scaler Multiclass nạp sẵn.
        return_debug (bool): Có trả về thông tin debug hay không.

    Returns:
        Tuple: (Ảnh đã tăng cường, Mặt nạ debug, Danh sách biển báo phát hiện được, [Tùy chọn: Debug Info]).

    Raises:
        FileNotFoundError: Nếu tệp mô hình không tồn tại.
        ValueError: Nếu ảnh không thể nạp.
    """
    params, project_root, _ = load_pipeline_config(project_root)

    if model_bin is None or scaler_bin is None:
        model_bin, scaler_bin = _load_required_model(
            params["task6"]["model_bin_path"], "mô hình Tầng 1"
        )

    if model_multi is None or scaler_multi is None:
        model_multi, scaler_multi = _load_required_model(
            params["task6"]["model_multi_path"], "mô hình Tầng 2"
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
            "union_components": union_components,
            "rejected_rois": rejected,
            "stage_funnel": {
                "rois_extracted": len(rois),
                "rois_rejected": len(rejected),
                "final_detections": len(detections),
            },
        }
        return enhanced, mask_debug, detections, debug_info

    return enhanced, mask_debug, detections


_VN_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}


def _get_vn_font(size: int = 18) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """Lấy phông chữ hỗ trợ gõ Tiếng Việt trên Windows / Linux / macOS."""
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
                # Bỏ qua font lỗi và thử ứng viên tiếp theo.
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
    """Vẽ bounding box và nhãn lớp (Tiếng Việt/Anh) có độ tin cậy lên ảnh BGR.

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        detections (List[Dict[str, Any]]): Danh sách phát hiện từ classify_rois.
        class_names (Optional[List[str]]): Danh sách tên lớp theo ID.
        color (Tuple[int, int, int]): Màu BGR cho bounding box (Mặc định xanh lá).

    Returns:
        np.ndarray: Ảnh BGR đã vẽ thông tin nhận diện.
    """
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
    """Vẽ các hộp ứng viên phục vụ chế độ demo chỉ phát hiện (--detect-only).

    Args:
        image_bgr (np.ndarray): Ảnh BGR gốc.
        candidates (List[Dict[str, Any]]): Danh sách ứng viên.
        color (Tuple[int, int, int]): Màu nét vẽ (BGR). Mặc định da cam.

    Returns:
        np.ndarray: Ảnh BGR mới đã vẽ khung ứng viên.
    """
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
