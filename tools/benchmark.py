"""Công Cụ Benchmark Độc Lập Dự Án VietSign Vision (Independent Benchmark Tool).

Đánh giá thực tế các chỉ số cốt lõi trên tập Test khóa (Locked Test Set):
1. Candidate Proposal Engine: Proposal Recall@IoU0.5, Proposals/Image, FP Proposals/Image.
2. Stage Funnel Evaluation: GT Signs -> Candidate Union -> ROI Filter -> Tier 1 Sign -> Tier 2 Correct Class.
3. Size Slice Breakdown: Phân tích độ phủ theo diện tích biển (Small < 32x32, Medium 32x32-96x96, Large > 96x96).
4. End-to-End Recognition: E2E Correct = (IoU >= 0.5) AND (class_pred == class_gt).
5. Baseline Comparison: So sánh với Majority Class và Linear SVM baseline (khi có đặc trưng).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Đảm bảo import được src khi chạy trực tiếp từ thư mục tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import load_model  # noqa: E402
from src.data_loader import load_image  # noqa: E402
from src.pipeline import classify_rois, load_pipeline_config, process_image_to_rois  # noqa: E402
from src.utils import box_xywh_to_xyxy, compute_iou, read_label_boxes  # noqa: E402


def evaluate_candidate_detection(
    ground_truth_boxes: List[List[float]],
    predicted_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Tính các chỉ số phát hiện ứng viên Candidate Proposal Engine.

    Lưu ý: Không gọi là mAP50 khi chưa quét đường cong Precision-Recall đầy đủ.
    Trả về đúng Proposal Recall@IoU0.5, Precision, TP, FP, FN.

    Args:
        ground_truth_boxes: Khung nhãn thật [[x, y, w, h], ...].
        predicted_boxes: Khung đề xuất [[x, y, w, h], ...].
        iou_threshold: Ngưỡng IoU coi là True Positive. Mặc định 0.5.

    Returns:
        Dict[str, float]: Từ điển chỉ số recall_at_iou50, precision, tp, fp, fn.
    """
    if not ground_truth_boxes:
        fp = len(predicted_boxes)
        return {
            "recall_at_iou50": 1.0,
            "precision": 0.0 if fp > 0 else 1.0,
            "tp": 0.0,
            "fp": float(fp),
            "fn": 0.0,
        }

    matched_gt = set()
    tp = 0
    fp = 0

    gt_xyxy = [box_xywh_to_xyxy((b[0], b[1], b[2], b[3])) for b in ground_truth_boxes]

    for pred in predicted_boxes:
        pred_xyxy = box_xywh_to_xyxy((pred[0], pred[1], pred[2], pred[3]))
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_xyxy):
            iou = compute_iou(pred_xyxy, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(matched_gt)
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "recall_at_iou50": float(recall),
        "precision": float(precision),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_end_to_end_matching(
    ground_truth_items: List[Dict[str, Any]],
    predicted_detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Đánh giá phát hiện & phân loại End-to-End chuẩn xác.

    Định nghĩa E2E Correct:
        IoU(pred, GT) >= iou_threshold VÀ class_pred == class_gt.

    Mỗi ground-truth sign chỉ được ghép với tối đa 1 dự đoán hợp lệ.

    Args:
        ground_truth_items: Danh sách nhãn GT [{"x1", "y1", "x2", "y2", "class"}, ...].
        predicted_detections: Danh sách dự đoán [{"bounding_box": [x, y, w, h], "predicted_class": int}, ...].
        iou_threshold: Ngưỡng IoU đánh giá. Mặc định 0.5.

    Returns:
        Dict[str, float]: e2e_recall, e2e_precision, e2e_f1, tp, fp, fn.
    """
    if not ground_truth_items:
        fp = len(predicted_detections)
        return {
            "e2e_recall": 1.0,
            "e2e_precision": 0.0 if fp > 0 else 1.0,
            "e2e_f1": 0.0,
            "tp": 0.0,
            "fp": float(fp),
            "fn": 0.0,
        }

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in predicted_detections:
        bx = pred["bounding_box"]
        pred_xyxy = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
        pred_cls = pred.get("predicted_class")

        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truth_items):
            gt_xyxy = (gt["x1"], gt["y1"], gt["x2"], gt["y2"])
            iou = compute_iou(pred_xyxy, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if (
            best_iou >= iou_threshold
            and best_gt_idx not in matched_gt
            and pred_cls is not None
            and pred_cls == ground_truth_items[best_gt_idx].get("class")
        ):
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth_items) - len(matched_gt)
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "e2e_recall": float(recall),
        "e2e_precision": float(precision),
        "e2e_f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Đánh giá các mô hình cơ sở Baseline trên tập Test khóa."""
    results = {}

    # Baseline 1: Majority Class
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    y_pred_majority = np.full_like(y_test, majority_class)

    results["Majority_Baseline"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_majority)),
        "f1_macro": float(f1_score(y_test, y_pred_majority, average="macro", zero_division=0)),
    }

    # Baseline 2: Linear SVM
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        linear_clf = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(dual="auto", max_iter=2000))])
        linear_clf.fit(X_train, y_train)
        y_pred_linear = linear_clf.predict(X_test)

        results["Linear_SVM_Baseline"] = {
            "accuracy": float(accuracy_score(y_test, y_pred_linear)),
            "f1_macro": float(f1_score(y_test, y_pred_linear, average="macro", zero_division=0)),
        }
    except Exception as e:
        results["Linear_SVM_Baseline"] = {"error": str(e)}

    return results


def run_benchmark(
    test_image_paths: List[Path],
    label_dir: Path,
    params: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    """Thực thi quy trình benchmark thực tế trên danh sách ảnh test."""
    total_images = len(test_image_paths)
    if total_images == 0:
        return {
            "status": "error",
            "message": "Không có tệp ảnh nào trong danh sách kiểm tra benchmark.",
        }

    # Thử nạp mô hình SVM nếu có sẵn
    models_loaded = False
    clf_bin = scaler_bin = clf_multi = scaler_multi = None
    try:
        bin_path = Path(params["task6"]["model_bin_path"])
        multi_path = Path(params["task6"]["model_multi_path"])
        if bin_path.is_file() and multi_path.is_file():
            clf_bin, scaler_bin = load_model(bin_path)
            clf_multi, scaler_multi = load_model(multi_path)
            models_loaded = True
    except Exception:
        models_loaded = False

    total_gt_boxes = 0
    total_proposals = 0
    proposal_tp = 0
    proposal_fp = 0

    e2e_tp = 0
    e2e_fp = 0

    # Phân tích theo lát cắt kích thước (Size Slices)
    size_slices = {
        "small (<32x32)": {"gt": 0, "matched_proposals": 0, "e2e_correct": 0},
        "medium (32-96)": {"gt": 0, "matched_proposals": 0, "e2e_correct": 0},
        "large (>96x96)": {"gt": 0, "matched_proposals": 0, "e2e_correct": 0},
    }

    # Stage Funnel Counts
    funnel = {
        "gt_signs_total": 0,
        "candidate_proposals_captured": 0,
        "roi_filter_passed": 0,
        "tier1_sign_passed": 0 if models_loaded else None,
        "tier2_correct_classified": 0 if models_loaded else None,
    }

    latencies: List[float] = []

    for img_path in test_image_paths:
        t0 = time.perf_counter()
        img = load_image(img_path)
        if img is None:
            continue

        label_file = label_dir / f"{img_path.stem}.txt"
        gt_boxes = read_label_boxes(label_file, img.shape)
        total_gt_boxes += len(gt_boxes)
        funnel["gt_signs_total"] += len(gt_boxes)

        # Chạy Stage 3 tiền xử lý + Stage 4 Proposal Generator + Stage 5 ROI filter
        _, _, union_components, rois, rejected = process_image_to_rois(img, params)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)

        # Hộp đề xuất từ candidate pool
        prop_boxes = [r["bounding_box"] for r in rois]
        total_proposals += len(prop_boxes)

        # 1. Đánh giá Candidate Proposals
        gt_xywh = [[g["x1"], g["y1"], g["w"], g["h"]] for g in gt_boxes]
        cand_eval = evaluate_candidate_detection(gt_xywh, prop_boxes, iou_threshold=0.5)
        proposal_tp += int(cand_eval["tp"])
        proposal_fp += int(cand_eval["fp"])

        # Cập nhật Funnel Stage 1 & 2
        funnel["candidate_proposals_captured"] += int(cand_eval["tp"])
        funnel["roi_filter_passed"] += len(rois)

        # Cập nhật Size Slices
        for g in gt_boxes:
            area = g["area"]
            if area < 1024:
                slice_key = "small (<32x32)"
            elif area <= 9216:
                slice_key = "medium (32-96)"
            else:
                slice_key = "large (>96x96)"
            size_slices[slice_key]["gt"] += 1

            # Kiểm tra xem GT này có proposal nào khớp không
            gt_xyxy = (g["x1"], g["y1"], g["x2"], g["y2"])
            matched_prop = any(
                compute_iou(gt_xyxy, (p[0], p[1], p[0] + p[2], p[1] + p[3])) >= 0.5
                for p in prop_boxes
            )
            if matched_prop:
                size_slices[slice_key]["matched_proposals"] += 1

        # 2. Nếu model SVM có sẵn, chạy End-to-End
        if models_loaded and rois:
            detections = classify_rois(rois, params, clf_bin, scaler_bin, clf_multi, scaler_multi)
            e2e_eval = evaluate_end_to_end_matching(gt_boxes, detections, iou_threshold=0.5)
            e2e_tp += int(e2e_eval["tp"])
            e2e_fp += int(e2e_eval["fp"])
            funnel["tier2_correct_classified"] += int(e2e_eval["tp"])

    proposal_fn = total_gt_boxes - proposal_tp
    prop_recall = proposal_tp / float(total_gt_boxes) if total_gt_boxes > 0 else 0.0
    prop_precision = proposal_tp / float(total_proposals) if total_proposals > 0 else 0.0
    props_per_image = total_proposals / float(total_images) if total_images > 0 else 0.0
    fp_props_per_image = proposal_fp / float(total_images) if total_images > 0 else 0.0

    # Tính tỷ lệ phủ theo Size Slices
    size_slice_report = {}
    for sk, sval in size_slices.items():
        g_cnt = sval["gt"]
        m_cnt = sval["matched_proposals"]
        rec = m_cnt / float(g_cnt) if g_cnt > 0 else 0.0
        size_slice_report[sk] = {
            "gt_count": g_cnt,
            "matched_proposals": m_cnt,
            "proposal_recall_at_iou50": round(rec, 4),
        }

    # Funnel conversion
    funnel_report = {
        "1_gt_signs_total": funnel["gt_signs_total"],
        "2_candidate_proposals_captured": funnel["candidate_proposals_captured"],
        "3_proposal_recall_percent": round(prop_recall * 100.0, 2),
        "4_roi_rois_extracted_total": funnel["roi_filter_passed"],
    }
    if models_loaded:
        e2e_recall = e2e_tp / float(total_gt_boxes) if total_gt_boxes > 0 else 0.0
        e2e_precision = e2e_tp / float(e2e_tp + e2e_fp) if (e2e_tp + e2e_fp) > 0 else 0.0
        e2e_f1 = (2 * e2e_precision * e2e_recall) / (e2e_precision + e2e_recall) if (e2e_precision + e2e_recall) > 0 else 0.0
        funnel_report["5_tier2_correct_classified"] = e2e_tp
        funnel_report["6_e2e_recall_percent"] = round(e2e_recall * 100.0, 2)

    status = "benchmark_completed_full" if models_loaded else "candidate_benchmark_completed_models_pending"

    return {
        "status": status,
        "models_loaded": models_loaded,
        "test_images_evaluated": total_images,
        "gt_signs_evaluated": total_gt_boxes,
        "candidate_proposal_metrics": {
            "proposal_recall_at_iou50": round(prop_recall, 4),
            "proposal_precision": round(prop_precision, 4),
            "proposals_per_image": round(props_per_image, 2),
            "fp_proposals_per_image": round(fp_props_per_image, 2),
            "tp_proposals": proposal_tp,
            "fp_proposals": proposal_fp,
            "fn_proposals": proposal_fn,
        },
        "size_slices": size_slice_report,
        "stage_funnel": funnel_report,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "p50": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "end_to_end_metrics": {
            "e2e_recall": round(e2e_tp / float(total_gt_boxes), 4) if (models_loaded and total_gt_boxes > 0) else None,
            "e2e_precision": round(e2e_tp / float(e2e_tp + e2e_fp), 4) if (models_loaded and (e2e_tp + e2e_fp) > 0) else None,
            "e2e_f1": round(e2e_f1, 4) if models_loaded else None,
            "e2e_tp": e2e_tp if models_loaded else None,
            "e2e_fp": e2e_fp if models_loaded else None,
        },
        "disclaimer": (
            "Benchmark được thực thi trên tập dữ liệu khả dụng hiện tại. "
            "Để tái lập toàn bộ chỉ số SVM 52 lớp lịch sử, vui lòng huấn luyện lại model với full dataset."
        ),
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Chạy benchmark kiểm định hệ thống VietSign Vision.")
    parser.add_argument("--test-list", type=str, default="data/processed/test_files.txt", help="Tệp chứa danh sách test.")
    parser.add_argument("--data-dir", type=str, default="data/raw/images", help="Thư mục ảnh gốc dự phòng.")
    parser.add_argument("--label-dir", type=str, default="data/raw/labels", help="Thư mục nhãn.")
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.json", help="Tệp xuất báo cáo.")
    args = parser.parse_args()

    print("=== VIETSIGN VISION INDEPENDENT BENCHMARK ===")
    params, project_root, _ = load_pipeline_config()

    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_root / path

    test_list_path = resolve_path(args.test_list)
    label_dir = resolve_path(args.label_dir)

    test_image_paths: List[Path] = []
    if test_list_path.is_file():
        lines = [line.strip() for line in test_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            p = project_root / line
            if not p.is_file():
                p = Path(line)
            if not p.is_file():
                p = project_root / "data" / line
            if p.is_file():
                test_image_paths.append(p)

    if not test_image_paths:
        data_dir = resolve_path(args.data_dir)
        if data_dir.is_dir():
            test_image_paths = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
            print(f"[NOTE] Không tìm thấy test_files.txt; fallback chạy benchmark trên {len(test_image_paths)} ảnh tại '{data_dir}'.")

    print(f"[INFO] Bắt đầu đánh giá benchmark trên {len(test_image_paths)} ảnh kiểm định...")
    results = run_benchmark(test_image_paths, label_dir, params, project_root)

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SUCCESS] Đã xuất báo cáo benchmark chi tiết tại '{out_path}':")
    c_met = results.get("candidate_proposal_metrics", {})
    print(f"  - Proposal Recall@IoU0.5 : {c_met.get('proposal_recall_at_iou50', 0):.4f}")
    print(f"  - Proposals per Image   : {c_met.get('proposals_per_image', 0):.2f}")
    print(f"  - FP Proposals / Image  : {c_met.get('fp_proposals_per_image', 0):.2f}")
    lat = results.get("latency_ms", {})
    print(f"  - Latency trung bình    : {lat.get('mean', 0):.2f} ms/ảnh (p50: {lat.get('p50', 0):.2f} ms)")
    if results.get("models_loaded"):
        e2e = results.get("end_to_end_metrics", {})
        print(f"  - End-to-End Recall     : {e2e.get('e2e_recall', 0):.4f}")
        print(f"  - End-to-End Precision  : {e2e.get('e2e_precision', 0):.4f}")
        print(f"  - End-to-End F1         : {e2e.get('e2e_f1', 0):.4f}")
    else:
        print("  - End-to-End Recognition: Đang chờ mô hình SVM được huấn luyện.")


if __name__ == "__main__":
    main()
