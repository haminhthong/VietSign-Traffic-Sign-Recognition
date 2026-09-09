"""Công Cụ Khai Thác Mẫu Âm Khó (Hard-Negative Mining Tool cho Tier-1 Binary SVM).

Thay vì chỉ huấn luyện Tier-1 SVM bằng các mảng nền cắt ngẫu nhiên (random crops),
công cụ này chạy Candidate Proposal Engine trên tập ảnh huấn luyện và thu thập:
1. Các đề xuất ứng viên có IoU < 0.2 đối với mọi biển báo nhãn thật (Ground Truth).
2. Các mẫu nền gây báo giả (False Positives) hoặc có điểm số cao từ Tier-1 SVM hiện tại
   (biển quảng cáo đỏ, đèn giao thông tròn, góc tòa nhà, decal xe cộ).
3. Đóng gói tập Hard Negative Pool để huấn luyện/tinh chỉnh Tier-1 Binary SVM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Đảm bảo import được src khi chạy trực tiếp từ thư mục tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import load_model  # noqa: E402
from src.data_loader import load_image  # noqa: E402
from src.feature_extraction import extract_hog_features  # noqa: E402
from src.pipeline import (  # noqa: E402
    _binary_sign_proba,
    load_pipeline_config,
    process_image_to_rois,
)
from src.utils import compute_iou, read_label_boxes  # noqa: E402


def mine_hard_negatives_from_image(
    image_path: Path,
    label_path: Path,
    params: Dict[str, Any],
    clf_bin: Optional[Any] = None,
    scaler_bin: Optional[Any] = None,
    max_iou_negative: float = 0.2,
    hard_threshold: float = 0.3,
    max_negatives_per_image: int = 15,
) -> List[Dict[str, Any]]:
    """Khai thác các vùng đề xuất nền bị nhận nhầm từ 1 ảnh huấn luyện."""
    img = load_image(image_path)
    if img is None:
        return []

    gt_boxes = read_label_boxes(label_path, img.shape)
    gt_xyxy = [(g["x1"], g["y1"], g["x2"], g["y2"]) for g in gt_boxes]

    _, _, _, rois, _ = process_image_to_rois(img, params)
    if not rois:
        return []

    resize_to = tuple(params["task4"]["resize"])
    hog_params = params["task5"]["hog_params"]

    mined: List[Dict[str, Any]] = []

    for roi in rois:
        bx = roi["bounding_box"]
        cand_xyxy = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])

        max_iou = 0.0
        if gt_xyxy:
            max_iou = max(compute_iou(cand_xyxy, gt) for gt in gt_xyxy)

        # Điều kiện tiên quyết: Vùng này không chứa biển báo thật (IoU < 0.2)
        if max_iou >= max_iou_negative:
            continue

        feat, _ = extract_hog_features(roi["crop"], resize_to=resize_to, visualize=False, **hog_params)

        p_sign = 0.0
        is_hard = False
        if clf_bin is not None and scaler_bin is not None:
            proba_arr, _ = _binary_sign_proba(clf_bin, scaler_bin, feat.reshape(1, -1))
            p_sign = float(proba_arr[0])
            is_hard = p_sign >= hard_threshold

        mined.append(
            {
                "image": str(image_path.name),
                "bounding_box": bx,
                "max_iou_to_gt": round(float(max_iou), 4),
                "model_sign_proba": round(float(p_sign), 4),
                "is_hard_negative": is_hard,
                "proposal_sources": roi.get("proposal_sources", []),
                "features": feat.tolist(),
            }
        )

    # Ưu tiên các mẫu có model_sign_proba cao nhất (những mẫu gây lừa model mạnh nhất)
    mined.sort(key=lambda x: x["model_sign_proba"], reverse=True)
    return mined[:max_negatives_per_image]


def run_hard_negative_mining(
    train_image_paths: List[Path],
    label_dir: Path,
    params: Dict[str, Any],
    output_path: Path,
    clf_bin: Optional[Any] = None,
    scaler_bin: Optional[Any] = None,
    max_iou_negative: float = 0.2,
    hard_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Chạy quy trình khai thác mẫu âm khó trên toàn bộ tập huấn luyện."""
    total_mined = 0
    hard_count = 0
    all_negatives_meta = []

    for idx, p in enumerate(train_image_paths, start=1):
        lbl = label_dir / f"{p.stem}.txt"
        negatives = mine_hard_negatives_from_image(
            p,
            lbl,
            params,
            clf_bin=clf_bin,
            scaler_bin=scaler_bin,
            max_iou_negative=max_iou_negative,
            hard_threshold=hard_threshold,
        )
        total_mined += len(negatives)
        hard_count += sum(1 for n in negatives if n["is_hard_negative"])
        for n in negatives:
            # Không lưu raw vector feature vào json tóm tắt để tránh file quá nặng
            feat_vec = n.pop("features", [])
            n["feature_len"] = len(feat_vec)
            all_negatives_meta.append(n)

    report = {
        "status": "success",
        "train_images_scanned": len(train_image_paths),
        "total_background_proposals_mined": total_mined,
        "hard_negatives_identified": hard_count,
        "iou_negative_threshold": max_iou_negative,
        "hard_confidence_threshold": hard_threshold,
        "sample_mined_candidates": all_negatives_meta[:30],
        "recommendation": (
            "Bổ sung các mẫu âm khó này vào tập huấn luyện Tier-1 SVM (nhãn 0) "
            "để triệt tiêu hoàn toàn báo giả trên các cấu trúc hình học phức tạp."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Khai thác mẫu âm khó (Hard-Negative Mining) cho Tier-1 SVM.")
    parser.add_argument("--train-list", type=str, default="data/processed/train_files.txt", help="Danh sách train.")
    parser.add_argument("--data-dir", type=str, default="data/raw/images", help="Thư mục ảnh gốc dự phòng.")
    parser.add_argument("--label-dir", type=str, default="data/raw/labels", help="Thư mục nhãn.")
    parser.add_argument("--output", type=str, default="outputs/hard_negatives.json", help="Tệp xuất báo cáo.")
    parser.add_argument("--hard-thresh", type=float, default=0.3, help="Ngưỡng điểm Tier-1 coi là hard negative.")
    args = parser.parse_args()

    print("=== VIETSIGN VISION HARD-NEGATIVE MINING ===")
    params, project_root, _ = load_pipeline_config()

    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_root / path

    train_list_path = resolve_path(args.train_list)
    label_dir = resolve_path(args.label_dir)

    train_images: List[Path] = []
    if train_list_path.is_file():
        lines = [line.strip() for line in train_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            p = project_root / line
            if not p.is_file():
                p = Path(line)
            if not p.is_file():
                p = project_root / "data" / line
            if p.is_file():
                train_images.append(p)

    if not train_images:
        data_dir = resolve_path(args.data_dir)
        if data_dir.is_dir():
            train_images = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))

    clf_bin = scaler_bin = None
    try:
        bin_path = Path(params["task6"]["model_bin_path"])
        if bin_path.is_file():
            clf_bin, scaler_bin = load_model(bin_path)
            print("[INFO] Đã nạp Tier-1 Binary SVM để sàng lọc False Positives có độ tự tin cao.")
    except Exception:
        print("[INFO] Chưa có Tier-1 model pre-trained; khai thác toàn bộ vùng đề xuất nền (IoU < 0.2).")

    print(f"[INFO] Bắt đầu quét {len(train_images)} ảnh huấn luyện...")
    report = run_hard_negative_mining(
        train_images,
        label_dir,
        params,
        output_path=resolve_path(args.output),
        clf_bin=clf_bin,
        scaler_bin=scaler_bin,
        hard_threshold=args.hard_thresh,
    )

    print(f"[SUCCESS] Đã xuất kết quả khai thác mẫu âm tại '{args.output}':")
    print(f"  - Mẫu nền thu thập được: {report['total_background_proposals_mined']}")
    print(f"  - Mẫu âm khó (Hard FP) : {report['hard_negatives_identified']}")


if __name__ == "__main__":
    main()
