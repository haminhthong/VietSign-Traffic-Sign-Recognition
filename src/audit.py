"""Module Kiểm Tra Tính Toàn Vẹn Dataset (Dataset Audit Tool).

Cung cấp công cụ tự động phát hiện lỗi bộ dữ liệu:
- Ảnh hỏng không đọc được
- Nhãn thiếu hoặc rỗng
- Nhãn không có ảnh tương ứng (Orphan labels)
- ID lớp vượt quá khai báo trong classes.txt
- Kiểm tra tỷ lệ Train/Test split khả dụng
- Xuất báo cáo chi tiết dạng JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.data_loader import get_image_dirs, list_image_paths, load_config, load_image
from src.utils import read_label_boxes


def audit_dataset(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Kiểm tra toàn bộ dataset và trả về báo cáo tổng hợp dạng dictionary.

    Args:
        project_root (Optional[Path]): Thư mục gốc dự án. Mặc định tự nhận diện.

    Returns:
        Dict[str, Any]: Báo cáo kiểm tra toàn vẹn bộ dữ liệu có thể xuất thành JSON.
    """
    config, root, _ = load_config(project_root)
    image_dir, label_dir = get_image_dirs(config, root)
    image_paths = list_image_paths(image_dir)

    unreadable_images: List[str] = []
    missing_labels: List[str] = []
    empty_labels: List[str] = []
    invalid_class_ids: List[Dict[str, Any]] = []
    malformed_labels: List[Dict[str, Any]] = []
    class_counts: Counter[int] = Counter()

    classes_path = root / config["paths"]["data_raw"] / "classes.txt"
    class_names = (
        [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if classes_path.is_file()
        else []
    )

    for image_path in image_paths:
        image = load_image(image_path)
        if image is None:
            unreadable_images.append(image_path.name)
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path.name)
            continue

        malformed_labels.extend(_validate_label_lines(label_path))

        boxes = read_label_boxes(label_path, image.shape)
        if not boxes:
            empty_labels.append(label_path.name)
        for box in boxes:
            class_id = box["class"]
            if class_id is None:
                continue
            class_counts[class_id] += 1
            if class_names and not 0 <= class_id < len(class_names):
                invalid_class_ids.append({"file": label_path.name, "class_id": class_id})

    orphan_labels: List[str] = []
    if label_dir.is_dir():
        image_stems = {path.stem for path in image_paths}
        orphan_labels = sorted(
            path.name for path in label_dir.glob("*.txt") if path.stem not in image_stems
        )

    split_report = _audit_splits(root, config, {path.name for path in image_paths})
    split_overlaps = _find_split_overlaps(split_report)
    for split in split_report.values():
        split.pop("files", None)
    issues = (
        len(unreadable_images)
        + len(missing_labels)
        + len(empty_labels)
        + len(invalid_class_ids)
        + len(orphan_labels)
        + len(malformed_labels)
        + sum(len(items) for items in split_overlaps.values())
    )
    return {
        "project_root": str(root),
        "summary": {
            "images": len(image_paths),
            "labels": len(list(label_dir.glob("*.txt"))) if label_dir.is_dir() else 0,
            "classes_declared": len(class_names),
            "annotations": sum(class_counts.values()),
            "issues": issues,
        },
        "class_distribution": {str(key): value for key, value in sorted(class_counts.items())},
        "issues": {
            "unreadable_images": unreadable_images,
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "orphan_labels": orphan_labels,
            "invalid_class_ids": invalid_class_ids,
            "malformed_labels": malformed_labels,
            "split_overlaps": split_overlaps,
        },
        "splits": split_report,
    }


def _audit_splits(root: Path, config: Dict[str, Any], available_images: Set[str]) -> Dict[str, Any]:
    """Đối chiếu train/val/test split với ảnh thực tế và giữ danh sách chuẩn hóa để audit."""
    split_dir = root / config["paths"]["data_raw"] / "split_dataset"
    report: Dict[str, Any] = {}
    split_names = ["train", "test"]
    if (split_dir / "val_files.txt").is_file():
        split_names.insert(1, "val")
    for split_name in split_names:
        path = split_dir / f"{split_name}_files.txt"
        entries = (
            [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if path.is_file()
            else []
        )
        normalized = [Path(entry).name for entry in entries]
        report[split_name] = {
            "declared": len(normalized),
            "available": sum(name in available_images for name in normalized),
            "missing": sum(name not in available_images for name in normalized),
            "duplicates": len(normalized) - len(set(normalized)),
            "files": sorted(set(normalized)),
        }
    return report


def _find_split_overlaps(split_report: Dict[str, Any]) -> Dict[str, List[str]]:
    """Tìm ảnh xuất hiện ở nhiều split — một dạng data leakage trực tiếp."""
    overlaps: Dict[str, List[str]] = {}
    names = list(split_report)
    for index, left in enumerate(names):
        left_files = set(split_report[left].get("files", []))
        for right in names[index + 1 :]:
            duplicated = sorted(left_files & set(split_report[right].get("files", [])))
            if duplicated:
                overlaps[f"{left}__{right}"] = duplicated
    return overlaps


def _validate_label_lines(label_path: Path) -> List[Dict[str, Any]]:
    """Phát hiện dòng nhãn YOLO hỏng thay vì âm thầm bỏ qua khi đọc box."""
    issues: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        reason = None
        try:
            values = [float(part) for part in parts]
        except ValueError:
            values = []
            reason = "chứa giá trị không phải số"

        if reason is None and len(values) not in (4, 5):
            reason = "phải có 4 tọa độ hoặc class_id + 4 tọa độ"
        if reason is None and not all(math.isfinite(value) for value in values):
            reason = "chứa NaN hoặc vô cực"
        if reason is None and len(values) == 5 and not values[0].is_integer():
            reason = "class_id phải là số nguyên"
        if reason is None:
            coords = values[-4:]
            if coords[2] <= 0 or coords[3] <= 0:
                reason = "chiều rộng và chiều cao phải lớn hơn 0"
            elif max(coords) <= 1.0 and any(value < 0 or value > 1 for value in coords):
                reason = "tọa độ YOLO chuẩn hóa phải nằm trong [0, 1]"

        if reason is not None:
            issues.append(
                {"file": label_path.name, "line": line_number, "reason": reason, "value": line}
            )
    return issues


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho công cụ audit."""
    parser = argparse.ArgumentParser(
        description="Công cụ kiểm tra tính toàn vẹn bộ dữ liệu VietSign Vision"
    )
    parser.add_argument("--project-root", type=Path, default=None, help="Thư mục gốc dự án")
    parser.add_argument(
        "--output", type=Path, default=None, help="Đường dẫn tệp JSON để lưu báo cáo"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Hàm thực thi chính của CLI audit."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    report = audit_dataset(args.project_root)
    summary = report["summary"]
    print(
        f"Ảnh: {summary['images']} | Nhãn: {summary['labels']} | "
        f"Annotation: {summary['annotations']} | Số lỗi phát hiện: {summary['issues']}"
    )
    for name, split in report["splits"].items():
        print(
            f"Tập split {name}: khai báo {split['declared']}, "
            f"hiện có {split['available']}, còn thiếu {split['missing']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu báo cáo JSON tại: {args.output.resolve()}")
    return 0 if summary["issues"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
