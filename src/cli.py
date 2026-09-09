"""Giao diện Dòng Lệnh (CLI Tool: Traffic Sign Recognition).

Cung cấp lệnh chạy nhận dạng biển báo giao thông Việt Nam trên một ảnh đơn hoặc toàn bộ thư mục ảnh.
Hỗ trợ cả chế độ nhận dạng đầy đủ (Recognition) và chế độ demo chỉ trích xuất vùng ứng viên (--detect-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_loader import list_image_paths, load_image, save_image
from src.pipeline import (
    draw_candidates,
    draw_detections,
    load_pipeline_config,
    load_pipeline_models,
    process_image_to_rois,
    run_pipeline_on_image,
)


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích cú pháp tham số dòng lệnh (ArgumentParser).

    Returns:
        argparse.ArgumentParser: Parser đã cấu hình các cờ lệnh.
    """
    parser = argparse.ArgumentParser(
        description="VietSign Vision - Hệ thống nhận dạng biển báo giao thông Việt Nam"
    )
    parser.add_argument("input", type=Path, help="Đường dẫn tới tệp ảnh đơn hoặc thư mục chứa ảnh")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions"),
        help="Thư mục lưu ảnh kết quả và tệp JSON (Mặc định: outputs/predictions)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Thư mục gốc của dự án (Mặc định: Tự động nhận diện)",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="Đường dẫn tệp danh sách tên lớp (Mặc định: data/raw/classes_vie.txt)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Chỉ trích xuất vùng ứng viên (Candidate ROIs), không yêu cầu nạp mô hình SVM",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Số lượng ứng viên tối đa hiển thị khi chạy cờ --detect-only (Mặc định: 50)",
    )
    return parser


def _read_class_names(path: Optional[Path]) -> Optional[List[str]]:
    """Đọc danh sách tên các lớp biển báo từ tệp text."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp tên lớp tại: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collect_images(input_path: Path) -> List[Path]:
    """Thu thập danh sách các tệp ảnh đầu vào từ tệp đơn hoặc thư mục."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return list_image_paths(input_path)
    raise FileNotFoundError(f"Không tìm thấy tệp hoặc thư mục đầu vào: {input_path}")


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    """Giải quyết đường dẫn CLI tương đối từ gốc dự án thay vì từ thư mục gọi lệnh."""
    return path if path.is_absolute() else project_root / path


def main(argv: Optional[List[str]] = None) -> int:
    """Hàm khởi chạy chính của CLI.

    Args:
        argv (Optional[List[str]]): Danh sách tham số đầu vào.

    Returns:
        int: Mã thoát (0 nếu thành công, 1 hoặc lỗi nếu thất bại).
    """
    # Đảm bảo PowerShell/Cmd hiển thị đúng ký tự Tiếng Việt Unicode
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    if args.max_results <= 0:
        raise ValueError("Tham số --max-results phải là số nguyên dương lớn hơn 0")

    images = _collect_images(args.input)
    if not images:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ (.jpg, .jpeg, .png) tại: {args.input}")

    params, project_root, _ = load_pipeline_config(args.project_root)
    output_dir = _resolve_project_path(args.output, project_root)
    class_file = (
        _resolve_project_path(args.classes, project_root)
        if args.classes
        else project_root / "data" / "raw" / "classes_vie.txt"
    )
    class_names = _read_class_names(class_file) if class_file.is_file() else None

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    model_components = None if args.detect_only else load_pipeline_models(params)

    print(f"Chạy VietSign Vision trên {len(images)} ảnh...")
    for image_path in images:
        started_at = time.perf_counter()
        if args.detect_only:
            image = load_image(image_path)
            if image is None:
                raise ValueError(f"Không đọc được ảnh tại: {image_path}")
            enhanced, _, _, rois, rejected = process_image_to_rois(image, params)
            detections = [
                {
                    key: value
                    for key, value in roi.items()
                    if key
                    in {
                        "bounding_box",
                        "source",
                        "confidence",
                        "vertices",
                        "x",
                        "y",
                        "radius",
                        "proposal_sources",
                        "proposal_score",
                    }
                }
                for roi in rois
            ]
            detections.sort(
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    item["bounding_box"][2] * item["bounding_box"][3],
                ),
                reverse=True,
            )
            detections = detections[: args.max_results]
            visualized = draw_candidates(enhanced, detections)
        else:
            # Model đã được nạp một lần trước vòng lặp để tránh I/O lặp theo số ảnh.
            assert model_components is not None
            model_bin, scaler_bin, model_multi, scaler_multi = model_components
            enhanced, _, detections, debug_info = run_pipeline_on_image(
                image_path,
                project_root=project_root,
                model_bin=model_bin,
                scaler_bin=scaler_bin,
                model_multi=model_multi,
                scaler_multi=scaler_multi,
                return_debug=True,
            )
            rejected = debug_info["rejected_rois"]
            visualized = draw_detections(enhanced, detections, class_names)

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        output_image = output_dir / image_path.name

        if not save_image(output_image, visualized):
            raise OSError(f"Không thể ghi ảnh kết quả tại: {output_image}")

        summary.append(
            {
                "image": str(image_path),
                "mode": "detect-only" if args.detect_only else "recognition",
                "processing_time_ms": elapsed_ms,
                "rejected_candidates": len(rejected),
                "detections": detections,
            }
        )
        label = "vùng ứng viên" if args.detect_only else "biển báo"
        print(f" -> {image_path.name}: Tìm thấy {len(detections)} {label} ({elapsed_ms:.2f} ms)")

    result_path = output_dir / "predictions.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Thành công] Đã lưu kết quả dự đoán và JSON tại thư mục: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
