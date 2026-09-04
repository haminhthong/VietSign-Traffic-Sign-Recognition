"""Tool Phân Chia Dữ Liệu Tái Lập (Reproducible Split & Provenance Generator).

Tự động phân chia tập Train / Validation / Test theo trình tự / thư mục nguồn
để tránh rò rỉ dữ liệu (Data Leakage) giữa các frame ảnh liền kề.
Tạo tệp manifest.json chứa mã băm SHA-256 của tất cả các tệp.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def compute_sha256(filepath: Path) -> str:
    """Tính mã băm SHA-256 của tệp.

    Args:
        filepath (Path): Đường dẫn tệp.

    Returns:
        str: Chuỗi hex SHA-256.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_sequence_split(
    image_paths: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Phân chia ảnh thành 3 tập Train/Val/Test theo thứ tự hạt giống cố định.

    Args:
        image_paths (List[Path]): Danh sách đường dẫn ảnh.
        train_ratio (float): Tỷ lệ tập train. Mặc định 0.7.
        val_ratio (float): Tỷ lệ tập validation. Mặc định 0.15.
        seed (int): Hạt giống ngẫu nhiên. Mặc định 42.

    Returns:
        Tuple[List[Path], List[Path], List[Path]]: (train_files, val_files, test_files).
    """
    rng = np.random.defaultrng(seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = sorted(shuffled[:n_train])
    val_files = sorted(shuffled[n_train : n_train + n_val])
    test_files = sorted(shuffled[n_train + n_val :])

    return train_files, val_files, test_files


def export_split_manifest(
    output_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
    base_dir: Path,
) -> Dict[str, Any]:
    """Xuất danh sách file và manifest SHA-256.

    Args:
        output_dir (Path): Thư mục lưu kết quả.
        train_files (List[Path]): Tập train.
        val_files (List[Path]): Tập validation.
        test_files (List[Path]): Tập test khóa.
        base_dir (Path): Thư mục gốc tương đối.

    Returns:
        Dict[str, Any]: Tóm tắt manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "split_summary": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": len(train_files) + len(val_files) + len(test_files),
        },
        "files": {},
    }

    def _write_split_list(filename: str, file_list: List[Path]) -> None:
        rel_paths = []
        for p in file_list:
            try:
                rel_path = str(p.relative_to(base_dir)).replace("\\", "/")
            except ValueError:
                rel_path = p.name
            rel_paths.append(rel_path)
            manifest["files"][rel_path] = {
                "sha256": compute_sha256(p),
                "bytes": p.stat().st_size,
            }
        (output_dir / filename).write_text("\n".join(rel_paths), encoding="utf-8")

    _write_split_list("train_files.txt", train_files)
    _write_split_list("val_files.txt", val_files)
    _write_split_list("test_files.txt", test_files)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest["split_summary"]


def main():
    parser = argparse.ArgumentParser(description="Tạo train/val/test split tái lập và SHA-256 manifest.")
    parser.add_argument("--data-dir", type=str, default="data/raw/images", help="Thư mục ảnh gốc.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục xuất split list.")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)

    if not data_path.is_dir():
        print(f"[ERROR] Thư mục dữ liệu không tồn tại: {data_path}")
        return

    images = sorted(list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")))
    if not images:
        print(f"[WARNING] Không tìm thấy ảnh .jpg/.png nào trong: {data_path}")
        return

    train, val, test = generate_sequence_split(images, seed=args.seed)
    summary = export_split_manifest(out_path, train, val, test, base_dir=data_path.parent.parent)

    print(f"[SUCCESS] Phân chia dữ liệu hoàn tất tại '{out_path}':")
    print(f"  - Train: {summary['train']} tệp")
    print(f"  - Val:   {summary['val']} tệp")
    print(f"  - Test:  {summary['test']} tệp (Khóa độc lập)")


if __name__ == "__main__":
    main()
