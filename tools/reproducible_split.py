"""Tool Phân Chia Dữ Liệu Tái Lập (Reproducible Split & Provenance Generator).

Tự động phân chia tập Train / Validation / Test theo trình tự / thư mục nguồn và
gom cụm ảnh tương đồng (pHash / Perceptual Hash) để loại bỏ hoàn toàn Data Leakage
giữa các frame ảnh liền kề hoặc ảnh trùng lặp.
Tạo tệp manifest.json lưu trữ lineage và mã băm SHA-256, pHash, group_id, class_ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

# Đảm bảo import được src khi chạy trực tiếp từ thư mục tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


def compute_sha256(filepath: Path) -> str:
    """Tính mã băm SHA-256 của tệp để phát hiện trùng lặp tuyệt đối (Exact duplicate).

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


def compute_phash(filepath: Path) -> str:
    """Tính mã băm cảm nhận (Perceptual Hash - pHash) để nhận diện ảnh gần trùng (Near-duplicate).

    Args:
        filepath (Path): Đường dẫn tệp ảnh.

    Returns:
        str: Chuỗi hex pHash (hoặc rỗng nếu không thể đọc).
    """
    if not HAS_IMAGEHASH:
        return ""
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except Exception:
        return ""


def extract_sequence_id(filepath: Path) -> str:
    """Trích xuất mã định danh chuỗi quay / hành trình (sequence_id) từ tên tệp hoặc thư mục.

    Quy tắc nhận diện:
    1. Thư mục cha nếu có tên dạng 'video_*', 'seq_*', 'route_*'.
    2. Tiền tố tên tệp nếu có định dạng 'video_XX_frame_YY', 'seqXX_YY', 'camXX_YY'.
    3. Mặc định 'seq_standalone' nếu là các ảnh rời rạc.

    Args:
        filepath (Path): Đường dẫn tệp.

    Returns:
        str: Định danh sequence_id.
    """
    parent_name = filepath.parent.name.lower()
    if any(parent_name.startswith(p) for p in ("video", "seq", "route", "trip", "cam")):
        return parent_name

    stem = filepath.stem.lower()
    match = re.match(r"^((?:video|seq|route|trip|cam)[_-]?[0-9a-zA-Z]+?)(?:[_-](?:frame|img|f)?[_-]?\d+)+$", stem)
    if match:
        return match.group(1)

    match_prefix = re.match(r"^((?:video|seq|route|trip|cam)[_-]?[0-9a-zA-Z]+)", stem)
    if match_prefix:
        return match_prefix.group(1)

    return "seq_standalone"


def read_file_classes(label_path: Path) -> List[int]:
    """Đọc danh sách class_id duy nhất từ tệp nhãn YOLO/text.

    Args:
        label_path (Path): Đường dẫn tệp nhãn.

    Returns:
        List[int]: Danh sách class_id đã sắp xếp.
    """
    if not label_path.is_file():
        return []

    classes: Set[int] = set()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 5:
                try:
                    classes.add(int(float(parts[0])))
                except ValueError:
                    pass
    except Exception:
        pass

    return sorted(classes)


class DisjointSet:
    """Cấu trúc dữ liệu Disjoint-Set Union (Union-Find) để gom nhóm ảnh liền kề / gần trùng."""

    def __init__(self, items: List[Any]):
        self.parent = {item: item for item in items}

    def find(self, item: Any) -> Any:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: Any, b: Any) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def cluster_images(
    image_paths: List[Path],
    label_dir: Optional[Path] = None,
    phash_thresh: int = 8,
) -> List[Dict[str, Any]]:
    """Gom cụm ảnh theo SHA-256 (exact duplicate), pHash (near-duplicate), và Sequence ID.

    Args:
        image_paths (List[Path]): Danh sách đường dẫn ảnh.
        label_dir (Optional[Path]): Thư mục nhãn để nạp class_ids.
        phash_thresh (int): Ngưỡng Hamming distance cho pHash. Mặc định 8.

    Returns:
        List[Dict[str, Any]]: Danh sách các nhóm (clusters), mỗi nhóm chứa danh sách ảnh.
    """
    file_info: Dict[str, Dict[str, Any]] = {}
    for p in image_paths:
        sha = compute_sha256(p)
        ph = compute_phash(p)
        seq = extract_sequence_id(p)
        c_ids = read_file_classes(label_dir / f"{p.stem}.txt") if label_dir else []
        file_info[str(p)] = {
            "path": p,
            "sha256": sha,
            "phash": ph,
            "sequence_id": seq,
            "class_ids": c_ids,
            "bytes": p.stat().st_size,
        }

    paths_str = list(file_info.keys())
    dsu = DisjointSet(paths_str)

    # 1. Gom nhóm exact duplicate (cùng SHA-256)
    sha_map: Dict[str, str] = {}
    for ps in paths_str:
        sha = file_info[ps]["sha256"]
        if sha in sha_map:
            dsu.union(sha_map[sha], ps)
        else:
            sha_map[sha] = ps

    # 2. Gom nhóm sequence (cùng sequence_id khác seq_standalone)
    seq_map: Dict[str, str] = {}
    for ps in paths_str:
        seq = file_info[ps]["sequence_id"]
        if seq != "seq_standalone":
            if seq in seq_map:
                dsu.union(seq_map[seq], ps)
            else:
                seq_map[seq] = ps

    # 3. Gom nhóm near-duplicate theo pHash
    if HAS_IMAGEHASH and phash_thresh >= 0:
        hash_objs = {}
        for ps in paths_str:
            ph_str = file_info[ps]["phash"]
            if ph_str:
                try:
                    hash_objs[ps] = imagehash.hex_to_hash(ph_str)
                except Exception:
                    pass

        valid_hash_keys = list(hash_objs.keys())
        for i in range(len(valid_hash_keys)):
            k1 = valid_hash_keys[i]
            h1 = hash_objs[k1]
            for j in range(i + 1, len(valid_hash_keys)):
                k2 = valid_hash_keys[j]
                h2 = hash_objs[k2]
                if abs(h1 - h2) <= phash_thresh:
                    dsu.union(k1, k2)

    # Tổng hợp thành các cụm
    clusters_dict: Dict[str, List[Dict[str, Any]]] = {}
    for ps in paths_str:
        root = dsu.find(ps)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(file_info[ps])

    # Sắp xếp và định danh group_id ổn định
    sorted_groups = sorted(clusters_dict.values(), key=lambda grp: grp[0]["path"].name)
    result_clusters: List[Dict[str, Any]] = []
    for idx, grp in enumerate(sorted_groups, start=1):
        group_id = f"group_{idx:04d}"
        for item in grp:
            item["group_id"] = group_id
        result_clusters.append(
            {
                "group_id": group_id,
                "items": grp,
                "size": len(grp),
                "sequence_ids": sorted(list({item["sequence_id"] for item in grp})),
            }
        )

    return result_clusters


def group_aware_split(
    clusters: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Phân chia các cụm ảnh thành 3 tập Train / Val / Test đảm bảo Leakage-Safe Invariant.

    Bất biến (Invariant): Mọi ảnh trong cùng một nhóm (cùng sequence_id hoặc near-duplicate)
    bắt buộc chỉ thuộc về DUY NHẤT một tập split.

    Args:
        clusters (List[Dict[str, Any]]): Danh sách các nhóm ảnh.
        train_ratio (float): Tỷ lệ tập train.
        val_ratio (float): Tỷ lệ tập validation.
        seed (int): Hạt giống ngẫu nhiên.

    Returns:
        Tuple[List, List, List]: (train_clusters, val_clusters, test_clusters).
    """
    rng = np.random.default_rng(seed)
    shuffled_clusters = list(clusters)
    rng.shuffle(shuffled_clusters)

    total_images = sum(c["size"] for c in clusters)
    target_train = int(total_images * train_ratio)
    target_val = int(total_images * val_ratio)

    train_c: List[Dict[str, Any]] = []
    val_c: List[Dict[str, Any]] = []
    test_c: List[Dict[str, Any]] = []

    curr_train = 0
    curr_val = 0

    # Phân bổ cụm đảm bảo không vượt quá dung lượng mục tiêu một cách hợp lý
    for c in shuffled_clusters:
        if curr_train + c["size"] <= target_train or (curr_train == 0 and target_train > 0):
            train_c.append(c)
            curr_train += c["size"]
        elif curr_val + c["size"] <= target_val or (curr_val == 0 and target_val > 0 and len(clusters) > 2):
            val_c.append(c)
            curr_val += c["size"]
        else:
            test_c.append(c)

    # Nếu tập nào bị rỗng do số lượng nhóm quá ít, tái cân bằng tối thiểu 1 nhóm
    if not val_c and len(train_c) > 1 and len(clusters) >= 2:
        val_c.append(train_c.pop())
    if not test_c and len(train_c) > 1 and len(clusters) >= 3:
        test_c.append(train_c.pop())

    return train_c, val_c, test_c


def export_split_manifest(
    output_dir: Path,
    train_clusters: List[Dict[str, Any]],
    val_clusters: List[Dict[str, Any]],
    test_clusters: List[Dict[str, Any]],
    base_dir: Path,
) -> Dict[str, Any]:
    """Xuất danh sách file split và manifest JSON hoàn chỉnh kèm Data Lineage.

    Args:
        output_dir (Path): Thư mục lưu kết quả.
        train_clusters: Các nhóm train.
        val_clusters: Các nhóm val.
        test_clusters: Các nhóm test.
        base_dir: Thư mục gốc tương đối.

    Returns:
        Dict[str, Any]: Tóm tắt kết quả split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "split_summary": {
            "train": sum(c["size"] for c in train_clusters),
            "val": sum(c["size"] for c in val_clusters),
            "test": sum(c["size"] for c in test_clusters),
            "total": sum(c["size"] for c in train_clusters + val_clusters + test_clusters),
        },
        "group_summary": {
            "train_groups": len(train_clusters),
            "val_groups": len(val_clusters),
            "test_groups": len(test_clusters),
            "total_groups": len(train_clusters + val_clusters + test_clusters),
        },
        "invariants": {
            "leakage_safe": True,
            "exact_duplicate_audit": True,
            "phash_grouping": HAS_IMAGEHASH,
        },
        "files": {},
    }

    def _write_split_list(filename: str, clusters: List[Dict[str, Any]], split_name: str) -> None:
        rel_paths = []
        for c in clusters:
            for item in c["items"]:
                p = item["path"]
                try:
                    rel_path = str(p.relative_to(base_dir)).replace("\\", "/")
                except ValueError:
                    rel_path = p.name
                rel_paths.append(rel_path)
                manifest["files"][rel_path] = {
                    "file": rel_path,
                    "sha256": item["sha256"],
                    "phash": item["phash"],
                    "group_id": item["group_id"],
                    "sequence_id": item["sequence_id"],
                    "route_id": "unknown",
                    "split": split_name,
                    "class_ids": item["class_ids"],
                    "bytes": item["bytes"],
                }
        rel_paths.sort()
        (output_dir / filename).write_text("\n".join(rel_paths), encoding="utf-8")

    _write_split_list("train_files.txt", train_clusters, "train")
    _write_split_list("val_files.txt", val_clusters, "val")
    _write_split_list("test_files.txt", test_clusters, "test")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest["split_summary"]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Phân chia train/val/test split theo nhóm chuỗi / pHash chống Data Leakage."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw/images", help="Thư mục ảnh gốc.")
    parser.add_argument("--label-dir", type=str, default="data/raw/labels", help="Thư mục nhãn gốc.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục xuất split list.")
    parser.add_argument("--phash-thresh", type=int, default=8, help="Ngưỡng pHash gom nhóm gần trùng.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Tỷ lệ tập Train.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ tập Validation.")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    label_path = Path(args.label_dir)
    out_path = Path(args.output_dir)

    if not data_path.is_dir():
        print(f"[ERROR] Thư mục dữ liệu không tồn tại: {data_path}")
        return

    images = sorted(list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")))
    if not images:
        print(f"[WARNING] Không tìm thấy ảnh .jpg/.png nào trong: {data_path}")
        return

    print(f"[INFO] Bắt đầu gom cụm {len(images)} ảnh với pHash threshold = {args.phash_thresh}...")
    clusters = cluster_images(images, label_dir=label_path, phash_thresh=args.phash_thresh)
    print(f"[INFO] Đã tạo thành công {len(clusters)} nhóm độc lập (clusters).")

    train_c, val_c, test_c = group_aware_split(
        clusters, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    summary = export_split_manifest(out_path, train_c, val_c, test_c, base_dir=data_path.parent.parent)

    print(f"[SUCCESS] Phân chia dữ liệu Leakage-Safe hoàn tất tại '{out_path}':")
    print(f"  - Train: {summary['train']} ảnh ({len(train_c)} nhóm)")
    print(f"  - Val:   {summary['val']} ảnh ({len(val_c)} nhóm)")
    print(f"  - Test:  {summary['test']} ảnh ({len(test_c)} nhóm - Khóa độc lập)")


if __name__ == "__main__":
    main()
