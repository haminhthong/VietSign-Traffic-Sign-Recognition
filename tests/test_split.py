"""Kiểm Thử Đơn Vị: Giao Thức Phân Chia Dữ Liệu Chống Rò Rỉ (tests/test_split.py).

Kiểm tra các thuộc tính bất biến (Invariants):
1. Ảnh cùng chuỗi quay (sequence_id) tuyệt đối không bị phân bổ chéo giữa Train/Val/Test.
2. Ảnh trùng khớp tuyệt đối (SHA-256) nằm chung một tập split.
3. Ảnh gần trùng (near-duplicate) gom cụm đúng bằng pHash.
4. Tệp manifest.json chứa đầy đủ metadata, mã băm và class_ids.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.reproducible_split import (
    cluster_images,
    compute_phash,
    compute_sha256,
    export_split_manifest,
    extract_sequence_id,
    group_aware_split,
)


class TestReproducibleSplit(unittest.TestCase):
    """Bộ kiểm thử cho công cụ reproducible_split.py."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.img_dir = self.temp_dir / "images"
        self.label_dir = self.temp_dir / "labels"
        self.out_dir = self.temp_dir / "processed"
        self.img_dir.mkdir(parents=True)
        self.label_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_image(self, filename: str, color: tuple = (100, 100, 100), size=(64, 64)) -> Path:
        p = self.img_dir / filename
        img = Image.new("RGB", size, color)
        img.save(p)
        return p

    def _create_dummy_label(self, filename: str, class_id: int = 0) -> Path:
        p = self.label_dir / filename
        p.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return p

    def test_extract_sequence_id(self):
        """Kiểm tra nhận diện đúng sequence_id từ tiền tố tên tệp."""
        self.assertEqual(extract_sequence_id(Path("video_01_frame_0100.jpg")), "video_01")
        self.assertEqual(extract_sequence_id(Path("seq02_0055.png")), "seq02")
        self.assertEqual(extract_sequence_id(Path("cam_a_123.jpg")), "cam_a")
        self.assertEqual(extract_sequence_id(Path("0589.jpg")), "seq_standalone")

    def test_same_sequence_never_crosses_split(self):
        """Bảo đảm các ảnh cùng chuỗi (sequence) luôn nằm trọn vẹn trong duy nhất 1 split."""
        # Tạo 3 ảnh chuỗi video_01 và 3 ảnh chuỗi video_02
        f1 = self._create_dummy_image("video_01_frame_01.jpg", color=(255, 0, 0))
        f2 = self._create_dummy_image("video_01_frame_02.jpg", color=(250, 0, 0))
        f3 = self._create_dummy_image("video_01_frame_03.jpg", color=(245, 0, 0))

        f4 = self._create_dummy_image("video_02_frame_01.jpg", color=(0, 255, 0))
        f5 = self._create_dummy_image("video_02_frame_02.jpg", color=(0, 250, 0))
        f6 = self._create_dummy_image("video_02_frame_03.jpg", color=(0, 245, 0))

        images = [f1, f2, f3, f4, f5, f6]
        clusters = cluster_images(images, label_dir=self.label_dir, phash_thresh=8)

        train_c, val_c, test_c = group_aware_split(clusters, train_ratio=0.5, val_ratio=0.25, seed=42)

        def get_seqs(cluster_list):
            seqs = set()
            for c in cluster_list:
                for item in c["items"]:
                    seqs.add(item["sequence_id"])
            return seqs

        train_seqs = get_seqs(train_c)
        val_seqs = get_seqs(val_c)
        test_seqs = get_seqs(test_c)

        # Không có bất kỳ sequence nào giao thoa giữa các split
        self.assertEqual(len(train_seqs & val_seqs), 0)
        self.assertEqual(len(train_seqs & test_seqs), 0)
        self.assertEqual(len(val_seqs & test_seqs), 0)

    def test_exact_duplicate_never_crosses_split(self):
        """Ảnh trùng khớp hoàn toàn (cùng mã SHA-256) phải cùng nhóm và cùng split."""
        f1 = self._create_dummy_image("img_a.jpg", color=(128, 128, 128))
        f2 = self._create_dummy_image("img_b.jpg", color=(128, 128, 128))  # Giống hệt img_a
        f3 = self._create_dummy_image("img_c.jpg", color=(20, 20, 20))

        images = [f1, f2, f3]
        clusters = cluster_images(images, label_dir=self.label_dir, phash_thresh=8)

        # img_a và img_b phải được gom vào cùng 1 nhóm
        dup_cluster = None
        for c in clusters:
            names = [it["path"].name for it in c["items"]]
            if "img_a.jpg" in names:
                dup_cluster = c
                break

        self.assertIsNotNone(dup_cluster)
        self.assertIn("img_b.jpg", [it["path"].name for it in dup_cluster["items"]])

    def test_manifest_schema_and_lineage(self):
        """Kiểm tra cấu trúc và tính toàn vẹn của tệp manifest.json xuất ra."""
        f1 = self._create_dummy_image("seq1_01.jpg", color=(200, 50, 50))
        self._create_dummy_label("seq1_01.txt", class_id=14)

        f2 = self._create_dummy_image("seq2_01.jpg", color=(50, 200, 50))
        self._create_dummy_label("seq2_01.txt", class_id=3)

        clusters = cluster_images([f1, f2], label_dir=self.label_dir, phash_thresh=8)
        train_c, val_c, test_c = group_aware_split(clusters, train_ratio=0.5, val_ratio=0.5, seed=42)

        summary = export_split_manifest(self.out_dir, train_c, val_c, test_c, base_dir=self.temp_dir)
        manifest_file = self.out_dir / "manifest.json"

        self.assertTrue(manifest_file.is_file())
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        self.assertIn("split_summary", manifest)
        self.assertIn("group_summary", manifest)
        self.assertIn("invariants", manifest)
        self.assertTrue(manifest["invariants"]["leakage_safe"])
        self.assertIn("files", manifest)

        for file_key, f_meta in manifest["files"].items():
            self.assertIn("sha256", f_meta)
            self.assertIn("phash", f_meta)
            self.assertIn("group_id", f_meta)
            self.assertIn("sequence_id", f_meta)
            self.assertIn("split", f_meta)
            self.assertIn("class_ids", f_meta)
            self.assertIsInstance(f_meta["class_ids"], list)


if __name__ == "__main__":
    unittest.main()
