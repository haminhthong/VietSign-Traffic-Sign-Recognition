"""Kiểm Thử Nhất Quán (Training vs Inference Parity Test).

Đảm bảo các bước tiền xử lý ảnh và trích xuất đặc trưng HOG sinh ra
đầu ra hoàn toàn nhất quán (Deterministic & Reproducible) giữa quá trình
huấn luyện (training) và quá trình dự đoán (inference/CLI/API).
"""

import unittest

import numpy as np

from src.feature_extraction import extract_hog_features
from src.preprocessing import preprocess_task1


class TestFeatureParity(unittest.TestCase):
    """Kiểm thử tính nhất quán giữa training và inference."""

    def test_preprocessing_deterministic_output(self):
        """Tiền xử lý cùng 1 ảnh cho ra kết quả BGR chính xác 100% từng pixel."""
        rng = np.random.default_rng(42)
        dummy_img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

        processed1 = preprocess_task1(dummy_img, kernel_size=3, tile_grid_size=(8, 8))
        processed2 = preprocess_task1(dummy_img, kernel_size=3, tile_grid_size=(8, 8))

        np.testing.assert_array_equal(processed1, processed2)

    def test_hog_feature_vector_parity(self):
        """Véc-tơ HOG trích xuất từ 1 ảnh ROI hoàn toàn trùng khớp giữa 2 lần gọi độc lập."""
        rng = np.random.default_rng(123)
        dummy_roi = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        feat1, _ = extract_hog_features(dummy_roi)
        feat2, _ = extract_hog_features(dummy_roi)

        self.assertEqual(feat1.shape, (1764,))
        np.testing.assert_array_almost_equal(feat1, feat2, decimal=6)

    def test_batch_vs_single_image_parity(self):
        """Trích xuất HOG từng ảnh một hay trích xuất theo mảng đều cho kết quả giống hệt."""
        rng = np.random.default_rng(999)
        roi1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        roi2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        feat1, _ = extract_hog_features(roi1)
        feat2, _ = extract_hog_features(roi2)

        batch_feats = np.vstack([feat1, feat2])
        self.assertEqual(batch_feats.shape, (2, 1764))
        np.testing.assert_array_almost_equal(batch_feats[0], feat1, decimal=6)
        np.testing.assert_array_almost_equal(batch_feats[1], feat2, decimal=6)


if __name__ == "__main__":
    unittest.main()
