"""Kiểm Thử Tình Huống Biên & Lỗi Đầu Vào (test_edge_cases.py).

Phủ các tình huống biên như: ảnh rỗng/hỏng, kích thước bất thường,
gọi CLI tệp không tồn tại, JSON output schema và mô hình chưa nạp.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classifier import predict_proba_safe, train_svm
from src.cli import main as cli_main
from src.feature_extraction import extract_hog_features
from src.preprocessing import apply_clahe_lab, apply_median_filter, preprocess_task1
from src.roi_extraction import crop_roi


class TestEdgeCases(unittest.TestCase):
    """Bộ kiểm thử cho các ngoại lệ và đường dẫn xử lý lỗi (Failure paths)."""

    def test_empty_or_none_image_preprocessing(self):
        """Xử lý đúng ValueError khi truyền ảnh None hoặc 0-byte vào preprocessing."""
        with self.assertRaises(ValueError):
            apply_median_filter(None)

        with self.assertRaises(ValueError):
            apply_clahe_lab(np.array([], dtype=np.uint8))

        with self.assertRaises(ValueError):
            preprocess_task1(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_invalid_kernel_and_grid_size(self):
        """Kiểm tra báo lỗi khi truyền kernel_size chẵn hoặc tile_grid_size không hợp lệ."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        with self.assertRaises(ValueError):
            apply_median_filter(img, kernel_size=4)

        with self.assertRaises(ValueError):
            apply_clahe_lab(img, tile_grid_size=(0, 8))

    def test_hog_feature_extraction_empty_image(self):
        """Kiểm tra báo lỗi trích xuất HOG khi truyền ảnh rỗng."""
        with self.assertRaises(ValueError):
            extract_hog_features(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_crop_roi_out_of_bounds(self):
        """Cắt ROI vượt khỏi biên ảnh vẫn không bị văng Exception (phải kẹp biên)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bounding box x, y, w, h
        roi = crop_roi(img, -20, -10, 150, 140)
        self.assertIsNotNone(roi)

    def test_predict_proba_safe_empty_input(self):
        """Dự đoán xác suất trên mảng rỗng trả về tuple mảng rỗng."""
        X_dummy = np.random.randn(10, 1764).astype(np.float32)
        y_dummy = np.array([0, 1] * 5)
        clf, scaler = train_svm(X_dummy, y_dummy)

        labels, conf = predict_proba_safe(clf, scaler, np.array([]))
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(conf), 0)

    def test_cli_missing_input_file(self):
        """CLI thông báo lỗi FileNotFoundError khi truyền tệp không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            cli_main(["non_existent_image_12345.jpg"])

    def test_json_output_schema_structure(self):
        """Đảm bảo cấu trúc xuất JSON của kết quả dự đoán đúng chuẩn schema."""
        sample_detection = {
            "bounding_box": [10, 20, 30, 40],
            "confidence": 0.95,
            "predicted_class_id": 5,
            "predicted_class": "P.102",
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.write_text(json.dumps([sample_detection], indent=2), encoding="utf-8")

        data = json.loads(tmp_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertIn("bounding_box", data[0])
        self.assertIn("confidence", data[0])
        self.assertIn("predicted_class", data[0])
        tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
