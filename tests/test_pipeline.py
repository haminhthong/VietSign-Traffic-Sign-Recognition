"""Bộ Kiểm Thử Đơn Vị Mở Rộng: HOG, Classifier và Pipeline (test_pipeline.py).

Kiểm tra tính đúng đắn của việc trích xuất đặc trưng HOG, huấn luyện/dự đoán SVM 2 tầng
và quy trình trích xuất ROI từ ảnh tổng hợp mà không cần mô hình pre-trained.
"""

import unittest

import numpy as np

from src.classifier import predict_proba_safe, train_svm, tune_svm
from src.feature_extraction import extract_hog_features
from src.pipeline import load_pipeline_config, load_pipeline_models, process_image_to_rois


class TestPipelineAndClassifier(unittest.TestCase):
    """Kiểm thử quy trình HOG, SVM và Pipeline với dữ liệu giả lập (Mock data)."""

    def test_hog_feature_vector_dimension(self):
        """Kiểm tra số chiều đặc trưng HOG cho ảnh 64x64 phải là 1,764 chiều."""
        dummy_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        feats, hog_img = extract_hog_features(dummy_img, visualize=True)
        self.assertEqual(feats.shape, (1764,))
        self.assertIsNotNone(hog_img)
        self.assertEqual(hog_img.shape, (64, 64))

    def test_svm_training_and_prediction(self):
        """Kiểm tra huấn luyện SVM nhị phân giả lập và dự đoán xác suất an toàn."""
        # Tạo dữ liệu ngẫu nhiên 20 mẫu, 1764 chiều
        X_dummy = np.random.randn(20, 1764).astype(np.float32)
        y_dummy = np.array([0, 1] * 10)

        clf, scaler = train_svm(X_dummy, y_dummy, C=1.0, probability=True)
        self.assertIsNotNone(clf)
        self.assertIsNotNone(scaler)

        labels, conf = predict_proba_safe(clf, scaler, X_dummy[:2])
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(conf), 2)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in conf))

    def test_tune_svm_uses_fold_safe_pipeline(self):
        """Grid search trả model/scaler đã refit mà không lộ prefix nội bộ của Pipeline."""
        rng = np.random.default_rng(42)
        X_dummy = rng.normal(size=(18, 12)).astype(np.float32)
        y_dummy = np.array([0, 1, 2] * 6)

        clf, scaler, best_params, score = tune_svm(
            X_dummy,
            y_dummy,
            param_grid={"C": [0.5, 1.0], "gamma": ["scale"]},
            cv=3,
            n_jobs=1,
            probability=False,
        )

        self.assertEqual(int(scaler.n_samples_seen_), len(X_dummy))
        self.assertIn("C", best_params)
        self.assertNotIn("svc__C", best_params)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertEqual(len(clf.predict(scaler.transform(X_dummy[:2]))), 2)

    def test_process_image_to_rois(self):
        """Kiểm tra hàm process_image_to_rois trích xuất được ROI từ ảnh giả lập."""
        # Tạo ảnh BGR màu đỏ 100x100 có chứa 1 ô vuông đỏ ở giữa
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bảng màu BGR: Đỏ ở giữa [0, 0, 255]
        dummy_img[30:70, 30:70] = [0, 0, 255]

        params, _, _ = load_pipeline_config()
        enhanced, mask_debug, union_components, rois, rejected = process_image_to_rois(
            dummy_img, params
        )

        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, (100, 100, 3))
        self.assertIsNotNone(mask_debug)
        self.assertIsInstance(rois, list)
        self.assertIsInstance(rejected, list)

    def test_load_pipeline_models_reports_missing_files(self):
        """Kiểm tra lỗi thiếu model có ngữ cảnh thay vì lỗi Joblib khó hiểu."""
        params, _, _ = load_pipeline_config()
        params["classifier"]["model_bin_path"] = "non_existent_model_123.joblib"
        params["task6"]["model_bin_path"] = "non_existent_model_123.joblib"
        with self.assertRaisesRegex(FileNotFoundError, "mô hình Tầng 1"):
            load_pipeline_models(params)


if __name__ == "__main__":
    unittest.main()
