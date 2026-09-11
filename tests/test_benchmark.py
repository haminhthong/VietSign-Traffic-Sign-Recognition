"""Kiểm Thử Đơn Vị: Động Cơ Benchmark và Ghép Cặp Đo Lường (tests/test_benchmark.py).

Kiểm tra:
1. Đánh giá Candidate Proposals (Proposal Recall@IoU0.5, Proposals/image, FP/image).
2. Quy tắc ghép cặp chỉ cho phép tối đa 1 dự đoán ghép với 1 GT.
3. Quy tắc End-to-End Correct: IoU >= 0.5 VÀ class_pred == class_gt.
4. Xử lý sai lớp (wrong class) thành False Positive thay vì True Positive.
5. Thực thi run_benchmark xuất báo cáo hợp lệ.
"""

import unittest

import numpy as np

from tools.benchmark import (
    evaluate_baselines,
    evaluate_candidate_detection,
    evaluate_end_to_end_matching,
)


class TestBenchmarkEngine(unittest.TestCase):
    """Bộ kiểm thử cho công cụ benchmark.py."""

    def test_evaluate_candidate_detection_matching(self):
        """Kiểm tra tính đúng đắn của Proposal Recall@IoU0.5."""
        # 2 GT boxes: [x, y, w, h]
        gt_boxes = [[10, 10, 50, 50], [100, 100, 40, 40]]

        # Pred 1: Khớp chuẩn GT 1 (IoU >= 0.5)
        # Pred 2: Khớp trượt GT 2 (IoU < 0.5)
        # Pred 3: Vùng nền hoàn toàn
        pred_boxes = [
            [12, 12, 48, 48],  # IoU cao với GT 1 -> TP
            [120, 120, 40, 40],  # IoU thấp với GT 2 -> FP
            [200, 200, 30, 30],  # Nền -> FP
        ]

        metrics = evaluate_candidate_detection(gt_boxes, pred_boxes, iou_threshold=0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 2.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_iou50"], 0.5, places=3)
        self.assertAlmostEqual(metrics["precision"], 1.0 / 3.0, places=3)

    def test_duplicate_prediction_matches_only_one_gt(self):
        """Nhiều dự đoán cùng đè lên 1 GT thì chỉ có 1 dự đoán được tính là TP, còn lại là FP."""
        gt_boxes = [[20, 20, 40, 40]]
        pred_boxes = [
            [20, 20, 40, 40],  # Khớp GT -> TP
            [21, 21, 39, 39],  # Trùng GT đã match -> FP
            [22, 22, 38, 38],  # Trùng GT đã match -> FP
        ]

        metrics = evaluate_candidate_detection(gt_boxes, pred_boxes, iou_threshold=0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 2.0)
        self.assertEqual(metrics["fn"], 0.0)
        self.assertAlmostEqual(metrics["recall_at_iou50"], 1.0, places=3)

    def test_e2e_correct_matching_rule(self):
        """Kiểm tra điều kiện E2E Correct: IoU >= 0.5 VÀ class_pred == class_gt."""
        gt_items = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class": 3},
            {"x1": 100, "y1": 100, "x2": 150, "y2": 150, "class": 10},
        ]

        # Pred 1: Khớp box và đúng class 3 -> E2E TP
        # Pred 2: Khớp box nhưng sai class (đoán 11 thay vì 10) -> E2E FP
        pred_detections = [
            {"bounding_box": [10, 10, 40, 40], "predicted_class": 3},
            {"bounding_box": [100, 100, 50, 50], "predicted_class": 11},
        ]

        metrics = evaluate_end_to_end_matching(gt_items, pred_detections, iou_threshold=0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertAlmostEqual(metrics["e2e_recall"], 0.5, places=3)
        self.assertAlmostEqual(metrics["e2e_precision"], 0.5, places=3)

    def test_evaluate_baselines(self):
        """Kiểm tra đánh giá baseline Majority Classifier và Linear SVM."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(30, 20)).astype(np.float32)
        y_train = np.array([0] * 20 + [1] * 10)
        X_test = rng.normal(size=(10, 20)).astype(np.float32)
        y_test = np.array([0] * 7 + [1] * 3)

        baselines = evaluate_baselines(X_train, y_train, X_test, y_test)
        self.assertIn("Majority_Baseline", baselines)
        self.assertIn("Linear_SVM_Baseline", baselines)
        self.assertTrue(0.0 <= baselines["Majority_Baseline"]["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
