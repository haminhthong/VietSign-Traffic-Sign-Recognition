"""Kiểm Thử Đơn Vị: Proposal Quality Score & Chẩn Đoán Classifier (tests/test_proposal_scoring.py).

Kiểm tra:
1. Tính điểm Proposal Quality Score (PQS) đa nguồn (Multi-cue evidence).
2. Hợp nhất bằng chứng nguồn gốc (Proposal Provenance) khi gộp ứng viên.
3. Thuật toán NMS ưu tiên ứng viên có Proposal Quality Score cao.
4. Chẩn đoán ma trận nhầm lẫn và F1 từng lớp (analyze_confusion_and_per_class_f1).
"""

import unittest
import numpy as np

from src.classifier import analyze_confusion_and_per_class_f1
from src.roi_extraction import apply_nms, compute_proposal_quality_score, merge_candidates


class TestProposalScoringAndDiagnostics(unittest.TestCase):
    """Kiểm thử tính điểm chất lượng proposal và chẩn đoán phân loại."""

    def test_multi_cue_proposal_quality_score(self):
        """Ứng viên có nhiều nguồn bằng chứng (HSV + MSER + Circle) phải có điểm PQS cao hơn ứng viên 1 nguồn."""
        single_cue_item = {
            "bounding_box": [10, 10, 40, 40],
            "source": "canny_hull",
            "proposal_sources": ["CANNY_HULL"],
            "confidence": 0.5,
        }

        multi_cue_item = {
            "bounding_box": [10, 10, 40, 40],
            "source": "hsv",
            "proposal_sources": ["HSV", "MSER", "HOUGH_CIRCLE"],
            "confidence": 0.5,
        }

        pqs_single = compute_proposal_quality_score(single_cue_item)
        pqs_multi = compute_proposal_quality_score(multi_cue_item)

        self.assertGreater(pqs_multi, pqs_single)
        self.assertGreater(pqs_multi, 0.5)

    def test_merge_candidates_fuses_sources(self):
        """Khi 2 ứng viên trùng lặp cao, merge_candidates phải hợp nhất mảng proposal_sources."""
        contour_items = [
            {
                "bounding_box": [20, 20, 50, 50],
                "source": "hsv",
                "proposal_sources": ["HSV"],
            }
        ]

        shape_items = [
            {
                "bounding_box": [21, 21, 49, 49],  # IoU rất cao với contour_items[0]
                "source": "circle",
                "proposal_sources": ["HOUGH_CIRCLE"],
            }
        ]

        merged = merge_candidates(contour_items, shape_items, iou_dedup_threshold=0.45)
        # Vì trùng lặp cao nên chỉ còn 1 ứng viên hợp nhất
        self.assertEqual(len(merged), 1)
        fused = merged[0]
        self.assertIn("HSV", fused["proposal_sources"])
        self.assertIn("HOUGH_CIRCLE", fused["proposal_sources"])
        self.assertIn("proposal_score", fused)

    def test_apply_nms_ranks_by_proposal_score(self):
        """NMS sắp xếp và chọn giữ lại ứng viên có Proposal Quality Score cao hơn."""
        item_weak = {
            "bounding_box": [30, 30, 60, 60],
            "source": "canny_hull",
            "proposal_sources": ["CANNY_HULL"],
            "proposal_score": 0.20,
            "confidence": 0.2,
        }

        item_strong = {
            "bounding_box": [31, 31, 59, 59],
            "source": "circle",
            "proposal_sources": ["HSV", "HOUGH_CIRCLE"],
            "proposal_score": 0.85,
            "confidence": 0.9,
        }

        kept = apply_nms([item_weak, item_strong], iou_threshold=0.4, prioritize_source=False)
        self.assertEqual(len(kept), 1)
        # item_strong phải được giữ lại
        self.assertEqual(kept[0]["source"], "circle")
        self.assertEqual(kept[0]["proposal_score"], 0.85)

    def test_analyze_confusion_and_per_class_f1(self):
        """Kiểm tra tính F1 từng lớp, worst classes và top confusion pairs."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 2, 3])
        y_pred = np.array([0, 0, 1, 2, 2, 2, 2, 0])  # class 1 nhầm sang 2, class 3 nhầm sang 0

        class_names = ["Speed_40", "Speed_50", "Speed_60", "No_Entry"]
        diag = analyze_confusion_and_per_class_f1(y_true, y_pred, class_names=class_names, top_k=2)

        self.assertIn("macro_f1", diag)
        self.assertIn("per_class", diag)
        self.assertIn("worst_classes", diag)
        self.assertIn("top_confusion_pairs", diag)

        # Lớp 3 bị dự đoán sai 100% -> f1 = 0
        worst_ids = [w["class_id"] for w in diag["worst_classes"]]
        self.assertIn(3, worst_ids)

        # Cặp nhầm lẫn xuất hiện
        top_pairs = diag["top_confusion_pairs"]
        self.assertTrue(len(top_pairs) > 0)


if __name__ == "__main__":
    unittest.main()
