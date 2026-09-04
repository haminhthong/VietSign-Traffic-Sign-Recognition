"""Kiểm thử nhanh các hàm lõi không cần mô hình đã huấn luyện."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_loader import list_image_paths, load_image, save_image
from src.preprocessing import apply_median_filter
from src.roi_extraction import apply_nms, crop_roi, is_valid_roi
from src.utils import box_xywh_to_xyxy, boxes_xywh_to_xyxy, compute_iou, read_label_boxes


class TestCoreUtilities(unittest.TestCase):
    def test_compute_iou(self):
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(compute_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_box_coordinate_conversion_reuses_single_box_logic(self):
        box = (2, 3, 5, 7)
        self.assertEqual(box_xywh_to_xyxy(box), (2, 3, 7, 10))
        self.assertEqual(boxes_xywh_to_xyxy([box]), [(2, 3, 7, 10)])

    def test_crop_roi_clips_to_image(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        crop = crop_roi(image, -5, 5, 30, 10)
        self.assertEqual(crop.shape, (5, 20, 3))

    def test_roi_validation(self):
        self.assertEqual(is_valid_roi(20, 20)[0], True)
        self.assertEqual(is_valid_roi(2, 20)[0], False)

    def test_nms_removes_overlapping_box(self):
        items = [
            {"bounding_box": [0, 0, 10, 10], "confidence": 0.9},
            {"bounding_box": [1, 1, 10, 10], "confidence": 0.5},
        ]
        self.assertEqual(len(apply_nms(items, 0.5, prioritize_source=False)), 1)

    def test_yolo_box_can_reach_image_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            box = read_label_boxes(label, (10, 20, 3))[0]
            self.assertEqual((box["x2"], box["y2"], box["w"], box["h"]), (20, 10, 20, 10))

    def test_missing_image_directory_returns_empty_list(self):
        self.assertEqual(list_image_paths(Path("khong-ton-tai")), [])

    def test_invalid_median_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_median_filter(np.zeros((8, 8, 3), dtype=np.uint8), 2)

    def test_unicode_image_path_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ảnh-biển-báo.png"
            image = np.full((5, 7, 3), 123, dtype=np.uint8)
            self.assertTrue(save_image(path, image))
            loaded = load_image(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
