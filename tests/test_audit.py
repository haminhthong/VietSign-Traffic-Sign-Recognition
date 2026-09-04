"""Kiểm thử báo cáo dataset trên dữ liệu mẫu của dự án."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audit import _find_split_overlaps, _validate_label_lines, audit_dataset


class TestDatasetAudit(unittest.TestCase):
    def test_sample_dataset_is_readable(self):
        report = audit_dataset()
        self.assertEqual(report["summary"]["images"], 5)
        self.assertEqual(report["summary"]["labels"], 5)
        self.assertEqual(report["summary"]["classes_declared"], 52)
        self.assertEqual(report["issues"]["unreadable_images"], [])
        self.assertEqual(report["issues"]["invalid_class_ids"], [])

    def test_split_report_exposes_incomplete_archive(self):
        report = audit_dataset()
        self.assertGreater(report["splits"]["train"]["missing"], 0)
        self.assertGreater(report["splits"]["test"]["missing"], 0)

    def test_malformed_label_is_reported_with_line_number(self):
        with TemporaryDirectory() as directory:
            label_path = Path(directory) / "bad.txt"
            label_path.write_text("0 0.5 0.5 -0.2 0.3\nnot-a-label\n", encoding="utf-8")

            issues = _validate_label_lines(label_path)

        self.assertEqual([issue["line"] for issue in issues], [1, 2])

    def test_split_overlap_is_detected(self):
        report = {
            "train": {"files": ["a.jpg", "b.jpg"]},
            "val": {"files": ["b.jpg"]},
            "test": {"files": ["c.jpg"]},
        }
        self.assertEqual(_find_split_overlaps(report), {"train__val": ["b.jpg"]})


if __name__ == "__main__":
    unittest.main()
