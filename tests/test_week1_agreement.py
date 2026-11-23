"""Check the statistic and prevent partial human labels from looking complete."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "week1"))
from week1_agreement import agreement, read_sheet


class AgreementTests(unittest.TestCase):
    def test_known_kappa(self):
        # Marginals are 50/50 for both; 3/4 agree, so kappa = .5.
        left = dict(enumerate(["a", "a", "a", "a", "b", "b", "b", "b"]))
        right = dict(enumerate(["a", "a", "a", "b", "a", "b", "b", "b"]))
        self.assertAlmostEqual(agreement(left, right)["kappa"], .5)

    def test_missing_labels_are_not_scored(self):
        report = agreement({"1": "explicit", "2": ""}, {"1": "explicit", "2": "framed"})
        self.assertEqual(report["status"], "pending")
        self.assertIsNone(report["kappa"])

    def test_constant_labels_have_undefined_kappa(self):
        self.assertIsNone(agreement({"1": "a"}, {"1": "a"})["kappa"])

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.csv"
            path.write_text("annotation_id,label\n1,a\n1,a\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_sheet(path, {"1", "2"}, "label", {"a"})


if __name__ == "__main__":
    unittest.main()
