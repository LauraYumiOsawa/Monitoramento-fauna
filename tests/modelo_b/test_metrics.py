from __future__ import annotations

import unittest

import numpy as np

from animal_monitor_b.metrics import (
    average_precision,
    classification_report,
    confusion_matrix,
    match_detections,
)


class ConfusionMatrixTest(unittest.TestCase):
    def test_counts_pairs(self) -> None:
        matrix = confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 3)
        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    def test_report_is_perfect_when_predictions_match(self) -> None:
        labels = ["a", "b"]
        report = classification_report(np.array([0, 1, 0]), np.array([0, 1, 0]), labels)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["per_class"]["a"]["support"], 2)

    def test_report_ignores_absent_classes_in_macro(self) -> None:
        labels = ["a", "b", "c"]
        report = classification_report(np.array([0, 0, 1]), np.array([0, 0, 1]), labels)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["per_class"]["c"]["support"], 0)


class AveragePrecisionTest(unittest.TestCase):
    def test_perfect_ranking_gives_one(self) -> None:
        ap, precision, recall = average_precision(
            np.array([0.9, 0.8]), np.array([1.0, 1.0]), total_ground_truth=2
        )
        self.assertAlmostEqual(ap, 1.0, places=6)
        self.assertAlmostEqual(precision, 1.0, places=6)
        self.assertAlmostEqual(recall, 1.0, places=6)

    def test_false_positive_before_true_positive_reduces_ap(self) -> None:
        ap, _, _ = average_precision(
            np.array([0.9, 0.8]), np.array([0.0, 1.0]), total_ground_truth=1
        )
        self.assertAlmostEqual(ap, 0.5, places=6)

    def test_missing_ground_truth_returns_nan(self) -> None:
        ap, _, _ = average_precision(np.array([]), np.array([]), total_ground_truth=0)
        self.assertTrue(np.isnan(ap))

    def test_no_prediction_with_ground_truth_gives_zero(self) -> None:
        ap, _, _ = average_precision(np.array([]), np.array([]), total_ground_truth=3)
        self.assertEqual(ap, 0.0)


class MatchDetectionsTest(unittest.TestCase):
    def test_one_ground_truth_matches_only_the_best_prediction(self) -> None:
        predicted = np.array([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=np.float64)
        scores = np.array([0.6, 0.9])
        truth = np.array([[0, 0, 10, 10]], dtype=np.float64)
        matched = match_detections(predicted, scores, truth, iou_threshold=0.5)
        self.assertEqual(matched.tolist(), [0.0, 1.0])

    def test_low_overlap_is_not_matched(self) -> None:
        predicted = np.array([[0, 0, 10, 10]], dtype=np.float64)
        truth = np.array([[20, 20, 30, 30]], dtype=np.float64)
        matched = match_detections(predicted, np.array([0.9]), truth, iou_threshold=0.5)
        self.assertEqual(matched.tolist(), [0.0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
