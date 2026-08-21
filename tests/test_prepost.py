import unittest

import numpy as np

from animal_monitor.inference.prepost import (
    LetterboxTransform,
    decode_yolo_output,
    non_max_suppression,
    undo_letterbox,
)


class PrePostTest(unittest.TestCase):
    def test_decodes_raw_yolov8_output_and_applies_nms(self) -> None:
        predictions = np.array(
            [
                [100, 100, 40, 40, 0.90, 0.05, 0.05],
                [102, 102, 40, 40, 0.80, 0.10, 0.10],
                [300, 300, 30, 30, 0.05, 0.85, 0.10],
            ],
            dtype=np.float32,
        )
        output = predictions.T[None, ...]
        boxes, scores, class_ids = decode_yolo_output(output, 3, 0.25, 0.45)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(set(class_ids.tolist()), {0, 1})
        self.assertAlmostEqual(float(scores.max()), 0.90, places=5)

    def test_decodes_end_to_end_output(self) -> None:
        output = np.array([[[1, 2, 10, 20, 0.75, 2], [2, 3, 9, 19, 0.10, 1]]])
        boxes, scores, class_ids = decode_yolo_output(output, 3, 0.25, 0.45)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertEqual(class_ids.tolist(), [2])
        self.assertAlmostEqual(float(scores[0]), 0.75)

    def test_class_aware_nms_keeps_different_classes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int64)
        selected = non_max_suppression(boxes, scores, classes, 0.5, False, 10)
        self.assertEqual(selected.tolist(), [0, 1])

    def test_undo_letterbox_restores_original_coordinates(self) -> None:
        transform = LetterboxTransform(100, 200, 640, 640, 3.2, 0.0, 160.0)
        restored = undo_letterbox(
            np.array([[0, 160, 640, 480]], dtype=np.float32), transform
        )
        np.testing.assert_allclose(restored[0], [0, 0, 199, 99], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
