from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_b.data_prep import (
    crop_square,
    expand_box,
    iou_xyxy,
    parse_label_file,
    prepare_model_b_data,
)

HAS_OPENCV = importlib.util.find_spec("cv2") is not None
requires_opencv = unittest.skipUnless(HAS_OPENCV, "OpenCV não está instalado neste ambiente.")


def _cv2():
    import cv2

    return cv2


class ExpandBoxTest(unittest.TestCase):
    def test_margin_grows_the_box_inside_the_image(self) -> None:
        result = expand_box((40.0, 40.0, 60.0, 60.0), 0.5, 200, 200)
        self.assertEqual(result, (30, 30, 70, 70))

    def test_clamps_to_image_limits(self) -> None:
        result = expand_box((0.0, 0.0, 20.0, 20.0), 1.0, 100, 100)
        self.assertEqual(result, (0, 0, 40, 40))

    def test_degenerate_box_still_returns_positive_area(self) -> None:
        x1, y1, x2, y2 = expand_box((10.0, 10.0, 10.0, 10.0), 0.0, 100, 100)
        self.assertGreater(x2 - x1, 0)
        self.assertGreater(y2 - y1, 0)


@requires_opencv
class CropSquareTest(unittest.TestCase):
    def test_returns_square_and_preserves_aspect_with_padding(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :] = (10, 20, 30)
        crop = crop_square(image, (0, 0, 200, 100), 64)
        self.assertEqual(crop.shape, (64, 64, 3))
        self.assertEqual(int(crop[0, 0, 0]), 114)
        self.assertEqual(tuple(int(value) for value in crop[32, 32]), (10, 20, 30))

    def test_empty_region_returns_neutral_canvas(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = crop_square(image, (5, 5, 5, 5), 16)
        self.assertEqual(crop.shape, (16, 16, 3))
        self.assertTrue(bool((crop == 114).all()))


class IouTest(unittest.TestCase):
    def test_no_overlap_is_zero(self) -> None:
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        others = np.array([[20, 20, 30, 30]], dtype=np.float32)
        self.assertEqual(float(iou_xyxy(box, others)[0]), 0.0)

    def test_half_overlap(self) -> None:
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        others = np.array([[0, 0, 10, 5]], dtype=np.float32)
        self.assertAlmostEqual(float(iou_xyxy(box, others)[0]), 0.5, places=5)


class ParseLabelFileTest(unittest.TestCase):
    def test_reads_boxes_and_ignores_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.5 0.2 0.2\n\n2 0.25 0.25 0.1 0.1\n", encoding="utf-8")
            boxes = parse_label_file(path)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[1].class_id, 2)

    def test_missing_file_returns_empty_list(self) -> None:
        self.assertEqual(parse_label_file(Path("/tmp/nao-existe-modelo-b.txt")), [])

    def test_malformed_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.5\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                parse_label_file(path)


@requires_opencv
class PrepareDataTest(unittest.TestCase):
    """Constrói um dataset sintético mínimo e verifica os dois datasets derivados."""

    def _build_dataset(self, root: Path) -> Path:
        cv2 = _cv2()
        for split, count in (("train", 3), ("val", 2), ("test", 1)):
            images_dir = root / "data" / "images" / split
            labels_dir = root / "data" / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = np.full((120, 160, 3), 60, dtype=np.uint8)
                image[40:80, 50:110] = 200
                cv2.imwrite(str(images_dir / f"img{index}.jpg"), image)
                class_id = index % 2
                (labels_dir / f"img{index}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.375 0.333\n", encoding="utf-8"
                )
        dataset_yaml = root / "dataset.yaml"
        dataset_yaml.write_text(
            "path: data\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n  0: especie_a\n  1: especie_b\n",
            encoding="utf-8",
        )
        return dataset_yaml

    def test_generates_agnostic_labels_and_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_yaml = self._build_dataset(root)
            report = prepare_model_b_data(
                dataset_yaml,
                agnostic_dir=root / "agnostic",
                crops_dir=root / "crops",
                crop_size=64,
                negatives_per_image=1.0,
                misaligned_per_box=0.0,
                seed=7,
            )

            label_file = root / "agnostic" / "labels" / "train" / "img0.txt"
            self.assertTrue(label_file.is_file())
            self.assertTrue(
                all(line.startswith("0 ") for line in label_file.read_text().splitlines())
            )

            agnostic_yaml = root / "agnostic" / "dataset_agnostic.yaml"
            self.assertIn("0: animal", agnostic_yaml.read_text(encoding="utf-8"))

            labels = json.loads((root / "crops" / "labels.json").read_text(encoding="utf-8"))
            self.assertEqual(labels, ["especie_a", "especie_b", "desconhecido"])

            train = report["splits"]["train"]
            self.assertEqual(train["agnostic_boxes"], 3)
            self.assertEqual(train["crops"]["especie_a"], 2)
            self.assertEqual(train["crops"]["especie_b"], 1)
            self.assertGreater(train["crops"]["desconhecido"], 0)

            crop_files = list((root / "crops" / "train" / "especie_a").iterdir())
            self.assertTrue(crop_files)
            crop = _cv2().imread(str(crop_files[0]))
            self.assertEqual(crop.shape, (64, 64, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
