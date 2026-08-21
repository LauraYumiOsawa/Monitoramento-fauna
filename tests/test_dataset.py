import tempfile
import unittest
from pathlib import Path

from animal_monitor.dataset import validate_dataset


class DatasetValidationTest(unittest.TestCase):
    def _make_dataset(self, root: Path, duplicate: bool = False) -> Path:
        for split in ("train", "val", "test"):
            (root / "data" / "images" / split).mkdir(parents=True)
            (root / "data" / "labels" / split).mkdir(parents=True)
            for class_id in range(3):
                image = root / "data" / "images" / split / f"animal_{class_id}.jpg"
                content = b"same" if duplicate and class_id == 0 else f"{split}-{class_id}".encode()
                image.write_bytes(content)
                label = root / "data" / "labels" / split / f"animal_{class_id}.txt"
                label.write_text(f"{class_id} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        yaml_path = root / "dataset.yaml"
        yaml_path.write_text(
            "path: data\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names: [anta, tamandua_bandeira, onca_pintada]\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_valid_dataset_passes_with_only_quantity_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(
                self._make_dataset(Path(directory)), minimum_boxes_per_class=1
            )
            self.assertTrue(report.ok)
            self.assertEqual(report.splits["train"].boxes, 3)

    def test_duplicate_between_splits_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(
                self._make_dataset(Path(directory), duplicate=True), minimum_boxes_per_class=1
            )
            self.assertFalse(report.ok)
            self.assertIn("cross_split_duplicate", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
