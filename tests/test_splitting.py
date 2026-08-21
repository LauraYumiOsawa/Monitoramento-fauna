import tempfile
import unittest
from pathlib import Path

from animal_monitor.splitting import split_dataset


class SplitDatasetTest(unittest.TestCase):
    def test_same_capture_group_never_crosses_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "raw_images"
            labels = root / "raw_labels"
            images.mkdir()
            labels.mkdir()
            for capture in range(6):
                for frame in range(3):
                    name = f"cam{capture}_frame{frame}.jpg"
                    (images / name).write_bytes(f"{capture}-{frame}".encode())
                    (labels / Path(name).with_suffix(".txt")).write_text(
                        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                    )
            result = split_dataset(
                images,
                labels,
                root / "dataset",
                group_regex=r"(cam\d+)_",
            )
            self.assertEqual(sum(result["images"].values()), 18)
            self.assertEqual(len(result["groups"]), 6)
            for capture in range(6):
                locations = set()
                for split in ("train", "val", "test"):
                    if list((root / "dataset" / "images" / split).glob(f"cam{capture}_*")):
                        locations.add(split)
                self.assertEqual(len(locations), 1)


if __name__ == "__main__":
    unittest.main()
