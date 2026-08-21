from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor.types import BoundingBox
from animal_monitor_b.augment import augment_crop, to_tensor
from animal_monitor_b.config import RuntimeConfigB, TrackerConfig
from animal_monitor_b.events import SightingStore
from animal_monitor_b.runtime.pipeline import SpeciesEvent

HAS_OPENCV = importlib.util.find_spec("cv2") is not None
requires_opencv = unittest.skipUnless(HAS_OPENCV, "OpenCV não está instalado neste ambiente.")


class RuntimeConfigBTest(unittest.TestCase):
    def _write(self, directory: Path, body: str) -> Path:
        path = directory / "runtime.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_reads_nested_tracker_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                "detector_confidence: 0.4\n"
                "detector_input: 320\n"
                "classifier_input: 96\n"
                "tracker:\n  max_age: 12\n  min_hits: 4\n  iou_match: 0.25\n",
            )
            config = RuntimeConfigB.from_yaml(path)
        self.assertEqual(config.detector_confidence, 0.4)
        self.assertEqual(config.detector_input, 320)
        self.assertEqual(config.classifier_input, 96)
        self.assertEqual(config.tracker.max_age, 12)
        self.assertEqual(config.tracker.min_hits, 4)

    def test_paths_are_resolved_relative_to_the_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), "output_dir: saida\ndatabase: saida/eventos.db\n")
            config = RuntimeConfigB.from_yaml(path)
        self.assertEqual(config.output_dir.name, "saida")
        self.assertTrue(str(config.database).endswith("saida/eventos.db"))

    def test_detector_input_must_be_multiple_of_32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), "detector_input: 300\n")
            with self.assertRaises(ConfigurationError):
                RuntimeConfigB.from_yaml(path)

    def test_out_of_range_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), "detector_confidence: 1.4\n")
            with self.assertRaises(ConfigurationError):
                RuntimeConfigB.from_yaml(path)

    def test_invalid_tracker_iou_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            TrackerConfig(iou_match=0.0).validate()


class AugmentTest(unittest.TestCase):
    @requires_opencv
    def test_output_keeps_shape_and_dtype(self) -> None:
        image = np.random.default_rng(3).integers(0, 255, (160, 160, 3), dtype=np.uint8)
        augmented = augment_crop(image, random.Random(1), 128)
        self.assertEqual(augmented.shape, (128, 128, 3))
        self.assertEqual(augmented.dtype, np.uint8)

    @requires_opencv
    def test_same_seed_gives_the_same_result(self) -> None:
        image = np.random.default_rng(5).integers(0, 255, (160, 160, 3), dtype=np.uint8)
        first = augment_crop(image, random.Random(11), 128)
        second = augment_crop(image, random.Random(11), 128)
        self.assertTrue(bool((first == second).all()))

    def test_tensor_is_chw_float_and_normalized(self) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        tensor = to_tensor(image)
        self.assertEqual(tensor.shape, (3, 32, 32))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertLess(abs(float(tensor.mean())), 0.6)


def _event(track_id: int, species: str) -> SpeciesEvent:
    return SpeciesEvent(
        track_id=track_id,
        species=species,
        confidence=0.82,
        vote_share=0.9,
        votes=5,
        box=BoundingBox(1.0, 2.0, 30.0, 40.0),
        frame_index=42,
        first_frame=10,
        frames_tracked=32,
        source_time_s=1.75,
        start_time_s=0.4,
        detection_score=0.91,
        votes_detail={species: 4.1},
    )


class SightingStoreTest(unittest.TestCase):
    def test_stores_and_counts_by_species(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            SightingStore(Path(directory) / "eventos.sqlite3") as store,
        ):
            store.add(
                _event(1, "anta"),
                source="video.mp4",
                camera_id="cam01",
                frames_tracked=32,
                evidence_path=None,
                detector_sha256="a" * 64,
                classifier_sha256="b" * 64,
                pipeline_version="0.1.0",
            )
            store.add(
                _event(2, "lobo_guara"),
                source="video.mp4",
                camera_id="cam01",
                frames_tracked=20,
                evidence_path=None,
                detector_sha256="a" * 64,
                classifier_sha256="b" * 64,
                pipeline_version="0.1.0",
            )
            self.assertEqual(store.count(), 2)
            self.assertEqual(store.count_by_species(), {"anta": 1, "lobo_guara": 1})

    def test_same_track_is_not_registered_twice(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            SightingStore(Path(directory) / "eventos.sqlite3") as store,
        ):
            arguments = {
                "source": "video.mp4",
                "camera_id": "cam01",
                "frames_tracked": 32,
                "evidence_path": None,
                "detector_sha256": None,
                "classifier_sha256": None,
                "pipeline_version": "0.1.0",
            }
            first = store.add(_event(1, "anta"), **arguments)
            second = store.add(_event(1, "anta"), **arguments)
        self.assertIsNotNone(first)
        self.assertIsNone(second)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
