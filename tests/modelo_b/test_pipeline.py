from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from animal_monitor_b.config import RuntimeConfigB, TrackerConfig
from animal_monitor_b.runtime.pipeline import TwoStagePipeline

LABELS = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]


class FakePresenceDetector:
    """Detector determinístico: devolve as caixas programadas por quadro."""

    label = "animal"
    model_path = Path("fake_detector.onnx")

    def __init__(self, boxes: np.ndarray, score: float = 0.9) -> None:
        self.boxes = boxes
        self.score = score
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        scores = np.full((self.boxes.shape[0],), self.score, dtype=np.float32)
        return self.boxes.astype(np.float32), scores

    def info(self) -> dict[str, object]:
        return {"model": str(self.model_path)}


class FakeSpeciesClassifier:
    """Classificador determinístico: devolve as probabilidades programadas."""

    model_path = Path("fake_classifier.onnx")

    def __init__(self, probabilities: list[np.ndarray], labels: list[str] | None = None) -> None:
        self.labels = labels or LABELS
        self.probabilities = probabilities
        self.calls = 0
        self.classified_boxes = 0

    def classify(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        row = self.probabilities[min(self.calls, len(self.probabilities) - 1)]
        self.calls += 1
        count = np.asarray(boxes).reshape(-1, 4).shape[0]
        self.classified_boxes += count
        return np.tile(row.astype(np.float32), (count, 1))

    def info(self) -> dict[str, object]:
        return {"model": str(self.model_path), "labels": self.labels}


def _probability(label: str, value: float = 0.9) -> np.ndarray:
    row = np.full(len(LABELS), (1.0 - value) / (len(LABELS) - 1), dtype=np.float32)
    row[LABELS.index(label)] = value
    return row


def _config(**overrides: object) -> RuntimeConfigB:
    defaults: dict[str, object] = {
        "classify_every": 1,
        "min_votes": 3,
        "min_vote_share": 0.65,
        "min_species_confidence": 0.50,
        "tracker": TrackerConfig(min_hits=2, iou_match=0.2, low_score=0.1),
    }
    defaults.update(overrides)
    config = RuntimeConfigB(**defaults)  # type: ignore[arg-type]
    config.validate()
    return config


def _pipeline(probabilities: list[np.ndarray], boxes: np.ndarray, **overrides: object):
    config = _config(**overrides)
    detector = FakePresenceDetector(boxes)
    classifier = FakeSpeciesClassifier(probabilities)
    return TwoStagePipeline(detector, classifier, config), detector, classifier


class VotingTest(unittest.TestCase):
    def test_consistent_votes_emit_one_event_per_individual(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("lobo_guara")], boxes)
        events = []
        for frame in range(8):
            result = pipeline.process(np.zeros((200, 200, 3), np.uint8), frame, frame / 24.0)
            events.extend(result.events)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].species, "lobo_guara")
        self.assertEqual(events[0].track_id, 1)
        self.assertGreaterEqual(events[0].votes, 3)

    def test_divided_vote_does_not_emit(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        alternating = [
            _probability("anta"),
            _probability("lobo_guara"),
            _probability("anta"),
            _probability("lobo_guara"),
            _probability("anta"),
            _probability("lobo_guara"),
        ]
        pipeline, _, _ = _pipeline(alternating, boxes, min_vote_share=0.70)
        events = []
        for frame in range(6):
            events.extend(
                pipeline.process(np.zeros((200, 200, 3), np.uint8), frame, frame / 24.0).events
            )
        self.assertEqual(events, [])

    def test_unknown_class_is_rejected(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("desconhecido")], boxes)
        events = []
        for frame in range(8):
            events.extend(
                pipeline.process(np.zeros((200, 200, 3), np.uint8), frame, frame / 24.0).events
            )
        self.assertEqual(events, [])

    def test_quorum_blocks_emission_with_few_votes(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("anta")], boxes, min_votes=10)
        events = []
        for frame in range(6):
            events.extend(
                pipeline.process(np.zeros((200, 200, 3), np.uint8), frame, frame / 24.0).events
            )
        self.assertEqual(events, [])

    def test_classify_every_reduces_classifier_calls(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, classifier = _pipeline(
            [_probability("anta")], boxes, classify_every=5, min_votes=2
        )
        for frame in range(10):
            pipeline.process(np.zeros((200, 200, 3), np.uint8), frame, frame / 24.0)
        self.assertEqual(classifier.calls, 2)

    def test_two_individuals_produce_two_events(self) -> None:
        boxes = np.array([[10, 10, 90, 90], [300, 300, 380, 380]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("anta")], boxes)
        events = []
        for frame in range(8):
            events.extend(
                pipeline.process(np.zeros((400, 400, 3), np.uint8), frame, frame / 24.0).events
            )
        self.assertEqual(len(events), 2)
        self.assertEqual({event.track_id for event in events}, {1, 2})


class DetectSpeciesTest(unittest.TestCase):
    def test_score_is_the_product_of_both_stages(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("anta", 0.8)], boxes)
        detections = pipeline.detect_species(np.zeros((200, 200, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "anta")
        self.assertAlmostEqual(detections[0].confidence, 0.9 * 0.8, places=5)

    def test_unknown_prediction_is_dropped(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("desconhecido")], boxes)
        self.assertEqual(pipeline.detect_species(np.zeros((200, 200, 3), np.uint8)), [])

    def test_species_indices_follow_the_class_order(self) -> None:
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        pipeline, _, _ = _pipeline([_probability("lobo_guara")], boxes)
        detections = pipeline.detect_species(np.zeros((200, 200, 3), np.uint8))
        self.assertEqual(detections[0].class_id, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
