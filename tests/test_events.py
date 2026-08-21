import tempfile
import unittest
from pathlib import Path

from animal_monitor.events import DetectionGate, EventStore
from animal_monitor.types import BoundingBox, Detection


class EventTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detection = Detection(0, "anta", 0.9, BoundingBox(1, 2, 10, 20))

    def test_gate_requires_consecutive_hits_and_cooldown(self) -> None:
        gate = DetectionGate(min_hits=2, cooldown_seconds=10)
        self.assertEqual(gate.update([self.detection], 0), [])
        self.assertEqual(gate.update([self.detection], 1), [self.detection])
        self.assertEqual(gate.update([self.detection], 2), [])
        self.assertEqual(gate.update([], 3), [])
        self.assertEqual(gate.update([self.detection], 10), [])
        self.assertEqual(gate.update([self.detection], 11), [self.detection])

    def test_store_persists_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            with EventStore(database) as store:
                event_id = store.add(self.detection, "camera-1", 2.0, 15, None)
                self.assertGreater(event_id, 0)
                self.assertEqual(store.count(), 1)


if __name__ == "__main__":
    unittest.main()
