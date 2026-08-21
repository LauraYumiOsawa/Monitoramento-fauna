from __future__ import annotations

import unittest

import numpy as np

from animal_monitor_b.runtime.tracker import ByteTrackLite, greedy_match, iou_matrix


class IouMatrixTest(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self) -> None:
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertAlmostEqual(float(iou_matrix(boxes, boxes)[0, 0]), 1.0, places=5)

    def test_empty_inputs_return_empty_matrix(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(iou_matrix(empty, empty).shape, (0, 0))


class GreedyMatchTest(unittest.TestCase):
    def test_picks_highest_similarity_first(self) -> None:
        similarity = np.array([[0.9, 0.4], [0.8, 0.85]], dtype=np.float32)
        pairs, unmatched_rows, unmatched_columns = greedy_match(similarity, 0.3)
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])
        self.assertEqual(unmatched_rows, [])
        self.assertEqual(unmatched_columns, [])

    def test_respects_threshold(self) -> None:
        similarity = np.array([[0.2]], dtype=np.float32)
        pairs, unmatched_rows, unmatched_columns = greedy_match(similarity, 0.3)
        self.assertEqual(pairs, [])
        self.assertEqual(unmatched_rows, [0])
        self.assertEqual(unmatched_columns, [0])


class ByteTrackLiteTest(unittest.TestCase):
    def test_keeps_same_id_for_a_moving_object(self) -> None:
        tracker = ByteTrackLite(min_hits=2, iou_match=0.2, high_score=0.4)
        identifiers = []
        for frame in range(6):
            offset = frame * 4.0
            boxes = np.array([[10 + offset, 10, 60 + offset, 60]], dtype=np.float32)
            active, _ = tracker.update(boxes, np.array([0.9]), frame, frame / 30.0)
            self.assertEqual(len(active), 1)
            identifiers.append(active[0].track_id)
        self.assertEqual(set(identifiers), {1})
        self.assertTrue(active[0].confirmed)

    def test_two_objects_receive_distinct_ids(self) -> None:
        tracker = ByteTrackLite(min_hits=1, high_score=0.4)
        boxes = np.array([[0, 0, 20, 20], [100, 100, 140, 140]], dtype=np.float32)
        active, _ = tracker.update(boxes, np.array([0.9, 0.8]), 0, 0.0)
        self.assertEqual({track.track_id for track in active}, {1, 2})

    def test_track_is_finished_after_max_age(self) -> None:
        tracker = ByteTrackLite(min_hits=1, max_age=2, high_score=0.4)
        tracker.update(np.array([[0, 0, 20, 20]], dtype=np.float32), np.array([0.9]), 0, 0.0)
        empty = np.zeros((0, 4), dtype=np.float32)
        finished: list = []
        for frame in range(1, 5):
            _, ended = tracker.update(empty, np.zeros((0,), dtype=np.float32), frame, frame / 30.0)
            finished.extend(ended)
        self.assertEqual(len(finished), 1)
        self.assertEqual(tracker.tracks, [])

    def test_low_score_detection_keeps_the_track_alive(self) -> None:
        tracker = ByteTrackLite(min_hits=1, high_score=0.5, low_score=0.1, iou_match=0.2)
        tracker.update(np.array([[0, 0, 40, 40]], dtype=np.float32), np.array([0.9]), 0, 0.0)
        active, _ = tracker.update(
            np.array([[2, 2, 42, 42]], dtype=np.float32), np.array([0.2]), 1, 0.03
        )
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].track_id, 1)
        self.assertEqual(active[0].hits, 2)

    def test_low_score_detection_alone_does_not_create_track(self) -> None:
        tracker = ByteTrackLite(min_hits=1, high_score=0.5, low_score=0.1)
        active, _ = tracker.update(
            np.array([[0, 0, 40, 40]], dtype=np.float32), np.array([0.2]), 0, 0.0
        )
        self.assertEqual(active, [])
        self.assertEqual(tracker.created_tracks, 0)

    def test_votes_accumulate_and_rank(self) -> None:
        tracker = ByteTrackLite(min_hits=1, high_score=0.4)
        active, _ = tracker.update(
            np.array([[0, 0, 40, 40]], dtype=np.float32), np.array([0.9]), 0, 0.0
        )
        track = active[0]
        track.add_vote("anta", 0.4)
        track.add_vote("lobo_guara", 0.9)
        track.add_vote("lobo_guara", 0.8)
        self.assertEqual(track.vote_count, 3)
        self.assertEqual(track.top_votes()[0][0], "lobo_guara")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
