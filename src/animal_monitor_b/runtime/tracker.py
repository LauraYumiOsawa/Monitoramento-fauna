"""Rastreador multi-objeto leve, inspirado no ByteTrack, em NumPy puro.

Substitui o `DetectionGate` do Modelo A, que contava quadros consecutivos por classe e
não tinha noção de indivíduo. Ali, um falso positivo persistente por 3 quadros virava
evento confirmado — foi exatamente o que produziu os dois eventos falsos de "anta".

Duas ideias do ByteTrack são mantidas, sem trazer SciPy/lap para o embarcado:

* **associação em dois estágios**: primeiro as detecções de alta confiança, depois as de
  baixa confiança contra as trilhas que sobraram. Isso recupera animais parcialmente
  ocluídos sem baixar o limiar global e sem criar trilhas espúrias;
* **ciclo de vida da trilha**: nasce provisória, é confirmada após `min_hits` e só morre
  depois de `max_age` quadros sem correspondência.

A associação usa custo IoU com atribuição gulosa. Com poucas caixas por quadro (o caso de
armadilha fotográfica) a solução gulosa coincide com a ótima na prática e evita
`scipy.optimize.linear_sum_assignment`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from animal_monitor.types import BoundingBox


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Matriz IoU (trilhas x detecções) em coordenadas xyxy."""
    if tracks.size == 0 or detections.size == 0:
        return np.zeros((tracks.shape[0], detections.shape[0]), dtype=np.float32)
    x1 = np.maximum(tracks[:, None, 0], detections[None, :, 0])
    y1 = np.maximum(tracks[:, None, 1], detections[None, :, 1])
    x2 = np.minimum(tracks[:, None, 2], detections[None, :, 2])
    y2 = np.minimum(tracks[:, None, 3], detections[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    track_area = np.maximum(0.0, tracks[:, 2] - tracks[:, 0]) * np.maximum(
        0.0, tracks[:, 3] - tracks[:, 1]
    )
    detection_area = np.maximum(0.0, detections[:, 2] - detections[:, 0]) * np.maximum(
        0.0, detections[:, 3] - detections[:, 1]
    )
    union = track_area[:, None] + detection_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).astype(np.float32)


def greedy_match(
    similarity: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Casamento guloso por similaridade decrescente."""
    rows, columns = similarity.shape
    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    if rows and columns:
        flat = np.argsort(similarity, axis=None)[::-1]
        for position in flat:
            row, column = divmod(int(position), columns)
            if similarity[row, column] < threshold:
                break
            if row in used_rows or column in used_columns:
                continue
            pairs.append((row, column))
            used_rows.add(row)
            used_columns.add(column)
    unmatched_rows = [row for row in range(rows) if row not in used_rows]
    unmatched_columns = [column for column in range(columns) if column not in used_columns]
    return pairs, unmatched_rows, unmatched_columns


@dataclass(slots=True)
class Track:
    """Trilha de um indivíduo, com o histórico de votos de espécie."""

    track_id: int
    box: np.ndarray
    score: float
    min_hits: int = 3
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    first_frame: int = 0
    last_frame: int = 0
    first_time_s: float = 0.0
    last_time_s: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    votes: dict[str, float] = field(default_factory=dict)
    vote_count: int = 0
    last_classified_frame: int = -10**9
    emitted: bool = False
    emitted_label: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def predict(self) -> np.ndarray:
        """Extrapolação linear simples da caixa (sem Kalman, sem custo de matriz)."""
        return self.box + self.velocity

    def update(self, box: np.ndarray, score: float, frame_index: int, time_s: float) -> None:
        previous = self.box
        self.box = box.astype(np.float32, copy=True)
        self.velocity = 0.5 * self.velocity + 0.5 * (self.box - previous)
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.last_time_s = time_s

    def add_vote(self, label: str, weight: float) -> None:
        self.votes[label] = self.votes.get(label, 0.0) + float(weight)
        self.vote_count += 1

    def top_votes(self) -> list[tuple[str, float]]:
        return sorted(self.votes.items(), key=lambda item: item[1], reverse=True)

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(*(float(value) for value in self.box))


class ByteTrackLite:
    """Rastreador de caixas sem classe, alimentado pelo detector de presença."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_match: float = 0.30,
        low_score: float = 0.15,
        high_score: float = 0.45,
        max_tracks: int = 32,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_match = iou_match
        self.low_score = low_score
        self.high_score = high_score
        self.max_tracks = max_tracks
        self.tracks: list[Track] = []
        self._next_id = 1

    @property
    def created_tracks(self) -> int:
        """Total de indivíduos distintos vistos desde o início."""
        return self._next_id - 1

    def _create(self, box: np.ndarray, score: float, frame_index: int, time_s: float) -> Track:
        track = Track(
            track_id=self._next_id,
            box=box.astype(np.float32, copy=True),
            score=float(score),
            min_hits=self.min_hits,
            first_frame=frame_index,
            last_frame=frame_index,
            first_time_s=time_s,
            last_time_s=time_s,
        )
        self._next_id += 1
        return track

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        frame_index: int,
        time_s: float,
    ) -> tuple[list[Track], list[Track]]:
        """Avança um quadro. Retorna (trilhas ativas, trilhas encerradas neste quadro)."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        high = scores >= self.high_score
        low = (scores >= self.low_score) & ~high
        predicted = (
            np.stack([track.predict() for track in self.tracks], axis=0)
            if self.tracks
            else np.zeros((0, 4), dtype=np.float32)
        )

        pairs, unmatched_tracks, unmatched_high = greedy_match(
            iou_matrix(predicted, boxes[high]), self.iou_match
        )
        high_indices = np.flatnonzero(high)
        for track_index, detection_index in pairs:
            index = int(high_indices[detection_index])
            self.tracks[track_index].update(boxes[index], scores[index], frame_index, time_s)

        low_indices = np.flatnonzero(low)
        if unmatched_tracks and low_indices.size:
            remaining = predicted[unmatched_tracks]
            low_pairs, _, _ = greedy_match(
                iou_matrix(remaining, boxes[low_indices]), max(0.2, self.iou_match * 0.8)
            )
            for local_track, detection_index in low_pairs:
                track_index = unmatched_tracks[local_track]
                index = int(low_indices[detection_index])
                self.tracks[track_index].update(boxes[index], scores[index], frame_index, time_s)

        for detection_index in unmatched_high:
            if len(self.tracks) >= self.max_tracks:
                break
            index = int(high_indices[detection_index])
            self.tracks.append(self._create(boxes[index], scores[index], frame_index, time_s))

        finished = [track for track in self.tracks if track.time_since_update > self.max_age]
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        active = [track for track in self.tracks if track.time_since_update == 0]
        return active, finished

    def flush(self) -> list[Track]:
        """Encerra todas as trilhas (fim do vídeo)."""
        finished = self.tracks
        self.tracks = []
        return finished
