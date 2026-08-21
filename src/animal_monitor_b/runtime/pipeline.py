"""Pipeline de dois estágios com agregação temporal por indivíduo.

Fluxo por quadro:

1. detector de presença devolve caixas de `animal`;
2. rastreador associa as caixas a trilhas (indivíduos);
3. o classificador roda apenas em algumas caixas e apenas a cada `classify_every` quadros
   por trilha — a espécie de um indivíduo não muda ao longo da trilha, então reclassificar
   todo quadro é desperdício de ciclo de CPU;
4. cada classificação vira um voto ponderado na trilha;
5. o evento é emitido **uma vez por indivíduo**, quando a votação atinge quórum e margem.

É a diferença estrutural em relação ao Modelo A: lá o alerta dependia de 3 quadros
consecutivos com a mesma classe, sem identidade e sem opção de recusa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animal_monitor.types import BoundingBox, Detection
from animal_monitor_b.config import UNKNOWN_LABEL, RuntimeConfigB
from animal_monitor_b.runtime.crop_classifier import SpeciesClassifier
from animal_monitor_b.runtime.presence_detector import PresenceDetector
from animal_monitor_b.runtime.tracker import ByteTrackLite, Track


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Estado de uma trilha em um quadro, para desenho e depuração."""

    track_id: int
    box: BoundingBox
    detection_score: float
    species: str
    species_confidence: float
    votes: int
    confirmed: bool
    emitted: bool

    @property
    def display_label(self) -> str:
        return f"#{self.track_id} {self.species}"


@dataclass(frozen=True, slots=True)
class SpeciesEvent:
    """Avistamento consolidado de um indivíduo."""

    track_id: int
    species: str
    confidence: float
    vote_share: float
    votes: int
    box: BoundingBox
    frame_index: int
    first_frame: int
    frames_tracked: int
    source_time_s: float
    start_time_s: float
    detection_score: float
    votes_detail: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "species": self.species,
            "confidence": round(self.confidence, 6),
            "vote_share": round(self.vote_share, 6),
            "votes": self.votes,
            "frame_index": self.frame_index,
            "first_frame": self.first_frame,
            "frames_tracked": self.frames_tracked,
            "source_time_s": round(self.source_time_s, 3),
            "start_time_s": round(self.start_time_s, 3),
            "detection_score": round(self.detection_score, 6),
            "votes_detail": {key: round(value, 6) for key, value in self.votes_detail.items()},
            "box": {
                "x1": round(self.box.x1, 2),
                "y1": round(self.box.y1, 2),
                "x2": round(self.box.x2, 2),
                "y2": round(self.box.y2, 2),
            },
        }


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    source_time_s: float
    observations: list[TrackObservation]
    events: list[SpeciesEvent]
    detections: int
    classified_crops: int


class TwoStagePipeline:
    """Junta detector de presença, rastreador e classificador de espécie."""

    def __init__(
        self,
        detector: PresenceDetector,
        classifier: SpeciesClassifier,
        config: RuntimeConfigB,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.config = config
        self.tracker = ByteTrackLite(
            max_age=config.tracker.max_age,
            min_hits=config.tracker.min_hits,
            iou_match=config.tracker.iou_match,
            low_score=config.tracker.low_score,
            high_score=max(config.detector_confidence, config.tracker.low_score),
            max_tracks=config.tracker.max_tracks,
        )
        self.species_labels = [
            label for label in classifier.labels if label != UNKNOWN_LABEL
        ]
        self.finished_tracks: list[Track] = []

    @classmethod
    def from_models(
        cls,
        detector_model: str | Path,
        classifier_model: str | Path,
        config: RuntimeConfigB,
        labels: list[str] | None = None,
    ) -> TwoStagePipeline:
        detector = PresenceDetector(detector_model, config)
        classifier = SpeciesClassifier(classifier_model, config, labels=labels)
        return cls(detector, classifier, config)

    def _vote_weight(self, detection_score: float, class_probability: float) -> float:
        return float(detection_score) * float(class_probability)

    def _decide(self, track: Track) -> tuple[str, float, float]:
        """Espécie vencedora da trilha, confiança média e participação nos votos."""
        ranking = track.top_votes()
        if not ranking:
            return UNKNOWN_LABEL, 0.0, 0.0
        total = sum(weight for _, weight in ranking)
        label, weight = ranking[0]
        share = weight / total if total > 0 else 0.0
        confidence = weight / max(1, track.vote_count)
        return label, confidence, share

    def _maybe_emit(self, track: Track) -> SpeciesEvent | None:
        if track.emitted or not track.confirmed:
            return None
        if track.vote_count < self.config.min_votes:
            return None
        label, confidence, share = self._decide(track)
        if self.config.reject_unknown and label == UNKNOWN_LABEL:
            return None
        if share < self.config.min_vote_share:
            return None
        if confidence < self.config.min_species_confidence:
            return None
        track.emitted = True
        track.emitted_label = label
        return SpeciesEvent(
            track_id=track.track_id,
            species=label,
            confidence=confidence,
            vote_share=share,
            votes=track.vote_count,
            box=track.bounding_box(),
            frame_index=track.last_frame,
            first_frame=track.first_frame,
            frames_tracked=track.hits,
            source_time_s=track.last_time_s,
            start_time_s=track.first_time_s,
            detection_score=track.score,
            votes_detail=dict(track.votes),
        )

    def process(self, frame: np.ndarray, frame_index: int, source_time_s: float) -> FrameResult:
        boxes, scores = self.detector.detect(frame)
        active, finished = self.tracker.update(boxes, scores, frame_index, source_time_s)
        self.finished_tracks.extend(finished)

        pending = [
            track
            for track in active
            if frame_index - track.last_classified_frame >= self.config.classify_every
        ]
        pending.sort(key=lambda track: track.score, reverse=True)
        pending = pending[: self.config.max_crops_per_frame]

        classified = 0
        if pending:
            crop_boxes = np.stack([track.box for track in pending], axis=0)
            probabilities = self.classifier.classify(frame, crop_boxes)
            classified = int(probabilities.shape[0])
            for track, row in zip(pending, probabilities, strict=True):
                best = int(np.argmax(row))
                label = self.classifier.labels[best]
                track.add_vote(label, self._vote_weight(track.score, float(row[best])))
                track.last_classified_frame = frame_index

        events: list[SpeciesEvent] = []
        for track in active:
            event = self._maybe_emit(track)
            if event is not None:
                events.append(event)

        observations = []
        for track in active:
            label, confidence, _ = self._decide(track)
            observations.append(
                TrackObservation(
                    track_id=track.track_id,
                    box=track.bounding_box(),
                    detection_score=track.score,
                    species=track.emitted_label or label,
                    species_confidence=confidence,
                    votes=track.vote_count,
                    confirmed=track.confirmed,
                    emitted=track.emitted,
                )
            )
        return FrameResult(
            frame_index=frame_index,
            source_time_s=source_time_s,
            observations=observations,
            events=events,
            detections=int(boxes.shape[0]),
            classified_crops=classified,
        )

    def close(self) -> list[Track]:
        finished = self.tracker.flush()
        self.finished_tracks.extend(finished)
        return finished

    def detect_species(self, frame: np.ndarray) -> list[Detection]:
        """Modo sem estado, quadro a quadro, para avaliar em imagens soltas.

        Cada caixa recebe a espécie mais provável e o escore combinado
        `confiança do detector x probabilidade da espécie`. É a forma de medir o Modelo B
        com a mesma métrica de detecção usada no Modelo A.
        """
        boxes, scores = self.detector.detect(frame)
        if boxes.size == 0:
            return []
        probabilities = self.classifier.classify(frame, boxes)
        detections: list[Detection] = []
        for box, score, row in zip(boxes, scores, probabilities, strict=True):
            best = int(np.argmax(row))
            label = self.classifier.labels[best]
            if self.config.reject_unknown and label == UNKNOWN_LABEL:
                continue
            if label not in self.species_labels:
                continue
            detections.append(
                Detection(
                    class_id=self.species_labels.index(label),
                    class_name=label,
                    confidence=float(score) * float(row[best]),
                    box=BoundingBox(*(float(value) for value in box)),
                )
            )
        return detections

    def info(self) -> dict[str, object]:
        return {
            "detector": self.detector.info(),
            "classifier": self.classifier.info(),
            "species": self.species_labels,
            "tracker": {
                "max_age": self.config.tracker.max_age,
                "min_hits": self.config.tracker.min_hits,
                "iou_match": self.config.tracker.iou_match,
                "low_score": self.config.tracker.low_score,
            },
            "voting": {
                "classify_every": self.config.classify_every,
                "min_votes": self.config.min_votes,
                "min_vote_share": self.config.min_vote_share,
                "min_species_confidence": self.config.min_species_confidence,
                "reject_unknown": self.config.reject_unknown,
            },
        }
