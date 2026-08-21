from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from animal_monitor.config import load_yaml, resolve_from
from animal_monitor.errors import ConfigurationError

UNKNOWN_LABEL = "desconhecido"
AGNOSTIC_LABEL = "animal"


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Parâmetros do rastreador ByteTrack-lite."""

    max_age: int = 30
    min_hits: int = 3
    iou_match: float = 0.30
    low_score: float = 0.15
    max_tracks: int = 32

    def validate(self) -> None:
        if self.max_age < 1 or self.min_hits < 1 or self.max_tracks < 1:
            raise ConfigurationError("max_age, min_hits e max_tracks devem ser positivos.")
        if not 0.0 < self.iou_match < 1.0:
            raise ConfigurationError("'iou_match' deve estar entre 0 e 1.")
        if not 0.0 <= self.low_score <= 1.0:
            raise ConfigurationError("'low_score' deve estar entre 0 e 1.")


@dataclass(frozen=True, slots=True)
class RuntimeConfigB:
    """Configuração de execução do pipeline de dois estágios."""

    detector_confidence: float = 0.30
    detector_iou: float = 0.40
    detector_input: int = 416
    classifier_input: int = 128
    frame_stride: int = 1
    max_detections: int = 20
    max_crops_per_frame: int = 3
    crop_margin: float = 0.12
    classify_every: int = 5
    min_species_confidence: float = 0.50
    min_votes: int = 4
    min_vote_share: float = 0.65
    reject_unknown: bool = True
    save_evidence: bool = True
    output_dir: Path = Path("outputs/modelo_b")
    database: Path = Path("outputs/modelo_b/events_b.sqlite3")
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    intra_op_threads: int = 0
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuntimeConfigB:
        data = load_yaml(path)
        providers = data.get("providers", ["CPUExecutionProvider"])
        if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
            raise ConfigurationError("'providers' deve ser uma lista de strings.")
        tracker_data: dict[str, Any] = data.get("tracker") or {}
        if not isinstance(tracker_data, dict):
            raise ConfigurationError("'tracker' deve ser um objeto.")
        tracker = TrackerConfig(
            max_age=int(tracker_data.get("max_age", 30)),
            min_hits=int(tracker_data.get("min_hits", 3)),
            iou_match=float(tracker_data.get("iou_match", 0.30)),
            low_score=float(tracker_data.get("low_score", 0.15)),
            max_tracks=int(tracker_data.get("max_tracks", 32)),
        )
        config = cls(
            detector_confidence=float(data.get("detector_confidence", 0.30)),
            detector_iou=float(data.get("detector_iou", 0.40)),
            detector_input=int(data.get("detector_input", 416)),
            classifier_input=int(data.get("classifier_input", 128)),
            frame_stride=int(data.get("frame_stride", 1)),
            max_detections=int(data.get("max_detections", 20)),
            max_crops_per_frame=int(data.get("max_crops_per_frame", 3)),
            crop_margin=float(data.get("crop_margin", 0.12)),
            classify_every=int(data.get("classify_every", 5)),
            min_species_confidence=float(data.get("min_species_confidence", 0.50)),
            min_votes=int(data.get("min_votes", 4)),
            min_vote_share=float(data.get("min_vote_share", 0.65)),
            reject_unknown=bool(data.get("reject_unknown", True)),
            save_evidence=bool(data.get("save_evidence", True)),
            output_dir=resolve_from(path, data.get("output_dir", "../../outputs/modelo_b")),
            database=resolve_from(
                path, data.get("database", "../../outputs/modelo_b/events_b.sqlite3")
            ),
            providers=tuple(providers),
            intra_op_threads=int(data.get("intra_op_threads", 0)),
            tracker=tracker,
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, value in (
            ("detector_confidence", self.detector_confidence),
            ("detector_iou", self.detector_iou),
            ("min_species_confidence", self.min_species_confidence),
            ("min_vote_share", self.min_vote_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"'{name}' deve estar entre 0 e 1.")
        if self.detector_input <= 0 or self.detector_input % 32 != 0:
            raise ConfigurationError("'detector_input' deve ser positivo e múltiplo de 32.")
        if self.classifier_input <= 0 or self.classifier_input % 16 != 0:
            raise ConfigurationError("'classifier_input' deve ser positivo e múltiplo de 16.")
        if self.frame_stride < 1 or self.classify_every < 1:
            raise ConfigurationError("'frame_stride' e 'classify_every' devem ser positivos.")
        if self.max_detections < 1 or self.max_crops_per_frame < 1:
            raise ConfigurationError("max_detections e max_crops_per_frame devem ser positivos.")
        if self.min_votes < 1:
            raise ConfigurationError("'min_votes' deve ser positivo.")
        if not 0.0 <= self.crop_margin < 1.0:
            raise ConfigurationError("'crop_margin' deve estar entre 0 e 1.")
        if self.intra_op_threads < 0:
            raise ConfigurationError("'intra_op_threads' não pode ser negativo.")
        self.tracker.validate()


@dataclass(frozen=True, slots=True)
class ClassifierTrainConfig:
    """Configuração do treinamento do classificador de espécie."""

    crops_dir: Path
    output_dir: Path
    backbone: str = "mobilenet_v3_small"
    input_size: int = 128
    batch_size: int = 32
    epochs_head: int = 12
    epochs_finetune: int = 28
    learning_rate_head: float = 1.5e-3
    learning_rate_finetune: float = 1.5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    unfreeze_from: int = 8
    patience: int = 10
    seed: int = 42
    balanced_sampling: bool = True
    threads: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassifierTrainConfig:
        data = load_yaml(path)
        config = cls(
            crops_dir=resolve_from(path, data.get("crops_dir", "../../data/crops")),
            output_dir=resolve_from(path, data.get("output_dir", "../../models/modelo_b")),
            backbone=str(data.get("backbone", "mobilenet_v3_small")),
            input_size=int(data.get("input_size", 128)),
            batch_size=int(data.get("batch_size", 32)),
            epochs_head=int(data.get("epochs_head", 12)),
            epochs_finetune=int(data.get("epochs_finetune", 28)),
            learning_rate_head=float(data.get("learning_rate_head", 1.5e-3)),
            learning_rate_finetune=float(data.get("learning_rate_finetune", 1.5e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            label_smoothing=float(data.get("label_smoothing", 0.05)),
            unfreeze_from=int(data.get("unfreeze_from", 8)),
            patience=int(data.get("patience", 10)),
            seed=int(data.get("seed", 42)),
            balanced_sampling=bool(data.get("balanced_sampling", True)),
            threads=int(data.get("threads", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_size <= 0 or self.input_size % 16 != 0:
            raise ConfigurationError("'input_size' deve ser positivo e múltiplo de 16.")
        if self.batch_size < 1:
            raise ConfigurationError("'batch_size' deve ser positivo.")
        if self.epochs_head < 0 or self.epochs_finetune < 0:
            raise ConfigurationError("As épocas não podem ser negativas.")
        if self.epochs_head + self.epochs_finetune == 0:
            raise ConfigurationError("É preciso treinar pelo menos uma época.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigurationError("'label_smoothing' deve estar em [0, 1).")
        if self.patience < 1:
            raise ConfigurationError("'patience' deve ser positivo.")
