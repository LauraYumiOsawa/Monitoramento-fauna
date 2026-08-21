from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from animal_monitor.errors import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {config_path}")
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML inválido em {config_path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigurationError(f"A raiz do YAML deve ser um objeto: {config_path}")
    return content


def resolve_from(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).expanduser().resolve().parent / path).resolve()


def load_class_names(dataset_yaml: str | Path) -> dict[int, str]:
    content = load_yaml(dataset_yaml)
    names = content.get("names")
    if isinstance(names, list):
        parsed = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            parsed = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("As chaves de 'names' devem ser números inteiros.") from exc
    else:
        raise ConfigurationError("O dataset YAML precisa de 'names' como lista ou mapa.")

    if not parsed:
        raise ConfigurationError("A lista de classes não pode estar vazia.")
    expected = list(range(len(parsed)))
    if sorted(parsed) != expected:
        raise ConfigurationError(
            f"Os IDs das classes devem ser consecutivos a partir de zero: esperado {expected}."
        )
    return dict(sorted(parsed.items()))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    confidence: float = 0.35
    iou_threshold: float = 0.45
    input_size: int = 640
    frame_stride: int = 1
    min_hits: int = 3
    cooldown_seconds: float = 15.0
    max_detections: int = 100
    class_agnostic_nms: bool = False
    save_evidence: bool = True
    output_dir: Path = Path("outputs")
    database: Path = Path("outputs/events.sqlite3")
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    intra_op_threads: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuntimeConfig:
        data = load_yaml(path)
        output_dir = resolve_from(path, data.get("output_dir", "../outputs"))
        database = resolve_from(path, data.get("database", "../outputs/events.sqlite3"))
        providers = data.get("providers", ["CPUExecutionProvider"])
        if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
            raise ConfigurationError("'providers' deve ser uma lista de strings.")

        config = cls(
            confidence=float(data.get("confidence", 0.35)),
            iou_threshold=float(data.get("iou_threshold", 0.45)),
            input_size=int(data.get("input_size", 640)),
            frame_stride=int(data.get("frame_stride", 1)),
            min_hits=int(data.get("min_hits", 3)),
            cooldown_seconds=float(data.get("cooldown_seconds", 15.0)),
            max_detections=int(data.get("max_detections", 100)),
            class_agnostic_nms=bool(data.get("class_agnostic_nms", False)),
            save_evidence=bool(data.get("save_evidence", True)),
            output_dir=output_dir,
            database=database,
            providers=tuple(providers),
            intra_op_threads=int(data.get("intra_op_threads", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ConfigurationError("'confidence' deve estar entre 0 e 1.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfigurationError("'iou_threshold' deve estar entre 0 e 1.")
        if self.input_size <= 0 or self.input_size % 32 != 0:
            raise ConfigurationError("'input_size' deve ser positivo e múltiplo de 32.")
        if self.frame_stride < 1 or self.min_hits < 1 or self.max_detections < 1:
            raise ConfigurationError("frame_stride, min_hits e max_detections devem ser positivos.")
        if self.cooldown_seconds < 0 or self.intra_op_threads < 0:
            raise ConfigurationError(
                "cooldown_seconds e intra_op_threads não podem ser negativos."
            )
