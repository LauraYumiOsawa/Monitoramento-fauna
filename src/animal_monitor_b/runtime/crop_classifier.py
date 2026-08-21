"""Classificador de espécie sobre recortes, executado por ONNX Runtime.

Só roda quando o detector encontra algo, em lote com as caixas do quadro. Em uma cena
vazia — a maior parte do tempo em campo — o custo é zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_b.augment import resize_square, to_tensor
from animal_monitor_b.config import RuntimeConfigB
from animal_monitor_b.data_prep import crop_square, expand_box
from animal_monitor_b.runtime.presence_detector import create_session


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-12)


class SpeciesClassifier:
    """Rede leve de classificação de espécie com classe explícita de rejeição."""

    def __init__(
        self,
        model_path: str | Path,
        config: RuntimeConfigB,
        labels: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config
        self.session = create_session(
            self.model_path, config.providers, config.intra_op_threads
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ConfigurationError(
                f"Esperado um tensor de entrada; o modelo possui {len(inputs)}."
            )
        self.input = inputs[0]
        shape = self.input.shape
        height, width = (shape[-2], shape[-1]) if len(shape) == 4 else (None, None)
        if isinstance(height, int) and isinstance(width, int) and height > 0:
            self.input_size = int(height)
        else:
            self.input_size = config.classifier_input
        self.dynamic_batch = not isinstance(shape[0], int)
        self.labels = labels if labels is not None else self._load_labels()

    def _load_labels(self) -> list[str]:
        candidate = self.model_path.with_name("classifier_metadata.json")
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            labels = data.get("labels")
            if isinstance(labels, list) and labels:
                return [str(label) for label in labels]
        raise ConfigurationError(
            f"Não foi possível descobrir os rótulos do classificador. Esperado {candidate}."
        )

    def crops_from(
        self, frame: np.ndarray, boxes: np.ndarray, margin: float | None = None
    ) -> list[np.ndarray]:
        """Extrai recortes quadrados com margem de contexto, no tamanho de entrada da rede."""
        height, width = frame.shape[:2]
        used_margin = self.config.crop_margin if margin is None else margin
        crops: list[np.ndarray] = []
        for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
            region = expand_box(tuple(float(value) for value in box), used_margin, width, height)
            crops.append(crop_square(frame, region, self.input_size))
        return crops

    def classify_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Retorna a matriz de probabilidades (N x classes)."""
        if not crops:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        tensors = np.stack(
            [to_tensor(resize_square(crop, self.input_size)) for crop in crops], axis=0
        )
        if self.dynamic_batch:
            outputs = self.session.run(None, {self.input.name: tensors})[0]
            return _softmax(np.asarray(outputs, dtype=np.float32))
        results = [
            self.session.run(None, {self.input.name: tensors[index : index + 1]})[0]
            for index in range(tensors.shape[0])
        ]
        return _softmax(np.concatenate(results, axis=0).astype(np.float32))

    def classify(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        return self.classify_crops(self.crops_from(frame, boxes))

    def info(self) -> dict[str, object]:
        return {
            "model": str(self.model_path),
            "input_name": self.input.name,
            "input_shape": list(self.input.shape),
            "runtime_input": [1, 3, self.input_size, self.input_size],
            "dynamic_batch": self.dynamic_batch,
            "providers": self.session.get_providers(),
            "labels": self.labels,
        }
