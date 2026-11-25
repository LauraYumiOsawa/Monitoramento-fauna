"""Detector YOLO e classificador de recortes independentes do runtime.

As duas classes expõem as mesmas interfaces que o projeto já usa, então entram sem
adaptação nos laços existentes:

* `YoloDetector.infer(frame) -> list[Detection]` — interface do Modelo A
  (`animal_monitor.monitoring.run_monitoring`);
* `YoloDetector.detect(frame) -> (caixas, escores)` e `CropClassifier.classify(frame, caixas)`
  — interfaces do Modelo B (`animal_monitor_b.runtime.pipeline.TwoStagePipeline`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor.inference.prepost import (
    build_detections,
    decode_yolo_output,
    letterbox,
    undo_letterbox,
)
from animal_monitor.types import Detection
from animal_monitor_b.augment import IMAGENET_MEAN, IMAGENET_STD, resize_square
from animal_monitor_b.data_prep import crop_square, expand_box
from animal_monitor_hw.backends.base import Runner, rgb_to_tensor
from animal_monitor_hw.yolo_heads import (
    decode_boxes_scores_classes,
    decode_hailo_nms_by_class,
    decode_split_head,
)

OUTPUT_FORMATS = {"standard", "split", "hailo_nms", "boxes_scores_classes"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-12)


class YoloDetector:
    """YOLOv8 sobre qualquer `Runner`, com o pós-processamento escolhido pelo formato.

    `output_format`:
      * `standard` — tensor único `(1, 4 + nc, N)` do export padrão (ONNX Runtime);
      * `split` — 6 saídas box/cls por escala (`yolo_heads.export_split_head_onnx`),
        usadas em RKNN, RV1106 e outras NPUs INT8;
      * `hailo_nms` — NMS por classe calculado no Hailo;
      * `boxes_scores_classes` — NMS embutido no grafo (IMX500/Ultralytics).
    """

    def __init__(
        self,
        runner: Runner,
        names: Mapping[int, str],
        confidence: float,
        iou_threshold: float,
        max_detections: int = 100,
        output_format: str = "standard",
        class_agnostic_nms: bool = False,
        split_layout: str = "nchw",
        model_path: str | Path | None = None,
    ) -> None:
        if output_format not in OUTPUT_FORMATS:
            raise ConfigurationError(
                f"'output_format' inválido: {output_format}. Use um de {sorted(OUTPUT_FORMATS)}."
            )
        if not names:
            raise ConfigurationError("O detector precisa de pelo menos uma classe.")
        self.runner = runner
        self.names = dict(names)
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.output_format = output_format
        self.class_agnostic_nms = class_agnostic_nms
        self.split_layout = split_layout
        self.model_path = Path(model_path) if model_path else Path(
            str(runner.describe().get("model", "modelo"))
        )
        self.label = self.names[0] if len(self.names) == 1 else "multiclasse"

    @property
    def input_hw(self) -> tuple[int, int]:
        return self.runner.spec.height, self.runner.spec.width

    def _decode(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.names)
        if self.output_format == "standard":
            return decode_yolo_output(
                outputs[0],
                class_count=count,
                confidence_threshold=self.confidence,
                iou_threshold=self.iou_threshold,
                class_agnostic_nms=self.class_agnostic_nms or count == 1,
                max_detections=self.max_detections,
            )
        if self.output_format == "split":
            return decode_split_head(
                outputs,
                self.input_hw,
                count,
                self.confidence,
                self.iou_threshold,
                self.max_detections,
                class_agnostic_nms=self.class_agnostic_nms or count == 1,
                layout=self.split_layout,
            )
        if self.output_format == "hailo_nms":
            per_class = outputs[0] if len(outputs) == 1 else outputs
            return decode_hailo_nms_by_class(
                per_class, self.input_hw, self.confidence, self.max_detections
            )
        boxes, scores, classes = outputs[0], outputs[1], outputs[2]
        return decode_boxes_scores_classes(
            boxes,
            scores,
            classes,
            self.input_hw,
            self.confidence,
            normalized=bool(np.nanmax(np.asarray(boxes, dtype=np.float32), initial=0.0) <= 1.5),
            max_detections=self.max_detections,
        )

    def detect_all(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Caixas xyxy em pixels do quadro original, escores e classes."""
        padded, transform = letterbox(frame, self.input_hw)
        tensor = rgb_to_tensor(padded[:, :, ::-1], self.runner.spec)
        boxes, scores, classes = self._decode(self.runner.run(tensor))
        if boxes.size == 0:
            return boxes.reshape(0, 4), scores, classes
        return undo_letterbox(boxes, transform), scores, classes

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interface do estágio 1 do Modelo B: caixas e escores, sem classe."""
        boxes, scores, _ = self.detect_all(frame)
        return boxes, scores

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Interface do Modelo A."""
        padded, transform = letterbox(frame, self.input_hw)
        tensor = rgb_to_tensor(padded[:, :, ::-1], self.runner.spec)
        boxes, scores, classes = self._decode(self.runner.run(tensor))
        return build_detections(boxes, scores, classes, self.names, transform)

    def warmup(self, repetitions: int = 2) -> None:
        dummy = np.zeros((*self.input_hw, 3), dtype=np.uint8)
        for _ in range(max(0, repetitions)):
            self.detect_all(dummy)

    def info(self) -> dict[str, object]:
        return {
            **self.runner.describe(),
            "output_format": self.output_format,
            "runtime_input": [1, 3, *self.input_hw],
            "classes": self.names,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
        }


class CropClassifier:
    """Classificador de espécie sobre recortes, com a classe `desconhecido` de rejeição.

    Reproduz o pré-processamento de `animal_monitor_b.runtime.crop_classifier`: recorte
    quadrado com margem, preenchimento 114 e normalização ImageNet. Em NPUs a normalização
    vai para o artefato (mean = 0.485·255..., std = 0.229·255...) e o host envia uint8.
    """

    def __init__(
        self,
        runner: Runner,
        labels: list[str],
        crop_margin: float = 0.12,
        model_path: str | Path | None = None,
        normalization: str = "imagenet",
    ) -> None:
        if not labels:
            raise ConfigurationError("O classificador precisa da lista de rótulos.")
        if normalization not in {"imagenet", "none"}:
            raise ConfigurationError("'normalization' deve ser 'imagenet' ou 'none'.")
        self.runner = runner
        self.labels = list(labels)
        self.crop_margin = float(crop_margin)
        self.normalization = normalization
        self.model_path = Path(model_path) if model_path else Path(
            str(runner.describe().get("model", "classificador"))
        )
        self.input_size = runner.spec.height
        self.dynamic_batch = bool(getattr(runner, "dynamic_batch", False))

    def crops_from(self, frame: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        crops = []
        for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
            corners = tuple(float(value) for value in box)
            region = expand_box(corners, self.crop_margin, width, height)
            crops.append(crop_square(frame, region, self.input_size))
        return crops

    def _tensor(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = resize_square(crop_bgr, self.input_size)[:, :, ::-1]
        spec = self.runner.spec
        if spec.dtype == "uint8" or self.normalization == "none":
            return rgb_to_tensor(rgb, spec)
        normalized = (np.asarray(rgb, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return rgb_to_tensor(normalized, spec, scale=1.0)

    def classify_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        tensors = [self._tensor(crop) for crop in crops]
        if self.dynamic_batch:
            logits = self.runner.run(np.concatenate(tensors, axis=0))[0]
        else:
            logits = np.concatenate([self.runner.run(tensor)[0] for tensor in tensors], axis=0)
        logits = np.asarray(logits, dtype=np.float32).reshape(len(crops), -1)
        return _softmax(logits)

    def classify(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        return self.classify_crops(self.crops_from(frame, boxes))

    def info(self) -> dict[str, object]:
        return {
            **self.runner.describe(),
            "labels": self.labels,
            "input_size": self.input_size,
            "normalization": (
                "no artefato" if self.runner.spec.dtype == "uint8" else self.normalization
            ),
        }
