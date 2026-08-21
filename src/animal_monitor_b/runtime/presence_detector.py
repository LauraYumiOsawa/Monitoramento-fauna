"""Detector de presença: uma classe, entrada reduzida, sem PyTorch no alvo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor.inference.prepost import (
    decode_yolo_output,
    prepare_input,
    undo_letterbox,
)
from animal_monitor_b.config import AGNOSTIC_LABEL, RuntimeConfigB


def create_session(model_path: Path, providers: tuple[str, ...], intra_op_threads: int):
    """Cria a sessão ONNX Runtime validando os providers pedidos."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime não está instalado.") from exc
    if not model_path.is_file():
        raise ConfigurationError(f"Modelo ONNX não encontrado: {model_path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads:
        options.intra_op_num_threads = intra_op_threads
    available = set(ort.get_available_providers())
    selected = [provider for provider in providers if provider in available]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    if not selected:
        raise ConfigurationError(
            f"Nenhum provider solicitado está disponível. Disponíveis: {sorted(available)}"
        )
    return ort.InferenceSession(str(model_path), sess_options=options, providers=selected)


class PresenceDetector:
    """YOLO de uma classe (`animal`) executado por ONNX Runtime.

    Colapsar as espécies em uma classe reduz a saída do modelo e, mais importante, faz
    todas as caixas rotuladas treinarem o mesmo objetivo. A entrada de 416 px custa cerca
    de 42% dos MACs de 640 px.
    """

    label = AGNOSTIC_LABEL

    def __init__(self, model_path: str | Path, config: RuntimeConfigB) -> None:
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
        if len(shape) != 4:
            raise ConfigurationError(f"Entrada ONNX deve ser NCHW; encontrado: {shape}")
        height, width = shape[-2], shape[-1]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            self.input_height, self.input_width = height, width
        else:
            self.input_height = self.input_width = config.detector_input

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (caixas xyxy em pixels da imagem original, escores)."""
        tensor, transform = prepare_input(frame, (self.input_height, self.input_width))
        outputs = self.session.run(None, {self.input.name: tensor})
        if not outputs:
            raise RuntimeError("O modelo ONNX não retornou saídas.")
        boxes, scores, _ = decode_yolo_output(
            outputs[0],
            class_count=1,
            confidence_threshold=self.config.detector_confidence,
            iou_threshold=self.config.detector_iou,
            class_agnostic_nms=True,
            max_detections=self.config.max_detections,
        )
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return undo_letterbox(boxes, transform), scores

    def info(self) -> dict[str, object]:
        return {
            "model": str(self.model_path),
            "input_name": self.input.name,
            "input_shape": list(self.input.shape),
            "runtime_input": [1, 3, self.input_height, self.input_width],
            "providers": self.session.get_providers(),
            "classes": {0: self.label},
        }
