from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from animal_monitor.config import RuntimeConfig
from animal_monitor.errors import ConfigurationError
from animal_monitor.inference.prepost import build_detections, decode_yolo_output, prepare_input
from animal_monitor.types import Detection


class OnnxYoloDetector:
    """Inferência YOLO com ONNX Runtime, sem dependência de PyTorch no alvo."""

    def __init__(
        self,
        model_path: str | Path,
        names: Mapping[int, str],
        config: RuntimeConfig,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime não está instalado. "
                "Instale o projeto com as dependências de runtime."
            ) from exc

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ConfigurationError(f"Modelo ONNX não encontrado: {self.model_path}")
        self.names = dict(names)
        self.config = config

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if config.intra_op_threads:
            session_options.intra_op_num_threads = config.intra_op_threads

        available = set(ort.get_available_providers())
        providers = [provider for provider in config.providers if provider in available]
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            raise ConfigurationError(
                f"Nenhum provider solicitado está disponível. Disponíveis: {sorted(available)}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=providers
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ConfigurationError(
                f"Esperado um tensor de entrada; o modelo possui {len(inputs)}."
            )
        self.input = inputs[0]
        self.input_height, self.input_width = self._resolve_input_size(self.input.shape)

    def _resolve_input_size(self, shape: list[object]) -> tuple[int, int]:
        if len(shape) != 4:
            raise ConfigurationError(f"Entrada ONNX deve ser NCHW; encontrado: {shape}")
        height, width = shape[-2], shape[-1]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return height, width
        return self.config.input_size, self.config.input_size

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def infer(self, frame: np.ndarray) -> list[Detection]:
        tensor, transform = prepare_input(frame, (self.input_height, self.input_width))
        outputs = self.session.run(None, {self.input.name: tensor})
        if not outputs:
            raise RuntimeError("O modelo ONNX não retornou saídas.")
        boxes, scores, class_ids = decode_yolo_output(
            outputs[0],
            class_count=len(self.names),
            confidence_threshold=self.config.confidence,
            iou_threshold=self.config.iou_threshold,
            class_agnostic_nms=self.config.class_agnostic_nms,
            max_detections=self.config.max_detections,
        )
        return build_detections(boxes, scores, class_ids, self.names, transform)

    def warmup(self, repetitions: int = 2) -> None:
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        for _ in range(max(0, repetitions)):
            self.infer(dummy)

    def info(self) -> dict[str, object]:
        return {
            "model": str(self.model_path),
            "input_name": self.input.name,
            "input_shape": list(self.input.shape),
            "runtime_input": [1, 3, self.input_height, self.input_width],
            "providers": self.providers,
            "classes": self.names,
        }
