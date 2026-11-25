"""ONNX Runtime: CPU (Pi 5, Pi Zero 2 W, Orange Pi 5) e GPU/TensorRT (Jetson).

Diferente de `animal_monitor_b.runtime.presence_detector.create_session`, aceita opções
por provider. No Jetson elas decidem se o TensorRT compila em FP16 e se guarda o motor
compilado em disco; sem cache, cada inicialização recompila o grafo por minutos.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.backends.base import TensorSpec


class OnnxRuntimeRunner:
    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        provider_options: Mapping[str, Mapping[str, object]] | None = None,
        intra_op_threads: int = 0,
        fallback_size: int = 640,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime não está instalado.") from exc
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ConfigurationError(f"Modelo ONNX não encontrado: {self.model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = int(intra_op_threads)

        available = set(ort.get_available_providers())
        requested = [name for name in providers if name in available]
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in requested:
            requested.append("CPUExecutionProvider")
        if not requested:
            raise ConfigurationError(
                f"Nenhum provider solicitado está disponível. Disponíveis: {sorted(available)}"
            )
        provider_options = provider_options or {}
        for options_name, values in provider_options.items():
            # Cache do TensorRT: o diretório precisa existir antes da primeira compilação.
            if options_name == "TensorrtExecutionProvider" and values.get("trt_engine_cache_path"):
                Path(str(values["trt_engine_cache_path"])).expanduser().mkdir(
                    parents=True, exist_ok=True
                )
        resolved = [
            (name, {key: str(value) for key, value in provider_options.get(name, {}).items()})
            for name in requested
        ]
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=resolved
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
        if not (isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0):
            height = width = fallback_size
        self.dynamic_batch = not isinstance(shape[0], int)
        dtype = "uint8" if "uint8" in str(self.input.type) else "float32"
        self.spec = TensorSpec(height=int(height), width=int(width), layout="nchw", dtype=dtype)
        self.spec.validate()

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(item) for item in self.session.run(None, {self.input.name: tensor})]

    def describe(self) -> dict[str, object]:
        return {
            "runtime": "onnxruntime",
            "model": str(self.model_path),
            "input_shape": list(self.input.shape),
            "providers": self.session.get_providers(),
            "outputs": [item.name for item in self.session.get_outputs()],
        }
