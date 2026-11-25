"""Contrato mínimo entre o pipeline e cada runtime de inferência.

O Modelo A e o Modelo B foram escritos sobre ONNX Runtime. Nas versões com acelerador o
grafo executado é outro artefato (HEF, RKNN, cvimodel...), com outro layout de entrada e,
às vezes, com o NMS já feito no chip. O `Runner` isola só essa diferença: recebe o tensor
no formato que o acelerador espera e devolve a lista de saídas. Letterbox, decodificação,
rastreamento, votação e banco continuam sendo o código já testado do projeto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from animal_monitor.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Formato do tensor de entrada esperado pelo artefato compilado."""

    height: int
    width: int
    layout: str = "nchw"  # "nchw" (ONNX) ou "nhwc" (Hailo, RKNN)
    dtype: str = "float32"  # "float32" (normalizado no host) ou "uint8" (normalização no chip)

    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ConfigurationError("A entrada do modelo deve ter altura e largura positivas.")
        if self.layout not in {"nchw", "nhwc"}:
            raise ConfigurationError("'layout' deve ser 'nchw' ou 'nhwc'.")
        if self.dtype not in {"float32", "uint8"}:
            raise ConfigurationError("'dtype' deve ser 'float32' ou 'uint8'.")


class Runner(Protocol):
    """Executa um artefato compilado. Não conhece YOLO nem espécies."""

    spec: TensorSpec

    def run(self, tensor: np.ndarray) -> list[np.ndarray]: ...

    def describe(self) -> dict[str, object]: ...


def rgb_to_tensor(rgb: np.ndarray, spec: TensorSpec, scale: float = 1.0 / 255.0) -> np.ndarray:
    """Converte HxWx3 RGB uint8 no tensor de lote 1 descrito por `spec`.

    Para `uint8` a imagem segue crua: média e escala foram embutidas no artefato na
    compilação (mean 0, std 255 no RKNN; normalização no próprio HEF/cvimodel).
    """
    if spec.dtype == "uint8":
        tensor = np.ascontiguousarray(rgb, dtype=np.uint8)
        if spec.layout == "nchw":
            tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
        return tensor[None, ...]
    tensor = np.ascontiguousarray(rgb, dtype=np.float32) * np.float32(scale)
    if spec.layout == "nchw":
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
    return tensor[None, ...]
