"""Quantização INT8 estática de detectores com entrada retangular.

Mesma receita de `animal_monitor_b.exporting.quantize_int8` (QDQ, ativações QUInt8,
pesos QInt8 por canal, calibração MinMax com imagens reais), com uma diferença: o leitor
de calibração usa a altura e a largura do modelo. O original deriva um lado só da última
dimensão e falha em entradas como 224×320 (padrão da MaixCAM).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor.inference.prepost import prepare_input


class _LeitorRetangular:
    def __init__(self, imagens: list[Path], nome_entrada: str, altura: int, largura: int) -> None:
        import cv2

        self.cv2 = cv2
        self.imagens = imagens
        self.nome_entrada = nome_entrada
        self.forma = (altura, largura)
        self._iterador = iter(imagens)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for caminho in self._iterador:
            imagem = self.cv2.imread(str(caminho))
            if imagem is None:
                continue
            tensor, _ = prepare_input(imagem, self.forma)
            return {self.nome_entrada: tensor}
        return None

    def rewind(self) -> None:  # pragma: no cover - exigido por algumas versões
        self._iterador = iter(self.imagens)


def quantize_detector_int8(
    modelo: str | Path,
    pasta_calibracao: str | Path,
    destino: str | Path,
    amostras: int = 36,
    apenas_conv: bool = True,
    semente: int = 42,
) -> dict[str, object]:
    """`apenas_conv=False` quantiza também a decodificação — só faz sentido na cabeça dividida."""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    origem = Path(modelo).expanduser().resolve()
    saida = Path(destino).expanduser().resolve()
    arquivos = sorted(
        caminho
        for caminho in Path(pasta_calibracao).expanduser().resolve().rglob("*")
        if caminho.suffix.lower() in IMAGE_SUFFIXES
    )
    if not arquivos:
        raise ConfigurationError(f"Nenhuma imagem de calibração em {pasta_calibracao}.")
    if len(arquivos) > amostras:
        random.Random(semente).shuffle(arquivos)
        arquivos = arquivos[:amostras]

    sessao = ort.InferenceSession(str(origem), providers=["CPUExecutionProvider"])
    entrada = sessao.get_inputs()[0]
    altura, largura = int(entrada.shape[-2]), int(entrada.shape[-1])
    del sessao

    preparado = saida.with_name(f"{saida.stem}_prep.onnx")
    quant_pre_process(str(origem), str(preparado), skip_symbolic_shape=False)
    quantize_static(
        model_input=str(preparado),
        model_output=str(saida),
        calibration_data_reader=_LeitorRetangular(arquivos, entrada.name, altura, largura),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"] if apenas_conv else None,
    )
    preparado.unlink(missing_ok=True)
    relatorio = {
        "source_model": str(origem),
        "quantized_model": str(saida),
        "input_hw": [altura, largura],
        "calibration_samples": len(arquivos),
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "calibrate_method": "minmax",
        "op_types_to_quantize": ["Conv"] if apenas_conv else "todos",
        "size_mb": {
            "fp32": round(origem.stat().st_size / 1024**2, 3),
            "int8": round(saida.stat().st_size / 1024**2, 3),
        },
    }
    saida.with_suffix(".json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return relatorio
