"""Exportação para ONNX e quantização INT8 estática dos dois modelos.

O Modelo A exporta apenas FP32. Para um embarcado de baixo custo, INT8 é o que decide se
o modelo cabe e roda: reduz o arquivo a cerca de 1/4 e é o formato exigido pelos
aceleradores das placas de baixo custo (a NPU do SG2002 da MaixCAM, por exemplo, só
executa inteiro). A quantização é **estática** (com calibração real, não dinâmica) porque
convoluções em modelos de visão perdem muito com quantização apenas de pesos.

Regra metodológica: aqui só se produz o artefato. A decisão de usar INT8 depende da
comparação de mAP e F1 medida por `animal-monitor-b evaluate`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor_b.augment import resize_square, to_tensor
from animal_monitor_b.classifier import build_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_presence_detector(
    weights: str | Path,
    output_dir: str | Path,
    image_size: int = 416,
    opset: int = 17,
    dynamic: bool = False,
) -> Path:
    """Exporta o detector de presença (uma classe) para ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics não está instalado.") from exc
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported = Path(
        str(
            model.export(
                format="onnx",
                imgsz=image_size,
                batch=1,
                dynamic=dynamic,
                simplify=True,
                opset=opset,
                nms=False,
            )
        )
    ).resolve()
    destination = destination_dir / "presence_detector.onnx"
    if exported != destination:
        shutil.copy2(exported, destination)
    _write_metadata(
        destination_dir / "presence_detector_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "deteccao_de_presenca",
            "source_weights": str(weights_path),
            "model_file": destination.name,
            "sha256": sha256_file(destination),
            "size_mb": round(destination.stat().st_size / 1024**2, 3),
            "image_size": image_size,
            "opset": opset,
            "dynamic": dynamic,
            "nms_embedded": False,
            "classes": {0: "animal"},
        },
    )
    return destination


def export_species_classifier(
    checkpoint: str | Path,
    output_dir: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Exporta o classificador de espécie para ONNX, com lote dinâmico."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch não está instalado.") from exc
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = [str(label) for label in payload["labels"]]
    input_size = int(payload["input_size"])
    backbone = str(payload["backbone"])

    model = build_model(backbone, len(labels), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "species_classifier.onnx"
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    arguments: dict[str, object] = {
        "input_names": ["crops"],
        "output_names": ["logits"],
        "opset_version": opset,
        "dynamic_axes": {"crops": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None,
        "do_constant_folding": True,
    }
    # A partir do torch 2.9 o exportador padrão é o baseado em Dynamo, que exige o pacote
    # onnxscript. O caminho TorchScript é suficiente para uma CNN estática e mantém o
    # ambiente enxuto, então ele é pedido explicitamente quando disponível.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        arguments["dynamo"] = False
    torch.onnx.export(model, dummy, str(destination), **arguments)  # type: ignore[arg-type]
    _write_metadata(
        destination_dir / "classifier_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "classificacao_de_especie",
            "source_checkpoint": str(checkpoint_path),
            "model_file": destination.name,
            "sha256": sha256_file(destination),
            "size_mb": round(destination.stat().st_size / 1024**2, 3),
            "backbone": backbone,
            "input_size": input_size,
            "opset": opset,
            "dynamic_batch": dynamic_batch,
            "normalization": "imagenet",
            "labels": labels,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
    )
    return destination


class _ImageCalibrationReader:
    """Fornece lotes de calibração para a quantização estática do detector."""

    def __init__(self, images: list[Path], input_name: str, size: int) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV não está instalado.") from exc
        from animal_monitor.inference.prepost import prepare_input

        self.cv2 = cv2
        self.prepare_input = prepare_input
        self.images = images
        self.input_name = input_name
        self.size = size
        self._iterator = iter(images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for path in self._iterator:
            image = self.cv2.imread(str(path))
            if image is None:
                continue
            tensor, _ = self.prepare_input(image, (self.size, self.size))
            return {self.input_name: tensor}
        return None

    def rewind(self) -> None:  # pragma: no cover - exigido por algumas versões
        self._iterator = iter(self.images)


class _CropCalibrationReader:
    """Fornece lotes de calibração para a quantização estática do classificador."""

    def __init__(self, crops: list[Path], input_name: str, size: int) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV não está instalado.") from exc
        self.cv2 = cv2
        self.crops = crops
        self.input_name = input_name
        self.size = size
        self._iterator = iter(crops)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for path in self._iterator:
            image = self.cv2.imread(str(path))
            if image is None:
                continue
            tensor = to_tensor(resize_square(image, self.size))[None, ...]
            return {self.input_name: tensor}
        return None

    def rewind(self) -> None:  # pragma: no cover
        self._iterator = iter(self.crops)


def _collect_files(directory: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ConfigurationError(f"Nenhuma imagem de calibração encontrada em {directory}.")
    if len(files) > limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]
    return files


def quantize_int8(
    model_path: str | Path,
    calibration_dir: str | Path,
    kind: str,
    output_path: str | Path | None = None,
    samples: int = 64,
    seed: int = 42,
    per_channel: bool = True,
    calibrate_method: str = "minmax",
    op_types: tuple[str, ...] | None = None,
    percentile: float = 99.99,
) -> dict[str, object]:
    """Quantiza um dos modelos ONNX para INT8 usando dados reais de calibração.

    Os dois parâmetros que decidem se INT8 é usável foram medidos neste projeto:

    * `op_types`: restringir a quantização a `Conv` deixa `Sigmoid`, `Concat` e a
      aritmética de decodificação de caixas em FP32. Sem isso, a cabeça do YOLO devolve
      escores fora de [0, 1] e o detector para de funcionar;
    * `calibrate_method`: `minmax` é sensível a ativação extrema. `percentile` e `entropy`
      recortam a cauda e preservam melhor o MobileNetV3, que tem hard-swish e
      squeeze-excite com faixa dinâmica larga.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:
        raise RuntimeError(
            "O módulo de quantização do ONNX Runtime não está disponível. "
            "Instale 'onnx' e 'onnxruntime'."
        ) from exc

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if calibrate_method not in methods:
        raise ConfigurationError(
            f"'calibrate_method' deve ser um de: {', '.join(sorted(methods))}."
        )
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Modelo ONNX não encontrado: {source}")
    if kind not in {"detector", "classifier"}:
        raise ConfigurationError("'kind' deve ser 'detector' ou 'classifier'.")
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else source.with_name(f"{source.stem}_int8.onnx")
    )
    files = _collect_files(Path(calibration_dir).expanduser().resolve(), samples, seed)

    import onnxruntime as ort

    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    size = int(shape[-1]) if isinstance(shape[-1], int) and shape[-1] > 0 else 416
    del session

    prepared = destination.with_name(f"{destination.stem}_prep.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
    reader = (
        _ImageCalibrationReader(files, model_input.name, size)
        if kind == "detector"
        else _CropCalibrationReader(files, model_input.name, size)
    )
    extra_options: dict[str, object] = {}
    if calibrate_method == "percentile":
        extra_options["CalibPercentile"] = percentile
        extra_options["CalibNumBins"] = 2048
    quantize_static(
        model_input=str(prepared),
        model_output=str(destination),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=methods[calibrate_method],
        op_types_to_quantize=list(op_types) if op_types else None,
        extra_options=extra_options or None,
    )
    prepared.unlink(missing_ok=True)

    result = {
        "source_model": str(source),
        "quantized_model": str(destination),
        "kind": kind,
        "calibration_samples": len(files),
        "calibration_dir": str(Path(calibration_dir).expanduser().resolve()),
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": per_channel,
        "calibrate_method": calibrate_method,
        "op_types_to_quantize": list(op_types) if op_types else "todos",
        "size_mb": {
            "fp32": round(source.stat().st_size / 1024**2, 3),
            "int8": round(destination.stat().st_size / 1024**2, 3),
        },
        "sha256_int8": sha256_file(destination),
    }
    _write_metadata(destination.with_suffix(".json"), result)
    return result
