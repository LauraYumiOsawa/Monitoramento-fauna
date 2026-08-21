"""Treinamento do detector de presença (uma única classe).

Reaproveita Ultralytics, mas com dois desvios deliberados em relação ao Modelo A:

* **uma classe** em vez de três: as 147 caixas de treino passam a somar no mesmo objetivo
  em vez de se dividir; a cabeça de classificação deixa de ser fonte de erro;
* **entrada 416** em vez de 640: em um embarcado de baixo custo o custo do detector domina
  a latência, e 416² são cerca de 42% dos MACs de 640².
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from animal_monitor.config import load_yaml, resolve_from


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def train_presence_detector(config_path: str | Path) -> Path:
    """Treina o detector de presença e grava o manifesto do experimento."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics não está instalado. Instale o projeto com o extra 'train'."
        ) from exc

    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml(config_path)
    data_path = resolve_from(config_path, config.get("data", "dataset_agnostic.yaml"))
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset de presença não encontrado: {data_path}. "
            "Rode 'animal-monitor-b prepare-data' primeiro."
        )

    model_name = str(config.get("model", "yolov8n.pt"))
    arguments: dict[str, Any] = {
        key: value for key, value in config.items() if key not in {"model", "data"}
    }
    arguments["data"] = str(data_path)
    if arguments.get("device") == "auto":
        arguments.pop("device")
    if "project" in arguments:
        arguments["project"] = str(resolve_from(config_path, arguments["project"]))

    model = YOLO(model_name)
    result = model.train(**arguments)
    save_dir_value = getattr(result, "save_dir", None) or getattr(model.trainer, "save_dir", None)
    if save_dir_value is None:  # pragma: no cover - Ultralytics sempre devolve save_dir
        save_dir = Path(arguments.get("project", "runs/train")) / str(
            arguments.get("name", "presence")
        )
    else:
        save_dir = Path(save_dir_value)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "deteccao_de_presenca",
        "config_path": str(config_path),
        "training_config": config,
        "dataset_agnostic": str(data_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ultralytics": _version("ultralytics"),
            "torch": _version("torch"),
        },
    }
    (save_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = save_dir / "weights" / "best.pt"
    return best if best.is_file() else save_dir


def evaluate_presence_detector(
    weights: str | Path,
    dataset_yaml: str | Path,
    split: str = "test",
    image_size: int = 416,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    """Mede o detector de presença isoladamente (mAP de uma classe)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics não está instalado.") from exc
    model = YOLO(str(Path(weights).expanduser().resolve()))
    metrics = model.val(
        data=str(Path(dataset_yaml).expanduser().resolve()),
        split=split,
        imgsz=image_size,
        plots=True,
    )
    result: dict[str, object] = {
        "stage": "deteccao_de_presenca",
        "split": split,
        "image_size": image_size,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "speed_ms": {key: float(value) for key, value in getattr(metrics, "speed", {}).items()},
    }
    results_dict = getattr(metrics, "results_dict", {})
    for key in ("metrics/precision(B)", "metrics/recall(B)"):
        if key in results_dict:
            result[key] = float(results_dict[key])
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
