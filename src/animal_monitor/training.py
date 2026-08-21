from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from animal_monitor.config import load_yaml, resolve_from
from animal_monitor.dataset import summarize_issue_counts, validate_dataset
from animal_monitor.errors import DatasetValidationError


def _ultralytics_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics não está instalado. Instale o projeto com o extra de treinamento."
        ) from exc
    return YOLO


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def train_model(config_path: str | Path) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml(config_path)
    data_path = resolve_from(config_path, config.get("data", "dataset.yaml"))
    report = validate_dataset(data_path)
    if not report.ok:
        counts = summarize_issue_counts(report.issues)
        raise DatasetValidationError(
            f"Dataset reprovado: {counts['errors']} erro(s). "
            "Execute 'validate-dataset' para detalhes."
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

    YOLO = _ultralytics_yolo()
    model = YOLO(model_name)
    result = model.train(**arguments)
    save_dir_value = getattr(result, "save_dir", None) or getattr(model.trainer, "save_dir", None)
    if save_dir_value is None:
        project = Path(arguments.get("project", "runs/detect"))
        save_dir = project / str(arguments.get("name", "train"))
    else:
        save_dir = Path(save_dir_value)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "dataset_report": report.as_dict(),
        "training_config": config,
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


def evaluate_model(
    weights: str | Path,
    dataset_yaml: str | Path,
    split: str = "test",
    image_size: int = 640,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    YOLO = _ultralytics_yolo()
    model = YOLO(str(Path(weights).expanduser().resolve()))
    metrics = model.val(
        data=str(Path(dataset_yaml).expanduser().resolve()),
        split=split,
        imgsz=image_size,
        plots=True,
    )
    result: dict[str, object] = {
        "split": split,
        "image_size": image_size,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95_by_class": [float(value) for value in metrics.box.maps],
        "speed_ms": {key: float(value) for key, value in getattr(metrics, "speed", {}).items()},
    }
    results_dict = getattr(metrics, "results_dict", {})
    for key in ("metrics/precision(B)", "metrics/recall(B)"):
        if key in results_dict:
            result[key] = float(results_dict[key])
    if output_json:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
