from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from animal_monitor.config import load_class_names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_onnx(
    weights: str | Path,
    dataset_yaml: str | Path,
    output_dir: str | Path,
    image_size: int = 640,
    opset: int = 17,
    dynamic: bool = False,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics não está instalado. Instale o projeto com o extra de treinamento."
        ) from exc

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    names = load_class_names(dataset_yaml)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported_value = model.export(
        format="onnx",
        imgsz=image_size,
        batch=1,
        dynamic=dynamic,
        simplify=True,
        opset=opset,
        nms=False,
    )
    exported_path = Path(str(exported_value)).expanduser().resolve()
    destination = destination_dir / "animal_detector.onnx"
    if exported_path != destination:
        shutil.copy2(exported_path, destination)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_weights": str(weights_path),
        "model_file": destination.name,
        "sha256": _sha256(destination),
        "image_size": image_size,
        "opset": opset,
        "dynamic": dynamic,
        "nms_embedded": False,
        "classes": names,
    }
    (destination_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
