"""Derivação dos dois datasets do Modelo B a partir do dataset YOLO original.

Do mesmo conjunto rotulado nascem duas tarefas mais fáceis que a detecção multiclasse:

1. **Detecção de presença** (`data/animals_agnostic`): as três espécies colapsam em uma
   única classe `animal`. Todas as caixas passam a treinar o mesmo objetivo, em vez de
   dividir 207 caixas entre três classes.
2. **Classificação de espécie** (`data/crops`): cada caixa vira um recorte com margem de
   contexto. Negativos de fundo e recortes desalinhados formam a classe `desconhecido`,
   que é o mecanismo explícito de rejeição que o Modelo A não possui.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animal_monitor.config import load_class_names, load_yaml, resolve_from
from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor_b.config import AGNOSTIC_LABEL, UNKNOWN_LABEL

SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class LabelledBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_pixels(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        half_w = self.width * image_width / 2.0
        half_h = self.height * image_height / 2.0
        cx = self.x_center * image_width
        cy = self.y_center * image_height
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - ambiente sem OpenCV
        raise RuntimeError("OpenCV não está instalado.") from exc
    return cv2


def parse_label_file(path: Path) -> list[LabelledBox]:
    """Lê um arquivo de rótulos YOLO ignorando linhas vazias."""
    boxes: list[LabelledBox] = []
    if not path.is_file():
        return boxes
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ConfigurationError(f"Linha inválida em {path}: {raw_line!r}")
        boxes.append(
            LabelledBox(
                class_id=int(float(parts[0])),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def _dataset_paths(dataset_yaml: str | Path) -> tuple[Path, dict[str, Path]]:
    dataset_yaml = Path(dataset_yaml).expanduser().resolve()
    content = load_yaml(dataset_yaml)
    root = resolve_from(dataset_yaml, content.get("path", "."))
    split_dirs: dict[str, Path] = {}
    for split in SPLITS:
        relative = content.get(split)
        if relative is None:
            continue
        candidate = Path(str(relative))
        images_dir = candidate if candidate.is_absolute() else root / candidate
        split_dirs[split] = images_dir.resolve()
    if not split_dirs:
        raise ConfigurationError(f"Nenhum split declarado em {dataset_yaml}.")
    return root, split_dirs


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    else:  # pragma: no cover - estrutura fora do padrão Ultralytics
        raise ConfigurationError(f"Não foi possível localizar 'labels' para {image_path}.")
    return Path(*parts).with_suffix(".txt")


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ConfigurationError(f"Diretório de imagens não encontrado: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        destination.write_bytes(source.read_bytes())


def iou_xyxy(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU entre uma caixa e um conjunto de caixas, ambos em pixels xyxy."""
    if others.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).astype(np.float32)


def expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Aplica margem de contexto e recorta nos limites da imagem."""
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * margin
    pad_y = height * margin
    nx1 = int(round(max(0.0, x1 - pad_x)))
    ny1 = int(round(max(0.0, y1 - pad_y)))
    nx2 = int(round(min(float(image_width), x2 + pad_x)))
    ny2 = int(round(min(float(image_height), y2 + pad_y)))
    if nx2 - nx1 < 2:
        nx1, nx2 = max(0, nx1 - 1), min(image_width, nx1 + 2)
    if ny2 - ny1 < 2:
        ny1, ny2 = max(0, ny1 - 1), min(image_height, ny1 + 2)
    return nx1, ny1, nx2, ny2


def crop_square(image: np.ndarray, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Recorta e redimensiona mantendo proporção, com preenchimento neutro."""
    cv2 = _cv2()
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.full((size, size, 3), 114, dtype=np.uint8)
    height, width = patch.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _sample_negative(
    rng: random.Random,
    gt: np.ndarray,
    image_width: int,
    image_height: int,
    max_iou: float,
    attempts: int = 40,
) -> tuple[float, float, float, float] | None:
    """Sorteia uma caixa de fundo com sobreposição baixa contra todas as caixas reais."""
    for _ in range(attempts):
        side_ratio = rng.uniform(0.20, 0.60)
        box_w = side_ratio * image_width * rng.uniform(0.8, 1.25)
        box_h = side_ratio * image_height * rng.uniform(0.8, 1.25)
        box_w = min(box_w, image_width - 2.0)
        box_h = min(box_h, image_height - 2.0)
        if box_w < 8 or box_h < 8:
            continue
        x1 = rng.uniform(0.0, image_width - box_w)
        y1 = rng.uniform(0.0, image_height - box_h)
        candidate = np.array([x1, y1, x1 + box_w, y1 + box_h], dtype=np.float32)
        if gt.size and float(iou_xyxy(candidate, gt).max()) > max_iou:
            continue
        return float(candidate[0]), float(candidate[1]), float(candidate[2]), float(candidate[3])
    return None


def _sample_misaligned(
    rng: random.Random,
    reference: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    attempts: int = 30,
) -> tuple[float, float, float, float] | None:
    """Gera recorte parcial do animal (IoU baixa) para ensinar rejeição geométrica."""
    gt = np.array(reference, dtype=np.float32)
    width = max(1.0, reference[2] - reference[0])
    height = max(1.0, reference[3] - reference[1])
    for _ in range(attempts):
        scale = rng.uniform(0.35, 0.75)
        new_w = width * scale
        new_h = height * scale
        shift_x = rng.uniform(-1.0, 1.0) * width * 0.8
        shift_y = rng.uniform(-1.0, 1.0) * height * 0.8
        cx = (reference[0] + reference[2]) / 2.0 + shift_x
        cy = (reference[1] + reference[3]) / 2.0 + shift_y
        x1 = max(0.0, cx - new_w / 2.0)
        y1 = max(0.0, cy - new_h / 2.0)
        x2 = min(float(image_width), cx + new_w / 2.0)
        y2 = min(float(image_height), cy + new_h / 2.0)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        candidate = np.array([x1, y1, x2, y2], dtype=np.float32)
        overlap = float(iou_xyxy(candidate, gt[None, :])[0])
        if 0.02 <= overlap <= 0.25:
            return float(x1), float(y1), float(x2), float(y2)
    return None


def prepare_model_b_data(
    dataset_yaml: str | Path,
    agnostic_dir: str | Path = "data/animals_agnostic",
    crops_dir: str | Path = "data/crops",
    crop_size: int = 160,
    crop_margin: float = 0.12,
    negatives_per_image: float = 1.0,
    misaligned_per_box: float = 0.5,
    negative_max_iou: float = 0.10,
    seed: int = 42,
    report_json: str | Path | None = None,
) -> dict[str, object]:
    """Gera o dataset de presença e o dataset de recortes por espécie."""
    cv2 = _cv2()
    names = load_class_names(dataset_yaml)
    _, split_dirs = _dataset_paths(dataset_yaml)
    agnostic_root = Path(agnostic_dir).expanduser().resolve()
    crops_root = Path(crops_dir).expanduser().resolve()
    rng = random.Random(seed)

    class_labels = [names[index] for index in sorted(names)]
    crop_labels = [*class_labels, UNKNOWN_LABEL]
    report: dict[str, object] = {
        "dataset_yaml": str(Path(dataset_yaml).expanduser().resolve()),
        "agnostic_dir": str(agnostic_root),
        "crops_dir": str(crops_root),
        "crop_size": crop_size,
        "crop_margin": crop_margin,
        "species": class_labels,
        "crop_labels": crop_labels,
        "splits": {},
    }

    for split, images_dir in split_dirs.items():
        images = _images_in(images_dir)
        agnostic_images = agnostic_root / "images" / split
        agnostic_labels = agnostic_root / "labels" / split
        agnostic_images.mkdir(parents=True, exist_ok=True)
        agnostic_labels.mkdir(parents=True, exist_ok=True)
        for label_dir in (crops_root / split / label for label in crop_labels):
            label_dir.mkdir(parents=True, exist_ok=True)

        crop_counts: Counter[str] = Counter()
        boxes_total = 0
        skipped_unreadable = 0
        for image_path in images:
            boxes = parse_label_file(_label_path(image_path))
            if not boxes:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_unreadable += 1
                continue
            image_height, image_width = image.shape[:2]

            _link_or_copy(image_path, agnostic_images / image_path.name)
            lines = [
                f"0 {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
                for box in boxes
            ]
            (agnostic_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            pixel_boxes = np.array(
                [box.to_pixels(image_width, image_height) for box in boxes], dtype=np.float32
            )
            stem_hash = hashlib.sha1(image_path.name.encode("utf-8")).hexdigest()[:8]
            for index, (box, pixels) in enumerate(zip(boxes, pixel_boxes, strict=True)):
                label = names.get(box.class_id, f"classe_{box.class_id}")
                region = expand_box(tuple(pixels), crop_margin, image_width, image_height)
                patch = crop_square(image, region, crop_size)
                name = f"{image_path.stem}_{stem_hash}_b{index:02d}.jpg"
                cv2.imwrite(str(crops_root / split / label / name), patch)
                crop_counts[label] += 1
                boxes_total += 1

                if rng.random() < misaligned_per_box:
                    partial = _sample_misaligned(rng, tuple(pixels), image_width, image_height)
                    if partial is not None:
                        region = expand_box(partial, 0.0, image_width, image_height)
                        patch = crop_square(image, region, crop_size)
                        name = f"{image_path.stem}_{stem_hash}_p{index:02d}.jpg"
                        cv2.imwrite(str(crops_root / split / UNKNOWN_LABEL / name), patch)
                        crop_counts[UNKNOWN_LABEL] += 1

            wanted = int(negatives_per_image) + (
                1 if rng.random() < (negatives_per_image % 1.0) else 0
            )
            for negative_index in range(wanted):
                negative = _sample_negative(
                    rng, pixel_boxes, image_width, image_height, negative_max_iou
                )
                if negative is None:
                    continue
                region = expand_box(negative, 0.0, image_width, image_height)
                patch = crop_square(image, region, crop_size)
                name = f"{image_path.stem}_{stem_hash}_n{negative_index:02d}.jpg"
                cv2.imwrite(str(crops_root / split / UNKNOWN_LABEL / name), patch)
                crop_counts[UNKNOWN_LABEL] += 1

        report["splits"][split] = {  # type: ignore[index]
            "images": len(images),
            "unreadable_images": skipped_unreadable,
            "agnostic_boxes": boxes_total,
            "crops": {label: crop_counts.get(label, 0) for label in crop_labels},
            "crops_total": int(sum(crop_counts.values())),
        }

    yaml_path = agnostic_root / "dataset_agnostic.yaml"
    yaml_path.write_text(
        "# Gerado por animal-monitor-b prepare-data. Uma única classe de presença.\n"
        f"path: {agnostic_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\nnames:\n"
        f"  0: {AGNOSTIC_LABEL}\n",
        encoding="utf-8",
    )
    labels_path = crops_root / "labels.json"
    labels_path.write_text(json.dumps(crop_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    report["dataset_agnostic_yaml"] = str(yaml_path)
    report["crop_labels_file"] = str(labels_path)

    if report_json:
        output = Path(report_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
