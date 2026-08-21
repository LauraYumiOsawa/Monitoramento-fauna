"""Avaliação ponta a ponta comparável entre Modelo A e Modelo B.

O ponto central: os dois modelos são medidos pelo **mesmo código de métrica** e sobre as
mesmas imagens. O `evaluation.json` do Modelo A veio do Ultralytics; comparar aquele número
com um mAP calculado por outra implementação misturaria diferença de método com diferença
de modelo. Aqui, qualquer objeto que saiba transformar um quadro em `list[Detection]` é
avaliado do mesmo jeito.

Além do mAP, são medidas duas coisas que o Modelo A não reportava e que decidem a
utilidade em campo: a matriz de confusão de espécie entre caixas corretamente localizadas
(era ali que estava o erro de "anta") e a contagem de eventos falsos por vídeo.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

from animal_monitor.config import load_class_names, load_yaml, resolve_from
from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor.types import Detection
from animal_monitor_b.data_prep import parse_label_file
from animal_monitor_b.metrics import average_precision, match_detections

NOT_DETECTED = "nao_detectado"
BACKGROUND = "fundo"


class SpeciesPredictor(Protocol):
    def __call__(self, frame: np.ndarray) -> list[Detection]: ...


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV não está instalado.") from exc
    return cv2


def _split_images(dataset_yaml: Path, split: str) -> list[Path]:
    content = load_yaml(dataset_yaml)
    root = resolve_from(dataset_yaml, content.get("path", "."))
    relative = content.get(split)
    if relative is None:
        raise ConfigurationError(f"O split '{split}' não está declarado em {dataset_yaml}.")
    candidate = Path(str(relative))
    images_dir = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not images_dir.is_dir():
        raise ConfigurationError(f"Diretório de imagens não encontrado: {images_dir}")
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _labels_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def evaluate_species_predictor(
    predictor: SpeciesPredictor,
    dataset_yaml: str | Path,
    split: str = "test",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.35,
    label: str = "modelo",
    output_json: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Roda o preditor em um split e devolve mAP50, P/R e a confusão de espécie."""
    cv2 = _cv2()
    dataset_yaml = Path(dataset_yaml).expanduser().resolve()
    names = load_class_names(dataset_yaml)
    species = [names[index] for index in sorted(names)]
    images = _split_images(dataset_yaml, split)

    scores_by_class: dict[str, list[float]] = defaultdict(list)
    matched_by_class: dict[str, list[float]] = defaultdict(list)
    ground_truth_by_class: dict[str, int] = defaultdict(int)
    confusion_rows = [*species, BACKGROUND]
    confusion_columns = [*species, NOT_DETECTED]
    confusion = np.zeros((len(confusion_rows), len(confusion_columns)), dtype=np.int64)
    true_positive = false_positive = false_negative = 0
    predictions_total = 0

    for position, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = parse_label_file(_labels_for(image_path))
        gt_boxes: dict[str, list[list[float]]] = defaultdict(list)
        for box in gt:
            name = names.get(box.class_id, f"classe_{box.class_id}")
            gt_boxes[name].append(list(box.to_pixels(width, height)))
            ground_truth_by_class[name] += 1

        detections = predictor(image)
        predictions_total += len(detections)
        for name in species:
            predicted = [
                item
                for item in detections
                if item.class_name == name and item.confidence >= 0.0
            ]
            if not predicted and name not in gt_boxes:
                continue
            predicted_boxes = np.array(
                [[item.box.x1, item.box.y1, item.box.x2, item.box.y2] for item in predicted],
                dtype=np.float64,
            ).reshape(-1, 4)
            predicted_scores = np.array(
                [item.confidence for item in predicted], dtype=np.float64
            ).reshape(-1)
            truth = np.array(gt_boxes.get(name, []), dtype=np.float64).reshape(-1, 4)
            flags = match_detections(predicted_boxes, predicted_scores, truth, iou_threshold)
            scores_by_class[name].extend(predicted_scores.tolist())
            matched_by_class[name].extend(flags.tolist())

        kept = [item for item in detections if item.confidence >= score_threshold]
        all_gt = [(name, box) for name, boxes in gt_boxes.items() for box in boxes]
        used = set()
        for name, box in all_gt:
            truth = np.array(box, dtype=np.float64)
            best_index, best_iou = -1, 0.0
            for index, item in enumerate(kept):
                if index in used:
                    continue
                candidate = np.array(
                    [item.box.x1, item.box.y1, item.box.x2, item.box.y2], dtype=np.float64
                )
                x1 = max(truth[0], candidate[0])
                y1 = max(truth[1], candidate[1])
                x2 = min(truth[2], candidate[2])
                y2 = min(truth[3], candidate[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = (
                    (truth[2] - truth[0]) * (truth[3] - truth[1])
                    + (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
                    - inter
                )
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_index, best_iou = index, iou
            row = confusion_rows.index(name)
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                predicted_name = kept[best_index].class_name
                column = (
                    confusion_columns.index(predicted_name)
                    if predicted_name in confusion_columns
                    else confusion_columns.index(NOT_DETECTED)
                )
                confusion[row, column] += 1
                if predicted_name == name:
                    true_positive += 1
                else:
                    false_positive += 1
                    false_negative += 1
            else:
                confusion[row, confusion_columns.index(NOT_DETECTED)] += 1
                false_negative += 1
        for index, item in enumerate(kept):
            if index in used:
                continue
            column = (
                confusion_columns.index(item.class_name)
                if item.class_name in confusion_columns
                else confusion_columns.index(NOT_DETECTED)
            )
            confusion[confusion_rows.index(BACKGROUND), column] += 1
            false_positive += 1
        if progress is not None:
            progress(position, len(images))

    per_class: dict[str, dict[str, float]] = {}
    aps: list[float] = []
    for name in species:
        ap, precision, recall = average_precision(
            np.asarray(scores_by_class.get(name, []), dtype=np.float64),
            np.asarray(matched_by_class.get(name, []), dtype=np.float64),
            ground_truth_by_class.get(name, 0),
        )
        per_class[name] = {
            "ap50": ap,
            "precision_final": precision,
            "recall_final": recall,
            "ground_truth": ground_truth_by_class.get(name, 0),
            "predictions": len(scores_by_class.get(name, [])),
        }
        if not np.isnan(ap):
            aps.append(ap)

    precision_at = true_positive / max(1, true_positive + false_positive)
    recall_at = true_positive / max(1, true_positive + false_negative)
    f1_at = (
        2 * precision_at * recall_at / (precision_at + recall_at)
        if precision_at + recall_at > 0
        else 0.0
    )
    result: dict[str, object] = {
        "model": label,
        "dataset_yaml": str(dataset_yaml),
        "split": split,
        "images": len(images),
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "map50": float(np.mean(aps)) if aps else 0.0,
        "ap50_by_class": per_class,
        f"precision@{score_threshold}": precision_at,
        f"recall@{score_threshold}": recall_at,
        f"f1@{score_threshold}": f1_at,
        "counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "predictions_total": predictions_total,
            "ground_truth_total": int(sum(ground_truth_by_class.values())),
        },
        "species_confusion": {
            "rows": confusion_rows,
            "columns": confusion_columns,
            "matrix": confusion.tolist(),
        },
    }
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def compare_reports(reports: Iterable[dict[str, object]]) -> dict[str, object]:
    """Monta a tabela comparativa a partir de relatórios já calculados."""
    rows = []
    for report in reports:
        threshold = report.get("score_threshold", 0.35)
        rows.append(
            {
                "model": report.get("model"),
                "split": report.get("split"),
                "map50": report.get("map50"),
                "precision": report.get(f"precision@{threshold}"),
                "recall": report.get(f"recall@{threshold}"),
                "f1": report.get(f"f1@{threshold}"),
                "false_positive": (report.get("counts") or {}).get("false_positive"),
                "false_negative": (report.get("counts") or {}).get("false_negative"),
                "ap50_by_class": {
                    name: values.get("ap50")
                    for name, values in (report.get("ap50_by_class") or {}).items()
                },
            }
        )
    return {"comparison": rows}
