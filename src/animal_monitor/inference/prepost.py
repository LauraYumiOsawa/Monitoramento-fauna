from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from animal_monitor.errors import ModelOutputError
from animal_monitor.types import BoundingBox, Detection


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale: float
    pad_x: float
    pad_y: float


def letterbox(
    image: np.ndarray, new_shape: tuple[int, int]
) -> tuple[np.ndarray, LetterboxTransform]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV não está instalado. Instale o projeto com as dependências de runtime."
        ) from exc

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve ter formato HxWx3 em BGR.")
    original_height, original_width = image.shape[:2]
    input_height, input_width = new_shape
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    transform = LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        input_height=input_height,
        input_width=input_width,
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )
    return padded, transform


def prepare_input(
    image: np.ndarray, new_shape: tuple[int, int]
) -> tuple[np.ndarray, LetterboxTransform]:
    padded, transform = letterbox(image, new_shape)
    rgb = padded[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor[None, ...], transform


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _iou(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    x1 = np.maximum(reference[0], candidates[:, 0])
    y1 = np.maximum(reference[1], candidates[:, 1])
    x2 = np.minimum(reference[2], candidates[:, 2])
    y2 = np.minimum(reference[3], candidates[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    reference_area = max(0.0, reference[2] - reference[0]) * max(0.0, reference[3] - reference[1])
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0, candidates[:, 3] - candidates[:, 1]
    )
    union = reference_area + candidate_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    class_agnostic: bool,
    max_detections: int,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        overlaps = _iou(boxes[current], boxes[remaining])
        if class_agnostic:
            suppress = overlaps > iou_threshold
        else:
            suppress = (overlaps > iou_threshold) & (class_ids[remaining] == class_ids[current])
        order = remaining[~suppress]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo_output(
    output: np.ndarray,
    class_count: int,
    confidence_threshold: float,
    iou_threshold: float,
    class_agnostic_nms: bool = False,
    max_detections: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise ModelOutputError("O runtime aceita batch igual a 1.")
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ModelOutputError(f"Formato de saída não suportado: {predictions.shape}")

    expected_columns = {4 + class_count, 5 + class_count, 6}
    if predictions.shape[1] not in expected_columns and predictions.shape[0] in expected_columns:
        predictions = predictions.T
    rows, columns = predictions.shape

    looks_end_to_end = (
        columns == 6
        and rows <= 1000
        and rows > 0
        and np.all(np.isfinite(predictions[:, :6]))
        and np.all(np.abs(predictions[:, 5] - np.rint(predictions[:, 5])) < 1e-4)
        and np.all((predictions[:, 5] >= 0) & (predictions[:, 5] < class_count))
    )
    if looks_end_to_end:
        boxes = predictions[:, :4].astype(np.float32, copy=False)
        scores = predictions[:, 4].astype(np.float32, copy=False)
        class_ids = np.rint(predictions[:, 5]).astype(np.int64)
    elif columns == 4 + class_count:
        boxes = _xywh_to_xyxy(predictions[:, :4])
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        scores = class_scores[np.arange(rows), class_ids].astype(np.float32)
    elif columns == 5 + class_count:
        boxes = _xywh_to_xyxy(predictions[:, :4])
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        scores = (predictions[:, 4] * class_scores[np.arange(rows), class_ids]).astype(np.float32)
    else:
        raise ModelOutputError(
            f"Saída {predictions.shape} incompatível com {class_count} classes. "
            "Exporte um modelo YOLO de detecção sem NMS embutido ou use saída end-to-end Nx6."
        )

    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (scores >= confidence_threshold)
        & (class_ids >= 0)
        & (class_ids < class_count)
    )
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    selected = non_max_suppression(
        boxes,
        scores,
        class_ids,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic_nms,
        max_detections=max_detections,
    )
    return boxes[selected], scores[selected], class_ids[selected]


def undo_letterbox(boxes: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    restored = boxes.astype(np.float32, copy=True)
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.pad_x) / transform.scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - transform.pad_y) / transform.scale
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, transform.original_width - 1)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, transform.original_height - 1)
    return restored


def build_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    names: Mapping[int, str],
    transform: LetterboxTransform,
) -> list[Detection]:
    restored = undo_letterbox(boxes, transform)
    detections: list[Detection] = []
    for box, score, class_id in zip(restored, scores, class_ids, strict=True):
        class_index = int(class_id)
        detections.append(
            Detection(
                class_id=class_index,
                class_name=names.get(class_index, f"classe_{class_index}"),
                confidence=float(score),
                box=BoundingBox(*(float(value) for value in box)),
            )
        )
    return detections
