"""Métricas em NumPy puro, compartilhadas entre classificação e detecção.

A AP é implementada aqui para que o Modelo A e o Modelo B sejam medidos pelo **mesmo
código**. Comparar o mAP calculado pelo Ultralytics com um mAP calculado por outra
biblioteca introduz diferença de interpolação e invalida a comparação do TCC.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(predicted)] += 1
    return matrix


def per_class_scores(matrix: np.ndarray) -> dict[str, np.ndarray]:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual}


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    scores = per_class_scores(matrix)
    total = int(matrix.sum())
    correct = int(np.diag(matrix).sum())
    present = scores["support"] > 0
    return {
        "labels": labels,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": float(scores["recall"][present].mean()) if present.any() else 0.0,
        "macro_f1": float(scores["f1"][present].mean()) if present.any() else 0.0,
        "per_class": {
            label: {
                "precision": float(scores["precision"][index]),
                "recall": float(scores["recall"][index]),
                "f1": float(scores["f1"][index]),
                "support": int(scores["support"][index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def average_precision(
    scores: np.ndarray, matched: np.ndarray, total_ground_truth: int
) -> tuple[float, float, float]:
    """AP com interpolação por todos os pontos (estilo COCO/VOC 2010+).

    Retorna (AP, precisão no fim da curva, recall no fim da curva).
    """
    if total_ground_truth == 0:
        return float("nan"), float("nan"), float("nan")
    if scores.size == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    matched = matched[order].astype(np.float64)
    true_positive = np.cumsum(matched)
    false_positive = np.cumsum(1.0 - matched)
    recall = true_positive / total_ground_truth
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)

    monotone = np.concatenate(([0.0], precision, [0.0]))
    for index in range(monotone.size - 2, -1, -1):
        monotone[index] = max(monotone[index], monotone[index + 1])
    recall_points = np.concatenate(([0.0], recall, [recall[-1] if recall.size else 0.0]))
    indices = np.where(recall_points[1:] != recall_points[:-1])[0]
    widths = recall_points[indices + 1] - recall_points[indices]
    ap = float(np.sum(widths * monotone[indices + 1]))
    return ap, float(precision[-1]), float(recall[-1])


def match_detections(
    predicted_boxes: np.ndarray,
    predicted_scores: np.ndarray,
    ground_truth_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Casa predições com verdades de uma imagem/classe. Retorna vetor 0/1 por predição."""
    matched = np.zeros(predicted_boxes.shape[0], dtype=np.float64)
    if predicted_boxes.size == 0 or ground_truth_boxes.size == 0:
        return matched
    taken = np.zeros(ground_truth_boxes.shape[0], dtype=bool)
    for index in np.argsort(-predicted_scores, kind="stable"):
        box = predicted_boxes[index]
        x1 = np.maximum(box[0], ground_truth_boxes[:, 0])
        y1 = np.maximum(box[1], ground_truth_boxes[:, 1])
        x2 = np.minimum(box[2], ground_truth_boxes[:, 2])
        y2 = np.minimum(box[3], ground_truth_boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        areas = np.maximum(0.0, ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]) * np.maximum(
            0.0, ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
        )
        union = area + areas - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        iou[taken] = -1.0
        best = int(np.argmax(iou))
        if iou[best] >= iou_threshold:
            taken[best] = True
            matched[index] = 1.0
    return matched
