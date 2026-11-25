"""Saídas da cabeça YOLOv8 nos formatos exigidos por cada acelerador.

O ONNX padrão do Ultralytics termina em um único tensor `(1, 4 + nc, N)` que mistura
coordenadas em pixels (0 a 640) com probabilidades de classe (0 a 1). Em FP32 isso não
importa. Em INT8 com um único fator de escala por tensor, importa muito: foi exatamente o
que o projeto mediu ao quantizar todos os operadores do detector (confiança 1.673,
`docs/MODELO_B.md`, seção 4.3). Por isso NPUs que só executam inteiro (RKNN, RV1106,
SG2002, Hailo) recebem o modelo **cortado antes da decodificação**:

* por escala `i`: `box_i` com `4 * reg_max` canais (logits da DFL) e `cls_i` com `nc`
  canais já passados pela sigmoide;
* a DFL, a grade de âncoras e o NMS rodam na CPU, em NumPy, neste módulo.

Também ficam aqui os decodificadores das saídas que já chegam prontas do acelerador:
NMS por classe do Hailo e as listas (caixas, escores, classes) do IMX500.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError, ModelOutputError
from animal_monitor.inference.prepost import non_max_suppression

REG_MAX = 16


def export_split_head_onnx(
    weights: str | Path,
    output: str | Path,
    imgsz: int | tuple[int, int] = 640,
    opset: int = 17,
) -> dict[str, object]:
    """Exporta o YOLOv8 com 6 saídas (box/cls por escala), sem DFL nem NMS no grafo.

    É o mesmo corte usado pelos exemplos oficiais de RKNN e Hailo para YOLOv8, feito
    aqui no PyTorch para não depender de forks do Ultralytics.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Exportação requer o extra de treino (ultralytics e torch).") from exc

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    if height % 32 or width % 32:
        raise ConfigurationError("A entrada do YOLOv8 deve ser múltipla de 32.")

    model = YOLO(str(weights_path)).model.float().eval()
    model = model.fuse(verbose=False)
    head = model.model[-1]
    if not hasattr(head, "cv2") or head.cv2 is None:
        raise ConfigurationError("A cabeça do modelo não expõe cv2/cv3 (não é YOLOv8 Detect).")

    def forward_split(self, features):
        outputs = []
        for index in range(self.nl):
            outputs.append(self.cv2[index](features[index]))
            outputs.append(self.cv3[index](features[index]).sigmoid())
        return outputs

    head.forward = types.MethodType(forward_split, head)
    names = [f"{kind}{index}" for index in range(head.nl) for kind in ("box", "cls")]
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)
    arguments: dict[str, object] = {
        "input_names": ["images"],
        "output_names": names,
        "opset_version": opset,
        "do_constant_folding": True,
    }
    # Mesmo motivo de animal_monitor_b.exporting: o caminho TorchScript evita onnxscript.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        arguments["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(destination), **arguments)  # type: ignore[arg-type]
    return {
        "onnx": str(destination),
        "input_hw": [height, width],
        "outputs": names,
        "num_classes": int(head.nc),
        "reg_max": int(head.reg_max),
        "strides": [int(value) for value in head.stride.tolist()],
        "class_names": {int(key): str(value) for key, value in model.names.items()},
    }


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _pair_split_outputs(
    outputs: Sequence[np.ndarray], num_classes: int, reg_max: int, layout: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Agrupa as saídas em pares (box, cls) por escala, da maior grade para a menor.

    Aceita as 6 saídas deste módulo e as 9 do rknn_model_zoo (box, cls, cls_sum por
    escala): o `cls_sum` é só um atalho de filtragem e é ignorado. O layout é explícito
    porque adivinhá-lo falha quando a grade tem 64 células, o mesmo número de canais da
    DFL (entrada 512 px).
    """
    if layout not in {"nchw", "nhwc"}:
        raise ConfigurationError("'layout' deve ser 'nchw' ou 'nhwc'.")
    boxes: dict[tuple[int, int], np.ndarray] = {}
    scores: dict[tuple[int, int], np.ndarray] = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ModelOutputError(
                f"Saída dividida deve ter 4 dimensões; encontrado {array.shape}."
            )
        if layout == "nhwc":
            array = np.transpose(array, (0, 3, 1, 2))
        channels, grid = array.shape[1], (array.shape[2], array.shape[3])
        if channels == 4 * reg_max:
            boxes[grid] = array
        elif channels == num_classes and grid not in scores:
            scores[grid] = array
        # canal único: cls_sum do rknn_model_zoo; se nc == 1 o primeiro já foi aceito.
    if not boxes or set(boxes) != set(scores):
        raise ModelOutputError(
            "Não foi possível parear box/cls por escala. Exporte com "
            "export_split_head_onnx ou confira nc/reg_max."
        )
    grids = sorted(boxes, key=lambda grid: grid[0] * grid[1], reverse=True)
    return [(boxes[grid], scores[grid]) for grid in grids]


def decode_split_head(
    outputs: Sequence[np.ndarray],
    input_hw: tuple[int, int],
    num_classes: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int = 100,
    class_agnostic_nms: bool = False,
    reg_max: int = REG_MAX,
    layout: str = "nchw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DFL + âncoras + NMS em NumPy. Caixas xyxy nas coordenadas da entrada da rede."""
    input_height, input_width = input_hw
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    bins = np.arange(reg_max, dtype=np.float32)
    for box_map, class_map in _pair_split_outputs(outputs, num_classes, reg_max, layout):
        _, _, grid_h, grid_w = class_map.shape
        probabilities = class_map[0].reshape(num_classes, -1).T
        class_ids = probabilities.argmax(axis=1)
        best = probabilities[np.arange(probabilities.shape[0]), class_ids]
        # Filtra antes da DFL: em armadilha fotográfica quase toda célula é fundo.
        keep = np.flatnonzero(best >= confidence_threshold)
        if keep.size == 0:
            continue
        logits = box_map[0].reshape(4, reg_max, -1)[:, :, keep]
        distances = (_softmax(logits, axis=1) * bins[None, :, None]).sum(axis=1)
        rows, columns = np.divmod(keep, grid_w)
        anchor_x = columns.astype(np.float32) + 0.5
        anchor_y = rows.astype(np.float32) + 0.5
        stride_x = input_width / grid_w
        stride_y = input_height / grid_h
        decoded = np.stack(
            [
                (anchor_x - distances[0]) * stride_x,
                (anchor_y - distances[1]) * stride_y,
                (anchor_x + distances[2]) * stride_x,
                (anchor_y + distances[3]) * stride_y,
            ],
            axis=1,
        )
        all_boxes.append(decoded.astype(np.float32))
        all_scores.append(best[keep].astype(np.float32))
        all_classes.append(class_ids[keep].astype(np.int64))
    if not all_boxes:
        return _empty()
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    selected = non_max_suppression(
        boxes, scores, classes, iou_threshold, class_agnostic_nms, max_detections
    )
    return boxes[selected], scores[selected], classes[selected]


def decode_hailo_nms_by_class(
    per_class: Sequence[np.ndarray],
    input_hw: tuple[int, int],
    confidence_threshold: float,
    max_detections: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte a saída `HAILO_NMS_BY_CLASS` em caixas xyxy na entrada da rede.

    O HailoRT entrega uma lista com um array por classe; cada linha é
    `[y_min, x_min, y_max, x_max, score]` normalizada em [0, 1]. O NMS já foi feito no
    chip, então aqui só se filtra, reordena e limita.
    """
    input_height, input_width = input_hw
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for class_id, detections in enumerate(per_class):
        array = np.asarray(detections, dtype=np.float32).reshape(-1, 5)
        array = array[array[:, 4] >= confidence_threshold]
        if array.size == 0:
            continue
        boxes.append(
            np.stack(
                [
                    array[:, 1] * input_width,
                    array[:, 0] * input_height,
                    array[:, 3] * input_width,
                    array[:, 2] * input_height,
                ],
                axis=1,
            )
        )
        scores.append(array[:, 4])
        classes.append(np.full(array.shape[0], class_id, dtype=np.int64))
    if not boxes:
        return _empty()
    all_boxes = np.concatenate(boxes)
    all_scores = np.concatenate(scores)
    all_classes = np.concatenate(classes)
    order = np.argsort(all_scores)[::-1][:max_detections]
    return all_boxes[order], all_scores[order], all_classes[order]


def decode_boxes_scores_classes(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    input_hw: tuple[int, int],
    confidence_threshold: float,
    normalized: bool,
    order: str = "xy",
    max_detections: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saídas já pós-processadas (NMS embutido), como as do IMX500 exportado pelo Ultralytics.

    `order="yx"` cobre redes que entregam `[y1, x1, y2, x2]`.
    """
    input_height, input_width = input_hw
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.rint(np.asarray(classes, dtype=np.float32).reshape(-1)).astype(np.int64)
    count = min(boxes.shape[0], scores.shape[0], classes.shape[0])
    boxes, scores, classes = boxes[:count], scores[:count], classes[:count]
    if order == "yx":
        boxes = boxes[:, [1, 0, 3, 2]]
    elif order != "xy":
        raise ConfigurationError("'order' deve ser 'xy' ou 'yx'.")
    if normalized:
        boxes = boxes * np.array(
            [input_width, input_height, input_width, input_height], dtype=np.float32
        )
    valid = np.isfinite(boxes).all(axis=1) & (scores >= confidence_threshold) & (classes >= 0)
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    order_index = np.argsort(scores)[::-1][:max_detections]
    return boxes[order_index], scores[order_index], classes[order_index]


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
