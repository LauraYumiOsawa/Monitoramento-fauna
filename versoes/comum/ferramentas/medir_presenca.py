"""Presença de animal (sem espécie) por detector, no split `test`: base do "filtro de vazias".

Na armadilha, um evento só vira `blank` se o DETECTOR não vê nada; uma detecção que o
classificador recusa vira `unknown` e é guardada. Por isso o filtro de vazias é medido
aqui no estágio do detector: casamento por IoU ≥ 0,5 sem olhar a classe.

    python versoes/comum/ferramentas/medir_presenca.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

from animal_monitor_b.data_prep import parse_label_file  # noqa: E402
from animal_monitor_b.evaluation import _labels_for, _split_images  # noqa: E402
from animal_monitor_b.metrics import match_detections  # noqa: E402
from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner  # noqa: E402
from animal_monitor_hw.models import YoloDetector  # noqa: E402

MODELOS = ROOT / "versoes" / "comum" / "modelos"
DATASET = ROOT / "config" / "dataset.yaml"
CLASSES_A = {0: "anta", 1: "tamandua_bandeira", 2: "lobo_guara"}
# detector -> (arquivo, formato, classes, confiança da versão)
DETECTORES = {
    "a_640": (ROOT / "models" / "animal_detector.onnx", "standard", CLASSES_A, 0.35),
    "a_640_split_int8full": (MODELOS / "a_640_split_int8full.onnx", "split", CLASSES_A, 0.35),
    "a_512_int8": (MODELOS / "a_512_int8.onnx", "standard", CLASSES_A, 0.35),
    "a_320_split_int8full": (MODELOS / "a_320_split_int8full.onnx", "split", CLASSES_A, 0.35),
    "a_224x320_int8": (MODELOS / "a_224x320_int8.onnx", "standard", CLASSES_A, 0.35),
    "a_256_int8": (MODELOS / "a_256_int8.onnx", "standard", CLASSES_A, 0.35),
    "b_416": (ROOT / "models" / "modelo_b" / "presence_detector.onnx", "standard",
              {0: "animal"}, 0.30),
    "b_416_int8": (ROOT / "models" / "modelo_b" / "presence_detector_int8.onnx", "standard",
                   {0: "animal"}, 0.30),
    "b_416_split_int8full": (MODELOS / "b_416_split_int8full.onnx", "split", {0: "animal"}, 0.30),
    "b_224x320_int8": (MODELOS / "b_224x320_int8.onnx", "standard", {0: "animal"}, 0.30),
}


def presenca(nome: str) -> dict[str, float]:
    import cv2

    caminho, formato, classes, confianca = DETECTORES[nome]
    detector = YoloDetector(OnnxRuntimeRunner(caminho, intra_op_threads=4), classes, confianca,
                            0.45, output_format=formato, class_agnostic_nms=True)
    verdadeiros = encontrados = previstos = acertos = 0
    for imagem in _split_images(DATASET, "test"):
        quadro = cv2.imread(str(imagem))
        altura, largura = quadro.shape[:2]
        gt = np.array([caixa.to_pixels(largura, altura)
                       for caixa in parse_label_file(_labels_for(imagem))], dtype=np.float64)
        caixas, escores, _ = detector.detect_all(quadro)
        sinal = match_detections(caixas.astype(np.float64), escores.astype(np.float64),
                                 gt.reshape(-1, 4), 0.5)
        verdadeiros += len(gt)
        previstos += len(caixas)
        acertos += int(np.sum(sinal))
        encontrados += min(len(gt), int(np.sum(sinal)))
    return {
        "recall_presenca": round(encontrados / max(1, verdadeiros), 4),
        "precisao_presenca": round(acertos / max(1, previstos), 4),
        "animais": verdadeiros,
        "deteccoes": previstos,
    }


def main() -> int:
    resultado = {nome: presenca(nome) for nome in DETECTORES}
    destino = ROOT / "versoes" / "ranking" / "medicoes" / "presenca_x86.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    for nome, valores in resultado.items():
        print(f"{nome:24s} recall {valores['recall_presenca']:.3f} "
              f"precisão {valores['precisao_presenca']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
