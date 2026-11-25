"""Simula disparos da armadilha sobre os vídeos de amostra e conta o desfecho de cada evento.

Cada evento é uma rajada de quadros consecutivos do vídeo, analisada exatamente pelo
código das versões (`animal_monitor_hw.analise`). O desfecho é classificado como:

* `correto` — espécie confirmada igual à do vídeo;
* `errado` — espécie confirmada diferente da do vídeo (o erro que custa caro em campo);
* `inconclusivo` — houve detecção sem espécie confirmada (vai para revisão);
* `vazio` — nada detectado.

Os vídeos são editados (7–12 cortes de cena, sobreposições gráficas): servem para medir
consistência entre modelos e precisões numéricas, não desempenho de campo. Proxy em x86_64.

    python versoes/comum/ferramentas/simular_rajadas.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

from animal_monitor.config import load_class_names  # noqa: E402
from animal_monitor_b.config import RuntimeConfigB, TrackerConfig  # noqa: E402
from animal_monitor_hw.analise import AnalisadorModeloA, AnalisadorModeloB  # noqa: E402
from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner  # noqa: E402
from animal_monitor_hw.captura import CameraArquivos  # noqa: E402
from animal_monitor_hw.models import CropClassifier, YoloDetector  # noqa: E402

MODELOS = ROOT / "versoes" / "comum" / "modelos"
DATASET = ROOT / "config" / "dataset.yaml"
VIDEOS = {
    "anta": ROOT / "samples" / "video_anta.mp4",
    "lobo_guara": ROOT / "samples" / "video_lobo_guara.mp4",
    "tamandua_bandeira": ROOT / "samples" / "video_tamandua_bandeira.mp4",
}
CLASSIFICADORES = {
    "mnv2": ROOT / "models" / "modelo_b" / "species_classifier.onnx",
    "mnv2_int8": ROOT / "models" / "modelo_b" / "species_classifier_int8.onnx",
    "shufflenet_int8": (
        ROOT / "models" / "modelo_b_shufflenet_v2_x0_5" / "species_classifier_int8.onnx"
    ),
}
LABELS_B = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]
# proxy -> (detector ONNX, formato, classificador ou None)
PROXIES = {
    "a_640": (ROOT / "models" / "animal_detector.onnx", "standard", None),
    "a_640_split_int8full": (MODELOS / "a_640_split_int8full.onnx", "split", None),
    "a_512_int8": (MODELOS / "a_512_int8.onnx", "standard", None),
    "a_320_split_int8full": (MODELOS / "a_320_split_int8full.onnx", "split", None),
    "a_224x320_int8": (MODELOS / "a_224x320_int8.onnx", "standard", None),
    "a_256_int8": (MODELOS / "a_256_int8.onnx", "standard", None),
    "b_416+mnv2": (ROOT / "models" / "modelo_b" / "presence_detector.onnx", "standard", "mnv2"),
    "b_416_int8+mnv2": (
        ROOT / "models" / "modelo_b" / "presence_detector_int8.onnx", "standard", "mnv2"
    ),
    "b_416_int8+mnv2_int8": (
        ROOT / "models" / "modelo_b" / "presence_detector_int8.onnx", "standard", "mnv2_int8"
    ),
    "b_416_split_int8full+mnv2_int8": (MODELOS / "b_416_split_int8full.onnx", "split", "mnv2_int8"),
    "b_416_split_int8full+mnv2": (MODELOS / "b_416_split_int8full.onnx", "split", "mnv2"),
    "b_224x320_int8+mnv2_int8": (MODELOS / "b_224x320_int8.onnx", "standard", "mnv2_int8"),
}


def config_rajada() -> RuntimeConfigB:
    """Os mesmos limiares de `config/runtime_b.yaml` das versões."""
    return RuntimeConfigB(
        detector_confidence=0.30, detector_iou=0.40, classify_every=1, min_votes=3,
        min_vote_share=0.65, min_species_confidence=0.50, max_crops_per_frame=3,
        tracker=TrackerConfig(max_age=10, min_hits=2, iou_match=0.30, low_score=0.15,
                              max_tracks=16),
    )


def analisador(proxy: str):
    caminho, formato, classificador = PROXIES[proxy]
    if classificador is None:
        detector = YoloDetector(OnnxRuntimeRunner(caminho, intra_op_threads=4),
                                load_class_names(DATASET), 0.35, 0.45, output_format=formato)
        return AnalisadorModeloA(detector, min_hits=3)
    config = config_rajada()
    detector = YoloDetector(OnnxRuntimeRunner(caminho, intra_op_threads=4), {0: "animal"},
                            config.detector_confidence, config.detector_iou,
                            max_detections=20, output_format=formato)
    especie = CropClassifier(OnnxRuntimeRunner(CLASSIFICADORES[classificador],
                                               intra_op_threads=2), LABELS_B)
    return AnalisadorModeloB(detector, especie, config)


def simular(proxy: str, quadros: int = 8, eventos: int = 30) -> dict[str, object]:
    modelo = analisador(proxy)
    por_video = {}
    total: Counter[str] = Counter()
    for especie, video in VIDEOS.items():
        camera = CameraArquivos(video)
        desfechos: Counter[str] = Counter()
        for _ in range(eventos):
            resultado = modelo.analisar(camera.rajada(quadros, 0.0))
            confirmadas = {item.rotulo for item in resultado.avistamentos if item.tipo == "animal"}
            if confirmadas - {especie}:
                desfechos["errado"] += 1
            elif especie in confirmadas:
                desfechos["correto"] += 1
            elif resultado.tipo == "unknown":
                desfechos["inconclusivo"] += 1
            else:
                desfechos["vazio"] += 1
        por_video[especie] = dict(desfechos)
        total.update(desfechos)
    com_animal = total["correto"] + total["errado"]
    return {
        "eventos": eventos * len(VIDEOS),
        "quadros_por_evento": quadros,
        "desfechos": dict(total),
        "por_video": por_video,
        "acerto_entre_confirmados": round(total["correto"] / com_animal, 4) if com_animal else None,
        "taxa_erro_confirmado": round(total["errado"] / (eventos * len(VIDEOS)), 4),
    }


def main() -> int:
    resultados: dict[str, object] = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "aviso": "Proxy em x86_64; vídeos editados, não capturas de armadilha.",
        "proxies": {},
    }
    for proxy in PROXIES:
        print(f"simulando {proxy}", flush=True)
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            resultados["proxies"][proxy] = simular(proxy)
    destino = ROOT / "versoes" / "ranking" / "medicoes" / "rajadas_x86.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"resultados em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
