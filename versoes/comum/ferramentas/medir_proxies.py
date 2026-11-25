"""Gera as variantes de modelo das versões e mede cada uma na máquina de desenvolvimento.

Tudo o que sai daqui é PROXY, não medição no hardware-alvo:

* acurácia: split `test` do projeto (18 imagens, 20 caixas), ONNX Runtime CPU, mesmo
  código de métrica do Modelo A/B (`animal_monitor_b.evaluation`). Mede o efeito de
  resolução, INT8 e corte de cabeça sobre o MESMO modelo — não o efeito do compilador
  de cada NPU, que pode acrescentar perda própria;
* latência: x86_64, só para razões entre variantes. Valores absolutos não se transferem.

Uso (na raiz do projeto, com o ambiente de treino ativo):

    python versoes/comum/ferramentas/medir_proxies.py            # tudo
    python versoes/comum/ferramentas/medir_proxies.py --so-medir # reaproveita os ONNX
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

from animal_monitor.config import RuntimeConfig, load_class_names  # noqa: E402
from animal_monitor.inference.prepost import prepare_input  # noqa: E402
from animal_monitor_b.config import RuntimeConfigB  # noqa: E402
from animal_monitor_b.evaluation import evaluate_species_predictor  # noqa: E402
from animal_monitor_b.exporting import sha256_file  # noqa: E402
from animal_monitor_b.runtime.pipeline import TwoStagePipeline  # noqa: E402
from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner  # noqa: E402
from animal_monitor_hw.models import CropClassifier, YoloDetector  # noqa: E402
from animal_monitor_hw.quantizacao import quantize_detector_int8  # noqa: E402
from animal_monitor_hw.yolo_heads import export_split_head_onnx  # noqa: E402

MODELOS = ROOT / "versoes" / "comum" / "modelos"
SAIDA = ROOT / "versoes" / "ranking" / "medicoes"
DATASET = ROOT / "config" / "dataset.yaml"
RUNTIME_A = ROOT / "config" / "runtime.yaml"
RUNTIME_B = ROOT / "config" / "modelo_b" / "runtime.yaml"
PESOS = {
    "a": ROOT / "runs" / "train" / "baseline_yolov8n-5" / "weights" / "best.pt",
    "b": ROOT / "runs" / "modelo_b" / "presence_yolov8n_416" / "weights" / "best.pt",
}
REFERENCIA = {
    "a_640": ROOT / "models" / "animal_detector.onnx",
    "b_416": ROOT / "models" / "modelo_b" / "presence_detector.onnx",
    "b_416_int8": ROOT / "models" / "modelo_b" / "presence_detector_int8.onnx",
}
CLASSIFICADORES = {
    "mnv2": ROOT / "models" / "modelo_b" / "species_classifier.onnx",
    "mnv2_int8": ROOT / "models" / "modelo_b" / "species_classifier_int8.onnx",
    "shufflenet": ROOT / "models" / "modelo_b_shufflenet_v2_x0_5" / "species_classifier.onnx",
    "shufflenet_int8": (
        ROOT / "models" / "modelo_b_shufflenet_v2_x0_5" / "species_classifier_int8.onnx"
    ),
}
CALIBRACAO = ROOT / "data" / "animals" / "images" / "val"
LABELS_B = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]

# (nome, modelo, entrada (h, w)). 640/512 são as resoluções de referência da pesquisa;
# 416 é a do Modelo B; 320 e 224x320 cobrem RV1106 e MaixCAM; 192 cobre o Grove Vision AI V2.
VARIANTES_PADRAO = [
    ("a_512", "a", (512, 512)),
    ("a_320", "a", (320, 320)),
    ("a_224x320", "a", (224, 320)),
    ("a_192", "a", (192, 192)),
    ("a_224", "a", (224, 224)),
    ("a_256", "a", (256, 256)),
    ("b_320", "b", (320, 320)),
    ("b_224x320", "b", (224, 320)),
    ("b_192", "b", (192, 192)),
]
VARIANTES_DIVIDIDAS = [
    ("a_640_split", "a", (640, 640)),
    ("a_320_split", "a", (320, 320)),
    ("b_416_split", "b", (416, 416)),
    ("b_320_split", "b", (320, 320)),
]


def _silencioso(funcao, *args, **kwargs):
    """Ultralytics e o quantizador imprimem muito; o log completo vai para arquivo."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        resultado = funcao(*args, **kwargs)
    (SAIDA / "log_exportacao.txt").open("a", encoding="utf-8").write(buffer.getvalue())
    return resultado


def exportar_padrao(nome: str, modelo: str, entrada: tuple[int, int]) -> Path:
    from ultralytics import YOLO

    destino = MODELOS / f"{nome}.onnx"
    if destino.is_file():
        return destino
    # Copia os pesos: o Ultralytics grava o .onnx ao lado do .pt e sobrescreveria runs/.
    trabalho = MODELOS / "pesos" / nome
    trabalho.mkdir(parents=True, exist_ok=True)
    pesos = trabalho / "best.pt"
    shutil.copy2(PESOS[modelo], pesos)
    exportado = _silencioso(
        YOLO(str(pesos)).export,
        format="onnx",
        imgsz=list(entrada),
        batch=1,
        dynamic=False,
        simplify=True,
        opset=17,
        nms=False,
    )
    shutil.move(str(exportado), destino)
    return destino


def exportar_dividida(nome: str, modelo: str, entrada: tuple[int, int]) -> Path:
    destino = MODELOS / f"{nome}.onnx"
    if not destino.is_file():
        _silencioso(export_split_head_onnx, PESOS[modelo], destino, imgsz=entrada)
    return destino


def quantizar(modelo_fp32: Path, apenas_conv: bool) -> Path:
    sufixo = "int8" if apenas_conv else "int8full"
    destino = modelo_fp32.with_name(f"{modelo_fp32.stem}_{sufixo}.onnx")
    if not destino.is_file():
        _silencioso(
            quantize_detector_int8,
            modelo_fp32,
            CALIBRACAO,
            destino,
            amostras=36,
            apenas_conv=apenas_conv,
        )
    return destino


def detector(caminho: Path, nomes: dict[int, str], formato: str, conf: float, iou: float,
             threads: int = 4) -> YoloDetector:
    runner = OnnxRuntimeRunner(caminho, intra_op_threads=threads)
    return YoloDetector(runner, nomes, conf, iou, max_detections=100, output_format=formato,
                        model_path=caminho)


def avaliar_a(caminho: Path, formato: str) -> dict[str, object]:
    runtime = RuntimeConfig.from_yaml(RUNTIME_A)
    nomes = load_class_names(DATASET)
    modelo = detector(caminho, nomes, formato, runtime.confidence, runtime.iou_threshold)
    return evaluate_species_predictor(modelo.infer, DATASET, split="test", label=caminho.stem)


def avaliar_b(caminho_detector: Path, formato: str, classificador: str) -> dict[str, object]:
    config = RuntimeConfigB.from_yaml(RUNTIME_B)
    presenca = detector(caminho_detector, {0: "animal"}, formato, config.detector_confidence,
                        config.detector_iou)
    runner = OnnxRuntimeRunner(CLASSIFICADORES[classificador], intra_op_threads=4)
    especie = CropClassifier(runner, LABELS_B, config.crop_margin,
                             model_path=CLASSIFICADORES[classificador])
    pipeline = TwoStagePipeline(presenca, especie, config)
    rotulo = f"{caminho_detector.stem}+{classificador}"
    return evaluate_species_predictor(pipeline.detect_species, DATASET, split="test", label=rotulo)


def resumo(relatorio: dict[str, object]) -> dict[str, object]:
    limiar = relatorio.get("score_threshold", 0.35)
    contagem = relatorio.get("counts", {})
    return {
        "map50": round(float(relatorio["map50"]), 4),
        "precisao": round(float(relatorio[f"precision@{limiar}"]), 4),
        "recall": round(float(relatorio[f"recall@{limiar}"]), 4),
        "f1": round(float(relatorio[f"f1@{limiar}"]), 4),
        "vp": contagem.get("true_positive"),
        "fp": contagem.get("false_positive"),
        "fn": contagem.get("false_negative"),
        "confusao": relatorio.get("species_confusion"),
    }


def classificacao_quadro_inteiro(classificador: str) -> dict[str, object]:
    """Classifica a imagem inteira (sem detector), como fariam ESP32-S3 e Nicla Vision.

    Rótulo verdadeiro = espécie da maior caixa anotada. `desconhecido` conta como erro de
    espécie, mas é reportado à parte: em campo ele mandaria o evento para revisão.
    """
    import cv2

    from animal_monitor_b.data_prep import parse_label_file
    from animal_monitor_b.evaluation import _labels_for, _split_images

    nomes = load_class_names(DATASET)
    runner = OnnxRuntimeRunner(CLASSIFICADORES[classificador], intra_op_threads=4)
    modelo = CropClassifier(runner, LABELS_B, crop_margin=0.0)
    acertos = desconhecidos = total = 0
    for caminho in _split_images(DATASET, "test"):
        quadro = cv2.imread(str(caminho))
        caixas = parse_label_file(_labels_for(caminho))
        if quadro is None or not caixas:
            continue
        altura, largura = quadro.shape[:2]
        maior = max(caixas, key=lambda caixa: caixa.width * caixa.height)
        verdade = nomes[maior.class_id]
        probabilidades = modelo.classify(quadro, np.array([[0, 0, largura - 1, altura - 1]]))
        previsto = LABELS_B[int(np.argmax(probabilidades[0]))]
        total += 1
        acertos += int(previsto == verdade)
        desconhecidos += int(previsto == "desconhecido")
    return {
        "imagens": total,
        "acuracia": round(acertos / max(1, total), 4),
        "previsto_desconhecido": desconhecidos,
    }


def equivalencia_dividida(padrao: Path, dividida: Path, nomes: dict[int, str]) -> dict[str, object]:
    """Confere se o corte de cabeça + DFL em NumPy reproduz o export padrão."""
    import cv2

    from animal_monitor_b.evaluation import _split_images

    a = detector(padrao, nomes, "standard", 0.25, 0.45)
    b = detector(dividida, nomes, "split", 0.25, 0.45)
    maior_dif_caixa = 0.0
    maior_dif_escore = 0.0
    divergentes = 0
    imagens = _split_images(DATASET, "test")
    for caminho in imagens:
        quadro = cv2.imread(str(caminho))
        caixas_a, escores_a, classes_a = a.detect_all(quadro)
        caixas_b, escores_b, classes_b = b.detect_all(quadro)
        if caixas_a.shape != caixas_b.shape or not np.array_equal(classes_a, classes_b):
            divergentes += 1
            continue
        if caixas_a.size:
            maior_dif_caixa = max(maior_dif_caixa, float(np.abs(caixas_a - caixas_b).max()))
            maior_dif_escore = max(maior_dif_escore, float(np.abs(escores_a - escores_b).max()))
    return {
        "imagens": len(imagens),
        "imagens_com_conjunto_diferente": divergentes,
        "maior_diferenca_caixa_px": round(maior_dif_caixa, 4),
        "maior_diferenca_escore": round(maior_dif_escore, 6),
    }


def latencia(caminho: Path, threads: int, repeticoes: int = 40) -> dict[str, float]:
    """Só `session.run`, sem pré/pós, como os benchmarks publicados pelo Ultralytics."""
    import cv2

    from animal_monitor_b.evaluation import _split_images

    runner = OnnxRuntimeRunner(caminho, intra_op_threads=threads)
    quadro = cv2.imread(str(_split_images(DATASET, "test")[0]))
    if runner.spec.height == runner.spec.width and runner.spec.height <= 160:
        tensor = np.zeros((1, 3, runner.spec.height, runner.spec.width), dtype=np.float32)
    else:
        tensor, _ = prepare_input(quadro, (runner.spec.height, runner.spec.width))
    for _ in range(5):
        runner.run(tensor)
    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        runner.run(tensor)
        tempos.append((time.perf_counter() - inicio) * 1000.0)
    valores = np.asarray(tempos)
    return {
        "media_ms": round(float(valores.mean()), 3),
        "p50_ms": round(float(np.percentile(valores, 50)), 3),
        "p95_ms": round(float(np.percentile(valores, 95)), 3),
    }


def gflops_yolo(modelo: str, entrada: tuple[int, int]) -> float | None:
    try:
        import torch
        from thop import profile
        from ultralytics import YOLO
    except ImportError:
        return None
    rede = YOLO(str(PESOS[modelo])).model.float().eval()
    rede = _silencioso(rede.fuse, verbose=False)
    with torch.no_grad():
        macs, _ = _silencioso(profile, rede, inputs=(torch.zeros(1, 3, *entrada),), verbose=False)
    return round(2 * macs / 1e9, 3)


def gflops_classificador(caminho_pt: Path, entrada: int = 128) -> float | None:
    try:
        import torch
        from thop import profile

        from animal_monitor_b.classifier import build_model
    except ImportError:
        return None
    carga = torch.load(caminho_pt, map_location="cpu", weights_only=False)
    rede = build_model(str(carga["backbone"]), len(carga["labels"]), pretrained=False).eval()
    with torch.no_grad():
        macs, _ = _silencioso(profile, rede, inputs=(torch.zeros(1, 3, entrada, entrada),),
                              verbose=False)
    return round(2 * macs / 1e9, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--so-medir", action="store_true", help="Não exporta; só mede.")
    args = parser.parse_args()
    MODELOS.mkdir(parents=True, exist_ok=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    nomes_a = load_class_names(DATASET)

    caminhos: dict[str, Path] = dict(REFERENCIA)
    if not args.so_medir:
        for nome, modelo, entrada in VARIANTES_PADRAO:
            print(f"exportando {nome} {entrada}", flush=True)
            caminhos[nome] = exportar_padrao(nome, modelo, entrada)
        for nome, modelo, entrada in VARIANTES_DIVIDIDAS:
            print(f"exportando {nome} (cabeça dividida) {entrada}", flush=True)
            caminhos[nome] = exportar_dividida(nome, modelo, entrada)
    else:
        for nome, _, _ in VARIANTES_PADRAO + VARIANTES_DIVIDIDAS:
            caminhos[nome] = MODELOS / f"{nome}.onnx"

    # INT8: só Conv no export padrão (achado do projeto); na cabeça dividida, todos os
    # operadores — é o que uma NPU só-inteiro faz e o que se quer testar.
    for nome in ["a_640", "a_512", "a_320", "a_224x320", "a_256", "a_224", "a_192", "b_320",
                 "b_224x320", "b_192"]:
        print(f"quantizando {nome} (apenas Conv)", flush=True)
        origem = caminhos[nome]
        if nome == "a_640":
            copia = MODELOS / "a_640.onnx"
            if not copia.is_file():
                shutil.copy2(origem, copia)
            origem = copia
        caminhos[f"{nome}_int8"] = quantizar(origem, apenas_conv=True)
    for nome in ["a_640_split", "a_320_split", "b_416_split", "b_320_split"]:
        print(f"quantizando {nome} (todos os operadores)", flush=True)
        caminhos[f"{nome}_int8full"] = quantizar(caminhos[nome], apenas_conv=False)

    resultados: dict[str, object] = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "ambiente": {
            "cpu": platform.processor() or platform.machine(),
            "plataforma": platform.platform(),
            "python": platform.python_version(),
        },
        "aviso": (
            "Proxy em x86_64 com ONNX Runtime CPU. Acurácia no split test (18 imagens, "
            "20 caixas): uma caixa vale 5 pontos de recall. Latência vale só como razão."
        ),
        "modelos": {},
        "acuracia": {},
        "equivalencia_cabeca_dividida": {},
        "latencia_x86": {},
        "gflops": {},
    }
    try:
        import onnxruntime

        resultados["ambiente"]["onnxruntime"] = onnxruntime.__version__
    except ImportError:  # pragma: no cover
        pass

    for nome, caminho in sorted(caminhos.items()):
        resultados["modelos"][nome] = {
            "arquivo": str(caminho.relative_to(ROOT)),
            "tamanho_mb": round(caminho.stat().st_size / 1024**2, 3),
            "sha256": sha256_file(caminho),
        }

    print("avaliando Modelo A", flush=True)
    for nome in ["a_640", "a_512", "a_320", "a_224x320", "a_256", "a_224", "a_192"]:
        resultados["acuracia"][nome] = resumo(avaliar_a(caminhos[nome], "standard"))
        resultados["acuracia"][f"{nome}_int8"] = resumo(
            avaliar_a(caminhos[f"{nome}_int8"], "standard")
        )
    for nome in ["a_640_split", "a_320_split"]:
        resultados["acuracia"][nome] = resumo(avaliar_a(caminhos[nome], "split"))
        resultados["acuracia"][f"{nome}_int8full"] = resumo(
            avaliar_a(caminhos[f"{nome}_int8full"], "split")
        )

    print("avaliando Modelo B", flush=True)
    combinacoes_b = [
        ("b_416", "standard", "mnv2"),
        ("b_416_int8", "standard", "mnv2_int8"),
        ("b_416_int8", "standard", "mnv2"),
        ("b_416", "standard", "shufflenet_int8"),
        ("b_416_split", "split", "mnv2"),
        ("b_416_split_int8full", "split", "mnv2_int8"),
        ("b_416_split_int8full", "split", "mnv2"),
        ("b_320_int8", "standard", "mnv2_int8"),
        ("b_320_split_int8full", "split", "mnv2_int8"),
        ("b_224x320_int8", "standard", "mnv2_int8"),
        ("b_192_int8", "standard", "mnv2_int8"),
        ("b_192_int8", "standard", "shufflenet_int8"),
    ]
    for detector_nome, formato, classificador in combinacoes_b:
        chave = f"{detector_nome}+{classificador}"
        resultados["acuracia"][chave] = resumo(
            avaliar_b(caminhos[detector_nome], formato, classificador)
        )

    print("classificação de quadro inteiro (cenário MCU sem detector)", flush=True)
    for nome in CLASSIFICADORES:
        resultados["acuracia"][f"quadro_inteiro_{nome}"] = classificacao_quadro_inteiro(nome)

    print("conferindo equivalência da cabeça dividida", flush=True)
    resultados["equivalencia_cabeca_dividida"]["a_640"] = equivalencia_dividida(
        caminhos["a_640"], caminhos["a_640_split"], nomes_a
    )
    resultados["equivalencia_cabeca_dividida"]["b_416"] = equivalencia_dividida(
        caminhos["b_416"], caminhos["b_416_split"], {0: "animal"}
    )

    print("medindo latência", flush=True)
    for nome, caminho in sorted(caminhos.items()):
        resultados["latencia_x86"][nome] = {
            "1_thread": latencia(caminho, 1),
            "4_threads": latencia(caminho, 4),
        }
    for nome, caminho in CLASSIFICADORES.items():
        resultados["latencia_x86"][f"classificador_{nome}"] = {
            "1_thread": latencia(caminho, 1),
            "4_threads": latencia(caminho, 4),
        }

    print("contando GFLOPs", flush=True)
    for nome, modelo, entrada in [
        ("a_640", "a", (640, 640)), ("a_512", "a", (512, 512)), ("a_320", "a", (320, 320)),
        ("a_224x320", "a", (224, 320)), ("a_256", "a", (256, 256)), ("a_224", "a", (224, 224)),
        ("a_192", "a", (192, 192)), ("b_416", "b", (416, 416)),
        ("b_320", "b", (320, 320)), ("b_224x320", "b", (224, 320)), ("b_192", "b", (192, 192)),
    ]:
        resultados["gflops"][nome] = gflops_yolo(modelo, entrada)
    resultados["gflops"]["classificador_mnv2_128"] = gflops_classificador(
        ROOT / "models" / "modelo_b" / "species_classifier.pt"
    )
    resultados["gflops"]["classificador_shufflenet_128"] = gflops_classificador(
        ROOT / "models" / "modelo_b_shufflenet_v2_x0_5" / "species_classifier.pt"
    )

    destino = SAIDA / "proxies_x86.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"resultados em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
