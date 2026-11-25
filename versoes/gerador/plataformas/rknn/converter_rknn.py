"""Converte os ONNX da versão @@ID@@ para RKNN INT8 (@@RKNN_PLATAFORMA@@) com o RKNN-Toolkit2.

Rode em Linux x86_64 (ou aarch64) com `pip install rknn-toolkit2` (2.5.x; Python
3.6–3.12) — em ambiente separado do treino, porque o toolkit fixa versões de onnx/torch.
A versão do toolkit deve casar com a do `librknnrt.so` da placa
(`strings /usr/lib/librknnrt.so | grep -i "librknnrt version"`).

Detector: ONNX com a cabeça dividida (6 saídas box/cls), mean 0 e std 255 embutidos —
a mesma normalização do rknn_model_zoo. Quantizar a cabeça inteira do YOLO quebra os
escores (achado do projeto); dividida, o INT8 em todos os operadores foi medido sem colapso
(proxy x86 em versoes/ranking/medicoes/proxies_x86.json).
Classificador (Modelo B): ImageNet embutido (mean 0,485·255..., std 0,229·255...).

    python converter_rknn.py                 # usa data/animals/images/val como calibração
    python converter_rknn.py --analise       # também grava snapshot/error_analysis.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
PLATAFORMA = "@@RKNN_PLATAFORMA@@"
DETECTOR_ONNX = AQUI / "@@RKNN_ONNX@@"
CLASSIFICADOR_ONNX = "@@RKNN_CLASSIFICADOR_ONNX@@"
IMAGENET_MEDIA = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
IMAGENET_DESVIO = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]


def lista_calibracao(pasta: Path, destino: Path, limite: int, semente: int = 42) -> Path:
    """dataset.txt: um caminho de imagem por linha (20–100 imagens no algoritmo normal)."""
    imagens = sorted(p for p in pasta.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imagens:
        raise SystemExit(f"Sem imagens de calibração em {pasta}")
    random.Random(semente).shuffle(imagens)
    destino.write_text("\n".join(str(p) for p in imagens[:limite]) + "\n", encoding="utf-8")
    return destino


def converter(onnx: Path, saida: Path, media, desvio, dataset: Path, analise: bool,
              entrada_fixa: tuple[str, list[int]] | None = None) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    if rknn.config(mean_values=media, std_values=desvio, target_platform=PLATAFORMA) != 0:
        raise SystemExit("rknn.config falhou")
    # O classificador tem lote dinâmico no ONNX; o RKNN precisa da forma fixa.
    argumentos = (
        {"inputs": [entrada_fixa[0]], "input_size_list": [entrada_fixa[1]]} if entrada_fixa else {}
    )
    if rknn.load_onnx(model=str(onnx), **argumentos) != 0:
        raise SystemExit(f"load_onnx falhou: {onnx}")
    if rknn.build(do_quantization=True, dataset=str(dataset)) != 0:
        raise SystemExit("build (INT8) falhou")
    if analise:
        primeira = dataset.read_text(encoding="utf-8").splitlines()[0]
        pasta = str(saida.with_suffix("")) + "_snapshot"
        rknn.accuracy_analysis(inputs=[primeira], output_dir=pasta)
    if rknn.export_rknn(str(saida)) != 0:
        raise SystemExit("export_rknn falhou")
    rknn.release()
    print(f"RKNN: {saida}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibracao", default=str(RAIZ / "data" / "animals" / "images" / "val"))
    parser.add_argument("--calibracao-recortes", default=str(RAIZ / "data" / "crops" / "val"))
    parser.add_argument("--imagens", type=int, default=60)
    parser.add_argument("--analise", action="store_true")
    args = parser.parse_args()
    modelos = AQUI / "modelos"
    modelos.mkdir(exist_ok=True)

    lista = lista_calibracao(Path(args.calibracao), modelos / "dataset_detector.txt", args.imagens)
    converter(DETECTOR_ONNX, modelos / f"{DETECTOR_ONNX.stem}_{PLATAFORMA}.rknn",
              [[0, 0, 0]], [[255, 255, 255]], lista, args.analise)
    if CLASSIFICADOR_ONNX:
        onnx = AQUI / CLASSIFICADOR_ONNX
        lista = lista_calibracao(Path(args.calibracao_recortes), modelos / "dataset_cls.txt",
                                 args.imagens)
        converter(onnx, modelos / f"{onnx.stem}_{PLATAFORMA}.rknn", IMAGENET_MEDIA,
                  IMAGENET_DESVIO, lista, args.analise, ("crops", [1, 3, 128, 128]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
