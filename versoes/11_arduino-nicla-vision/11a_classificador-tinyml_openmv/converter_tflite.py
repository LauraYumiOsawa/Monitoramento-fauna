"""Converte o ShuffleNetV2-x0.5 (128×128) da versão 11a para TFLite INT8 (OpenMV/TFLM).

Ambiente separado: `pip install onnx2tf tensorflow onnx onnxsim` (o onnx2tf acompanha
versões específicas do TensorFlow; veja o README dele). Calibração com quadros inteiros,
letterbox 114 e normalização ImageNet — o mesmo que o modelo verá na placa.

Por que ShuffleNetV2 e não MobileNetV2 aqui: a Nicla tem ~476 KB de memória compartilhada
entre quadro e modelo; só a expansão do segundo bloco do MobileNetV2 a 128 px passa de
390 KB, enquanto o maior tensor do ShuffleNetV2-x0.5 fica perto de 100 KB.
Risco conhecido: o channel shuffle vira Reshape/Transpose de 5 dimensões; confira se o
TFLite resultante roda no OpenMV antes de ir a campo. Alternativa documentada: treinar no
Edge Impulse (MobileNetV2 0,35 a 96 px ou FOMO) com os mesmos recortes.

    python converter_tflite.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


def lote_calibracao(pasta: Path, destino: Path, lado: int = 128, limite: int = 64) -> Path:
    import cv2

    arquivos = sorted(p for p in pasta.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    quadros = []
    for caminho in arquivos[:limite]:
        imagem = cv2.imread(str(caminho))
        altura, largura = imagem.shape[:2]
        escala = lado / max(altura, largura)
        novo = cv2.resize(imagem, (round(largura * escala), round(altura * escala)))
        tela = np.full((lado, lado, 3), 114, dtype=np.uint8)
        topo, esquerda = (lado - novo.shape[0]) // 2, (lado - novo.shape[1]) // 2
        tela[topo:topo + novo.shape[0], esquerda:esquerda + novo.shape[1]] = novo
        quadros.append(tela[:, :, ::-1].astype(np.float32) / 255.0)  # NHWC, sem normalizar
    np.save(destino, np.stack(quadros))
    return destino


def main() -> int:
    modelos = AQUI / "modelos"
    onnx = modelos / "species_classifier_shufflenet.onnx"
    calibracao = lote_calibracao(RAIZ / "data" / "animals" / "images" / "val",
                                 modelos / "calibracao_nhwc.npy")
    saida = modelos / "tflite"
    # -oiqt: gera também o TFLite totalmente inteiro; -cind: dados de calibração do input
    # "crops" com média e desvio (o onnx2tf normaliza antes de calibrar).
    comando = [
        "onnx2tf", "-i", str(onnx), "-o", str(saida), "-b", "1", "-oiqt",
        "-cind", "crops", str(calibracao), str([[[MEDIA]]]), str([[[DESVIO]]]),
    ]
    print(" ".join(comando))
    subprocess.run(comando, check=True)
    inteiro = sorted(saida.glob("*_full_integer_quant.tflite"))
    if not inteiro:
        print("onnx2tf não gerou *_full_integer_quant.tflite", file=sys.stderr)
        return 1
    destino = modelos / "classificador.tflite"
    destino.write_bytes(inteiro[0].read_bytes())
    (modelos / "classificador.txt").write_text(
        "anta\ntamandua_bandeira\nlobo_guara\ndesconhecido\n", encoding="utf-8"
    )
    print(f"TFLite INT8: {destino} — copie para /rom pelo OpenMV IDE (Tools → ROM File System)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
