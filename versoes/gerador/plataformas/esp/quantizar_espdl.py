"""Quantiza o MobileNetV2 128×128 da versão @@ID@@ para ESP-DL (ESP32-S3, INT8) com o ESP-PPQ.

Ambiente separado da máquina de treino:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install esp-ppq

A calibração usa quadros INTEIROS (o firmware classifica a foto toda, não um recorte):
letterbox com preenchimento 114 até 128 px e normalização ImageNet, a mesma do
`ImagePreprocessor` do firmware. Saída: modelos/classificador.espdl (+ .info e .json).
O MobileNetV2 é o modelo do tutorial oficial do ESP-DL; o ShuffleNetV2 do projeto usa
Reshape 5-D no channel shuffle, sem suporte documentado.

    python quantizar_espdl.py [--calibracao data/animals/images/val]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def quadro_para_tensor(caminho: Path, lado: int = 128) -> np.ndarray:
    import cv2

    imagem = cv2.imread(str(caminho))
    altura, largura = imagem.shape[:2]
    escala = lado / max(altura, largura)
    novo = cv2.resize(imagem, (max(1, round(largura * escala)), max(1, round(altura * escala))),
                      interpolation=cv2.INTER_AREA)
    tela = np.full((lado, lado, 3), 114, dtype=np.uint8)
    topo, esquerda = (lado - novo.shape[0]) // 2, (lado - novo.shape[1]) // 2
    tela[topo:topo + novo.shape[0], esquerda:esquerda + novo.shape[1]] = novo
    rgb = tela[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEDIA) / DESVIO).transpose(2, 0, 1).astype(np.float32)


def main() -> int:
    import torch
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", default=str(AQUI / "modelos" / "species_classifier.onnx"))
    parser.add_argument("--calibracao", default=str(RAIZ / "data" / "animals" / "images" / "val"))
    parser.add_argument("--passos", type=int, default=32)
    args = parser.parse_args()

    arquivos = sorted(p for p in Path(args.calibracao).rglob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tensores = [torch.from_numpy(quadro_para_tensor(caminho)) for caminho in arquivos]
    carregador = DataLoader(tensores, batch_size=1, shuffle=False)
    destino = AQUI / "modelos" / "classificador.espdl"
    destino.parent.mkdir(exist_ok=True)
    configuracao = QuantizationSettingFactory.espdl_setting()
    espdl_quantize_onnx(
        onnx_import_file=args.onnx,
        espdl_export_file=str(destino),
        calib_dataloader=carregador,
        calib_steps=min(args.passos, len(tensores)),
        input_shape=[1, 3, 128, 128],
        target="esp32s3",
        num_of_bits=8,
        setting=configuracao,
        collate_fn=lambda lote: lote.to("cpu"),
        device="cpu",
        export_test_values=True,  # permite model->test() no dispositivo
    )
    print(f"ESP-DL: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
