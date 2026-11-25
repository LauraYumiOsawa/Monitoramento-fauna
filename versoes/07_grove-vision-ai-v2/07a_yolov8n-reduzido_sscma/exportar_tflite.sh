#!/usr/bin/env bash
# Exporta o YOLOv8n da versão 07a para TFLite INT8 e compila para o Ethos-U55 (Vela).
#
# Caminho da Seeed (wiki ma_deploy_yolov8 / notebook YOLO26 de 2026): o export
# `saved_model` com INT8 gera `*_full_integer_quant.tflite`; o Vela produz `*_vela.tflite`,
# que se envia ao módulo pelo SenseCraft AI (Models → My Models → Add Model).
# Requer TensorFlow (instalado pelo Ultralytics no export) e `pip install ethos-u-vela`.
# Entradas documentadas para o módulo: 192, 224 e 256; aqui 256.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
PESOS="${PESOS:-$RAIZ/runs/train/baseline_yolov8n-5/weights/best.pt}"
TRABALHO="$(mktemp -d)"
trap 'rm -rf "$TRABALHO"' EXIT
cp "$PESOS" "$TRABALHO/grove.pt"
cat > "$TRABALHO/dados.yaml" <<EOF
path: $RAIZ/data/animals
train: images/train
val: images/val
names: {0: anta, 1: tamandua_bandeira, 2: lobo_guara}
EOF

"$PYTHON" - "$TRABALHO/grove.pt" "$TRABALHO/dados.yaml" <<'EOF'
import sys
from ultralytics import YOLO

pesos, dados = sys.argv[1], sys.argv[2]
# `quantize=8` substitui `int8=True` desde o Ultralytics 8.4.80.
YOLO(pesos).export(format="saved_model", imgsz=256, batch=1, quantize=8, data=dados)
EOF

TFLITE="$(find "$TRABALHO" -name '*_full_integer_quant.tflite' | head -n 1)"
[[ -n "$TFLITE" ]] || { echo "Export INT8 não gerou *_full_integer_quant.tflite"; exit 1; }
mkdir -p "$AQUI/modelos"
vela --accelerator-config ethos-u55-64 --config "$AQUI/vela_config.ini" \
     --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent \
     --verbose-performance --output-dir "$AQUI/modelos" "$TFLITE"
echo "Modelo para o SenseCraft AI: $(ls "$AQUI"/modelos/*_vela.tflite)"
echo "A Himax recomenda 'Total SRAM used' abaixo de ~1 MB no relatório do Vela acima."
