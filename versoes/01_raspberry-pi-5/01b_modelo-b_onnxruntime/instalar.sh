#!/usr/bin/env bash
# Instala as dependências da versão 01b (Raspberry Pi 5 · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · ONNX Runtime (CPU, FP32)).
# Rode NO DISPOSITIVO, dentro da pasta da versão (ou do pacote gerado por empacotar.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-numpy python3-opencv python3-yaml python3-picamera2 python3-gpiozero python3-lgpio

# --system-site-packages: Picamera2, libcamera e os bindings de GPIO vêm do sistema
# (apt), não do PyPI; o ambiente virtual precisa enxergá-los.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install onnxruntime

./executar.sh info
