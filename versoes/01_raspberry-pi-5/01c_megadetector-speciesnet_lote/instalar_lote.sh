#!/usr/bin/env bash
# Instala MegaDetector + SpeciesNet para a versão 01c (Raspberry Pi 5 de 8 GB, PyTorch em cpu).
# O SpeciesNet traz o MegaDetector v5a como detector e baixa os pesos do Kaggle na
# primeira execução (~500 MB; classificador v4.0.3a de 224 MB).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "cpu" == "cuda" ]]; then
  # Jetson (JetPack 6, Python 3.10): torch/torchvision com CUDA das wheels do jetson-ai-lab,
  # antes do speciesnet, para o pip não trocar por uma versão só-CPU.
  .venv/bin/python -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
else
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install "speciesnet==5.0.6"
.venv/bin/python -m speciesnet.scripts.gpu_test || true   # informa se há GPU disponível
