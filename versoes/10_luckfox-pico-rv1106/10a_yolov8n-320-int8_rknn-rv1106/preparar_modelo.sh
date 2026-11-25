#!/usr/bin/env bash
# Prepara os artefatos ONNX da versão 10a (máquina de desenvolvimento, x86_64).
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
PYTHON="${PYTHON:-$RAIZ/.venv/bin/python}"
"$PYTHON" "$RAIZ/versoes/comum/ferramentas/preparar_modelos.py" "$AQUI"
