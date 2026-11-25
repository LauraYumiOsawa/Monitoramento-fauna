#!/usr/bin/env bash
# Executa a versão 02a com a CLI comum (animal_monitor_hw).
#   ./executar.sh                       # armadilha em campo (PIR, câmera, IR)
#   ./executar.sh trap --simular PASTA  # bancada: imagens/vídeo no lugar da câmera
#   ./executar.sh benchmark --fonte PASTA_OU_VIDEO
#   ./executar.sh export-camtrapdp
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$AQUI/lib" ]]; then
  BIBLIOTECAS="$AQUI/lib"                      # pacote gerado por empacotar.sh
else
  RAIZ="$(cd "$AQUI/../../.." && pwd)"         # dentro do repositório
  BIBLIOTECAS="$RAIZ/src:$RAIZ/versoes/comum"
fi
PYTHON="$AQUI/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
export PYTHONPATH="$BIBLIOTECAS${PYTHONPATH:+:$PYTHONPATH}"

COMANDO="${1:-trap}"
shift || true
exec "$PYTHON" -m animal_monitor_hw "$COMANDO" --versao "$AQUI" "$@"
