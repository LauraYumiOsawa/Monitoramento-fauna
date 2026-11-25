#!/usr/bin/env bash
# Classificação em lote (arquitetura C) da versão @@ID@@.
#   ./executar_lote.sh /media/cartao/DCIM            # MegaDetector + SpeciesNet (Brasil)
#   ./executar_lote.sh /media/cartao/DCIM --detector  # só o MegaDetector (filtro de vazias)
# Depois: registros em Camtrap DP com ./importar_resultados.py (mesmo diretório).
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASTA="${1:?informe a pasta com as fotos}"
MODO="${2:-}"
SAIDA="$AQUI/dados/speciesnet_$(date +%Y%m%dT%H%M%S).json"
mkdir -p "$AQUI/dados"
EXTRA=()
[[ "$MODO" == "--detector" ]] && EXTRA+=(--detector_only)
# --country BRA ativa o filtro geográfico: só táxons com ocorrência no Brasil.
"$AQUI/.venv/bin/python" -m speciesnet.scripts.run_model \
  --folders "$PASTA" \
  --predictions_json "$SAIDA" \
  --country BRA \
  --bypass_prompts \
  "${EXTRA[@]}"
echo "Predições: $SAIDA"
echo "Camtrap DP: $AQUI/.venv/bin/python $AQUI/importar_resultados.py $SAIDA"
