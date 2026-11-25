#!/usr/bin/env bash
# Gera modelos/b_416_imx500/network.rpk a partir do packerOut.zip. Roda NO Raspberry Pi
# (o imx500-package só existe lá): sudo apt install imx500-all imx500-tools
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASTA="$AQUI/modelos/b_416_imx500"
[[ -f "$PASTA/packerOut.zip" ]] || { echo "Falta $PASTA/packerOut.zip (rode exportar_imx500.sh)"; exit 1; }
imx500-package -i "$PASTA/packerOut.zip" -o "$PASTA"
ls -l "$PASTA/network.rpk"
echo "Primeira execução: o firmware da rede pode levar minutos para carregar (fica em cache)."
