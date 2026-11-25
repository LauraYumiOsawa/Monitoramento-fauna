#!/usr/bin/env bash
# Compila a armadilha da versão @@ID@@ para o RV1106 (Luckfox Pico Max/Ultra, uclibc).
#   export LUCKFOX_SDK_PATH=/caminho/luckfox-pico          # SDK oficial da Luckfox
#   export RKNN_TOOLKIT2=/caminho/rknn-toolkit2            # repositório airockchip (rknpu2/)
#   export OPENCV_MOBILE=/caminho/opencv-mobile-*-luckfox-pico   # pacote pré-compilado
#   ./compilar.sh
# Na placa: RkLunch-stop.sh (libera a câmera do rkipc) e ./armadilha modelos/<modelo>.rknn
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LUCKFOX_SDK_PATH:?defina LUCKFOX_SDK_PATH}" "${RKNN_TOOLKIT2:?defina RKNN_TOOLKIT2}" \
  "${OPENCV_MOBILE:?defina OPENCV_MOBILE}"
PREFIXO="$LUCKFOX_SDK_PATH/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
cmake -S "$AQUI/app" -B "$AQUI/build" \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_C_COMPILER="$PREFIXO-gcc" -DCMAKE_CXX_COMPILER="$PREFIXO-g++" \
  -DOpenCV_DIR="$OPENCV_MOBILE/lib/cmake/opencv4" \
  -DRKNN_API_DIR="$RKNN_TOOLKIT2/rknpu2/runtime/Linux/librknn_api" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$AQUI/instalar"
cmake --build "$AQUI/build" -j
cmake --install "$AQUI/build"
mkdir -p "$AQUI/instalar/modelos"
cp "$AQUI"/modelos/*_rv1106.rknn "$AQUI/instalar/modelos/" 2>/dev/null || \
  echo "AVISO: rode converter_rknn.py antes (modelo .rknn ausente)"
echo "Copie $AQUI/instalar para /userdata/armadilha na placa (adb push ou scp)."
