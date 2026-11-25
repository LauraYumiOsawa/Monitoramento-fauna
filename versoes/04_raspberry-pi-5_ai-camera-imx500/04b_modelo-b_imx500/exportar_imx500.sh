#!/usr/bin/env bash
# Exporta o YOLOv8n da versão 04b para o IMX500 (entrada 416 px).
#
# Rode em Linux com Java 17+ (o Ultralytics instala o openjdk se faltar) e o extra de
# treino. O export só aceita YOLOv8n e YOLO11n, força INT8 (MCT) e embute o NMS.
# Saída: <nome>_imx_model/packerOut.zip — o empacotamento final (network.rpk) roda no Pi:
#   ./empacotar_rpk.sh
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
PESOS="${PESOS:-$RAIZ/runs/modelo_b/presence_yolov8n_416/weights/best.pt}"
PASTA="b_416_imx500"
if [[ "modelo-b" == "modelo-a" ]]; then
  DADOS="$RAIZ/data/animals"; NOMES='{0: anta, 1: tamandua_bandeira, 2: lobo_guara}'
else
  DADOS="$RAIZ/data/animals_agnostic"; NOMES='{0: animal}'
fi
TRABALHO="$(mktemp -d)"
trap 'rm -rf "$TRABALHO"' EXIT
cp "$PESOS" "$TRABALHO/modelo.pt"
cat > "$TRABALHO/dados.yaml" <<EOF
path: $DADOS
train: images/train
val: images/val
names: $NOMES
EOF

"$PYTHON" - "$TRABALHO/modelo.pt" "$TRABALHO/dados.yaml" <<'EOF'
import sys
from ultralytics import YOLO

pesos, dados = sys.argv[1], sys.argv[2]
YOLO(pesos).export(format="imx", imgsz=416, data=dados)
EOF

mkdir -p "$AQUI/modelos/$PASTA"
cp -r "$TRABALHO/modelo_imx_model/." "$AQUI/modelos/$PASTA/"
echo "Pacote IMX500: $AQUI/modelos/$PASTA/packerOut.zip (falta ./empacotar_rpk.sh no Pi)"
