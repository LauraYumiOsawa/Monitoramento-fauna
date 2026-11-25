#!/usr/bin/env bash
# Compila o HEF da versão 12a para hailo10h (entrada 640 px, 3 classe(s)).
#
# Rode em Linux x86_64 (o export do Ultralytics recusa ARM64), com:
#   - Hailo-8/8L : Dataflow Compiler 3.x (Model Zoo v2.x); no Pi: HailoRT 4.x (`hailo-all`)
#   - Hailo-10H  : Dataflow Compiler 5.x (Model Zoo v5.x); no Pi: HailoRT 5.x (`hailo-h10-all`)
#   HEFs das duas gerações não são intercambiáveis.
#   pip install ultralytics /caminho/hailo_dataflow_compiler-*.whl   # wheel do Developer Zone
# GPU NVIDIA é recomendada: sem ela o otimizador cai para o nível 0 (pior quantização).
#
# Calibração: as imagens de validação do projeto (poucas). A Hailo recomenda >= 1024
# imagens do domínio; com o dataset atual, confira a perda com `./executar.sh benchmark`.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODELO="modelo-a"
if [[ "$MODELO" == "modelo-a" ]]; then
  PESOS="${PESOS:-$RAIZ/runs/train/baseline_yolov8n-5/weights/best.pt}"
  DADOS="$RAIZ/data/animals"; NOMES='{0: anta, 1: tamandua_bandeira, 2: lobo_guara}'
else
  PESOS="${PESOS:-$RAIZ/runs/modelo_b/presence_yolov8n_416/weights/best.pt}"
  DADOS="$RAIZ/data/animals_agnostic"; NOMES='{0: animal}'
fi
NOME="$(basename "modelos/a_640.onnx" .onnx)"
TRABALHO="$(mktemp -d)"
trap 'rm -rf "$TRABALHO"' EXIT
cp "$PESOS" "$TRABALHO/$NOME.pt"   # o Ultralytics grava a saída ao lado dos pesos
cat > "$TRABALHO/dados.yaml" <<EOF
path: $DADOS
train: images/train
val: images/val
names: $NOMES
EOF

"$PYTHON" - "$TRABALHO/$NOME.pt" "$TRABALHO/dados.yaml" <<'EOF'
import sys
from ultralytics import YOLO

pesos, dados = sys.argv[1], sys.argv[2]
# INT8 é o único modo do Hailo; NMS do YOLOv8 vai para o HailoRT com estes limiares
# (o runtime da versão pode sobrescrever com set_nms_score/iou_threshold).
YOLO(pesos).export(format="hailo", name="hailo10h", imgsz=640, data=dados,
                   conf=0.25, iou=0.45)
EOF

mkdir -p "$AQUI/modelos"
cp "$TRABALHO/${NOME}_hailo_model/$NOME.hef" "$AQUI/modelos/${NOME}_hailo10h.hef"
cp "$TRABALHO/${NOME}_hailo_model/metadata.yaml" "$AQUI/modelos/${NOME}_hailo10h.metadata.yaml" 2>/dev/null || true
echo "HEF: $AQUI/modelos/${NOME}_hailo10h.hef"

# Alternativa sem o Ultralytics (Model Zoo), a partir do ONNX desta pasta:
#   hailomz compile yolov8n --ckpt "$AQUI/modelos/a_640.onnx" --hw-arch hailo10h \
#     --calib-path "$DADOS/images/val" --classes 3
# Para entrada diferente de 640, copie cfg/networks/yolov8n.yaml, ajuste input_shape e
# image_dims do JSON de NMS e passe --yaml/--model-script (não documentado pela Hailo).
