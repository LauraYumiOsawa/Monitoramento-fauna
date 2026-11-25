#!/usr/bin/env bash
# Converte os ONNX da versão 06b para cvimodel INT8 (MaixCAM, SG2002 = processor cv181x).
# Fluxo e argumentos da documentação da Sipeed (maixpy/doc/en/ai_model_converter/maixcam.html).
#
# Rode em x86_64 com Docker. O script entra no contêiner oficial do TPU-MLIR:
#   docker pull sophgo/tpuc_dev:latest
# Os nomes de saída do YOLOv8 foram conferidos nos ONNX deste projeto
# (/model.22/dfl/conv/Conv_output_0 e /model.22/Sigmoid_output_0): o corte antes da
# decodificação mantém caixas e escores em tensores separados no INT8.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
IMAGEM="${TPU_MLIR_IMAGEM:-sophgo/tpuc_dev:latest}"

docker run --rm -v "$RAIZ":/projeto -w "/projeto/versoes/$(realpath --relative-to="$RAIZ/versoes" "$AQUI")" \
  "$IMAGEM" bash -euo pipefail -c '
pip install -q tpu_mlir[onnx]
mkdir -p trabalho modelos
cd trabalho
CALIB=/projeto/data/animals/images/val
TESTE=$(ls $CALIB/*.jpg | head -n 1)

model_transform.py \
  --model_name detector \
  --model_def ../modelos/b_224x320.onnx \
  --input_shapes [[1,3,224,320]] \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --channel_format nchw \
  --output_names "/model.22/dfl/conv/Conv_output_0,/model.22/Sigmoid_output_0" \
  --test_input "$TESTE" \
  --test_result detector_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --mlir detector.mlir
run_calibration.py detector.mlir --dataset "$CALIB" --input_num 36 -o detector_cali_table
model_deploy.py \
  --mlir detector.mlir \
  --quantize INT8 \
  --quant_input \
  --calibration_table detector_cali_table \
  --processor cv181x \
  --test_input detector_in_f32.npz \
  --test_reference detector_top_outputs.npz \
  --tolerance 0.9,0.6 \
  --model ../modelos/b_224x320_int8.cvimodel

if [ -n "modelos/species_classifier.onnx" ]; then
  RECORTES=/projeto/data/crops/val
  TESTE_C=$(find $RECORTES -name "*.jpg" | head -n 1)
  # ImageNet: mean = 255·(0,485; 0,456; 0,406); scale = 1/(255·desvio).
  model_transform.py \
    --model_name classificador \
    --model_def ../modelos/species_classifier.onnx \
    --input_shapes [[1,3,128,128]] \
    --mean "123.675,116.28,103.53" \
    --scale "0.0171248,0.0175070,0.0174292" \
    --pixel_format rgb \
    --channel_format nchw \
    --output_names "logits" \
    --test_input "$TESTE_C" \
    --test_result classificador_top_outputs.npz \
    --tolerance 0.99,0.99 \
    --mlir classificador.mlir
  run_calibration.py classificador.mlir --dataset "$RECORTES" --input_num 98 -o classificador_cali_table
  model_deploy.py \
    --mlir classificador.mlir \
    --quantize INT8 \
    --quant_input \
    --calibration_table classificador_cali_table \
    --processor cv181x \
    --test_input classificador_in_f32.npz \
    --test_reference classificador_top_outputs.npz \
    --tolerance 0.9,0.6 \
    --model ../modelos/species_classifier_int8.cvimodel
fi
'
echo "cvimodel(s) em $AQUI/modelos/ — copie a pasta modelos/ e app/ para /root/armadilha na MaixCAM."
