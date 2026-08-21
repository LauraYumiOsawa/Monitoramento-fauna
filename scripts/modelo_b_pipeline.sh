#!/usr/bin/env bash
# Executa o pipeline completo do Modelo B, do dataset ao comparativo com o Modelo A.
# Cada etapa grava um JSON em outputs/modelo_b/ para o relatório do TCC.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

DATASET="${DATASET:-config/dataset.yaml}"
DETECTOR_CONFIG="${DETECTOR_CONFIG:-config/modelo_b/detector.yaml}"
CLASSIFIER_CONFIG="${CLASSIFIER_CONFIG:-config/modelo_b/classifier.yaml}"
RUNTIME="${RUNTIME:-config/modelo_b/runtime.yaml}"
MODELS_DIR="${MODELS_DIR:-models/modelo_b}"
OUT="${OUT:-outputs/modelo_b}"
MODEL_A_ONNX="${MODEL_A_ONNX:-models/animal_detector.onnx}"

mkdir -p "$OUT"

echo "==> 1/8 Derivando dataset de presença e recortes de espécie"
animal-monitor-b prepare-data --data "$DATASET" --json-output "$OUT/data_prep.json"

echo "==> 2/8 Treinando o detector de presença (1 classe, 416 px)"
animal-monitor-b train-detector --config "$DETECTOR_CONFIG" | tee "$OUT/train_detector.json"

DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-$(python3 -c "
import json,sys
print(json.load(open('$OUT/train_detector.json'))['best_weights'])
")}"
echo "    pesos: $DETECTOR_WEIGHTS"

echo "==> 3/8 Treinando o classificador de espécie"
animal-monitor-b train-classifier --config "$CLASSIFIER_CONFIG"

echo "==> 4/8 Exportando os dois modelos para ONNX"
animal-monitor-b export \
  --detector-weights "$DETECTOR_WEIGHTS" \
  --classifier-checkpoint "$MODELS_DIR/species_classifier.pt" \
  --output-dir "$MODELS_DIR" \
  --imgsz 416

echo "==> 5/8 Quantizando para INT8 (calibração com dados reais de validação)"
# --op-types Conv e obrigatorio no detector: quantizar a cabeca do YOLO faz os escores
# sairem de [0,1] e o detector para de funcionar (medido: conf_max 1.673).
animal-monitor-b quantize \
  --model "$MODELS_DIR/presence_detector.onnx" \
  --kind detector \
  --calibration-dir data/animals_agnostic/images/val \
  --samples 36 \
  --op-types Conv
animal-monitor-b quantize \
  --model "$MODELS_DIR/species_classifier.onnx" \
  --kind classifier \
  --calibration-dir data/crops/val \
  --samples 98

echo "==> 6/8 Avaliando: detector isolado, pipeline completo e Modelo A (mesma métrica)"
animal-monitor-b evaluate-detector \
  --weights "$DETECTOR_WEIGHTS" \
  --data data/animals_agnostic/dataset_agnostic.yaml \
  --split test \
  --imgsz 416 \
  --output "$OUT/evaluation_detector.json"

animal-monitor-b evaluate \
  --detector "$MODELS_DIR/presence_detector.onnx" \
  --classifier "$MODELS_DIR/species_classifier.onnx" \
  --data "$DATASET" --runtime "$RUNTIME" --split test \
  --label modelo_b_fp32 --output "$OUT/evaluation_e2e.json"

animal-monitor-b evaluate \
  --detector "$MODELS_DIR/presence_detector_int8.onnx" \
  --classifier "$MODELS_DIR/species_classifier_int8.onnx" \
  --data "$DATASET" --runtime "$RUNTIME" --split test \
  --label modelo_b_int8 --output "$OUT/evaluation_e2e_int8.json"

if [[ -f "$MODEL_A_ONNX" ]]; then
  animal-monitor-b evaluate-baseline \
    --model "$MODEL_A_ONNX" \
    --data "$DATASET" --runtime config/runtime.yaml --split test \
    --label modelo_a_fp32 --output "$OUT/evaluation_baseline.json"
  animal-monitor-b compare \
    --reports "$OUT/evaluation_baseline.json" "$OUT/evaluation_e2e.json" \
      "$OUT/evaluation_e2e_int8.json" \
    --output "$OUT/comparison.json"
fi

echo "==> 7/8 Inferência nos vídeos de amostra"
for video in samples/*.mp4; do
  name="$(basename "$video" .mp4)"
  animal-monitor-b infer \
    --detector "$MODELS_DIR/presence_detector.onnx" \
    --classifier "$MODELS_DIR/species_classifier.onnx" \
    --source "$video" --runtime "$RUNTIME" --camera-id "cam_teste" \
    --output-video "$OUT/${name}_anotado.mp4" > "$OUT/infer_${name}.json"
  echo "    $name -> $OUT/infer_${name}.json"
done

echo "==> 8/8 Benchmark por estágio"
for video in samples/*.mp4; do
  name="$(basename "$video" .mp4)"
  animal-monitor-b benchmark \
    --detector "$MODELS_DIR/presence_detector.onnx" \
    --classifier "$MODELS_DIR/species_classifier.onnx" \
    --source "$video" --runtime "$RUNTIME" --frames 200 \
    --output "$OUT/benchmark_${name}.json" > /dev/null
  echo "    $name -> $OUT/benchmark_${name}.json"
done

echo "Concluído. Artefatos em $OUT e $MODELS_DIR."
