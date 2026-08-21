# Modelos

Depois do treinamento, execute:

```bash
animal-monitor export \
  --weights runs/train/baseline_yolov8n/weights/best.pt \
  --data config/dataset.yaml \
  --output-dir models
```

O diretório receberá `animal_detector.onnx` e `model_metadata.json`.
