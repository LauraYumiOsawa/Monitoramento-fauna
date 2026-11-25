# Orange Pi 5

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

RK3588S, 4× Cortex-A76 + 4× Cortex-A55; NPU anunciada de 6 TOPS [21].

**Papel na pesquisa:** Contraponto com NPU no SoC; fluxo RKNN próprio de conversão e execução [22].

**Câmera:** Câmera USB/UVC ou módulo MIPI (OV13850/OV13855) via GStreamer.

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [08a](08a_yolov8n-multiclasse_rknn/README.md) | Contraponto com NPU no SoC; o fluxo RKNN tem conversão e execução próprias [21; 22]. INT8 exige cortar a cabeça (achado do projeto). |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [08b](08b_modelo-b_rknn/README.md) | Os dois estágios na NPU; CPU livre para rastreamento e registro. |
| MegaDetector + SpeciesNet | faz sentido — gerado | [08c](08c_megadetector-speciesnet_lote/README.md) | 8/16 GB de RAM permitem a classificação de referência em lote, como na 01c. |
| ONNX + ONNX Runtime | faz sentido — gerado | 08a | ONNX Runtime CPU fica como alternativa sem conversão nas versões 08a/08b (comparação CPU × NPU da QP3). |
