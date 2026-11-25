# Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Acelerador PCIe Hailo-8L, 13 TOPS [14]; hospedeiro Pi 5 entra no custo e na energia.

**Papel na pesquisa:** IA em acelerador PCIe (perfil 'Visão acelerada em Linux').

**Câmera:** Camera Module 3 NoIR [26].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [02a](02a_yolov8n-multiclasse_hailort/README.md) | YOLOv8n é arquitetura suportada pelo fluxo ONNX → otimização → HEF [39]; a NPU tira a convolução da CPU (QP3) [14]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [02b](02b_modelo-b_hailort/README.md) | Detector de presença no Hailo; o classificador roda só nos recortes (~0,19 por quadro medido no projeto), barato na CPU do Pi 5. |
| MegaDetector + SpeciesNet | condicional — não gerado |  | Sem HEF documentado para MegaDetector/SpeciesNet; compilar YOLOv5x6 a 1280 px e EfficientNetV2 no DFC exigiria validação própria. O hospedeiro Pi 5 pode rodar a versão 01c em lote. |
