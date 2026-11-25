# Luckfox Pico (RV1106)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Família RV1103/RV1106 com CPU, ISP e NPU; variantes não equivalentes [24].

**Papel na pesquisa:** Linux + NPU em baixo consumo; exige caracterizar o modelo exato (aqui RV1106G3) [24].

**Câmera:** SC3336 (MIPI) da Luckfox.

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [10a](10a_yolov8n-320-int8_rknn-rv1106/README.md) | A NPU do RV1106 executa INT8; o modelo exato (RV1106G3, 256 MB) é caracterizado antes de comparar [24]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | Exigiria segundo contexto RKNN e portar rastreamento/votação para C; não gerado nesta revisão. |
| MegaDetector + SpeciesNet | não faz sentido |  | Memória insuficiente. |
