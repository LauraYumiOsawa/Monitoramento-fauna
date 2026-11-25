# Raspberry Pi Zero 2 W

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

4× Cortex-A53 a 1 GHz, 512 MB de RAM [13].

**Papel na pesquisa:** Contraponto de capacidade e preço de placa; memória limita modelos [13].

**Câmera:** Camera Module 3 NoIR com cabo para Zero [26].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [05a](05a_yolov8n-512-int8_onnxruntime/README.md) | Cabe em 512 MB com Raspberry Pi OS Lite; segundos por quadro são aceitáveis para fotos acionadas por PIR. Entrada 512 é a alternativa leve da pesquisa [P1; 13]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [05b](05b_modelo-b-int8_onnxruntime/README.md) | Modelo B INT8 (5,7 MB) foi a configuração recomendada pelo projeto para hardware limitado; MobileNetV2 não perde acurácia em INT8. |
| MegaDetector + SpeciesNet | não faz sentido |  | 512 MB de RAM não comportam os pesos do MegaDetector e do SpeciesNet com o sistema [13]. |
