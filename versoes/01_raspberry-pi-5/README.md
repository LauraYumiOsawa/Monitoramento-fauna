# Raspberry Pi 5

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

BCM2712, 4× Cortex-A76 a 2,4 GHz, 4/8 GB; conectores de câmera (CSI) e PCIe [12].

**Papel na pesquisa:** Referência de sistema Linux flexível (arquitetura de referência da pesquisa).

**Câmera:** Camera Module 3 NoIR (IMX708, 12 MP, sem filtro IR) + iluminador IR externo [26].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [01a](01a_yolov8n-multiclasse_onnxruntime/README.md) | Configuração de referência da pesquisa: YOLOv8n em ONNX executado pelo ONNX Runtime no Pi 5 [P1; 5; 6]. O Ultralytics publica benchmark ONNX no Pi 5 [38]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [01b](01b_modelo-b_onnxruntime/README.md) | Mesma placa, pipeline em dois estágios: separa presença de espécie (QP4) e permite 'desconhecido'/'inconclusivo' [3; 7; 8]. Medido no projeto: maior F1 entre os modelos próprios. |
| MegaDetector + SpeciesNet | faz sentido — gerado | [01c](01c_megadetector-speciesnet_lote/README.md) | Modelos grandes demais para tempo real, mas viáveis em lote no Pi 5 de 8 GB: classificação posterior das fotos de um subsistema de captura (arquitetura C) [4; 7; 8; 30]. |
| TinyML (SDK do fabricante) | não faz sentido |  | Os SDKs TinyML visam microcontroladores; no Pi 5 o mesmo modelo roda sem as restrições de memória que justificam TinyML. |
