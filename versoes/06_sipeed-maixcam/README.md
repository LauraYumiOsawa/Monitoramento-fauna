# Sipeed MaixCAM

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

SoC SG2002 com NPU INT8; ecossistemas MaixPy e MaixCDK [17].

**Papel na pesquisa:** Câmera com ambiente próprio; exige fluxo de conversão próprio (cvimodel) [17].

**Câmera:** Câmera do kit (GC4653); trocar por módulo sem filtro IR para uso noturno.

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [06a](06a_yolov8n-multiclasse_maixpy/README.md) | MaixPy executa YOLOv8 convertido por TPU-MLIR; o fluxo de conversão é verificado antes de assumir execução [17]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [06b](06b_modelo-b_maixpy/README.md) | Os dois estágios cabem na NPU; INT8 do MobileNetV2 medido sem perda no projeto. |
| MegaDetector + SpeciesNet | não faz sentido |  | Memória e NPU incompatíveis com MegaDetector (1280 px) e SpeciesNet. |
