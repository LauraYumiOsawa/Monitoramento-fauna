# Raspberry Pi 5 + AI Camera (IMX500)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Sensor Sony IMX500 de 12,3 MP com processamento neural no módulo [16].

**Papel na pesquisa:** IA no sensor versus IA no processador; não se presume comportamento NoIR [16].

**Câmera:** A própria AI Camera (foco manual, com filtro IR: noturno depende de luz visível).

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [04a](04a_yolov8n-multiclasse_imx500/README.md) | O Ultralytics exporta YOLOv8n para o IMX500; a inferência sai do processador hospedeiro [16]. Compara IA no sensor com IA no processador. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [04b](04b_modelo-b_imx500/README.md) | Divisão natural: o sensor localiza o animal; a CPU classifica o recorte em alta resolução do mesmo quadro. |
| MegaDetector + SpeciesNet | não faz sentido |  | MegaDetector e SpeciesNet excedem em muito a memória da rede no sensor do IMX500. |
