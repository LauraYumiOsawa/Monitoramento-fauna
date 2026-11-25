# NVIDIA Jetson Orin Nano Super

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Até 67 TOPS e modos de potência de 7 a 25 W [23].

**Papel na pesquisa:** Teto de desempenho/estação base; modos de potência não são autonomia de armadilha [23].

**Câmera:** IMX219 NoIR (CSI) via nvarguscamerasrc.

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [13a](13a_yolov8n-multiclasse_tensorrt/README.md) | Teto de desempenho do mesmo modelo [23]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [13b](13b_modelo-b_tensorrt/README.md) | Teto de desempenho do pipeline em dois estágios. |
| MegaDetector + SpeciesNet | faz sentido — gerado | [13c](13c_megadetector-speciesnet_gpu/README.md) | Único hardware do estudo em que a referência roda perto do tempo real; papel de estação base, não de armadilha a bateria [23]. |
