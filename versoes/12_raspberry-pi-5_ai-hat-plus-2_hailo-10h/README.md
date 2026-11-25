# Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Hailo-10H, 40 TOPS em INT4 e 8 GB no módulo; visão comparável ao HAT+ de 26 TOPS [15].

**Papel na pesquisa:** Sem prioridade especial para detecção de fauna; 40 TOPS INT4 ≠ vantagem proporcional [15].

**Câmera:** Camera Module 3 NoIR [26].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [12a](12a_yolov8n-multiclasse_hailort5/README.md) | Executa visão comparável ao HAT+ de 26 TOPS [15]; gerado para completar o cruzamento, sem prioridade especial. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [12b](12b_modelo-b_hailort5/README.md) | Como 03b, compilado para hailo10h. |
| MegaDetector + SpeciesNet | condicional — não gerado |  | Os 8 GB no módulo não resolvem a falta de artefato compilado documentado. |
