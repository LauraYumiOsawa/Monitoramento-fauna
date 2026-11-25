# Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Acelerador PCIe Hailo-8, 26 TOPS [14].

**Papel na pesquisa:** IA em acelerador PCIe (perfil 'Visão acelerada em Linux').

**Câmera:** Camera Module 3 NoIR [26].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [03a](03a_yolov8n-multiclasse_hailort/README.md) | Mesmo fluxo da 02a para o Hailo-8 (26 TOPS) [14; 39]; compara capacidade declarada sem converter TOPS em FPS. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [03b](03b_modelo-b_hailort/README.md) | Como 02b, compilado para hailo8. |
| MegaDetector + SpeciesNet | condicional — não gerado |  | Como no Hailo-8L: possível em tese, sem artefato documentado nem validação. |
