# Seeed Grove Vision AI Module V2

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

WiseEye2 HX6538: 2× Cortex-M55 + Ethos-U55; câmera CSI separada [18].

**Papel na pesquisa:** Exemplo TinyML com aceleração; câmera, controlador e firmware compõem o sistema [18].

**Câmera:** OV5647 (CSI) — existe variante sem filtro IR.

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [07a](07a_yolov8n-reduzido_sscma/README.md) | Detector YOLOv8n em resolução reduzida é o caso de uso documentado do módulo; câmera, controlador e firmware entram no sistema [18]. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | O firmware executa um modelo por vez; o classificador teria de rodar no hospedeiro sobre o JPEG reduzido. Não gerado. |
| TinyML (SDK do fabricante) | faz sentido — gerado | 07a | O SDK TinyML (SSCMA) é o próprio runtime da versão 07a. |
| MegaDetector + SpeciesNet | não faz sentido |  | Microcontrolador; ordens de grandeza abaixo da memória necessária. |
