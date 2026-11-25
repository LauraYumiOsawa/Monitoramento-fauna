# Espressif ESP32-S3-EYE

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

ESP32-S3 com câmera OV2640 de 2 MP e microSD [19]; sono profundo documentado [20].

**Papel na pesquisa:** Referência de microcontrolador com câmera; 'porteiro' da arquitetura C.

**Câmera:** OV2640 integrada (até 1600 × 1200) [19].

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| TinyML (SDK do fabricante) | faz sentido — gerado | [09a](09a_captura-independente_tinyml-esp-dl/README.md) | Subsistema de captura da arquitetura C com sono profundo [19; 20]; a pré-classificação TinyML é opcional e não decide descarte. |
| YOLOv8n multiclasse (Modelo A) | não faz sentido |  | A pesquisa veda presumir YOLOv8n completo a 640 px sem documentação de implantação [19]; detecção fica para o processamento em lote. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | Só o classificador (sobre o quadro inteiro) é plausível; o detector de presença ficaria em segundos por foto. Representado pela pré-classificação da 09a. |
| MegaDetector + SpeciesNet | não faz sentido |  | Microcontrolador; inviável. As fotos vão para a 01c/08c em lote. |
