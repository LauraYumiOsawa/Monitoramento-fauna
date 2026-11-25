# Arduino Nicla Vision

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Placa de visão com câmera e recursos integrados para aplicações embarcadas [25].

**Papel na pesquisa:** Alternativa TinyML; não confundir com câmeras OpenMV de outras gerações [25].

**Câmera:** GC2145 integrada (2 MP).

| Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|
| TinyML (SDK do fabricante) | faz sentido — gerado | [11a](11a_classificador-tinyml_openmv/README.md) | Alternativa TinyML [25]: classificação do quadro com modelo de 0,35 M parâmetros, acionada por movimento. |
| YOLOv8n multiclasse (Modelo A) | não faz sentido |  | YOLOv8n não cabe na memória da placa. |
| Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | não faz sentido |  | Sem detector de presença viável; o classificador isolado é a versão 11a. |
