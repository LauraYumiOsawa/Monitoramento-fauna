# Matriz de compatibilidade hardware × software/modelo

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

Cruzamento dos hardwares candidatos (seção 4 da pesquisa) com os softwares e modelos (seção 6). ✅ faz sentido e foi gerada uma versão; ⚠️ condicional (possível, com ressalva documentada, não gerado); — não faz sentido. Clique no identificador para abrir a versão.

| Hardware | YOLOv8n multiclasse (Modelo A) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | MegaDetector + SpeciesNet | TinyML (SDK do fabricante) |
|---|---|---|---|---|
| Raspberry Pi 5 | ✅ [01a](01_raspberry-pi-5/01a_yolov8n-multiclasse_onnxruntime/README.md) | ✅ [01b](01_raspberry-pi-5/01b_modelo-b_onnxruntime/README.md) | ✅ [01c](01_raspberry-pi-5/01c_megadetector-speciesnet_lote/README.md) | — |
| Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) | ✅ [02a](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02a_yolov8n-multiclasse_hailort/README.md) | ✅ [02b](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02b_modelo-b_hailort/README.md) | ⚠️ | — |
| Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) | ✅ [03a](03_raspberry-pi-5_ai-hat-plus_hailo-8/03a_yolov8n-multiclasse_hailort/README.md) | ✅ [03b](03_raspberry-pi-5_ai-hat-plus_hailo-8/03b_modelo-b_hailort/README.md) | ⚠️ | — |
| Raspberry Pi 5 + AI Camera (IMX500) | ✅ [04a](04_raspberry-pi-5_ai-camera-imx500/04a_yolov8n-multiclasse_imx500/README.md) | ✅ [04b](04_raspberry-pi-5_ai-camera-imx500/04b_modelo-b_imx500/README.md) | — | — |
| Raspberry Pi Zero 2 W | ✅ [05a](05_raspberry-pi-zero-2-w/05a_yolov8n-512-int8_onnxruntime/README.md) | ✅ [05b](05_raspberry-pi-zero-2-w/05b_modelo-b-int8_onnxruntime/README.md) | — | — |
| Sipeed MaixCAM | ✅ [06a](06_sipeed-maixcam/06a_yolov8n-multiclasse_maixpy/README.md) | ✅ [06b](06_sipeed-maixcam/06b_modelo-b_maixpy/README.md) | — | — |
| Seeed Grove Vision AI Module V2 | ✅ [07a](07_grove-vision-ai-v2/07a_yolov8n-reduzido_sscma/README.md) | ⚠️ | — | ✅ [07a](07_grove-vision-ai-v2/07a_yolov8n-reduzido_sscma/README.md) |
| Orange Pi 5 | ✅ [08a](08_orange-pi-5/08a_yolov8n-multiclasse_rknn/README.md) | ✅ [08b](08_orange-pi-5/08b_modelo-b_rknn/README.md) | ✅ [08c](08_orange-pi-5/08c_megadetector-speciesnet_lote/README.md) | — |
| Espressif ESP32-S3-EYE | — | ⚠️ | — | ✅ [09a](09_esp32-s3-eye/09a_captura-independente_tinyml-esp-dl/README.md) |
| Luckfox Pico (RV1106) | ✅ [10a](10_luckfox-pico-rv1106/10a_yolov8n-320-int8_rknn-rv1106/README.md) | ⚠️ | — | — |
| Arduino Nicla Vision | — | — | — | ✅ [11a](11_arduino-nicla-vision/11a_classificador-tinyml_openmv/README.md) |
| Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) | ✅ [12a](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12a_yolov8n-multiclasse_hailort5/README.md) | ✅ [12b](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12b_modelo-b_hailort5/README.md) | ⚠️ | — |
| NVIDIA Jetson Orin Nano Super | ✅ [13a](13_jetson-orin-nano-super/13a_yolov8n-multiclasse_tensorrt/README.md) | ✅ [13b](13_jetson-orin-nano-super/13b_modelo-b_tensorrt/README.md) | ✅ [13c](13_jetson-orin-nano-super/13c_megadetector-speciesnet_gpu/README.md) | — |
| Coral USB e OpenMV | — | — | — | — |

## Camadas transversais

| Software | Onde se aplica | Tratamento |
|---|---|---|
| ONNX + ONNX Runtime [6] | Pi 5, Pi Zero 2 W, Orange Pi 5 (CPU), Jetson (GPU) | Runtime direto das versões 01a/01b/05a/05b/13a/13b e do classificador nas versões Hailo/IMX500; alternativa sem NPU nas 08a/08b. Nas NPUs o ONNX é só o formato de entrada da conversão |
| Conversão para NPU [22; 39] | Hailo, RKNN (RK3588S e RV1106), IMX500, SG2002, Ethos-U55 | Um script por versão (`compilar_hef.sh`, `converter_rknn.py`, `exportar_imx500.sh`, `converter_cvimodel.sh`, `exportar_tflite.sh`). INT8 com a cabeça YOLO dividida (achado medido: INT8 em todos os operadores com a cabeça inteira quebra os escores) |
| Picamera2 [26; 30] | Todas as versões Raspberry Pi (01–05, 12) | `captura.tipo: picamera2`; nas demais placas, OpenCV/GStreamer, MaixPy, SSCMA, esp32-camera ou OpenMV |
| Camtrap DP [11] | Todas | Linux: registro direto (`export-camtrapdp`); MCU: CSV no cartão importado em lote |
| LILA BC / WCS [9; 10] | Nenhum hardware | Bases para treino e avaliação, uso offline; não há experimento novo com elas |
| DeepFaune, Zamba, Timelapse, Wildlife Insights, AddaxAI [P0; 7] | Nenhum hardware embarcado | Ferramentas de desktop/nuvem; consomem os registros exportados (o AddaxAI é interface do MegaDetector) |

## Justificativa de cada cruzamento

| Hardware | Software/modelo | Veredito | Versão | Justificativa |
|---|---|---|---|---|
| Raspberry Pi 5 | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [01a](01_raspberry-pi-5/01a_yolov8n-multiclasse_onnxruntime/README.md) | Configuração de referência da pesquisa: YOLOv8n em ONNX executado pelo ONNX Runtime no Pi 5 [P1; 5; 6]. O Ultralytics publica benchmark ONNX no Pi 5 [38]. |
| Raspberry Pi 5 | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [01b](01_raspberry-pi-5/01b_modelo-b_onnxruntime/README.md) | Mesma placa, pipeline em dois estágios: separa presença de espécie (QP4) e permite 'desconhecido'/'inconclusivo' [3; 7; 8]. Medido no projeto: maior F1 entre os modelos próprios. |
| Raspberry Pi 5 | MegaDetector + SpeciesNet | faz sentido — gerado | [01c](01_raspberry-pi-5/01c_megadetector-speciesnet_lote/README.md) | Modelos grandes demais para tempo real, mas viáveis em lote no Pi 5 de 8 GB: classificação posterior das fotos de um subsistema de captura (arquitetura C) [4; 7; 8; 30]. |
| Raspberry Pi 5 | TinyML (SDK do fabricante) | não faz sentido |  | Os SDKs TinyML visam microcontroladores; no Pi 5 o mesmo modelo roda sem as restrições de memória que justificam TinyML. |
| Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [02a](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02a_yolov8n-multiclasse_hailort/README.md) | YOLOv8n é arquitetura suportada pelo fluxo ONNX → otimização → HEF [39]; a NPU tira a convolução da CPU (QP3) [14]. |
| Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [02b](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02b_modelo-b_hailort/README.md) | Detector de presença no Hailo; o classificador roda só nos recortes (~0,19 por quadro medido no projeto), barato na CPU do Pi 5. |
| Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) | MegaDetector + SpeciesNet | condicional — não gerado |  | Sem HEF documentado para MegaDetector/SpeciesNet; compilar YOLOv5x6 a 1280 px e EfficientNetV2 no DFC exigiria validação própria. O hospedeiro Pi 5 pode rodar a versão 01c em lote. |
| Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [03a](03_raspberry-pi-5_ai-hat-plus_hailo-8/03a_yolov8n-multiclasse_hailort/README.md) | Mesmo fluxo da 02a para o Hailo-8 (26 TOPS) [14; 39]; compara capacidade declarada sem converter TOPS em FPS. |
| Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [03b](03_raspberry-pi-5_ai-hat-plus_hailo-8/03b_modelo-b_hailort/README.md) | Como 02b, compilado para hailo8. |
| Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) | MegaDetector + SpeciesNet | condicional — não gerado |  | Como no Hailo-8L: possível em tese, sem artefato documentado nem validação. |
| Raspberry Pi 5 + AI Camera (IMX500) | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [04a](04_raspberry-pi-5_ai-camera-imx500/04a_yolov8n-multiclasse_imx500/README.md) | O Ultralytics exporta YOLOv8n para o IMX500; a inferência sai do processador hospedeiro [16]. Compara IA no sensor com IA no processador. |
| Raspberry Pi 5 + AI Camera (IMX500) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [04b](04_raspberry-pi-5_ai-camera-imx500/04b_modelo-b_imx500/README.md) | Divisão natural: o sensor localiza o animal; a CPU classifica o recorte em alta resolução do mesmo quadro. |
| Raspberry Pi 5 + AI Camera (IMX500) | MegaDetector + SpeciesNet | não faz sentido |  | MegaDetector e SpeciesNet excedem em muito a memória da rede no sensor do IMX500. |
| Raspberry Pi Zero 2 W | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [05a](05_raspberry-pi-zero-2-w/05a_yolov8n-512-int8_onnxruntime/README.md) | Cabe em 512 MB com Raspberry Pi OS Lite; segundos por quadro são aceitáveis para fotos acionadas por PIR. Entrada 512 é a alternativa leve da pesquisa [P1; 13]. |
| Raspberry Pi Zero 2 W | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [05b](05_raspberry-pi-zero-2-w/05b_modelo-b-int8_onnxruntime/README.md) | Modelo B INT8 (5,7 MB) foi a configuração recomendada pelo projeto para hardware limitado; MobileNetV2 não perde acurácia em INT8. |
| Raspberry Pi Zero 2 W | MegaDetector + SpeciesNet | não faz sentido |  | 512 MB de RAM não comportam os pesos do MegaDetector e do SpeciesNet com o sistema [13]. |
| Sipeed MaixCAM | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [06a](06_sipeed-maixcam/06a_yolov8n-multiclasse_maixpy/README.md) | MaixPy executa YOLOv8 convertido por TPU-MLIR; o fluxo de conversão é verificado antes de assumir execução [17]. |
| Sipeed MaixCAM | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [06b](06_sipeed-maixcam/06b_modelo-b_maixpy/README.md) | Os dois estágios cabem na NPU; INT8 do MobileNetV2 medido sem perda no projeto. |
| Sipeed MaixCAM | MegaDetector + SpeciesNet | não faz sentido |  | Memória e NPU incompatíveis com MegaDetector (1280 px) e SpeciesNet. |
| Seeed Grove Vision AI Module V2 | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [07a](07_grove-vision-ai-v2/07a_yolov8n-reduzido_sscma/README.md) | Detector YOLOv8n em resolução reduzida é o caso de uso documentado do módulo; câmera, controlador e firmware entram no sistema [18]. |
| Seeed Grove Vision AI Module V2 | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | O firmware executa um modelo por vez; o classificador teria de rodar no hospedeiro sobre o JPEG reduzido. Não gerado. |
| Seeed Grove Vision AI Module V2 | TinyML (SDK do fabricante) | faz sentido — gerado | [07a](07_grove-vision-ai-v2/07a_yolov8n-reduzido_sscma/README.md) | O SDK TinyML (SSCMA) é o próprio runtime da versão 07a. |
| Seeed Grove Vision AI Module V2 | MegaDetector + SpeciesNet | não faz sentido |  | Microcontrolador; ordens de grandeza abaixo da memória necessária. |
| Orange Pi 5 | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [08a](08_orange-pi-5/08a_yolov8n-multiclasse_rknn/README.md) | Contraponto com NPU no SoC; o fluxo RKNN tem conversão e execução próprias [21; 22]. INT8 exige cortar a cabeça (achado do projeto). |
| Orange Pi 5 | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [08b](08_orange-pi-5/08b_modelo-b_rknn/README.md) | Os dois estágios na NPU; CPU livre para rastreamento e registro. |
| Orange Pi 5 | MegaDetector + SpeciesNet | faz sentido — gerado | [08c](08_orange-pi-5/08c_megadetector-speciesnet_lote/README.md) | 8/16 GB de RAM permitem a classificação de referência em lote, como na 01c. |
| Orange Pi 5 | ONNX + ONNX Runtime | faz sentido — gerado | [08a](08_orange-pi-5/08a_yolov8n-multiclasse_rknn/README.md) | ONNX Runtime CPU fica como alternativa sem conversão nas versões 08a/08b (comparação CPU × NPU da QP3). |
| Espressif ESP32-S3-EYE | TinyML (SDK do fabricante) | faz sentido — gerado | [09a](09_esp32-s3-eye/09a_captura-independente_tinyml-esp-dl/README.md) | Subsistema de captura da arquitetura C com sono profundo [19; 20]; a pré-classificação TinyML é opcional e não decide descarte. |
| Espressif ESP32-S3-EYE | YOLOv8n multiclasse (Modelo A) | não faz sentido |  | A pesquisa veda presumir YOLOv8n completo a 640 px sem documentação de implantação [19]; detecção fica para o processamento em lote. |
| Espressif ESP32-S3-EYE | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | Só o classificador (sobre o quadro inteiro) é plausível; o detector de presença ficaria em segundos por foto. Representado pela pré-classificação da 09a. |
| Espressif ESP32-S3-EYE | MegaDetector + SpeciesNet | não faz sentido |  | Microcontrolador; inviável. As fotos vão para a 01c/08c em lote. |
| Luckfox Pico (RV1106) | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [10a](10_luckfox-pico-rv1106/10a_yolov8n-320-int8_rknn-rv1106/README.md) | A NPU do RV1106 executa INT8; o modelo exato (RV1106G3, 256 MB) é caracterizado antes de comparar [24]. |
| Luckfox Pico (RV1106) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | condicional — não gerado |  | Exigiria segundo contexto RKNN e portar rastreamento/votação para C; não gerado nesta revisão. |
| Luckfox Pico (RV1106) | MegaDetector + SpeciesNet | não faz sentido |  | Memória insuficiente. |
| Arduino Nicla Vision | TinyML (SDK do fabricante) | faz sentido — gerado | [11a](11_arduino-nicla-vision/11a_classificador-tinyml_openmv/README.md) | Alternativa TinyML [25]: classificação do quadro com modelo de 0,35 M parâmetros, acionada por movimento. |
| Arduino Nicla Vision | YOLOv8n multiclasse (Modelo A) | não faz sentido |  | YOLOv8n não cabe na memória da placa. |
| Arduino Nicla Vision | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | não faz sentido |  | Sem detector de presença viável; o classificador isolado é a versão 11a. |
| Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [12a](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12a_yolov8n-multiclasse_hailort5/README.md) | Executa visão comparável ao HAT+ de 26 TOPS [15]; gerado para completar o cruzamento, sem prioridade especial. |
| Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [12b](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12b_modelo-b_hailort5/README.md) | Como 03b, compilado para hailo10h. |
| Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) | MegaDetector + SpeciesNet | condicional — não gerado |  | Os 8 GB no módulo não resolvem a falta de artefato compilado documentado. |
| NVIDIA Jetson Orin Nano Super | YOLOv8n multiclasse (Modelo A) | faz sentido — gerado | [13a](13_jetson-orin-nano-super/13a_yolov8n-multiclasse_tensorrt/README.md) | Teto de desempenho do mesmo modelo [23]. |
| NVIDIA Jetson Orin Nano Super | Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) | faz sentido — gerado | [13b](13_jetson-orin-nano-super/13b_modelo-b_tensorrt/README.md) | Teto de desempenho do pipeline em dois estágios. |
| NVIDIA Jetson Orin Nano Super | MegaDetector + SpeciesNet | faz sentido — gerado | [13c](13_jetson-orin-nano-super/13c_megadetector-speciesnet_gpu/README.md) | Único hardware do estudo em que a referência roda perto do tempo real; papel de estação base, não de armadilha a bateria [23]. |
| Coral USB e OpenMV | YOLOv8n multiclasse (Modelo A) | não faz sentido |  | A pesquisa mantém Coral USB e OpenMV só como contexto, sem modelo, suporte e custo homogêneos [P0]. |
