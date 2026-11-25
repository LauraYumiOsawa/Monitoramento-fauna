# Consumo e desempenho documentados — MCUs, câmeras inteligentes e NPUs

Coleta de 20–21/10/2026. **Oficial** = fabricante; **terceiro** = medição publicada.

| Plataforma | Grandeza | Valor | Condição | Fonte | Tipo |
|---|---|---|---|---|---|
| MaixCAM (SG2002) | NPU / RAM | 1 TOPS INT8 / 256 MB DDR3 | — | https://wiki.sipeed.com/hardware/en/maixcam/maixcam.html | oficial |
| MaixCAM | Potência típica do chip | ~500 mW | "1080P + Video Encode + AI"; demais cenários "TBD" | datasheet SG2002 v1.0-alpha | oficial |
| MaixCAM | YOLOv8n 640×640 | 23 FPS | condição não informada | https://github.com/sipeed/MaixPy (README) | oficial |
| MaixCAM | YOLO11n 224² / 320×224 / 320² / 640² | 175 / 120 / 95 / 23 FPS | idem | idem | oficial |
| MaixCAM | Câmera do kit | GC4653, 2560×1440 | IR-cut/NoIR não informado | https://wiki.sipeed.com/hardware/en/maixcam/cameras.html | oficial |
| MaixCAM | Gerenciamento de energia | inexistente | "does not have an onboard power management unit" | wiki Sipeed (pmu) | oficial |
| Grove Vision AI V2 (HX6538) | CPU / NPU / SRAM | 2× M55 (400 e 150 MHz) / Ethos-U55-64 / 2432 KB | — | datasheet HX6538 | oficial |
| Grove Vision AI V2 | Potência ativa | 0,35 W (70 mA) | YOLOv8 1 classe, 192², INT8, 25 FPS; "minimal difference between processing and standing by" | https://core-electronics.com.au/guides/getting-started-with-the-grove-vision-ai-v2-power-efficient-object-detection/ | terceiro |
| Grove Vision AI V2 | Swift-YOLO 192² (pessoa) | 76 ms | Vela INT8 | sscma-model-zoo | oficial |
| Grove Vision AI V2 | Câmera IR | OV5647-160 "support IR" | sem IR-cut | Seeed Studio | oficial |
| Grove Vision AI V2 | Temperatura | −40 a 85 °C | chip | datasheet HX6538 | oficial |
| Orange Pi 5 (RK3588S) | Ocioso / carga | 1,0 W / 8,9–11,5 W | Armbian, 4 GB | https://github.com/geerlingguy/sbc-reviews/issues/5 | terceiro |
| RK3588 | YOLOv8n INT8 640 | 76,3 FPS | 1 núcleo, só inferência | https://github.com/airockchip/rknn_model_zoo (v2.5.0) | oficial |
| RK3588 | MobileNetV2 INT8 224 | 458,8 FPS | idem | idem | oficial |
| ESP32-S3 | Sono profundo | 7 µA | RTC mem ligada, 3,3 V, 25 °C | datasheet ESP32-S3 v2.2, tab. 5-10 | oficial |
| ESP32-S3 | Light-sleep | 240 µA (+140 µA PSRAM octal) | — | idem | oficial |
| ESP32-S3 | Ativo 240 MHz, Wi-Fi off | 66–108 mA | dual-core, periféricos off/on | idem | oficial |
| OV2640 | Ativa / standby | 125–140 mW / 600 µA | 15 fps UXGA | datasheet OmniVision v1.6 | oficial |
| ESP-DL no ESP32-S3 | MobileNetV2 INT8 224 | 353,5 ms (multi) / 423,3 ms (single) | só o modelo | https://github.com/espressif/esp-dl/tree/master/models/imagenet_cls | oficial |
| ESP-DL no ESP32-S3 | YOLO11n INT8 320 / 640 | 1335 / 5556 ms | só o modelo | esp-dl/models/coco_detect | oficial |
| ESP32-S3-EYE | Flash / PSRAM / microSD | 8 MB / 8 MB octal / sim | — | guia ESP32-S3-EYE (esp-who) | oficial |
| Luckfox Pico Max/Ultra (RV1106G3) | NPU / RAM | 1 TOPS / 256 MB DDR3L | — | https://wiki.luckfox.com/Luckfox-Pico-Pro-Max | oficial |
| Luckfox Pico (RV1103) | NPU / RAM | 0,5 TOPS / 64 MB | Mini/Plus não rodam o exemplo YOLO | wiki Luckfox | oficial |
| Luckfox | YOLOv5s 640 | ~100 ms | variante não informada | fórum EEPW | terceiro |
| Luckfox | Potência | não encontrada | — | — | — |
| Nicla Vision | Sono profundo / captura | 374 µA / 105 mA | bateria 200 mAh, OpenMV 4.3.4 | datasheet ABX00051 | oficial |
| Nicla Vision | Sono após usar a câmera | ~20 mA | pino de power-down do GC2145 não ligado | fórum OpenMV | terceiro |
| Nicla Vision | Memória / armazenamento | 1 MB SRAM, 2 MB flash, 16 MB QSPI; sem microSD | — | datasheet | oficial |
| Nicla Vision | Temperatura | −40 a 85 °C | placa | datasheet | oficial |
| Nicla Vision | FOMO 96² (MobileNetV2 0,35) | 30 FPS / 70 ms | Edge Impulse (docs / blog 2022) | Edge Impulse | oficial |
| Coral USB | Desempenho / potência | 4 TOPS / ~2 W | 0,5 W por TOPS | revendedor (datasheet coral.ai inacessível) | oficial |
| PyCoral | Suporte | última versão 2.0.0 (2021); wheels cp36–cp39; repositórios arquivados em 2025–2026 | incompatível com o Python 3.11 do Pi OS Bookworm | https://github.com/google-coral/pycoral | oficial |

Não encontrado: potência das placas MaixCAM, ESP32-S3-EYE e Luckfox; potência em espera
do Grove; latência oficial de YOLOv8n no Grove e no RV1106; tempos de boot de MaixCAM
(fora o primeiro), Orange Pi 5 e Luckfox.
