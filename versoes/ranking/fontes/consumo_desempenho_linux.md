# Consumo e desempenho documentados — família Raspberry Pi, Hailo, IMX500 e Jetson

Levantamento documental feito em 20–21/10/2026 para completar o ranking. **Oficial** =
fabricante (docs, datasheet, product brief, Model Zoo); **terceiro** = medição publicada
por revisor (CNX Software, Jeff Geerling, RidgeRun) ou fórum oficial. Nenhum valor foi
medido pela autora. Benchmarks de inferência excluem pré/pós-processamento quando a fonte
diz isso.

## Raspberry Pi 5

| Grandeza | Valor | Condição | Fonte | Tipo |
|---|---|---|---|---|
| Corrente típica ativa (placa nua) | 800 mA (≈ 4 W a 5 V) | Tabela "Typical power requirements"; condição de medição não definida | https://www.raspberrypi.com/documentation/computers/raspberry-pi.html | oficial |
| Potência desligada (padrão) | ~1 a 1,4 W | POWER_OFF_ON_HALT=0 | idem, "Decrease Raspberry Pi 5 wattage when turned off" | oficial |
| Potência desligada com POWER_OFF_ON_HALT=1 | ~0,01 W | `sudo rpi-eeprom-config -e`; PMIC em STANDBY | idem | oficial |
| Baixo consumo com alarme RTC | ~3 mA | POWER_OFF_ON_HALT=1 + WAKE_ON_GPIO=0; `echo +600 \| sudo tee /sys/class/rtc/rtc0/wakealarm` e `sudo halt` | idem, seção RTC | oficial |
| Potência mínima | 4–5 W | Whitepaper RP-010139-WP-1, tabela 1 | https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf | oficial |
| Ocioso | 3,0 W | Headless, só Wi-Fi, 8 GB, Active Cooler, medidor de tomada | https://www.cnx-software.com/2023/11/05/raspberry-pi-5-review-raspberry-pi-os-bookworm-benchmarks-power-consumption/ | terceiro |
| Ocioso | 2,4 / 3,3 / 3,2 W | 2 GB / 4 GB / 8 GB | https://www.jeffgeerling.com/blog/2024/new-2gb-pi-5-has-33-smaller-die-30-idle-power-savings/ | terceiro |
| Carga plena de CPU | 8,8 W (CNX) / 9,8 W (Geerling, 4 e 8 GB) | `stress -c 4` / stress-ng | URLs acima | terceiro |
| Boot até o início do SO | ~7 s | CM5 com configuração padrão (não é o userland) | Whitepaper RP-010196-WP-1 | oficial |
| Boot até o desktop | 21,74 s | Raspberry Pi OS Desktop, microSD | https://www.cnx-software.com/2024/11/19/raspberry-pi-ssd-kit-and-class-a2-microsd-cards-review-with-raspberry-pi-5/ | terceiro |
| YOLOv8n ONNX / NCNN, 640, FP32 | 198,69 / 94,28 ms | Docs Ultralytics (snapshot da tag v8.3.0); versão não declarada | https://github.com/ultralytics/ultralytics/blob/v8.3.0/docs/en/guides/raspberry-pi.md | oficial |
| YOLO11n ONNX / NCNN, 640 | 191,09 / 94,03 ms | Ultralytics 8.3.152 | https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md | oficial |
| YOLO26n ONNX / NCNN, 640 | 125,99 / 67,03 ms | Ultralytics 8.4.108 | https://docs.ultralytics.com/guides/raspberry-pi/ | oficial |
| PCIe | Gen 2 padrão; Gen 3 recomendado para NPU (automático no AI HAT+) | — | https://www.raspberrypi.com/documentation/computers/ai.html | oficial |
| Temperatura de operação | não informada | Product brief: "normal ambient temperatures" | brief oficial | oficial |

## Raspberry Pi Zero 2 W

| Grandeza | Valor | Condição | Fonte | Tipo |
|---|---|---|---|---|
| Corrente típica ativa (placa nua) | 350 mA | Mesma tabela oficial | docs oficiais | oficial |
| Temperatura de operação | −20 a +70 °C | Product brief (abr/2024) | https://pip-assets.raspberrypi.com/categories/584-raspberry-pi-zero-2-w/documents/RP-008359-DS-1-raspberry-pi-zero-2-w-product-brief.pdf | oficial |
| Potência mínima | 0,6–1,2 W | Whitepaper RP-010139-WP-1 | — | oficial |
| Ocioso | ~120 mA (0,6 W) | Pi OS Lite padrão, Otii Arc | https://www.cnx-software.com/2021/12/09/raspberry-pi-zero-2-w-power-consumption/ | terceiro |
| Ocioso otimizado | 74,5 mA (~0,37 W) | HDMI, Wi-Fi/BT, áudio e núcleos extras desligados | idem | terceiro |
| Carga | 370–460 mA (~2,3 W) | `stress-ng -c 4` | https://www.jeffgeerling.com/blog/2021/disabling-cores-reduce-pi-zero-2-ws-power-consumption-half/ | terceiro |

## Câmeras

| Grandeza | Valor | Fonte | Tipo |
|---|---|---|---|
| Camera Module 3 NoIR | sem filtro IR ("Infrared-sensitive: Yes"); 0–50 °C | product brief jun/2024; docs de câmera | oficial |
| AI Camera (IMX500) | filtro IR integrado, "Infrared sensitive: No", sem versão NoIR | brief set/2024; docs | oficial |
| AI Camera: entrada máxima da rede | 640 × 640 (Raspberry Pi) — 640 × 480 na página da Sony | brief; https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500 | oficial (divergente) |
| AI Camera: memória no sensor | 8 388 480 bytes (firmware + pesos + trabalho) | brief e Sony | oficial |
| AI Camera: carga do firmware da rede | "several minutes" se não estiver em cache (o RP2040 faz cache) | https://www.raspberrypi.com/documentation/accessories/ai-camera.html | oficial |
| IMX500: YOLOv8n / YOLO11n, 640, INT8 | 58,79 / 58,82 ms por imagem; frame rate recomendado 16 fps | https://docs.ultralytics.com/integrations/sony-imx500/ | oficial |
| Consumo das câmeras | não encontrado | — | — |

## Hailo (AI HAT+ e AI HAT+ 2)

| Grandeza | Hailo-8L | Hailo-8 | Hailo-10H | Fonte |
|---|---|---|---|---|
| Desempenho declarado | 13 TOPS (8 bits) | 26 TOPS (8 bits) | 40 TOPS INT4 / 20 INT8 | datasheets Hailo (abr/2026; jun/2025) |
| Potência típica | 1,5 W | 2,5 W | < 2,5 W (até 8,25 máx.) | páginas de produto/datasheets |
| Potência média medida pelo fabricante | 1,9–3,14 W | 3,4–4,7 W | — | datasheet, tabela 12 |
| YOLOv8n 640 no Model Zoo (host x86, PCIe Gen3) | 201 FPS (b1) | 1036 FPS (b1) | 375 FPS (b1) | hailo_model_zoo v2.20.0 / master |
| No Raspberry Pi 5 (fórum oficial Hailo) | YOLOv8n 61,9 FPS (b1) | YOLO11n ~105 FPS (hw_only, x1) | — | community.hailo.ai (mar–abr/2026) |

A própria Hailo esclarece que o Model Zoo mede "raw chip throughput, not application-level
FPS": no Pi 5 o enlace PCIe de uma via limita o resultado. Por isso o ranking usa os
valores do fórum para o Pi 5 e não converte TOPS em FPS (pesquisa, seção 4).

## Jetson Orin Nano Super

| Grandeza | Valor | Condição | Fonte | Tipo |
|---|---|---|---|---|
| Desempenho | 67 TOPS INT8 (esparso) / 33 (denso) | — | NVIDIA, dez/2024 | oficial |
| Modos nvpmodel (JetPack 6.2.x, config Super) | 15 W (ID 0), 25 W (ID 1, padrão), MAXN_SUPER (ID 2) | o modo 7 W só existe na config não-Super | docs Jetson Linux r36.4.4 | oficial |
| Potência sem carga | 6,85 W | MAXN, tegrastats VDD_IN | RidgeRun | terceiro |
| Ocioso | 4,7 W → 2,5 W otimizado | JetPack 6.2, periféricos desligados | fórum NVIDIA (2026) | terceiro |
| YOLO11n TensorRT FP16 / INT8, 640 | 4,53 / 3,70 ms | Ultralytics 8.3.157, JetPack 6.1 | docs Ultralytics (v8.3.200) | oficial |

## O que não foi encontrado

Boot até o userland do Raspberry Pi OS Lite (Pi 5 e Zero 2 W); consumo da Camera Module 3 e
da AI Camera; potência da placa AI HAT+/AI HAT+ 2 (só do chip); YOLOv8n com TensorRT no
Orin Nano Super (há YOLO11n, de custo semelhante); temperatura de operação do Pi 5.
