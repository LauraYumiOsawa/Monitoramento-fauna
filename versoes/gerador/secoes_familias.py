"""Trechos de README específicos de cada família de runtime.

Cada função recebe o contexto da versão e devolve seções que substituem ou completam as
genéricas de gerar_versoes.py. Os fatos citados aqui vêm da documentação oficial
consultada em 20–21/10/2026 (fontes em versoes/ranking/fontes/).
"""

from __future__ import annotations

from typing import Any

IMPORTAR_CSV = """### 5. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \\
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id {id} \\
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c)."""

ENERGIA_PI = """
**Energia (Raspberry Pi).** Na arquitetura A a placa fica ligada: o Pi 5 consome ~3 W
ocioso (medições de terceiros) e o Zero 2 W ~0,5–0,6 W. Para a arquitetura B, desligue a
placa entre eventos e deixe o circuito do PIR religá-la: com `POWER_OFF_ON_HALT=1`
(`sudo rpi-eeprom-config -e`) o Pi 5 desligado cai para ~0,01 W (documentação oficial).
Ative `energia.desligar_ao_final: true` e `gatilho.tipo: nenhum` em `versao.yaml`: o
serviço faz um disparo por boot e desliga. O custo é o boot (~7 s até o SO no CM5,
whitepaper oficial; mais até o userland) antes da primeira foto.
"""


def _disparo(ctx, acelerador: str) -> str:
    evento = ctx.spec.get("evento", ctx.padroes["evento"])
    return f"PIR → rajada de {evento['quadros']} quadros → {acelerador} → registro"


def _ort(ctx) -> dict[str, str]:
    detector = ctx.spec["detector"]
    secoes = {"disparo": _disparo(ctx, f"ONNX Runtime na CPU ({detector.get('threads', 4)} threads)")}
    secoes["instalar_extra"] = ENERGIA_PI
    if ctx.hardware_chave == "pizero2w":
        secoes["limitacoes_extra"] = (
            "- 512 MB de RAM: use Raspberry Pi OS Lite 64 bits sem interface gráfica; a câmera "
            "em 1280×720 e 5 quadros por evento limitam a memória (confirme com `benchmark`).\n"
            "- O Cortex-A53 a 1 GHz leva segundos por quadro: a análise acontece depois da "
            "rajada, nunca durante a captura.\n"
        )
    else:
        secoes["limitacoes_extra"] = (
            "- O Ultralytics indica NCNN como o formato mais rápido no Pi 5 (YOLOv8n: 94 ms "
            "contra 199 ms do ONNX, docs v8.3.0); esta versão mantém ONNX Runtime porque é o "
            "runtime de referência da pesquisa [6].\n"
        )
    return secoes


def _hailo(ctx) -> dict[str, str]:
    hailo = ctx.spec["hailo"]
    arch = hailo["arch"]
    pacote = "hailo-h10-all" if arch == "hailo10h" else "hailo-all"
    geracao = ("DFC 5.x / HailoRT 5.x" if arch == "hailo10h" else "DFC 3.x / HailoRT 4.x")
    classificador = " + MobileNetV2 na CPU (ONNX Runtime)" if ctx.modelo == "modelo-b" else ""
    return {
        "disparo": _disparo(ctx, f"HEF no {arch} (NMS no HailoRT){classificador}"),
        "passos_preparar": [
            f"./compilar_hef.sh         # HEF {arch}, {hailo['entrada']} px (Ultralytics format=\"hailo\")",
        ],
        "preparar_extra": (
            f"`compilar_hef.sh` exige Linux x86_64 com o Dataflow Compiler ({geracao}; wheel no "
            "Developer Zone da Hailo, registro gratuito). HEFs de gerações diferentes não são "
            "intercambiáveis. Sem GPU NVIDIA o otimizador cai para o nível 0."
        ),
        "instalar_extra": (
            f"\nNo Pi 5 (Raspberry Pi OS 64 bits): `sudo apt install dkms {pacote}` e "
            "`hailortcli fw-control identify`. Os pacotes `hailo-all` e `hailo-h10-all` não "
            "coexistem. O AI HAT+ ativa PCIe Gen 3 automaticamente.\n" + ENERGIA_PI
        ),
        "limitacoes_extra": (
            "- A Hailo recomenda ≥ 1024 imagens de calibração; o dataset atual tem bem menos. "
            "A qualidade informada usa como proxy o INT8 em todos os operadores com a cabeça "
            "dividida (x86), não o HEF.\n"
            "- O Model Zoo mede vazão bruta do chip em host x86 (YOLOv8n: Hailo-8L 201 FPS, "
            "Hailo-8 1036 FPS); no Pi 5 o fórum oficial relata 61,9 FPS (8L) — o enlace PCIe "
            "de uma via limita.\n"
            "- Os limiares de NMS do `versao.yaml` são aplicados em tempo de execução "
            "(`set_nms_score_threshold`/`set_nms_iou_threshold`).\n"
        ),
    }


def _imx500(ctx) -> dict[str, str]:
    classificador = " + MobileNetV2 na CPU do Pi 5" if ctx.modelo == "modelo-b" else ""
    return {
        "disparo": _disparo(ctx, f"rede no sensor IMX500 (INT8, NMS embutido){classificador}"),
        "passos_preparar": [
            "./exportar_imx500.sh       # Ultralytics format=\"imx\" → packerOut.zip (Java 17+)",
        ],
        "preparar_extra": (
            "O export só aceita YOLOv8n e YOLO11n e força INT8 (Model Compression Toolkit, "
            "NMS embutido)."
        ),
        "instalar_extra": (
            "\nNo Pi 5: `sudo apt install imx500-all imx500-tools` e, na pasta da versão, "
            "`./empacotar_rpk.sh` (o `imx500-package` só roda no Pi) para gerar o `network.rpk`.\n"
        ),
        "executar_extra": (
            "\n`--simular` não se aplica: o resultado da rede chega como metadado do sensor. "
            "Para bancada, use as versões 01a/01b com o mesmo modelo em ONNX.\n"
        ),
        "limitacoes_extra": (
            "- **Sem visão noturna com IR**: a AI Camera tem filtro IR e não existe versão NoIR "
            "(documentação oficial). Para fauna noturna, é a principal limitação desta versão.\n"
            "- O firmware da rede pode levar minutos para carregar na primeira vez (fica em "
            "cache): a câmera fica sempre ligada e a arquitetura B não se aplica.\n"
            "- Entrada máxima 640×640 (Raspberry Pi; a Sony cita 640×480) e 8 MB de memória no "
            "sensor. Inferência publicada pelo Ultralytics: YOLOv8n 58,8 ms (640, INT8).\n"
        ),
    }


def _rknn(ctx) -> dict[str, str]:
    classificador = " e classificador na NPU" if ctx.modelo == "modelo-b" else ""
    return {
        "disparo": _disparo(ctx, f"YOLOv8n INT8 na NPU do RK3588S{classificador}"),
        "passos_preparar": [
            "python converter_rknn.py --analise   # INT8 rk3588 (ambiente com rknn-toolkit2 2.5.0)",
        ],
        "preparar_extra": (
            "Rode o conversor num ambiente separado (`pip install rknn-toolkit2==2.5.0`; ele fixa "
            "versões de onnx/torch). A versão do toolkit deve casar com a do `librknnrt.so` da "
            "placa (`strings /usr/lib/librknnrt.so | grep -i version`)."
        ),
        "instalar_extra": (
            "\nNo Orange Pi 5 (Debian/Orange Pi OS; o toolkit-lite2 lista Debian 10–12): o "
            "`instalar.sh` instala `rknn-toolkit-lite2`. Descubra o chip/linha do PIR com "
            "`gpioinfo` e ajuste `gatilho` e `iluminacao_ir` em `versao.yaml`. Câmera: UVC "
            "(`dispositivo: 0`) ou módulo MIPI por pipeline GStreamer em `captura.dispositivo`.\n"
        ),
        "limitacoes_extra": (
            "- rknn_model_zoo: YOLOv8n INT8 640 a 76,3 FPS em um núcleo do RK3588 (só "
            "inferência). A qualidade informada é o proxy x86 do INT8 com cabeça dividida.\n"
            "- `gatilho.linha`/`iluminacao_ir.pino` são exemplos: confira com `gpioinfo`.\n"
        ),
    }


def _tensorrt(ctx) -> dict[str, str]:
    return {
        "disparo": _disparo(ctx, "ONNX Runtime GPU (TensorRT FP16 → CUDA → CPU, nesta ordem)"),
        "instalar_extra": (
            "\nJetPack 6.2.x. O `instalar.sh` usa `requisitos_jetson.txt` (onnxruntime-gpu do "
            "jetson-ai-lab). Modo de potência econômico da configuração Super: "
            "`sudo nvpmodel -m 0` (15 W; o modo de 7 W só existe na configuração não-Super). "
            "A primeira execução compila o motor TensorRT (minutos) e o guarda em "
            "`modelos/cache_trt`.\n"
        ),
        "limitacoes_extra": (
            "- Não foi confirmado que as wheels do jetson-ai-lab expõem o "
            "`TensorrtExecutionProvider`; sem ele a versão roda em CUDA. Alternativa oficial: "
            "export Ultralytics `format=\"engine\", quantize=16` no próprio Jetson.\n"
            "- 15–25 W de modo de potência e ~5–7 W ocioso (terceiros): papel de estação base "
            "com energia da rede ou solar dimensionado, não de armadilha a bateria [23].\n"
        ),
    }


def _lote(ctx) -> dict[str, str]:
    return {
        "disparo": "fotos de um subsistema de captura (ex.: 09a) → lote MegaDetector + SpeciesNet "
                   "→ Camtrap DP",
        "preparar": (
            "### 1. Instalar\n\n```bash\n./instalar_lote.sh   # venv + PyTorch "
            f"({ctx.spec['dispositivo_torch']}) + speciesnet 5.0.6\n```"
        ),
        "executar": (
            "### 4. Executar\n\n```bash\n./executar_lote.sh /media/cartao/DCIM          # "
            "MegaDetector v5a + SpeciesNet v4.0.3a, --country BRA\n./executar_lote.sh "
            "/media/cartao/DCIM --detector   # só o filtro de vazias\n.venv/bin/python "
            "importar_resultados.py dados/speciesnet_*.json --latitude <lat> --longitude <lon>\n"
            "```\n\nAlternativa mais leve para o Pi: os MegaDetector MDv1000-spruce (YOLOv5s, "
            "640 px, 16,5 GFLOPs) e -sorrel (YOLOv11s, 960 px) do pacote `megadetector`."
        ),
        "limitacoes": (
            "- **Medido no projeto (x86, i5-1235U):** nas 18 imagens de teste o MegaDetector "
            "localizou os 20 animais (presença 20/20) e o SpeciesNet acertou a espécie exata em "
            "13 (F1 0,65); os erros foram arredondamentos para família (\"tapiridae\", "
            "\"canine\") ou rótulos sem sentido, nunca troca entre as três espécies. Lote de 18 "
            "imagens em ~42 s com 12 threads.\n"
            "- As imagens de teste não são de armadilha fotográfica (animal em close, 14 geradas "
            "por IA): o resultado não estima o desempenho de campo do SpeciesNet.\n"
            "- Pesos baixados do Kaggle na primeira execução (~500 MB). Licenças: SpeciesNet "
            "Apache-2.0; inferência do MDv5 via pacote yolov5 (GPL).\n"
        ),
        "exportar": "",
        "instalar": "",
    }


def _maixpy(ctx) -> dict[str, str]:
    maix = ctx.spec["maix"]
    classificador = " + nn.Classifier no recorte (votação na rajada)" if ctx.modelo == "modelo-b" else ""
    return {
        "disparo": _disparo(ctx, f"nn.YOLOv8 na NPU (cvimodel INT8, {maix['entrada'][1]}×"
                                 f"{maix['entrada'][0]}){classificador}"),
        "preparar": (
            "### 1. Preparar o modelo (x86_64 com Docker)\n\n```bash\n./preparar_modelo.sh       "
            "# exporta o ONNX 224×320\n./converter_cvimodel.sh    # TPU-MLIR: model_transform → "
            "run_calibration → model_deploy --processor cv181x\n```\n\nOs nomes de saída "
            "(`/model.22/dfl/conv/Conv_output_0`, `/model.22/Sigmoid_output_0`) foram conferidos "
            "nos ONNX deste projeto."
        ),
        "instalar": (
            "### 2. Instalar na MaixCAM\n\n```bash\nscp -r app modelos root@<ip>:/root/armadilha/"
            "\nssh root@<ip> python /root/armadilha/app/main.py\n```\n\nOu instale `app/` como "
            "aplicativo pelo MaixVision (`app.yaml`)."
        ),
        "ligacoes": (
            "### 3. Ligações\n\nPIR (3,3 V) no pino `A19` (ajuste `PINO_PIR`; evite A14, A17 e o "
            "botão). O MaixPy não documenta interrupção de GPIO: o PIR é lido por polling a cada "
            "50 ms."
        ),
        "executar": "",
        "exportar": IMPORTAR_CSV.format(id=ctx.id),
        "limitacoes": (
            "- A MaixCAM não tem gerenciamento de energia nem API de sono documentada: fica "
            "ligada (o chip SG2002 consome ~500 mW típicos com vídeo e IA, datasheet).\n"
            "- Modelo B sem rastreador: a votação usa o maior animal de cada quadro da rajada "
            "(simplificação para um animal por evento).\n"
            "- Desempenho publicado pela Sipeed: YOLOv8n 640×640 a 23 FPS; YOLO11n 320×224 a "
            "120 FPS (condição não informada).\n"
            "- A câmera GC4653 do kit não tem informação de IR-cut; para noite, troque por "
            "sensor sem filtro IR.\n" + LIMITES_DISPOSITIVO
        ),
    }


def _grove(ctx) -> dict[str, str]:
    return {
        "disparo": "PIR acorda o XIAO → liga o Grove → AT+INVOKE em rajada (YOLOv8n INT8 no "
                   "Ethos-U55) → porta de 3 invocações → JPEG + CSV no microSD do XIAO",
        "preparar": (
            "### 1. Preparar o modelo\n\n```bash\npip install ethos-u-vela\n./exportar_tflite.sh "
            f"   # Ultralytics saved_model INT8 a {ctx.spec['grove']['entrada']} px → Vela "
            "ethos-u55-64\n```\n\nEnvie o `*_vela.tflite` pelo SenseCraft AI (Chrome/Edge: Models "
            "→ My Models → Add Model), com as classes anta, tamandua_bandeira, lobo_guara."
        ),
        "instalar": (
            "### 2. Firmware do hospedeiro\n\nArduino IDE, pacote esp32 da Espressif, placa "
            "XIAO_ESP32S3 (OPI PSRAM habilitada), biblioteca Seeed_Arduino_SSCMA. Carregue "
            "`host_xiao/host_xiao.ino`."
        ),
        "ligacoes": (
            "### 3. Ligações\n\nXIAO ESP32S3 Sense montado no conector do Grove (I2C 0x62). PIR "
            "→ D0 (GPIO1, RTC). D1 comanda o MOSFET que alimenta o Grove só durante o evento. "
            "microSD na placa Sense (CS 21)."
        ),
        "executar": "",
        "exportar": IMPORTAR_CSV.format(id=ctx.id),
        "limitacoes": (
            "- O módulo consome ~0,35 W rodando a ~25 FPS (YOLOv8 192 px, medição da Core "
            "Electronics) e quase o mesmo parado; por isso a alimentação é cortada entre "
            "eventos. O tempo de partida após religar não está documentado.\n"
            "- Resoluções documentadas: 192, 224 e 256. No proxy x86 o YOLOv8n do projeto cai "
            "de F1 0,872 (640) para 0,647 (256, INT8): o animal fica pequeno demais.\n"
            "- Não está documentado se `x`/`y` das caixas são centro ou canto; o firmware só usa "
            "classe e escore.\n" + LIMITES_DISPOSITIVO
        ),
    }


def _esp(ctx) -> dict[str, str]:
    return {
        "disparo": "sono profundo → PIR (ext0) → 3 JPEG UXGA no microSD → MobileNetV2 INT8 no "
                   "quadro inteiro (ESP-DL) → CSV → sono profundo",
        "preparar": (
            "### 1. Preparar o modelo\n\n```bash\n./preparar_modelo.sh          # copia o "
            "MobileNetV2 ONNX\npip install esp-ppq          # ambiente separado (torch CPU)\n"
            "python quantizar_espdl.py     # → modelos/classificador.espdl\n```"
        ),
        "instalar": (
            "### 2. Compilar e gravar (ESP-IDF ≥ 5.3)\n\n```bash\ncd firmware\nidf.py "
            "set-target esp32s3\nidf.py build flash monitor\n```\n\nAntes de instalar, grave a "
            "data em `/sdcard/relogio.txt` (ex.: `2026-11-02T08:30:00`): a placa não tem RTC "
            "com bateria; o horário sobrevive ao sono profundo, não à falta de energia."
        ),
        "ligacoes": (
            "### 3. Ligações\n\nPIR → GPIO3 (o guia do S3-EYE diz que é o único livre; é pino de "
            "strapping e acende o LED verde — o PIR deve estar em nível baixo no reset). "
            "microSD da própria placa (SDMMC 1 bit)."
        ),
        "executar": "",
        "exportar": IMPORTAR_CSV.format(id=ctx.id),
        "limitacoes": (
            "- A pré-classificação não descarta nada: é um rótulo provisório. Proxy x86: "
            "MobileNetV2 INT8 no quadro inteiro acerta 13 de 18 imagens (72%).\n"
            "- ESP-DL publica MobileNetV2 224 px a 353–423 ms no ESP32-S3; a 128 px estima-se "
            "~1/3 disso (proporção de pixels), não medido.\n"
            "- Corrente da placa em sono profundo não documentada (o chip: 7 µA; a OV2640 em "
            "standby: 600 µA).\n"
            "- Conferir: ordem de bytes do RGB565 e assinatura de `enable_letterbox`.\n"
            + LIMITES_DISPOSITIVO
        ),
    }


def _openmv(ctx) -> dict[str, str]:
    return {
        "disparo": "PIR (IRQ, lightsleep) ou diferença de quadros → 3 fotos QVGA → "
                   "ShuffleNetV2 INT8 (quadro inteiro) → JPEG + CSV em /flash",
        "preparar": (
            "### 1. Preparar o modelo\n\n```bash\n./preparar_modelo.sh\npip install onnx2tf "
            "tensorflow onnx onnxsim     # ambiente separado\npython converter_tflite.py   # → "
            "modelos/classificador.tflite + classificador.txt\n```\n\nNo OpenMV IDE: Tools → "
            "ROM File System → copie o `.tflite` e o `.txt` para `/rom`."
        ),
        "instalar": (
            "### 2. Instalar\n\nFirmware OpenMV v5.0.1 na Nicla Vision; copie `main.py` para "
            "`/flash`. Ajuste `PINO_PIR` (None = gatilho por diferença de quadros)."
        ),
        "executar": "",
        "exportar": IMPORTAR_CSV.format(id=ctx.id),
        "limitacoes": (
            "- Sem microSD: fotos e CSV vão para os 11 MB de `/flash` (FAT, sensível a queda de "
            "energia durante a escrita).\n"
            "- Sono: 374 µA no datasheet, mas o pino de power-down do GC2145 não está ligado e o "
            "consumo sobe para ~20 mA depois de usar a câmera (fórum OpenMV).\n"
            "- Proxy x86: ShuffleNetV2 INT8 no quadro inteiro acerta 11 de 18 imagens (61%). O "
            "channel shuffle usa Reshape de 5 dimensões; confirme que o TFLite roda no OpenMV.\n"
            + LIMITES_DISPOSITIVO
        ),
    }


def _luckfox(ctx) -> dict[str, str]:
    return {
        "disparo": _disparo(ctx, "YOLOv8n INT8 (320 px, cabeça dividida) na NPU do RV1106"),
        "preparar": (
            "### 1. Preparar o modelo\n\n```bash\n./preparar_modelo.sh           # ONNX com "
            "cabeça dividida, 320 px\npip install rknn-toolkit2==2.5.0  # ambiente separado\n"
            "python converter_rknn.py       # target rv1106, INT8 (o RV1106 só executa inteiro)\n"
            "```"
        ),
        "instalar": (
            "### 2. Compilar e instalar\n\n```bash\nexport LUCKFOX_SDK_PATH=... RKNN_TOOLKIT2=... "
            "OPENCV_MOBILE=...\n./compilar.sh\nadb push instalar /userdata/armadilha   # ou scp\n"
            "# na placa:\nRkLunch-stop.sh && cd /userdata/armadilha && ./armadilha "
            "modelos/a_320_split_rv1106.rknn\n```\n\nUse uma Luckfox Pico Max/Ultra (RV1106G3, "
            "256 MB, 1 TOPS): Mini/Plus (RV1103, 64 MB) não comportam o exemplo YOLO."
        ),
        "ligacoes": (
            "### 3. Ligações\n\nPIR → GPIO1_C7 (sysfs 55; ajuste `GPIO_PIR`). Câmera SC3336 no "
            "conector MIPI."
        ),
        "executar": "",
        "exportar": IMPORTAR_CSV.format(id=ctx.id),
        "limitacoes": (
            "- O rknn_model_zoo traz o exemplo `yolov8_rv1106_1103.cc`, mas não publica FPS "
            "para RV1106; consumo da placa também não documentado.\n"
            "- Proxy x86 (INT8 em todos os operadores, 320 px): F1 0,824.\n"
            "- C++ verificado só por análise de sintaxe com cabeçalhos simulados.\n"
            + LIMITES_DISPOSITIVO
        ),
    }


LIMITES_DISPOSITIVO = (
    "- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são "
    "estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).\n"
    "- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de "
    "armadilha fotográfica).\n"
)


def registrar(destino: dict[str, Any], especificacoes: dict[str, Any]) -> None:
    destino.update({
        "ort": _ort, "hailo": _hailo, "imx500": _imx500, "rknn": _rknn, "tensorrt": _tensorrt,
        "lote": _lote, "maixpy": _maixpy, "grove": _grove, "esp": _esp, "openmv": _openmv,
        "luckfox": _luckfox,
    })
