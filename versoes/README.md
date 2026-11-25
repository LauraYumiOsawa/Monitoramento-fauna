# Versões embarcadas: cruzamento hardware × software/modelo

Esta pasta responde a duas tarefas sobre a pesquisa teórica
(*Estudo teórico de sistemas de monitoramento de fauna com visão computacional embarcada*,
20/10/2026):

1. **Cruzar** os hardwares candidatos (seção 4) com os softwares e modelos (seção 6) e, onde
   o uso faz sentido, **gerar uma versão** do sistema para aquele par — uma pasta por versão;
2. **Completar e refazer a tabela de ranking** (seção 7), que na pesquisa ficou quase toda
   "indeterminada".

| Documento | Conteúdo |
|---|---|
| [`MATRIZ_COMPATIBILIDADE.md`](MATRIZ_COMPATIBILIDADE.md) | Todos os cruzamentos, com veredito e justificativa — inclusive os que não fazem sentido e por quê |
| [`RANKING.md`](RANKING.md) | Os 13 rankings refeitos (1º, 2º, 3º), tabela do esqueleto preenchida, melhor geral nos três perfis de peso, dados por candidato e sensibilidade |
| `NN_<hardware>/` | Uma pasta por hardware, com uma subpasta por versão gerada |

## As 25 versões

Arquitetura de energia (seção 3): **A** placa ligada aguardando o PIR; **C** captura
independente e classificação posterior. Todas as versões Linux aceitam também a
arquitetura B (desligar entre eventos) por configuração.

| Versão | Hardware | Modelo | Runtime | Arq. | Estado |
|---|---|---|---|---|---|
| [01a](01_raspberry-pi-5/01a_yolov8n-multiclasse_onnxruntime/README.md) | Raspberry Pi 5 | YOLOv8n multiclasse 640 | ONNX Runtime CPU | A | executada em x86 (simulação) |
| [01b](01_raspberry-pi-5/01b_modelo-b_onnxruntime/README.md) | Raspberry Pi 5 | Modelo B (presença 416 + MobileNetV2) | ONNX Runtime CPU | A | executada em x86 (simulação) |
| [01c](01_raspberry-pi-5/01c_megadetector-speciesnet_lote/README.md) | Raspberry Pi 5 8 GB | MegaDetector + SpeciesNet | PyTorch CPU, lote | C | SpeciesNet executado em x86 |
| [02a](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02a_yolov8n-multiclasse_hailort/README.md) | Pi 5 + AI HAT+ (Hailo-8L) | YOLOv8n multiclasse | HailoRT (HEF) | A | backend testado com SDK simulado |
| [02b](02_raspberry-pi-5_ai-hat-plus_hailo-8l/02b_modelo-b_hailort/README.md) | Pi 5 + AI HAT+ (Hailo-8L) | Modelo B | HailoRT + ONNX Runtime | A | backend testado com SDK simulado |
| [03a](03_raspberry-pi-5_ai-hat-plus_hailo-8/03a_yolov8n-multiclasse_hailort/README.md) | Pi 5 + AI HAT+ (Hailo-8) | YOLOv8n multiclasse | HailoRT (HEF) | A | backend testado com SDK simulado |
| [03b](03_raspberry-pi-5_ai-hat-plus_hailo-8/03b_modelo-b_hailort/README.md) | Pi 5 + AI HAT+ (Hailo-8) | Modelo B | HailoRT + ONNX Runtime | A | backend testado com SDK simulado |
| [04a](04_raspberry-pi-5_ai-camera-imx500/04a_yolov8n-multiclasse_imx500/README.md) | Pi 5 + AI Camera (IMX500) | YOLOv8n multiclasse | rede no sensor | A | backend testado com SDK simulado |
| [04b](04_raspberry-pi-5_ai-camera-imx500/04b_modelo-b_imx500/README.md) | Pi 5 + AI Camera (IMX500) | Modelo B | IMX500 + ONNX Runtime | A | backend testado com SDK simulado |
| [05a](05_raspberry-pi-zero-2-w/05a_yolov8n-512-int8_onnxruntime/README.md) | Raspberry Pi Zero 2 W | YOLOv8n 512 INT8 | ONNX Runtime CPU | A | executada em x86 (simulação) |
| [05b](05_raspberry-pi-zero-2-w/05b_modelo-b-int8_onnxruntime/README.md) | Raspberry Pi Zero 2 W | Modelo B INT8 | ONNX Runtime CPU | A | executada em x86 (simulação) |
| [06a](06_sipeed-maixcam/06a_yolov8n-multiclasse_maixpy/README.md) | Sipeed MaixCAM | YOLOv8n 224×320 | MaixPy (cvimodel) | A | lógica testada com `maix` simulado |
| [06b](06_sipeed-maixcam/06b_modelo-b_maixpy/README.md) | Sipeed MaixCAM | Modelo B 224×320 | MaixPy (cvimodel) | A | lógica testada com `maix` simulado |
| [07a](07_grove-vision-ai-v2/07a_yolov8n-reduzido_sscma/README.md) | Grove Vision AI V2 + XIAO ESP32S3 | YOLOv8n 256 INT8 | TFLite + Vela (SSCMA) | A* | firmware não compilado |
| [08a](08_orange-pi-5/08a_yolov8n-multiclasse_rknn/README.md) | Orange Pi 5 | YOLOv8n (cabeça dividida) | RKNN INT8 | A | backend testado com SDK simulado |
| [08b](08_orange-pi-5/08b_modelo-b_rknn/README.md) | Orange Pi 5 | Modelo B | RKNN INT8 | A | backend testado com SDK simulado |
| [08c](08_orange-pi-5/08c_megadetector-speciesnet_lote/README.md) | Orange Pi 5 | MegaDetector + SpeciesNet | PyTorch CPU, lote | C | como 01c |
| [09a](09_esp32-s3-eye/09a_captura-independente_tinyml-esp-dl/README.md) | ESP32-S3-EYE | MobileNetV2 128 INT8 (quadro inteiro) | ESP-IDF + ESP-DL | C | firmware não compilado |
| [10a](10_luckfox-pico-rv1106/10a_yolov8n-320-int8_rknn-rv1106/README.md) | Luckfox Pico Max (RV1106) | YOLOv8n 320 INT8 | RKNN C API | A | C++ verificado só na sintaxe |
| [11a](11_arduino-nicla-vision/11a_classificador-tinyml_openmv/README.md) | Arduino Nicla Vision | ShuffleNetV2-x0.5 INT8 | OpenMV (TFLite Micro) | A | script não executado |
| [12a](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12a_yolov8n-multiclasse_hailort5/README.md) | Pi 5 + AI HAT+ 2 (Hailo-10H) | YOLOv8n multiclasse | HailoRT 5 | A | backend testado com SDK simulado |
| [12b](12_raspberry-pi-5_ai-hat-plus-2_hailo-10h/12b_modelo-b_hailort5/README.md) | Pi 5 + AI HAT+ 2 (Hailo-10H) | Modelo B | HailoRT 5 + ONNX Runtime | A | backend testado com SDK simulado |
| [13a](13_jetson-orin-nano-super/13a_yolov8n-multiclasse_tensorrt/README.md) | Jetson Orin Nano Super | YOLOv8n multiclasse | ONNX Runtime GPU/TensorRT | A | executada em x86 (CPU) |
| [13b](13_jetson-orin-nano-super/13b_modelo-b_tensorrt/README.md) | Jetson Orin Nano Super | Modelo B | ONNX Runtime GPU/TensorRT | A | executada em x86 (CPU) |
| [13c](13_jetson-orin-nano-super/13c_megadetector-speciesnet_gpu/README.md) | Jetson Orin Nano Super | MegaDetector + SpeciesNet | PyTorch CUDA | C | como 01c |

\* No 07a o microcontrolador hospedeiro dorme e só liga o módulo de IA no disparo.

Não geradas (detalhes na matriz): Coral USB e OpenMV (contexto da pesquisa; o PyCoral parou
no Python 3.9), MegaDetector/SpeciesNet em Hailo (sem artefato documentado), Modelo B no
Grove e no Luckfox (condicionais), YOLOv8n em ESP32-S3 e Nicla (não cabe), qualquer modelo
grande em placas de 512 MB ou menos.

## Resultado do ranking em uma linha por perfil

Melhor geral (pesos da seção 7): **Equilíbrio** → 06b MaixCAM · Modelo B, praticamente
empatada com 05a Pi Zero 2 W (diferença 0,003, menor que a incerteza das premissas);
**Longa duração** → 09a ESP32-S3-EYE; **Orçamento** → 05a Pi Zero 2 W. Os vencedores não
mudam com 10 ou 1000 eventos/dia nem com o preço do Orange Pi 5 entre US$ 60 e 270. Na
identificação de espécies, nenhuma variante do Modelo B confirmou espécie errada em 90
rajadas simuladas; todas as do Modelo A confirmaram (8–16% dos eventos). Detalhes,
confiança de cada ranking e o que **não** se pode concluir: [`RANKING.md`](RANKING.md).

## Como as versões são feitas

```text
versoes/
├── matriz.yaml                 # cruzamentos, veredito e justificativa (fonte única)
├── gerador/
│   ├── especificacoes.yaml     # como cada versão é configurada
│   ├── gerar_versoes.py        # gera as pastas, a matriz e os READMEs
│   ├── secoes_familias.py      # textos de README por família de runtime
│   └── plataformas/            # arquivos de cada família (scripts, firmware, apps)
├── comum/
│   ├── animal_monitor_hw/      # núcleo comum das versões Linux (ver abaixo)
│   ├── ferramentas/            # medições e preparação de modelos
│   └── tests/                  # 32 testes
└── ranking/
    ├── candidatos.yaml         # dados de cada candidato, com tipo e fonte
    ├── calcular_ranking.py     # rankings → RANKING.md e resultados/
    ├── medicoes/               # proxies medidos no projeto (JSON)
    └── fontes/                 # levantamento de preços, consumo e APIs (20–21/10/2026)
```

As pastas de versão são **geradas**: para mudar uma versão, edite `matriz.yaml`,
`gerador/especificacoes.yaml` ou `gerador/plataformas/` e rode
`python versoes/gerador/gerar_versoes.py`. Depois de mudar dados do ranking, rode antes
`python versoes/ranking/calcular_ranking.py` (os READMEs mostram os dados e as posições).

O núcleo `comum/animal_monitor_hw` reaproveita o Modelo A (`src/animal_monitor`) e o
Modelo B (`src/animal_monitor_b`) **sem alterá-los** e acrescenta só o que muda entre
placas: runners por runtime (ONNX Runtime/TensorRT, HailoRT, RKNN, IMX500), cabeça YOLO
dividida para NPUs INT8, ciclo da armadilha (PIR → rajada → análise → registro, arquiteturas
A/B/C), registro em Camtrap DP, modelo de autonomia da seção 7 e a CLI
`python -m animal_monitor_hw` (`trap`, `batch`, `benchmark`, `export-camtrapdp`,
`import-csv`, `autonomy`, `info`). As versões de microcontrolador e câmera inteligente
trazem o próprio firmware/app e gravam um `registros.csv` importável pelo `import-csv`.

## Medições feitas no projeto (proxies em x86_64)

Ferramentas em `comum/ferramentas/`; resultados em `ranking/medicoes/`.

- **Cabeça YOLO dividida** (6 saídas box/cls, DFL + NMS em NumPy): reproduz o export padrão
  nas 18 imagens de teste com diferença máxima de 0,0004 px nas caixas.
- **INT8 em todos os operadores**: com a cabeça inteira o detector quebra (achado original do
  projeto); com a cabeça dividida funciona (F1 0,800 contra 0,872 em FP32 no Modelo A; 0,842
  no Modelo B INT8 completo) — é o que viabiliza RKNN, RV1106, SG2002 e Hailo.
- **Resolução**: o YOLOv8n do projeto mantém F1 em 512 px, perde pouco em 224×320 e cai em
  entradas quadradas pequenas (0,727 em 256; 0,429 em 192).
- **Rajadas**: 90 eventos de 8 quadros por proxy, analisados pelo mesmo código das versões.
- **MegaDetector v5a + SpeciesNet v4.0.3a** (`--country BRA`): presença 20/20, espécie exata
  13/20 (F1 0,65) nas mesmas imagens.

Todos os números de qualidade usam o split `test` do projeto (18 imagens, 20 caixas; não são
capturas de armadilha fotográfica e 14 imagens do dataset são geradas por IA): servem para
comparar variantes entre si, não para prever o desempenho de campo.

## Reproduzir

```bash
# na raiz do projeto, com o ambiente de treino (ultralytics, torch, onnx, onnxruntime)
python versoes/comum/ferramentas/medir_proxies.py     # exporta variantes, quantiza, mede
python versoes/comum/ferramentas/simular_rajadas.py
python versoes/comum/ferramentas/medir_presenca.py
python versoes/comum/ferramentas/avaliar_speciesnet.py <predicoes_speciesnet.json>
python versoes/ranking/calcular_ranking.py
python versoes/gerador/gerar_versoes.py
PYTHONPATH=src:versoes/comum python -m unittest discover -s versoes/comum/tests
```

**Atenção ao ambiente:** o `.venv` do projeto foi criado com Python 3.12, que não existe
mais neste sistema (o `python3` agora é 3.14), então `.venv/bin/python` não encontra os
pacotes. As medições deste trabalho rodaram com um Python 3.12 isolado
(`uv python install 3.12`) apontando `PYTHONPATH` para `.venv/lib/python3.12/site-packages`.
Para voltar a usar o projeto, recrie o ambiente com um Python 3.12 (por exemplo
`uv venv --python 3.12 .venv && uv pip install -e ".[train,modelo-b]"`).
