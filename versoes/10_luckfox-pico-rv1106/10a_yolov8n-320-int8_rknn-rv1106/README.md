# 10a — Luckfox Pico (RV1106) · YOLOv8n multiclasse (Modelo A) · RKNN C API (RV1106, INT8, 320 px)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Luckfox Pico (RV1106) × YOLOv8n multiclasse (Modelo A) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

A NPU do RV1106 executa INT8; o modelo exato (RV1106G3, 256 MB) é caracterizado antes de comparar [24].

Papel do hardware na pesquisa: Linux + NPU em baixo consumo; exige caracterizar o modelo exato (aqui RV1106G3) [24].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Família RV1103/RV1106 com CPU, ISP e NPU; variantes não equivalentes [24] |
| Câmera | SC3336 (MIPI) da Luckfox |
| Modelo | YOLOv8n multiclasse (Modelo A): anta, tamanduá-bandeira, lobo-guará |
| Runtime / artefato | RKNN C API (RV1106, INT8, 320 px) |
| Disparo | PIR → rajada de 8 quadros → YOLOv8n INT8 (320 px, cabeça dividida) na NPU do RV1106 → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 866,76 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,600 | estimado | consumo não documentado; Linux + ISP com câmera aberta (premissa) |
| Inferência por quadro (ms) | 20,0 | estimado | YOLOv5s 640 ~100 ms no RV1106 (fórum) escalado para YOLOv8n 320 (2,02 GFLOPs) + DFL na CPU |
| Energia por imagem (J) | 0,016 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 4,4 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,824 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `a_320_split_int8full` |
| Recall de presença (proxy) | 70% | medido (proxy x86) | medicoes/presenca_x86.json, detector `a_320_split_int8full` |
| Acerto entre eventos confirmados (rajadas) | 86% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `a_320_split_int8full` |
| Capacidade noturna (0–2) | 1 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 3/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 9.228,73 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Luckfox Pico Max (RV1106G3) + Câmera Luckfox SC3336 (importação, pedido luckfox) | 394,24 | estimado | US$ 47,98 + frete US$ 10, II US$ 4,79, ICMS 17% por dentro, câmbio 5,2132; Luckfox (faixa US$ 32,99–38,99 para Pro/Max); Luckfox |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 2º; Melhor custo-benefício: 2º; Maior eficiência energética: 3º; Maior autonomia: 4º; Disparo mais rápido: 1º; Maior precisão: 7º; Melhor filtro de vazias: 6º; Melhor identificação de espécies: 3º; Melhor desempenho noturno: 2º; Menor custo em dois anos: 4º; Mais fácil de montar: 3º; Melhor geral — Equilíbrio: 5º; Melhor geral — Longa duração: 7º; Melhor geral — Orçamento: 4º.

## Passo a passo

### 1. Preparar o modelo

```bash
./preparar_modelo.sh           # ONNX com cabeça dividida, 320 px
pip install rknn-toolkit2==2.5.0  # ambiente separado
python converter_rknn.py       # target rv1106, INT8 (o RV1106 só executa inteiro)
```

### 2. Compilar e instalar

```bash
export LUCKFOX_SDK_PATH=... RKNN_TOOLKIT2=... OPENCV_MOBILE=...
./compilar.sh
adb push instalar /userdata/armadilha   # ou scp
# na placa:
RkLunch-stop.sh && cd /userdata/armadilha && ./armadilha modelos/a_320_split_rv1106.rknn
```

Use uma Luckfox Pico Max/Ultra (RV1106G3, 256 MB, 1 TOPS): Mini/Plus (RV1103, 64 MB) não comportam o exemplo YOLO.

### 3. Ligações

PIR → GPIO1_C7 (sysfs 55; ajuste `GPIO_PIR`). Câmera SC3336 no conector MIPI.

### 4. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id 10a \
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c).

## Limitações e o que não foi validado

- O rknn_model_zoo traz o exemplo `yolov8_rv1106_1103.cc`, mas não publica FPS para RV1106; consumo da placa também não documentado.
- Proxy x86 (INT8 em todos os operadores, 320 px): F1 0,824.
- C++ verificado só por análise de sintaxe com cabeçalhos simulados.
- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).
- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de armadilha fotográfica).

## Arquivos desta pasta

- `README.md`
- `app/CMakeLists.txt`
- `app/src/main.cc`
- `compilar.sh`
- `converter_rknn.py`
- `preparar_modelo.sh`
- `versao.yaml`
