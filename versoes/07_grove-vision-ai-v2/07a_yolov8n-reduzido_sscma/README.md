# 07a — Seeed Grove Vision AI Module V2 · YOLOv8n multiclasse (Modelo A) · TFLite INT8 + Vela (Ethos-U55), firmware SSCMA; hospedeiro XIAO ESP32S3

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Seeed Grove Vision AI Module V2 × YOLOv8n multiclasse (Modelo A) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Detector YOLOv8n em resolução reduzida é o caso de uso documentado do módulo; câmera, controlador e firmware entram no sistema [18].

Papel do hardware na pesquisa: Exemplo TinyML com aceleração; câmera, controlador e firmware compõem o sistema [18].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | WiseEye2 HX6538: 2× Cortex-M55 + Ethos-U55; câmera CSI separada [18] |
| Câmera | OV5647 (CSI) — existe variante sem filtro IR |
| Modelo | YOLOv8n multiclasse (Modelo A): anta, tamanduá-bandeira, lobo-guará |
| Runtime / artefato | TFLite INT8 + Vela (Ethos-U55), firmware SSCMA; hospedeiro XIAO ESP32S3 |
| Disparo | PIR acorda o XIAO → liga o Grove → AT+INVOKE em rajada (YOLOv8n INT8 no Ethos-U55) → porta de 3 invocações → JPEG + CSV no microSD do XIAO |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 900,19 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,001 | estimado | XIAO em sono profundo + PIR; Grove sem alimentação (premissa) |
| Inferência por quadro (ms) | 71,0 | estimado | YOLOv8 192 px a ~25 FPS (Core Electronics) × (256/192)² |
| Energia por imagem (J) | 0,046 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 453,2 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 1,6 | estimado | despertar do XIAO + partida do Grove (não documentada; premissa) |
| Qualidade (proxy) | 0,647 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `a_256_int8` |
| Recall de presença (proxy) | 60% | medido (proxy x86) | medicoes/presenca_x86.json, detector `a_256_int8` |
| Acerto entre eventos confirmados (rajadas) | 69% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `a_256_int8` |
| Capacidade noturna (0–2) | 2 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 1/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 980,73 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| XIAO ESP32S3 Sense | 169,00 | documentado | RoboCore, 21/10/2026 (esgotado) |
| Iluminador IR 850 nm (36 LEDs, 12 V) | 14,40 | documentado | SatFrota, 21/10/2026 |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Grove Vision AI Module V2 + Câmera OV5647-160 IR (importação, pedido seeed) | 244,27 | estimado | US$ 28,89 + frete US$ 10, II US$ 0,00, ICMS 17% por dentro, câmbio 5,2132; Seeed Studio, out/2026; Seeed Studio, out/2026 |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 3º; Melhor custo-benefício: 7º; Maior eficiência energética: 6º; Maior autonomia: 2º; Disparo mais rápido: 5º; Maior precisão: 10º; Melhor filtro de vazias: 7º; Melhor identificação de espécies: 6º; Melhor desempenho noturno: 1º; Maior robustez: 1º; Menor custo em dois anos: 2º; Mais fácil de montar: 5º; Melhor geral — Equilíbrio: 12º; Melhor geral — Longa duração: 2º; Melhor geral — Orçamento: 7º.

## Passo a passo

### 1. Preparar o modelo

```bash
pip install ethos-u-vela
./exportar_tflite.sh    # Ultralytics saved_model INT8 a 256 px → Vela ethos-u55-64
```

Envie o `*_vela.tflite` pelo SenseCraft AI (Chrome/Edge: Models → My Models → Add Model), com as classes anta, tamandua_bandeira, lobo_guara.

### 2. Firmware do hospedeiro

Arduino IDE, pacote esp32 da Espressif, placa XIAO_ESP32S3 (OPI PSRAM habilitada), biblioteca Seeed_Arduino_SSCMA. Carregue `host_xiao/host_xiao.ino`.

### 3. Ligações

XIAO ESP32S3 Sense montado no conector do Grove (I2C 0x62). PIR → D0 (GPIO1, RTC). D1 comanda o MOSFET que alimenta o Grove só durante o evento. microSD na placa Sense (CS 21).

### 4. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id 07a \
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c).

## Limitações e o que não foi validado

- O módulo consome ~0,35 W rodando a ~25 FPS (YOLOv8 192 px, medição da Core Electronics) e quase o mesmo parado; por isso a alimentação é cortada entre eventos. O tempo de partida após religar não está documentado.
- Resoluções documentadas: 192, 224 e 256. No proxy x86 o YOLOv8n do projeto cai de F1 0,872 (640) para 0,647 (256, INT8): o animal fica pequeno demais.
- Não está documentado se `x`/`y` das caixas são centro ou canto; o firmware só usa classe e escore.
- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).
- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de armadilha fotográfica).

## Arquivos desta pasta

- `README.md`
- `exportar_tflite.sh`
- `host_xiao/host_xiao.ino`
- `vela_config.ini`
