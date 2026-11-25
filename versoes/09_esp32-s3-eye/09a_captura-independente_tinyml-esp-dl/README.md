# 09a — Espressif ESP32-S3-EYE · TinyML (SDK do fabricante) · ESP-IDF + ESP-DL (MobileNetV2-128 INT8, opcional)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Espressif ESP32-S3-EYE × TinyML (SDK do fabricante) — **faz sentido — gerado**. Arquitetura de energia: C — captura independente e classificação em lote.

## Por que este cruzamento faz sentido

Subsistema de captura da arquitetura C com sono profundo [19; 20]; a pré-classificação TinyML é opcional e não decide descarte.

Papel do hardware na pesquisa: Referência de microcontrolador com câmera; 'porteiro' da arquitetura C.

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | ESP32-S3 com câmera OV2640 de 2 MP e microSD [19]; sono profundo documentado [20] |
| Câmera | OV2640 integrada (até 1600 × 1200) [19] |
| Modelo | Classificador TinyML do quadro inteiro (3 espécies + `desconhecido`), sem detector |
| Runtime / artefato | ESP-IDF + ESP-DL (MobileNetV2-128 INT8, opcional) |
| Disparo | sono profundo → PIR (ext0) → 3 JPEG UXGA no microSD → MobileNetV2 INT8 no quadro inteiro (ESP-DL) → CSV → sono profundo |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 836,82 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,003 | estimado | ESP32-S3 7 µA + OV2640 em standby 600 µA + regulador e PIR (premissa) |
| Inferência por quadro (ms) | 170,0 | estimado | MobileNetV2 224 a 353,5 ms (ESP-DL) × (128/224)² + decodificação JPEG 1/8 (~50 ms, premissa) |
| Energia por imagem (J) | 0,085 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 597,5 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,8 | estimado | saída do sono + iniciar câmera (premissa) |
| Qualidade (proxy) | 0,722 | medido (proxy x86) | acurácia do quadro inteiro (18 imagens); proxy `quadro_inteiro_mnv2_int8` |
| Recall de presença (proxy) | 89% | medido (proxy x86) | medicoes/presenca_x86.json, detector `quadro_inteiro_mnv2_int8` |
| Acerto entre eventos confirmados (rajadas) | não disponível nesta revisão | — | — |
| Capacidade noturna (0–2) | 1 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 2/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 897,91 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| ESP32-S3-EYE (importação, pedido digikey) | 364,30 | estimado | US$ 45,00 + frete US$ 10, II US$ 3,00, ICMS 17% por dentro, câmbio 5,2132; DigiKey, out/2026 |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 1º; Melhor custo-benefício: 3º; Maior eficiência energética: 12º; Maior autonomia: 1º; Disparo mais rápido: 3º; Maior precisão: 9º; Melhor filtro de vazias: 3º; Melhor desempenho noturno: 2º; Maior robustez: 3º; Menor custo em dois anos: 1º; Mais fácil de montar: 4º; Melhor geral — Equilíbrio: 7º; Melhor geral — Longa duração: 1º; Melhor geral — Orçamento: 5º.

## Passo a passo

### 1. Preparar o modelo

```bash
./preparar_modelo.sh          # copia o MobileNetV2 ONNX
pip install esp-ppq          # ambiente separado (torch CPU)
python quantizar_espdl.py     # → modelos/classificador.espdl
```

### 2. Compilar e gravar (ESP-IDF ≥ 5.3)

```bash
cd firmware
idf.py set-target esp32s3
idf.py build flash monitor
```

Antes de instalar, grave a data em `/sdcard/relogio.txt` (ex.: `2026-11-02T08:30:00`): a placa não tem RTC com bateria; o horário sobrevive ao sono profundo, não à falta de energia.

### 3. Ligações

PIR → GPIO3 (o guia do S3-EYE diz que é o único livre; é pino de strapping e acende o LED verde — o PIR deve estar em nível baixo no reset). microSD da própria placa (SDMMC 1 bit).

### 4. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id 09a \
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c).

## Limitações e o que não foi validado

- A pré-classificação não descarta nada: é um rótulo provisório. Proxy x86: MobileNetV2 INT8 no quadro inteiro acerta 13 de 18 imagens (72%).
- ESP-DL publica MobileNetV2 224 px a 353–423 ms no ESP32-S3; a 128 px estima-se ~1/3 disso (proporção de pixels), não medido.
- Corrente da placa em sono profundo não documentada (o chip: 7 µA; a OV2640 em standby: 600 µA).
- Conferir: ordem de bytes do RGB565 e assinatura de `enable_letterbox`.
- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).
- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de armadilha fotográfica).

## Arquivos desta pasta

- `README.md`
- `firmware/CMakeLists.txt`
- `firmware/main/CMakeLists.txt`
- `firmware/main/idf_component.yml`
- `firmware/main/main.cpp`
- `firmware/partitions.csv`
- `firmware/sdkconfig.defaults`
- `preparar_modelo.sh`
- `quantizar_espdl.py`
- `versao.yaml`
