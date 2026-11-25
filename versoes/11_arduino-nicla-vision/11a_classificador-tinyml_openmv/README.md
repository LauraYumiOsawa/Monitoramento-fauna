# 11a — Arduino Nicla Vision · TinyML (SDK do fabricante) · OpenMV + TFLite Micro (ShuffleNetV2-x0.5 INT8)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Arduino Nicla Vision × TinyML (SDK do fabricante) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Alternativa TinyML [25]: classificação do quadro com modelo de 0,35 M parâmetros, acionada por movimento.

Papel do hardware na pesquisa: Alternativa TinyML; não confundir com câmeras OpenMV de outras gerações [25].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Placa de visão com câmera e recursos integrados para aplicações embarcadas [25] |
| Câmera | GC2145 integrada (2 MP) |
| Modelo | Classificador TinyML do quadro inteiro (3 espécies + `desconhecido`), sem detector |
| Runtime / artefato | OpenMV + TFLite Micro (ShuffleNetV2-x0.5 INT8) |
| Disparo | PIR (IRQ, lightsleep) ou diferença de quadros → 3 fotos QVGA → ShuffleNetV2 INT8 (quadro inteiro) → JPEG + CSV em /flash |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 1.190,56 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,074 | documentado | ~20 mA em sono após usar a câmera (fórum OpenMV) × 3,7 V |
| Inferência por quadro (ms) | 200,0 | estimado | ShuffleNetV2-x0.5 128 (0,028 GFLOPs) no Cortex-M7 a 480 MHz (premissa) |
| Energia por imagem (J) | 0,078 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 35,1 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,3 | estimado | saída do lightsleep + câmera (premissa) |
| Qualidade (proxy) | 0,611 | medido (proxy x86) | acurácia do quadro inteiro (18 imagens); proxy `quadro_inteiro_shufflenet_int8` |
| Recall de presença (proxy) | 89% | medido (proxy x86) | medicoes/presenca_x86.json, detector `quadro_inteiro_shufflenet_int8` |
| Acerto entre eventos confirmados (rajadas) | não disponível nesta revisão | — | — |
| Capacidade noturna (0–2) | 1 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 3/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 2.231,34 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Arduino Nicla Vision (importação, pedido arduino) | 718,04 | estimado | US$ 80,20 + frete US$ 10, II US$ 24,12, ICMS 17% por dentro, câmbio 5,2132; Arduino Store US; MakerHero R$ 1.014,90 em resumo de busca (não verificado) |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 6º; Melhor custo-benefício: 9º; Maior eficiência energética: 9º; Maior autonomia: 3º; Disparo mais rápido: 2º; Maior precisão: 11º; Melhor filtro de vazias: 3º; Melhor desempenho noturno: 2º; Maior robustez: 1º; Menor custo em dois anos: 3º; Mais fácil de montar: 3º; Melhor geral — Equilíbrio: 11º; Melhor geral — Longa duração: 8º; Melhor geral — Orçamento: 11º.

## Passo a passo

### 1. Preparar o modelo

```bash
./preparar_modelo.sh
pip install onnx2tf tensorflow onnx onnxsim     # ambiente separado
python converter_tflite.py   # → modelos/classificador.tflite + classificador.txt
```

No OpenMV IDE: Tools → ROM File System → copie o `.tflite` e o `.txt` para `/rom`.

### 2. Instalar

Firmware OpenMV v5.0.1 na Nicla Vision; copie `main.py` para `/flash`. Ajuste `PINO_PIR` (None = gatilho por diferença de quadros).

### 3. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id 11a \
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c).

## Limitações e o que não foi validado

- Sem microSD: fotos e CSV vão para os 11 MB de `/flash` (FAT, sensível a queda de energia durante a escrita).
- Sono: 374 µA no datasheet, mas o pino de power-down do GC2145 não está ligado e o consumo sobe para ~20 mA depois de usar a câmera (fórum OpenMV).
- Proxy x86: ShuffleNetV2 INT8 no quadro inteiro acerta 11 de 18 imagens (61%). O channel shuffle usa Reshape de 5 dimensões; confirme que o TFLite roda no OpenMV.
- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).
- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de armadilha fotográfica).

## Arquivos desta pasta

- `README.md`
- `converter_tflite.py`
- `main.py`
- `preparar_modelo.sh`
- `versao.yaml`
