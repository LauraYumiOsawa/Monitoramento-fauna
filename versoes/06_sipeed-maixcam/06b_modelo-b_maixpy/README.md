# 06b — Sipeed MaixCAM · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · MaixPy nn.YOLOv8 + nn.Classifier (cvimodel INT8)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Sipeed MaixCAM × Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Os dois estágios cabem na NPU; INT8 do MobileNetV2 medido sem perda no projeto.

Papel do hardware na pesquisa: Câmera com ambiente próprio; exige fluxo de conversão próprio (cvimodel) [17].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | SoC SG2002 com NPU INT8; ecossistemas MaixPy e MaixCDK [17] |
| Câmera | Câmera do kit (GC4653); trocar por módulo sem filtro IR para uso noturno |
| Modelo | Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`) |
| Runtime / artefato | MaixPy nn.YOLOv8 + nn.Classifier (cvimodel INT8) |
| Disparo | PIR → rajada de 8 quadros → nn.YOLOv8 na NPU (cvimodel INT8, 320×224) + nn.Classifier no recorte (votação na rajada) → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 1.097,10 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,700 | estimado | chip SG2002 ~0,5 W típicos com vídeo e IA (datasheet) + placa; sem sono documentado |
| Inferência por quadro (ms) | 15,0 | estimado | detector 224×320 (~12 ms) + classificador na NPU (~3 ms) |
| Energia por imagem (J) | 0,015 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 3,7 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,889 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `b_224x320_int8+mnv2_int8` |
| Recall de presença (proxy) | 95% | medido (proxy x86) | medicoes/presenca_x86.json, detector `b_224x320_int8` |
| Acerto entre eventos confirmados (rajadas) | 100% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `b_224x320_int8+mnv2_int8` |
| Capacidade noturna (0–2) | 1 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 3/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 10.853,85 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Sipeed MaixCAM (kit com câmera) (importação, pedido sipeed) | 624,58 | estimado | US$ 70,90 + frete US$ 10, II US$ 18,54, ICMS 17% por dentro, câmbio 5,2132; Spotpear (distribuidor), out/2026; sem loja brasileira |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 5º; Melhor custo-benefício: 5º; Maior eficiência energética: 2º; Maior autonomia: 7º; Disparo mais rápido: 1º; Maior precisão: 2º; Melhor filtro de vazias: 1º; Melhor identificação de espécies: 1º; Melhor desempenho noturno: 2º; Maior robustez: 2º; Menor custo em dois anos: 7º; Mais fácil de montar: 3º; Melhor geral — Equilíbrio: 1º; Melhor geral — Longa duração: 4º; Melhor geral — Orçamento: 3º.

## Passo a passo

### 1. Preparar o modelo (x86_64 com Docker)

```bash
./preparar_modelo.sh       # exporta o ONNX 224×320
./converter_cvimodel.sh    # TPU-MLIR: model_transform → run_calibration → model_deploy --processor cv181x
```

Os nomes de saída (`/model.22/dfl/conv/Conv_output_0`, `/model.22/Sigmoid_output_0`) foram conferidos nos ONNX deste projeto.

### 2. Instalar na MaixCAM

```bash
scp -r app modelos root@<ip>:/root/armadilha/
ssh root@<ip> python /root/armadilha/app/main.py
```

Ou instale `app/` como aplicativo pelo MaixVision (`app.yaml`).

### 3. Ligações

PIR (3,3 V) no pino `A19` (ajuste `PINO_PIR`; evite A14, A17 e o botão). O MaixPy não documenta interrupção de GPIO: o PIR é lido por polling a cada 50 ms.

### 4. Exportar os registros (Camtrap DP)

Copie `registros.csv` e a pasta de mídias do dispositivo e, na máquina de desenvolvimento:

```bash
PYTHONPATH=src:versoes/comum python -m animal_monitor_hw import-csv \
  --csv /caminho/registros.csv --midias /caminho/pasta_do_cartao --versao-id 06b \
  --latitude <lat> --longitude <lon>
```

Eventos sem espécie confirmada entram como `unknown`; sem rótulo, como `unclassified` —
classifique-os depois em lote (versões 01c, 08c ou 13c).

## Limitações e o que não foi validado

- A MaixCAM não tem gerenciamento de energia nem API de sono documentada: fica ligada (o chip SG2002 consome ~500 mW típicos com vídeo e IA, datasheet).
- Modelo B sem rastreador: a votação usa o maior animal de cada quadro da rajada (simplificação para um animal por evento).
- Desempenho publicado pela Sipeed: YOLOv8n 640×640 a 23 FPS; YOLO11n 320×224 a 120 FPS (condição não informada).
- A câmera GC4653 do kit não tem informação de IR-cut; para noite, troque por sensor sem filtro IR.
- Nada foi executado no hardware-alvo; consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`).
- Qualidade = proxy no split de teste do projeto (18 imagens; não são capturas de armadilha fotográfica).

## Arquivos desta pasta

- `README.md`
- `app/app.yaml`
- `app/main.py`
- `converter_cvimodel.sh`
- `modelos/classificador.mud`
- `modelos/detector.mud`
- `preparar_modelo.sh`
- `versao.yaml`
