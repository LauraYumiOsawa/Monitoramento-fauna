# 05a — Raspberry Pi Zero 2 W · YOLOv8n multiclasse (Modelo A) · ONNX Runtime (CPU, INT8, entrada 512)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Raspberry Pi Zero 2 W × YOLOv8n multiclasse (Modelo A) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Cabe em 512 MB com Raspberry Pi OS Lite; segundos por quadro são aceitáveis para fotos acionadas por PIR. Entrada 512 é a alternativa leve da pesquisa [P1; 13].

Papel do hardware na pesquisa: Contraponto de capacidade e preço de placa; memória limita modelos [13].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | 4× Cortex-A53 a 1 GHz, 512 MB de RAM [13] |
| Câmera | Camera Module 3 NoIR com cabo para Zero [26] |
| Modelo | YOLOv8n multiclasse (Modelo A): anta, tamanduá-bandeira, lobo-guará |
| Runtime / artefato | ONNX Runtime (CPU, INT8, entrada 512) |
| Disparo | PIR → rajada de 5 quadros → ONNX Runtime na CPU (4 threads) → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 921,65 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 0,600 | documentado | Zero 2 W ocioso, Pi OS Lite (CNX Software, 2021) |
| Inferência por quadro (ms) | 500,0 | estimado | Pi 5 (198,69 ms) × ~7 (A53 1 GHz vs A76 2,4 GHz) × 0,64 (512 px) × 0,53 (INT8, razão medida em x86) |
| Energia por imagem (J) | 1,150 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 4,3 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 1,2 | estimado | premissa |
| Qualidade (proxy) | 0,865 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `a_512_int8` |
| Recall de presença (proxy) | 85% | medido (proxy x86) | medicoes/presenca_x86.json, detector `a_512_int8` |
| Acerto entre eventos confirmados (rajadas) | 88% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `a_512_int8` |
| Capacidade noturna (0–2) | 2 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 6/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 9.360,96 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| Raspberry Pi Zero 2 W | 199,90 | documentado | RoboCore, 20/10/2026 (esgotado) |
| Cabo de câmera 22→15 vias (Pi 5/Zero) | 15,00 | documentado | RoboCore, 20/10/2026 |
| Iluminador IR 850 nm (36 LEDs, 12 V) | 14,40 | documentado | SatFrota, 21/10/2026 |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Camera Module 3 NoIR (importação, pedido raspberrypi) | 219,83 | estimado | US$ 25,00 + frete US$ 10, II US$ 0,00, ICMS 17% por dentro, câmbio 5,2132; preço oficial Raspberry Pi; não encontrada à venda no Brasil (importação estimada) |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 4º; Melhor custo-benefício: 1º; Maior eficiência energética: 18º; Maior autonomia: 6º; Disparo mais rápido: 4º; Maior precisão: 4º; Melhor filtro de vazias: 4º; Melhor identificação de espécies: 2º; Melhor desempenho noturno: 1º; Maior robustez: 3º; Menor custo em dois anos: 6º; Mais fácil de montar: 1º; Melhor geral — Equilíbrio: 2º; Melhor geral — Longa duração: 3º; Melhor geral — Orçamento: 1º.

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/05_raspberry-pi-zero-2-w/05a_yolov8n-512-int8_onnxruntime
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
./empacotar.sh            # gera versoes/dist/05a.tar.gz
```

### 2. Instalar no dispositivo

```bash
scp versoes/dist/05a.tar.gz pi@<ip>:/tmp/
ssh pi@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/05a.tar.gz
sudo chown -R $USER /opt/armadilha/05a && cd /opt/armadilha/05a
./instalar.sh
```

**Energia (Raspberry Pi).** Na arquitetura A a placa fica ligada: o Pi 5 consome ~3 W
ocioso (medições de terceiros) e o Zero 2 W ~0,5–0,6 W. Para a arquitetura B, desligue a
placa entre eventos e deixe o circuito do PIR religá-la: com `POWER_OFF_ON_HALT=1`
(`sudo rpi-eeprom-config -e`) o Pi 5 desligado cai para ~0,01 W (documentação oficial).
Ative `energia.desligar_ao_final: true` e `gatilho.tipo: nenhum` em `versao.yaml`: o
serviço faz um disparo por boot e desliga. O custo é o boot (~7 s até o SO no CM5,
whitepaper oficial; mais até o userland) antes da primeira foto.

### 3. Ligações elétricas

| Sinal | Pino físico | GPIO | Observação |
|---|---|---|---|
| PIR (saída digital) | 11 | GPIO17 | PIR de 3,3 V (ex.: AM312) ou HC-SR501 com saída de 3,3 V |
| Iluminador IR (comando) | 13 | GPIO27 | Via MOSFET/transistor; o GPIO não alimenta LEDs IR |
| Câmera | CSI | — | Pi 5: cabo de 22 para 15 vias; Zero 2 W: cabo de câmera do Zero |

`iluminacao_ir.modo: auto` usa o metadado `Lux` da câmera (limiar em `limiar_lux`). Use
`sempre`/`nunca` para fixar.

### 4. Executar

```bash
# bancada, sem sensor nem câmera: imagens/vídeo simulam os disparos
./executar.sh trap --simular /caminho/imagens_ou_video.mp4 --eventos 5
./executar.sh benchmark --fonte /caminho/video.mp4 --eventos 20
# campo
./executar.sh trap
sudo cp armadilha.service /etc/systemd/system/armadilha-05a.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-05a
```

### 5. Exportar os registros

```bash
./executar.sh export-camtrapdp   # dados/camtrap-dp/{deployments,media,observations}.csv
```

Antes do campo, preencha `implantacao` em `versao.yaml` (coordenadas, local, altura).

## Limitações e o que não foi validado

- Nada aqui foi executado no hardware-alvo: a pesquisa é teórica e não há aquisição de equipamento. O código foi testado em x86_64 (testes unitários e simulação de câmera com `--simular`).
- A qualidade informada é proxy: split `test` do projeto (18 imagens, 20 caixas; uma caixa = 5 pontos de recall), imagens que não são de armadilha fotográfica (o animal ocupa ~47% do quadro) e 14 delas geradas por IA (`docs/MODELO_B.md`, seção 5).
- Consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`); não são medição elétrica.
- 512 MB de RAM: use Raspberry Pi OS Lite 64 bits sem interface gráfica; a câmera em 1280×720 e 5 quadros por evento limitam a memória (confirme com `benchmark`).
- O Cortex-A53 a 1 GHz leva segundos por quadro: a análise acontece depois da rajada, nunca durante a captura.

## Arquivos desta pasta

- `README.md`
- `armadilha.service`
- `empacotar.sh`
- `executar.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `versao.yaml`
