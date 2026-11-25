# 03b — Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · HailoRT (detector HEF) + ONNX Runtime (classificador)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) × Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Como 02b, compilado para hailo8.

Papel do hardware na pesquisa: IA em acelerador PCIe (perfil 'Visão acelerada em Linux').

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Acelerador PCIe Hailo-8, 26 TOPS [14] |
| Câmera | Camera Module 3 NoIR [26] |
| Modelo | Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`) |
| Runtime / artefato | HailoRT (detector HEF) + ONNX Runtime (classificador) |
| Disparo | PIR → rajada de 8 quadros → HEF no hailo8 (NMS no HailoRT) + MobileNetV2 na CPU (ONNX Runtime) → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 3.746,45 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 3,700 | estimado | Pi 5 4 GB ocioso (Jeff Geerling, 2024); câmera parada entre eventos + Hailo ocioso (premissa) |
| Inferência por quadro (ms) | 11,0 | estimado | detector 416 no Hailo-8 (~6 ms) + classificador na CPU ~5 ms |
| Energia por imagem (J) | 0,071 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 0,7 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,8 | estimado | partida do libcamera com a câmera parada (premissa) |
| Qualidade (proxy) | 0,865 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `b_416_split_int8full+mnv2` |
| Recall de presença (proxy) | 90% | medido (proxy x86) | medicoes/presenca_x86.json, detector `b_416_split_int8full` |
| Acerto entre eventos confirmados (rajadas) | 100% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `b_416_split_int8full+mnv2` |
| Capacidade noturna (0–2) | 2 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 4/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 55.300,12 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| Raspberry Pi 5 4 GB | 1.574,90 | documentado | RoboCore, 20/10/2026 (esgotado); Orix R$ 1.349,90 e Alturion R$ 1.280,00 disponíveis |
| AI HAT+ 26 TOPS (Hailo-8) | 1.399,90 | documentado | RoboCore, 20/10/2026 (esgotado) |
| Cabo de câmera 22→15 vias (Pi 5/Zero) | 15,00 | documentado | RoboCore, 20/10/2026 |
| Active Cooler (Pi 5 em caixa fechada) | 49,90 | documentado | RoboCore, 20/10/2026 (esgotado) |
| Iluminador IR 850 nm (36 LEDs, 12 V) | 14,40 | documentado | SatFrota, 21/10/2026 |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Camera Module 3 NoIR (importação, pedido raspberrypi) | 219,83 | estimado | US$ 25,00 + frete US$ 10, II US$ 0,00, ICMS 17% por dentro, câmbio 5,2132; preço oficial Raspberry Pi; não encontrada à venda no Brasil (importação estimada) |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 11º; Melhor custo-benefício: 16º; Maior eficiência energética: 8º; Maior autonomia: 12º; Disparo mais rápido: 3º; Maior precisão: 4º; Melhor filtro de vazias: 2º; Melhor identificação de espécies: 1º; Melhor desempenho noturno: 1º; Maior robustez: 3º; Menor custo em dois anos: 12º; Mais fácil de montar: 2º; Melhor geral — Equilíbrio: 15º; Melhor geral — Longa duração: 14º; Melhor geral — Orçamento: 16º.

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/03_raspberry-pi-5_ai-hat-plus_hailo-8/03b_modelo-b_hailort
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
./compilar_hef.sh         # HEF hailo8, 416 px (Ultralytics format="hailo")
./empacotar.sh            # gera versoes/dist/03b.tar.gz
```
`compilar_hef.sh` exige Linux x86_64 com o Dataflow Compiler (DFC 3.x / HailoRT 4.x; wheel no Developer Zone da Hailo, registro gratuito). HEFs de gerações diferentes não são intercambiáveis. Sem GPU NVIDIA o otimizador cai para o nível 0.

### 2. Instalar no dispositivo

```bash
scp versoes/dist/03b.tar.gz pi@<ip>:/tmp/
ssh pi@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/03b.tar.gz
sudo chown -R $USER /opt/armadilha/03b && cd /opt/armadilha/03b
./instalar.sh
```

No Pi 5 (Raspberry Pi OS 64 bits): `sudo apt install dkms hailo-all` e `hailortcli fw-control identify`. Os pacotes `hailo-all` e `hailo-h10-all` não coexistem. O AI HAT+ ativa PCIe Gen 3 automaticamente.

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
sudo cp armadilha.service /etc/systemd/system/armadilha-03b.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-03b
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
- A Hailo recomenda ≥ 1024 imagens de calibração; o dataset atual tem bem menos. A qualidade informada usa como proxy o INT8 em todos os operadores com a cabeça dividida (x86), não o HEF.
- O Model Zoo mede vazão bruta do chip em host x86 (YOLOv8n: Hailo-8L 201 FPS, Hailo-8 1036 FPS); no Pi 5 o fórum oficial relata 61,9 FPS (8L) — o enlace PCIe de uma via limita.
- Os limiares de NMS do `versao.yaml` são aplicados em tempo de execução (`set_nms_score_threshold`/`set_nms_iou_threshold`).

## Arquivos desta pasta

- `README.md`
- `armadilha.service`
- `compilar_hef.sh`
- `config/runtime_b.yaml`
- `empacotar.sh`
- `executar.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `versao.yaml`
