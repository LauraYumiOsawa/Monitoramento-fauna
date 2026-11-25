# 04b — Raspberry Pi 5 + AI Camera (IMX500) · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · IMX500 (detector de presença) + ONNX Runtime (classificador no Pi 5)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Raspberry Pi 5 + AI Camera (IMX500) × Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Divisão natural: o sensor localiza o animal; a CPU classifica o recorte em alta resolução do mesmo quadro.

Papel do hardware na pesquisa: IA no sensor versus IA no processador; não se presume comportamento NoIR [16].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Sensor Sony IMX500 de 12,3 MP com processamento neural no módulo [16] |
| Câmera | A própria AI Camera (foco manual, com filtro IR: noturno depende de luz visível) |
| Modelo | Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`) |
| Runtime / artefato | IMX500 (detector de presença) + ONNX Runtime (classificador no Pi 5) |
| Disparo | PIR → rajada de 8 quadros → rede no sensor IMX500 (INT8, NMS embutido) + MobileNetV2 na CPU do Pi 5 → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 3.065,22 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 4,500 | estimado | Pi 5 4 GB ocioso (Jeff Geerling, 2024); câmera parada entre eventos + câmera sempre ligada com a rede rodando (consumo não documentado; premissa) |
| Inferência por quadro (ms) | 63,0 | estimado | rede no sensor (~59 ms, em paralelo) + classificador na CPU ~5 ms |
| Energia por imagem (J) | 0,328 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 0,6 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,865 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `b_416_split_int8full+mnv2` |
| Recall de presença (proxy) | 90% | medido (proxy x86) | medicoes/presenca_x86.json, detector `b_416_split_int8full` |
| Acerto entre eventos confirmados (rajadas) | 100% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `b_416_split_int8full+mnv2` |
| Capacidade noturna (0–2) | 0 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 4/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 65.780,00 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| Raspberry Pi 5 4 GB | 1.574,90 | documentado | RoboCore, 20/10/2026 (esgotado); Orix R$ 1.349,90 e Alturion R$ 1.280,00 disponíveis |
| Raspberry Pi AI Camera (IMX500) | 967,90 | documentado | RoboCore, 20/10/2026 (disponível) |
| Active Cooler (Pi 5 em caixa fechada) | 49,90 | documentado | RoboCore, 20/10/2026 (esgotado) |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 9º; Melhor custo-benefício: 14º; Maior eficiência energética: 15º; Maior autonomia: 13º; Disparo mais rápido: 1º; Maior precisão: 4º; Melhor filtro de vazias: 2º; Melhor identificação de espécies: 1º; Melhor desempenho noturno: 3º; Maior robustez: 3º; Menor custo em dois anos: 14º; Mais fácil de montar: 2º; Melhor geral — Equilíbrio: 14º; Melhor geral — Longa duração: 17º; Melhor geral — Orçamento: 14º.

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/04_raspberry-pi-5_ai-camera-imx500/04b_modelo-b_imx500
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
./exportar_imx500.sh       # Ultralytics format="imx" → packerOut.zip (Java 17+)
./empacotar.sh            # gera versoes/dist/04b.tar.gz
```
O export só aceita YOLOv8n e YOLO11n e força INT8 (Model Compression Toolkit, NMS embutido).

### 2. Instalar no dispositivo

```bash
scp versoes/dist/04b.tar.gz pi@<ip>:/tmp/
ssh pi@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/04b.tar.gz
sudo chown -R $USER /opt/armadilha/04b && cd /opt/armadilha/04b
./instalar.sh
```

No Pi 5: `sudo apt install imx500-all imx500-tools` e, na pasta da versão, `./empacotar_rpk.sh` (o `imx500-package` só roda no Pi) para gerar o `network.rpk`.

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
sudo cp armadilha.service /etc/systemd/system/armadilha-04b.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-04b
```

`--simular` não se aplica: o resultado da rede chega como metadado do sensor. Para bancada, use as versões 01a/01b com o mesmo modelo em ONNX.

### 5. Exportar os registros

```bash
./executar.sh export-camtrapdp   # dados/camtrap-dp/{deployments,media,observations}.csv
```

Antes do campo, preencha `implantacao` em `versao.yaml` (coordenadas, local, altura).

## Limitações e o que não foi validado

- Nada aqui foi executado no hardware-alvo: a pesquisa é teórica e não há aquisição de equipamento. O código foi testado em x86_64 (testes unitários e simulação de câmera com `--simular`).
- A qualidade informada é proxy: split `test` do projeto (18 imagens, 20 caixas; uma caixa = 5 pontos de recall), imagens que não são de armadilha fotográfica (o animal ocupa ~47% do quadro) e 14 delas geradas por IA (`docs/MODELO_B.md`, seção 5).
- Consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`); não são medição elétrica.
- **Sem visão noturna com IR**: a AI Camera tem filtro IR e não existe versão NoIR (documentação oficial). Para fauna noturna, é a principal limitação desta versão.
- O firmware da rede pode levar minutos para carregar na primeira vez (fica em cache): a câmera fica sempre ligada e a arquitetura B não se aplica.
- Entrada máxima 640×640 (Raspberry Pi; a Sony cita 640×480) e 8 MB de memória no sensor. Inferência publicada pelo Ultralytics: YOLOv8n 58,8 ms (640, INT8).

## Arquivos desta pasta

- `README.md`
- `armadilha.service`
- `config/runtime_b.yaml`
- `empacotar.sh`
- `empacotar_rpk.sh`
- `executar.sh`
- `exportar_imx500.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `versao.yaml`
