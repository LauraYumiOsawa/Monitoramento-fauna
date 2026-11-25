# 04a — Raspberry Pi 5 + AI Camera (IMX500) · YOLOv8n multiclasse (Modelo A) · IMX500 (rede no sensor, INT8) via Picamera2

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Raspberry Pi 5 + AI Camera (IMX500) × YOLOv8n multiclasse (Modelo A) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

O Ultralytics exporta YOLOv8n para o IMX500; a inferência sai do processador hospedeiro [16]. Compara IA no sensor com IA no processador.

Papel do hardware na pesquisa: IA no sensor versus IA no processador; não se presume comportamento NoIR [16].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Sensor Sony IMX500 de 12,3 MP com processamento neural no módulo [16] |
| Câmera | A própria AI Camera (foco manual, com filtro IR: noturno depende de luz visível) |
| Modelo | YOLOv8n multiclasse (Modelo A): anta, tamanduá-bandeira, lobo-guará |
| Runtime / artefato | IMX500 (rede no sensor, INT8) via Picamera2 |
| Disparo | PIR → rajada de 8 quadros → rede no sensor IMX500 (INT8, NMS embutido) → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 3.065,22 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 4,500 | estimado | Pi 5 4 GB ocioso (Jeff Geerling, 2024); câmera parada entre eventos + câmera sempre ligada com a rede rodando (consumo não documentado; premissa) |
| Inferência por quadro (ms) | 58,8 | documentado | YOLOv8n IMX500 640 INT8 (Ultralytics), no sensor, em paralelo à captura |
| Energia por imagem (J) | 0,306 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 0,6 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,800 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `a_640_split_int8full` |
| Recall de presença (proxy) | 90% | medido (proxy x86) | medicoes/presenca_x86.json, detector `a_640_split_int8full` |
| Acerto entre eventos confirmados (rajadas) | 84% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `a_640_split_int8full` |
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

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 9º; Melhor custo-benefício: 17º; Maior eficiência energética: 14º; Maior autonomia: 13º; Disparo mais rápido: 1º; Maior precisão: 8º; Melhor filtro de vazias: 2º; Melhor identificação de espécies: 4º; Melhor desempenho noturno: 3º; Maior robustez: 3º; Menor custo em dois anos: 14º; Mais fácil de montar: 2º; Melhor geral — Equilíbrio: 18º; Melhor geral — Longa duração: 20º; Melhor geral — Orçamento: 17º.

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/04_raspberry-pi-5_ai-camera-imx500/04a_yolov8n-multiclasse_imx500
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
./exportar_imx500.sh       # Ultralytics format="imx" → packerOut.zip (Java 17+)
./empacotar.sh            # gera versoes/dist/04a.tar.gz
```
O export só aceita YOLOv8n e YOLO11n e força INT8 (Model Compression Toolkit, NMS embutido).

### 2. Instalar no dispositivo

```bash
scp versoes/dist/04a.tar.gz pi@<ip>:/tmp/
ssh pi@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/04a.tar.gz
sudo chown -R $USER /opt/armadilha/04a && cd /opt/armadilha/04a
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
sudo cp armadilha.service /etc/systemd/system/armadilha-04a.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-04a
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
- `empacotar.sh`
- `empacotar_rpk.sh`
- `executar.sh`
- `exportar_imx500.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `versao.yaml`
