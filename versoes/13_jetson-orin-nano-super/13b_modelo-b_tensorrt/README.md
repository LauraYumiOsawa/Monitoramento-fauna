# 13b — NVIDIA Jetson Orin Nano Super · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · ONNX Runtime GPU + TensorRT EP (FP16)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** NVIDIA Jetson Orin Nano Super × Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Teto de desempenho do pipeline em dois estágios.

Papel do hardware na pesquisa: Teto de desempenho/estação base; modos de potência não são autonomia de armadilha [23].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | Até 67 TOPS e modos de potência de 7 a 25 W [23] |
| Câmera | IMX219 NoIR (CSI) via nvarguscamerasrc |
| Modelo | Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`) |
| Runtime / artefato | ONNX Runtime GPU + TensorRT EP (FP16) |
| Disparo | PIR → rajada de 8 quadros → ONNX Runtime GPU (TensorRT FP16 → CUDA → CPU, nesta ordem) → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 4.175,25 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 5,000 | documentado | 4,7 W ocioso (fórum NVIDIA, JetPack 6.2) |
| Inferência por quadro (ms) | 4,0 | estimado | detector 416 + classificador, TensorRT FP16 (escalado) |
| Energia por imagem (J) | 0,040 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 0,5 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,947 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `b_416+mnv2` |
| Recall de presença (proxy) | 90% | medido (proxy x86) | medicoes/presenca_x86.json, detector `b_416` |
| Acerto entre eventos confirmados (rajadas) | 100% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `b_416+mnv2` |
| Capacidade noturna (0–2) | 2 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 6/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 73.964,92 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| Jetson Orin Nano Super Dev Kit | 3.499,90 | documentado | RoboCore (esgotado); NVIDIA elevou o MSRP para US$ 399 em jul/2026 |
| Iluminador IR 850 nm (36 LEDs, 12 V) | 14,40 | documentado | SatFrota, 21/10/2026 |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Câmera IMX219 NoIR (CSI) (importação, pedido generica) | 188,43 | estimado | US$ 20,00 + frete US$ 10, II US$ 0,00, ICMS 17% por dentro, câmbio 5,2132; premissa |

**Fora dos rankings de armadilha completa:** referência de desempenho (Jetson), não candidata a armadilha a bateria [23].

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/13_jetson-orin-nano-super/13b_modelo-b_tensorrt
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
./empacotar.sh            # gera versoes/dist/13b.tar.gz
```

### 2. Instalar no dispositivo

```bash
scp versoes/dist/13b.tar.gz jetson@<ip>:/tmp/
ssh jetson@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/13b.tar.gz
sudo chown -R $USER /opt/armadilha/13b && cd /opt/armadilha/13b
./instalar.sh
```

JetPack 6.2.x. O `instalar.sh` usa `requisitos_jetson.txt` (onnxruntime-gpu do jetson-ai-lab). Modo de potência econômico da configuração Super: `sudo nvpmodel -m 0` (15 W; o modo de 7 W só existe na configuração não-Super). A primeira execução compila o motor TensorRT (minutos) e o guarda em `modelos/cache_trt`.

### 3. Executar

```bash
# bancada, sem sensor nem câmera: imagens/vídeo simulam os disparos
./executar.sh trap --simular /caminho/imagens_ou_video.mp4 --eventos 5
./executar.sh benchmark --fonte /caminho/video.mp4 --eventos 20
# campo
./executar.sh trap
sudo cp armadilha.service /etc/systemd/system/armadilha-13b.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-13b
```

### 4. Exportar os registros

```bash
./executar.sh export-camtrapdp   # dados/camtrap-dp/{deployments,media,observations}.csv
```

Antes do campo, preencha `implantacao` em `versao.yaml` (coordenadas, local, altura).

## Limitações e o que não foi validado

- Nada aqui foi executado no hardware-alvo: a pesquisa é teórica e não há aquisição de equipamento. O código foi testado em x86_64 (testes unitários e simulação de câmera com `--simular`).
- A qualidade informada é proxy: split `test` do projeto (18 imagens, 20 caixas; uma caixa = 5 pontos de recall), imagens que não são de armadilha fotográfica (o animal ocupa ~47% do quadro) e 14 delas geradas por IA (`docs/MODELO_B.md`, seção 5).
- Consumo, autonomia e tempo de disparo são estimativas com premissas explícitas (`versoes/ranking/candidatos.yaml`); não são medição elétrica.
- Não foi confirmado que as wheels do jetson-ai-lab expõem o `TensorrtExecutionProvider`; sem ele a versão roda em CUDA. Alternativa oficial: export Ultralytics `format="engine", quantize=16` no próprio Jetson.
- 15–25 W de modo de potência e ~5–7 W ocioso (terceiros): papel de estação base com energia da rede ou solar dimensionado, não de armadilha a bateria [23].

## Arquivos desta pasta

- `README.md`
- `armadilha.service`
- `config/runtime_b.yaml`
- `empacotar.sh`
- `executar.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `requisitos_jetson.txt`
- `versao.yaml`
