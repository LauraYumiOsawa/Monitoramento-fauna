# 08b — Orange Pi 5 · Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) · RKNN (detector e classificador, INT8)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Orange Pi 5 × Modelo B — YOLOv8n presença (416) + MobileNetV2 (128) — **faz sentido — gerado**. Arquitetura de energia: A — processador ativo: a placa fica ligada aguardando o PIR.

## Por que este cruzamento faz sentido

Os dois estágios na NPU; CPU livre para rastreamento e registro.

Papel do hardware na pesquisa: Contraponto com NPU no SoC; fluxo RKNN próprio de conversão e execução [22].

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | RK3588S, 4× Cortex-A76 + 4× Cortex-A55; NPU anunciada de 6 TOPS [21] |
| Câmera | Câmera USB/UVC ou módulo MIPI (OV13850/OV13855) via GStreamer |
| Modelo | Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`) |
| Runtime / artefato | RKNN (detector e classificador, INT8) |
| Disparo | PIR → rajada de 8 quadros → YOLOv8n INT8 na NPU do RK3588S e classificador na NPU → registro |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

**Alternativas na mesma versão:**

- Sem NPU: `runtime: onnxruntime` com `modelos/b_416_split.onnx` e `modelos/species_classifier.onnx`.

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo completo (R$) | R$ 1.589,20 | documentado + estimado (importação) | composição na tabela abaixo |
| Potência em espera (W) | 1,500 | estimado | 1,0 W ocioso (Jeff Geerling, Armbian) + câmera USB aberta ~0,5 W (premissa) |
| Inferência por quadro (ms) | 7,0 | estimado | detector 416 (~5,5 ms, escalado) + MobileNetV2 128 na NPU (~1 ms) |
| Energia por imagem (J) | 0,018 | estimado | potência de inferência × tempo de inferência |
| Autonomia, 74 Wh e 100 eventos/dia (dias) | 1,7 | estimado | equação da seção 7 (candidatos.yaml: cenario) |
| Tempo até a 1ª captura (s) | 0,1 | estimado | câmera sempre ligada: próximo quadro a 10 fps (premissa) |
| Qualidade (proxy) | 0,842 | medido (proxy x86) | F1@0,35 por espécie (18 imagens, 20 caixas); proxy `b_416_split_int8full+mnv2_int8` |
| Recall de presença (proxy) | 90% | medido (proxy x86) | medicoes/presenca_x86.json, detector `b_416_split_int8full` |
| Acerto entre eventos confirmados (rajadas) | 100% | medido (proxy x86) | medicoes/rajadas_x86.json, proxy `b_416_split_int8full+mnv2_int8` |
| Capacidade noturna (0–2) | 1 | documentado (rubrica) | rubrica 0–2 (RANKING.md, seção 7) |
| Integração/documentação (0–6) | 4/6 | rubrica | rubrica 0–6 (RANKING.md, seção 7) |
| TCO 2 anos (R$) | R$ 22.506,11 | estimado | custo + (730/D) visitas × R$ 50 |

Composição do custo:

| Item | R$ | Tipo | Fonte |
|---|---|---|---|
| Câmera Orange Pi OV13850 | 199,66 | documentado | Amazon.com.br via resumo de busca (página não abriu) — não verificado |
| PIR HC-SR501 | 8,90 | documentado | RoboCore, 21/10/2026 |
| Caixa IP66 Intelbras VBOX 1100 E | 23,72 | documentado | Kabum, 21/10/2026 |
| microSD 32 GB SanDisk | 126,00 | documentado | RoboCore, 21/10/2026 |
| Power bank 20 000 mAh (74 Wh) | 298,90 | documentado | Kabum, Xiaomi PB2020MI, 21/10/2026 |
| Jumpers, MOSFET e conectores | 15,00 | estimado | premissa |
| Orange Pi 5 4 GB (importação, pedido orangepi) | 917,02 | estimado | US$ 100,00 + frete US$ 10, II US$ 36,00, ICMS 17% por dentro, câmbio 5,2132; sem preço atual: lançamento US$ 60–75 (2022) e revenda UK £ 209,53 (jul/2026); premissa US$ 100 |

**Posição nos rankings** (entre 20 configurações completas): Menor custo: 7º; Melhor custo-benefício: 8º; Maior eficiência energética: 4º; Maior autonomia: 8º; Disparo mais rápido: 1º; Maior precisão: 6º; Melhor filtro de vazias: 2º; Melhor identificação de espécies: 1º; Melhor desempenho noturno: 2º; Menor custo em dois anos: 8º; Mais fácil de montar: 2º; Melhor geral — Equilíbrio: 8º; Melhor geral — Longa duração: 10º; Melhor geral — Orçamento: 8º.

## Passo a passo

### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)

```bash
cd versoes/08_orange-pi-5/08b_modelo-b_rknn
./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml
python converter_rknn.py --analise   # INT8 rk3588 (ambiente com rknn-toolkit2 2.5.0)
./empacotar.sh            # gera versoes/dist/08b.tar.gz
```
Rode o conversor num ambiente separado (`pip install rknn-toolkit2==2.5.0`; ele fixa versões de onnx/torch). A versão do toolkit deve casar com a do `librknnrt.so` da placa (`strings /usr/lib/librknnrt.so | grep -i version`).

### 2. Instalar no dispositivo

```bash
scp versoes/dist/08b.tar.gz orangepi@<ip>:/tmp/
ssh orangepi@<ip>
sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/08b.tar.gz
sudo chown -R $USER /opt/armadilha/08b && cd /opt/armadilha/08b
./instalar.sh
```

No Orange Pi 5 (Debian/Orange Pi OS; o toolkit-lite2 lista Debian 10–12): o `instalar.sh` instala `rknn-toolkit-lite2`. Descubra o chip/linha do PIR com `gpioinfo` e ajuste `gatilho` e `iluminacao_ir` em `versao.yaml`. Câmera: UVC (`dispositivo: 0`) ou módulo MIPI por pipeline GStreamer em `captura.dispositivo`.

### 3. Executar

```bash
# bancada, sem sensor nem câmera: imagens/vídeo simulam os disparos
./executar.sh trap --simular /caminho/imagens_ou_video.mp4 --eventos 5
./executar.sh benchmark --fonte /caminho/video.mp4 --eventos 20
# campo
./executar.sh trap
sudo cp armadilha.service /etc/systemd/system/armadilha-08b.service
sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-08b
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
- rknn_model_zoo: YOLOv8n INT8 640 a 76,3 FPS em um núcleo do RK3588 (só inferência). A qualidade informada é o proxy x86 do INT8 com cabeça dividida.
- `gatilho.linha`/`iluminacao_ir.pino` são exemplos: confira com `gpioinfo`.

## Arquivos desta pasta

- `README.md`
- `armadilha.service`
- `config/runtime_b.yaml`
- `converter_rknn.py`
- `empacotar.sh`
- `executar.sh`
- `instalar.sh`
- `preparar_modelo.sh`
- `versao.yaml`
