# Ranking das versões embarcadas (completo e refeito)

> Gerado por `versoes/ranking/calcular_ranking.py` em 21/10/2026 a partir de `versoes/ranking/candidatos.yaml` e `versoes/ranking/medicoes/`. Para recalcular com outras premissas, edite o YAML e rode o script.

## O que mudou

Na pesquisa teórica, a tabela "O que cada ranking permite concluir" (seção 7) ficou quase toda **indeterminada**, porque faltavam custos completos e desempenho comparável. Esta versão ordena as **20 configurações completas de armadilha** que o cruzamento hardware × software/modelo gerou (ver `MATRIZ_COMPATIBILIDADE.md`), usando a mesma regra da pesquisa: cada número mantém seu tipo.

- **documentado**: preço de loja ou fabricante e medição publicada, com fonte e data;
- **medido**: proxy medido neste projeto em x86_64 com os modelos reais (Modelo A e B, INT8, cabeça dividida, resoluções de 192 a 640 px, MegaDetector + SpeciesNet);
- **estimado**: conta com premissa explícita (consumo de placas sem dado publicado, tempo de disparo, importação).

O que continua sem resposta — e o ranking não finge responder: desempenho em campo nas três espécies, recall noturno, consumo medido e durabilidade. As posições valem para a configuração descrita, sob estas premissas, e devem ser confirmadas em bancada.

Fora do ranking de armadilha completa (mas com dados na seção 6): Jetson Orin Nano Super (13a, 13b — teto de desempenho, não candidato a bateria) e as estações de lote MegaDetector + SpeciesNet (01c, 08c, 13c — complementam uma armadilha).

## 1. Tabela de ranking

| Ranking | Métrica | 1º | 2º | 3º | Base dos dados | Confiança | O que se pode concluir |
|---|---|---|---|---|---|---|---|
| Menor custo | custo completo da armadilha (R$) | **09a** Espressif ESP32-S3-EYE · TinyML (R$ 836,82) | **10a** Luckfox Pico (RV1106) · YOLOv8n (R$ 866,76) | **07a** Seeed Grove Vision AI Module V2 · YOLOv8n (R$ 900,19) | Documentado (lojas BR, 20–21/10/2026) + importação estimada | Média: vários itens esgotados; Orange Pi 5 sem preço atual | As três mais baratas (09a, 10a, 07a) ficam entre R$ 836,82 e R$ 900,19 completas; a Raspberry Pi mais barata é a 05a (R$ 921,65). Só a placa Pi 5 de 4 GB custa R$ 1.574,90. |
| Melhor custo-benefício | benefício de desempenho ÷ custo (por R$ 1.000) | **05a** Raspberry Pi Zero 2 W · YOLOv8n (0,660) | **10a** Luckfox Pico (RV1106) · YOLOv8n (0,658) | **09a** Espressif ESP32-S3-EYE · TinyML (0,646) | Derivado | Baixa a média: herda as incertezas de todos os critérios | Lideram configurações baratas (05a, 10a, 09a); no fim ficam 03a, 12b, 12a: o Pi 5 com acelerador paga por vazão que uma rajada de poucos quadros não usa. |
| Maior eficiência energética | energia por imagem analisada, E_img = P·t (J) | **06a** Sipeed MaixCAM · YOLOv8n (0,012 J) | **06b** Sipeed MaixCAM · Modelo B (0,015 J) | **10a** Luckfox Pico (RV1106) · YOLOv8n (0,016 J) | Estimado (P·t; t documentado em parte) | Baixa: potência de inferência quase toda estimada | Com acelerador, a energia por imagem cai de 1,75 J (CPU do Pi 5) para 0,06–0,09 J (20× menos); mas a espera continua dominando o dia, então isso quase não muda a autonomia (QP2, QP3). |
| Maior autonomia | dias com 74 Wh a 100 eventos/dia | **09a** Espressif ESP32-S3-EYE · TinyML (597,5 d) | **07a** Seeed Grove Vision AI Module V2 · YOLOv8n (453,2 d) | **11a** Arduino Nicla Vision · TinyML (35,1 d) | Estimado (seção 7) com espera documentada em parte | Média para a ordem de grandeza; baixa para a ordem fina | Só quem dorme entre eventos passa de duas semanas (09a, 07a, 11a). Linux sempre ligado fica em dias (Zero 2 W, Luckfox, MaixCAM) ou em 14–19 h (Pi 5 com ou sem acelerador). |
| Disparo mais rápido | tempo até a primeira captura (s) | **04a**, **04b**, **06a**, **06b**, **08a**, **08b**, **10a** (empate, 0,1 s) | **11a** Arduino Nicla Vision · TinyML (0,3 s) | **01a**, **01b**, **02a**, **02b**, **03a**, **03b**, **09a**, **12a**, **12b** (empate, 0,8 s) | Estimado por arquitetura | Baixa: nenhum tempo medido | Câmera sempre ligada dispara em ~0,1 s ao custo de espera alta; desligar a câmera (0,8–1,2 s) ou a placa inteira (arquitetura B, ~20 s) troca atraso por autonomia (QP1). |
| Maior precisão | F1 por espécie no teste (MCU: acurácia do quadro) | **01b** Raspberry Pi 5 · Modelo B (0,947) | **06b** Sipeed MaixCAM · Modelo B (0,889) | **01a** Raspberry Pi 5 · YOLOv8n (0,872) | Medido no projeto (proxy x86, 18 imagens) | Baixa: 1 caixa = 5 pontos de recall; não é imagem de armadilha | O Modelo B tem o maior F1 em 7 das 8 plataformas que rodam os dois modelos (exceção: 05a 0,865 × 05b 0,842 — diferença de uma caixa). Em 256 px o YOLOv8n cai para 0,647 (Grove). |
| Melhor filtro de vazias | recall de presença do detector no teste | **06b** Sipeed MaixCAM · Modelo B (0,95) | **01b**, **02a**, **02b**, **03a**, **03b**, **04a**, **04b**, **05b**, **08a**, **08b**, **12a**, **12b** (empate, 0,90) | **09a**, **11a** (empate, 0,89) | Medido no projeto (presença no detector) | Baixa: sem imagens vazias no teste | Os detectores de presença do Modelo B acham 90%–95% dos animais; o YOLOv8n multiclasse, 60%–90%, caindo com a resolução. Sem imagens vazias no teste, não há como medir descarte seguro. |
| Melhor identificação de espécies | acerto entre eventos confirmados (rajadas) | **01b**, **02b**, **03b**, **04b**, **05b**, **06b**, **08b**, **12b** (empate, 100%) | **05a** Raspberry Pi Zero 2 W · YOLOv8n (88%) | **01a**, **10a** (empate, 86%) | Medido no projeto (90 rajadas por proxy) | Média para A × B; vídeos editados, não de campo | Nenhuma variante do Modelo B confirmou espécie errada em 90 rajadas; todas as do Modelo A confirmaram (8%–16% dos eventos). O preço é ~1/3 de eventos inconclusivos, guardados para revisão. |
| Melhor desempenho noturno | nota 0–2 (câmera sensível a IR + iluminador) | **01a**, **01b**, **02a**, **02b**, **03a**, **03b**, **05a**, **05b**, **07a**, **12a**, **12b** (empate, nota 2) | **06a**, **06b**, **08a**, **08b**, **09a**, **10a**, **11a** (empate, nota 1) | **04a**, **04b** (empate, nota 0) | Documentado (filtro IR da câmera) | Média: diz se há como fotografar com IR, não a qualidade | Câmeras sensíveis a IR com iluminador (Camera Module 3 NoIR, OV5647 IR, IMX219 NoIR) habilitam a noite; a AI Camera não, por ter filtro IR fixo e não existir versão NoIR. |
| Maior robustez | limite superior de temperatura declarado do elo mais fraco (°C) | **07a**, **11a** (empate, 85 °C) | **06a**, **06b** (empate, 70 °C) | **01a**, **01b**, **02a**, **02b**, **03a**, **03b**, **04a**, **04b**, **05a**, **05b**, **09a**, **12a**, **12b** (empate, 50 °C) | Documentado parcialmente (temperatura de operação) | Baixa: várias placas não declaram; não há histórico de falhas | Só se ordena o que está declarado, e com cautela: Grove e MaixCAM informam só o chip; nas versões Raspberry Pi o elo fraco é a câmera (0 a 50 °C), abaixo do que uma caixa ao sol pode atingir. |
| Menor custo em dois anos | TCO = custo + (730/D)·R$ 50 | **09a** Espressif ESP32-S3-EYE · TinyML (R$ 898) | **07a** Seeed Grove Vision AI Module V2 · YOLOv8n (R$ 981) | **11a** Arduino Nicla Vision · TinyML (R$ 2.231) | Estimado (custo + visitas de troca) | Baixa: depende do custo de visita adotado | Com troca paga por visita, 17 das 20 configurações duram menos de uma semana por carga e somam dezenas de milhares de reais em visitas: a bateria não é o caminho para elas — exigem solar ou rede. |
| Mais fácil de montar | rubrica de integração e documentação (0–6) | **01a**, **01b**, **05a**, **05b** (empate, 6/6) | **02a**, **02b**, **03a**, **03b**, **04a**, **04b**, **08a**, **08b**, **12a**, **12b** (empate, 4/6) | **06a**, **06b**, **10a**, **11a** (empate, 3/6) | Rubrica sobre as versões geradas | Média | ONNX Runtime direto no Raspberry Pi é o caminho mais curto; firmware em C/C++ e conversão por toolchain externa (TPU-MLIR, Vela, ESP-PPQ, SDK da Luckfox) são os mais longos. |
| Melhor geral — Equilíbrio | escore ponderado min-max (0–1) | **06b** Sipeed MaixCAM · Modelo B (0,685) | **05a** Raspberry Pi Zero 2 W · YOLOv8n (0,681) | **05b** Raspberry Pi Zero 2 W · Modelo B (0,665) | Composição dos critérios acima | Baixa a média: ver sensibilidade (seção 5) | 06b à frente de 05a por 0,003: diferença menor que a incerteza das premissas, a ordem entre os dois não é robusta. Ver seções 3 e 5. |
| Melhor geral — Longa duração | escore ponderado min-max (0–1) | **09a** Espressif ESP32-S3-EYE · TinyML (0,729) | **07a** Seeed Grove Vision AI Module V2 · YOLOv8n (0,653) | **05a** Raspberry Pi Zero 2 W · YOLOv8n (0,590) | Composição dos critérios acima | Baixa a média: ver sensibilidade (seção 5) | 09a à frente de 07a por 0,077. Ver seções 3 e 5. |
| Melhor geral — Orçamento | escore ponderado min-max (0–1) | **05a** Raspberry Pi Zero 2 W · YOLOv8n (0,761) | **05b** Raspberry Pi Zero 2 W · Modelo B (0,748) | **06b** Sipeed MaixCAM · Modelo B (0,745) | Composição dos critérios acima | Baixa a média: ver sensibilidade (seção 5) | 05a à frente de 05b por 0,013: diferença menor que a incerteza das premissas, a ordem entre os dois não é robusta. Ver seções 3 e 5. |

Na precisão, as versões de microcontrolador sem detector (09a, 11a) entram com a acurácia do quadro inteiro, não com F1 — a tarefa é outra. A identificação de espécies só ordena quem tem simulação de rajadas (as duas também ficam de fora). Empates são mostrados como empates.

## 2. Resultados (tabela do esqueleto preenchida)

| Posição | Mais barato | Custo-benefício | Mais eficiente | Maior autonomia | Mais preciso | Melhor geral (Equilíbrio) |
|---|---|---|---|---|---|---|
| 1º | 09a Espressif ESP32-S3-EYE · TinyML | 05a Raspberry Pi Zero 2 W · YOLOv8n | 06a Sipeed MaixCAM · YOLOv8n | 09a Espressif ESP32-S3-EYE · TinyML | 01b Raspberry Pi 5 · Modelo B | 06b Sipeed MaixCAM · Modelo B |
| 2º | 10a Luckfox Pico (RV1106) · YOLOv8n | 10a Luckfox Pico (RV1106) · YOLOv8n | 06b Sipeed MaixCAM · Modelo B | 07a Seeed Grove Vision AI Module V2 · YOLOv8n | 06b Sipeed MaixCAM · Modelo B | 05a Raspberry Pi Zero 2 W · YOLOv8n |
| 3º | 07a Seeed Grove Vision AI Module V2 · YOLOv8n | 09a Espressif ESP32-S3-EYE · TinyML | 10a Luckfox Pico (RV1106) · YOLOv8n | 11a Arduino Nicla Vision · TinyML | 01a Raspberry Pi 5 · YOLOv8n | 05b Raspberry Pi Zero 2 W · Modelo B |

## 3. Melhor geral por perfil de pesos

Pesos da seção 7 da pesquisa (cada perfil soma 1). Normalização min-max; autonomia em log10(dias). Um critério sem variação sairia com redistribuição dos pesos (não ocorreu).

| Critério | Equilíbrio | Longa duração | Orçamento |
|---|---|---|---|
| Qualidade de detecção | 0,25 | 0,20 | 0,20 |
| Autonomia | 0,20 | 0,40 | 0,15 |
| Custo completo | 0,20 | 0,15 | 0,40 |
| Tempo de disparo | 0,15 | 0,10 | 0,10 |
| Capacidade noturna | 0,10 | 0,10 | 0,10 |
| Integração e documentação | 0,10 | 0,05 | 0,05 |

| Posição | Equilíbrio | Longa duração | Orçamento |
|---|---|---|---|
| 1º | 06b Sipeed MaixCAM · Modelo B (0,685) | 09a Espressif ESP32-S3-EYE · TinyML (0,729) | 05a Raspberry Pi Zero 2 W · YOLOv8n (0,761) |
| 2º | 05a Raspberry Pi Zero 2 W · YOLOv8n (0,681) | 07a Seeed Grove Vision AI Module V2 · YOLOv8n (0,653) | 05b Raspberry Pi Zero 2 W · Modelo B (0,748) |
| 3º | 05b Raspberry Pi Zero 2 W · Modelo B (0,665) | 05a Raspberry Pi Zero 2 W · YOLOv8n (0,590) | 06b Sipeed MaixCAM · Modelo B (0,745) |
| 4º | 06a Sipeed MaixCAM · YOLOv8n (0,661) | 06b Sipeed MaixCAM · Modelo B (0,581) | 10a Luckfox Pico (RV1106) · YOLOv8n (0,736) |
| 5º | 10a Luckfox Pico (RV1106) · YOLOv8n (0,654) | 05b Raspberry Pi Zero 2 W · Modelo B (0,576) | 09a Espressif ESP32-S3-EYE · TinyML (0,729) |

O vencedor **muda** com o perfil: Equilíbrio → 06b; Longa duração → 09a; Orçamento → 05a.

## 4. Dados por candidato

Tipos: ver `candidatos.yaml` (cada número com fonte ou premissa). Qualidade: F1 por espécie no teste (09a e 11a: acurácia do quadro inteiro). Rajadas: acerto entre eventos com espécie confirmada e taxa de eventos com espécie errada confirmada (90 rajadas de 8 quadros).

| Versão | Configuração | Custo | Espera (W) | Inferência (ms) | E_img (J) | Autonomia (d) | Disparo (s) | Qualidade | Presença | Rajadas: acerto / erro | Noturno | Integração | TCO 2 anos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01a | Raspberry Pi 5 · YOLOv8n | R$ 2.346,55 | 3,300 | 198,7 | 1,748 | 0,8 | 0,8 | 0,872 | 0,85 | 86% / 9% | 2 | 6/6 | R$ 48.490,67 |
| 01b | Raspberry Pi 5 · Modelo B | R$ 2.346,55 | 3,300 | 89,0 | 0,783 | 0,8 | 0,8 | 0,947 | 0,90 | 100% / 0% | 2 | 6/6 | R$ 48.374,29 |
| 02a | Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) · YOLOv8n | R$ 3.246,45 | 3,600 | 16,2 | 0,089 | 0,7 | 0,8 | 0,800 | 0,90 | 84% / 10% | 2 | 4/6 | R$ 53.383,81 |
| 02b | Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) · Modelo B | R$ 3.246,45 | 3,600 | 15,0 | 0,083 | 0,7 | 0,8 | 0,865 | 0,90 | 100% / 0% | 2 | 4/6 | R$ 53.383,81 |
| 03a | Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) · YOLOv8n | R$ 3.746,45 | 3,700 | 9,5 | 0,062 | 0,7 | 0,8 | 0,800 | 0,90 | 84% / 10% | 2 | 4/6 | R$ 55.300,12 |
| 03b | Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) · Modelo B | R$ 3.746,45 | 3,700 | 11,0 | 0,071 | 0,7 | 0,8 | 0,865 | 0,90 | 100% / 0% | 2 | 4/6 | R$ 55.300,12 |
| 04a | Raspberry Pi 5 + AI Camera (IMX500) · YOLOv8n | R$ 3.065,22 | 4,500 | 58,8 | 0,306 | 0,6 | 0,1 | 0,800 | 0,90 | 84% / 10% | 0 | 4/6 | R$ 65.780,00 |
| 04b | Raspberry Pi 5 + AI Camera (IMX500) · Modelo B | R$ 3.065,22 | 4,500 | 63,0 | 0,328 | 0,6 | 0,1 | 0,865 | 0,90 | 100% / 0% | 0 | 4/6 | R$ 65.780,00 |
| 05a | Raspberry Pi Zero 2 W · YOLOv8n | R$ 921,65 | 0,600 | 500,0 | 1,150 | 4,3 | 1,2 | 0,865 | 0,85 | 88% / 8% | 2 | 6/6 | R$ 9.360,96 |
| 05b | Raspberry Pi Zero 2 W · Modelo B | R$ 921,65 | 0,600 | 300,0 | 0,690 | 4,3 | 1,2 | 0,842 | 0,90 | 100% / 0% | 2 | 6/6 | R$ 9.333,73 |
| 06a | Sipeed MaixCAM · YOLOv8n | R$ 1.097,10 | 0,700 | 12,0 | 0,012 | 3,7 | 0,1 | 0,857 | 0,75 | 84% / 11% | 1 | 3/6 | R$ 10.853,85 |
| 06b | Sipeed MaixCAM · Modelo B | R$ 1.097,10 | 0,700 | 15,0 | 0,015 | 3,7 | 0,1 | 0,889 | 0,95 | 100% / 0% | 1 | 3/6 | R$ 10.853,85 |
| 07a | Seeed Grove Vision AI Module V2 · YOLOv8n | R$ 900,19 | 0,001 | 71,0 | 0,046 | 453,2 | 1,6 | 0,647 | 0,60 | 69% / 16% | 2 | 1/6 | R$ 980,73 |
| 08a | Orange Pi 5 · YOLOv8n | R$ 1.589,20 | 1,500 | 13,1 | 0,033 | 1,7 | 0,1 | 0,800 | 0,90 | 84% / 10% | 1 | 4/6 | R$ 22.506,11 |
| 08b | Orange Pi 5 · Modelo B | R$ 1.589,20 | 1,500 | 7,0 | 0,018 | 1,7 | 0,1 | 0,842 | 0,90 | 100% / 0% | 1 | 4/6 | R$ 22.506,11 |
| 09a | Espressif ESP32-S3-EYE · TinyML | R$ 836,82 | 0,003 | 170,0 | 0,085 | 597,5 | 0,8 | 0,722 | 0,89 | — | 1 | 2/6 | R$ 897,91 |
| 10a | Luckfox Pico (RV1106) · YOLOv8n | R$ 866,76 | 0,600 | 20,0 | 0,016 | 4,4 | 0,1 | 0,824 | 0,70 | 86% / 9% | 1 | 3/6 | R$ 9.228,73 |
| 11a | Arduino Nicla Vision · TinyML | R$ 1.190,56 | 0,074 | 200,0 | 0,078 | 35,1 | 0,3 | 0,611 | 0,89 | — | 1 | 3/6 | R$ 2.231,34 |
| 12a | Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) · YOLOv8n | R$ 4.245,55 | 3,700 | 12,0 | 0,078 | 0,7 | 0,8 | 0,800 | 0,90 | 84% / 10% | 2 | 4/6 | R$ 55.799,22 |
| 12b | Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) · Modelo B | R$ 4.245,55 | 3,700 | 13,0 | 0,085 | 0,7 | 0,8 | 0,865 | 0,90 | 100% / 0% | 2 | 4/6 | R$ 55.799,22 |

## 5. Sensibilidade

**Eventos por dia** (vencedor do melhor geral):

| Eventos/dia | Equilíbrio | Longa duração | Orçamento |
|---|---|---|---|
| 10 | 06b | 09a | 05a |
| 100 | 06b | 09a | 05a |
| 1000 | 06b | 09a | 05a |

Autonomia (dias) com 10 / 100 / 1000 eventos por dia:

| Versão | 10 | 100 | 1000 |
|---|---|---|---|
| 09a | 835,0 | 597,5 | 155,4 |
| 07a | 1.772,8 | 453,2 | 53,7 |
| 11a | 35,4 | 35,1 | 32,2 |
| 10a | 4,4 | 4,4 | 4,3 |
| 05b | 4,4 | 4,3 | 4,1 |
| 05a | 4,4 | 4,3 | 4,0 |
| 06a | 3,7 | 3,7 | 3,7 |
| 06b | 3,7 | 3,7 | 3,7 |
| 08a | 1,7 | 1,7 | 1,7 |
| 08b | 1,7 | 1,7 | 1,7 |
| 01b | 0,8 | 0,8 | 0,8 |
| 01a | 0,8 | 0,8 | 0,8 |
| 02a | 0,7 | 0,7 | 0,7 |
| 02b | 0,7 | 0,7 | 0,7 |
| 03a | 0,7 | 0,7 | 0,7 |
| 03b | 0,7 | 0,7 | 0,7 |
| 12a | 0,7 | 0,7 | 0,7 |
| 12b | 0,7 | 0,7 | 0,7 |
| 04a | 0,6 | 0,6 | 0,6 |
| 04b | 0,6 | 0,6 | 0,6 |

**Preço do Orange Pi 5** (sem cotação atual; premissa US$ 100):

| Preço FOB | Custo 08a | Posição de 08a em custo | Vencedores (Equilíbrio / Longa / Orçamento) |
|---|---|---|---|
| US$ 60 | R$ 1.187,22 | 6º | 06b / 09a / 05a |
| US$ 270 | R$ 3.297,62 | 10º | 06b / 09a / 05a |

**Custo de uma visita de troca de bateria** (três menores TCO em 2 anos):

- R$ 20: 09a, 07a, 11a
- R$ 200: 09a, 07a, 11a

**Arquitetura B no Pi 5** (versão 01b desligada entre eventos com `POWER_OFF_ON_HALT=1`, ~0,01 W oficial, religada pelo circuito do PIR; boot de 20 s estimado): autonomia de 0,8 para **18,7 dias** a 100 eventos/dia, com o tempo até a primeira foto passando de 0,8 s para ~20 s — o dilema da QP1/QP2 em números. A versão suporta esse modo (`gatilho.tipo: nenhum`, `energia.desligar_ao_final: true`).

## 6. Fora do ranking de armadilha

| Versão | Papel | Custo | Dados |
|---|---|---|---|
| 13a | Teto de desempenho (Jetson) | R$ 4.175,25 | inferência 4,53 ms; espera 5,0 W; autonomia 0,52 d com 74 Wh; qualidade 0,872 |
| 13b | Teto de desempenho (Jetson) | R$ 4.175,25 | inferência 4,00 ms; espera 5,0 W; autonomia 0,52 d com 74 Wh; qualidade 0,947 |
| 01c | Estação de lote (MegaDetector + SpeciesNet) | R$ 2.655,90 | ~15,0 s/imagem (estimado: 2,3 s/imagem medidos em x86 (i5-1235U, 12 threads) × ~6,5 (Pi 5, premissa)) |
| 08c | Estação de lote (MegaDetector + SpeciesNet) | R$ 1.043,02 | ~12,0 s/imagem (estimado: 2,3 s/imagem em x86 × ~5 (RK3588S, 8 núcleos; premissa)) |
| 13c | Estação de lote (MegaDetector + SpeciesNet) | R$ 3.499,90 | ~0,5 s/imagem (estimado: GPU; o SpeciesNet carrega em 0,96 s com CUDA contra 9,45 s em CPU (README) — premissa) |

O custo da 08c usa o Orange Pi 5 de 4 GB como piso (preço estimado); o lote pede 8 GB de RAM.

**MegaDetector v5a + SpeciesNet v4.0.3a medidos no projeto** (`--country BRA`, x86, mesmas 18 imagens): presença 20/20 animais; espécie exata em 13 de 20 (F1 0,65). Níveis atribuídos: espécie: 13, tapiridae family: 2, vampire bat: 1, canine family: 1, mammal: 1, human: 1, animal: 1. Nenhuma troca entre as três espécies; os erros foram arredondamentos para família ou rótulos sem sentido. É a referência da área sem treino na fauna local — e fica abaixo do Modelo B neste conjunto, que não é de armadilha fotográfica.

## 7. Método e limites

- **Unidade de comparação:** a configuração completa (placa, acelerador, câmera, modelo, precisão numérica, runtime e arquitetura de energia), como define a seção 3.
- **Custo completo:** itens da versão + comuns (PIR, caixa IP66, microSD 32 GB, power bank de 74 Wh, diversos). Importação: Remessa Conforme vigente, frete de US$ 10 por pedido, ICMS 17% (SP), câmbio 5.2132 (PTAX venda de 28/09/2026 (Banco Central) — a mais recente encontrada na coleta).
- **Autonomia:** equação da seção 7 com o mesmo pacote para todos (74 Wh, f = 1.0, η = 0.85) e 100 eventos/dia; ciclo = partida + rajada + escrita + inferência de cada quadro. Power banks comuns desligam com carga de µA: nas versões de microcontrolador use células com regulador de mesma energia.
- **Qualidade:** proxies medidos em x86 com ONNX Runtime sobre o mesmo modelo, na resolução e precisão da versão (o INT8 de NPUs usa o proxy de cabeça dividida com todos os operadores em INT8). Não inclui a perda própria de cada compilador.
- **Disparo:** tempo até a primeira captura, separado da latência de inferência e do alerta (seção 3). Estimado por arquitetura; o atraso do próprio PIR é comum a todos e fica fora.
- **Noturno (0–2):** 2 = câmera sensível a IR com iluminador; 1 = não documentado ou exige troca de câmera; 0 = filtro IR fixo documentado.
- **Integração (0–6):** conversão (2 ONNX direto, 1 export de uma linha, 0 toolchain externa) + API Python no dispositivo + conjunto único + exemplo oficial + caminho testado neste projeto.
- **Custo-benefício:** benefício só com critérios de desempenho (pesos do perfil Equilíbrio sem o custo, renormalizados) dividido pelo custo em milhares de reais — o custo não entra duas vezes.
- **TCO:** custo + (730 / D) visitas × R$ 50. Para quem dura menos de uma semana, o número só mostra que a bateria não é o caminho: esses sistemas exigem energia solar ou da rede.

## 8. Fontes

- Preços e regra de importação: `versoes/ranking/fontes/precos.md`.
- Consumo e benchmarks: `versoes/ranking/fontes/consumo_desempenho_linux.md` e `consumo_desempenho_mcu_npu.md`.
- Toolchains e APIs de cada versão: `versoes/ranking/fontes/toolchains_e_apis.md`.
- Dado usado em cada candidato, com fonte ou premissa: `versoes/ranking/candidatos.yaml`.
- Proxies medidos: `versoes/ranking/medicoes/` (`proxies_x86.json`, `rajadas_x86.json`, `presenca_x86.json`, `speciesnet_x86.json`), gerados pelas ferramentas de `versoes/comum/ferramentas/`.
- Referências numeradas [n] como na pesquisa teórica.
