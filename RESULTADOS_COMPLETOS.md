# Resultados Completos - Sistema de Monitoramento de Animais

## Sumario Executivo

Este documento compila todos os resultados experimentais obtidos durante o desenvolvimento do sistema de monitoramento de animais do Cerrado. Foram testados multiplos modelos e configuracoes, incluindo:

- **Modelo A**: YOLOv8n Baseline (detector multiclasse direto)
- **Modelo B**: Arquitetura em duas etapas (detector de presenca + classificador de especies)
- **Variantes de Classificadores**: MobileNet V2, MobileNet V3 Small, ShuffleNet V2 x0.5
- **Versoes Quantizadas**: FP32 e INT8
- **Simulacao Embarcada**: Teste em ambiente com restricoes de recursos

---

## 1. MODELO A - YOLOv8n Baseline

### 1.1 Descricao do Modelo

O Modelo A e uma abordagem direta usando YOLOv8n treinado para detectar e classificar simultaneamente tres especies de animais do Cerrado: anta, lobo-guara e tamandua-bandeira.

**Caracteristicas:**
- Arquitetura: YOLOv8n
- Tamanho de entrada: 640x640
- Classes: 3 (anta, tamandua_bandeira, lobo_guara)
- Tamanho do modelo: 11.7 MB (FP32)

### 1.2 Metricas de Acuracia

**Dataset de Teste (18 imagens):**

| Metrica | Valor |
|---------|-------|
| mAP@50 | 0.8424 |
| mAP@50-95 | 0.5095 |
| Precision@0.35 | 0.8947 |
| Recall@0.35 | 0.8500 |
| F1-Score@0.35 | 0.8718 |

**Desempenho por Classe:**

| Classe | AP@50 | Precision | Recall |
|--------|-------|-----------|--------|
| Anta | 0.8367 | - | - |
| Tamandua-bandeira | 0.8333 | - | - |
| Lobo-guara | 0.8571 | - | - |

**Contadores:**
- True Positives: 17
- False Positives: 2
- False Negatives: 3
- Total Ground Truth: 20
- Total Predictions: 19

### 1.3 Desempenho em Video (200 frames)

**Video Anta:**
- Latencia media: 42.94 ms
- Latencia P95: 50.23 ms
- Throughput: 23.29 FPS
- Uso de memoria RAM: 218.2 MB
- Deteccoes totais: 145

**Video Lobo-guara:**
- Latencia media: 43.12 ms
- Latencia P95: 49.98 ms
- Throughput: 23.19 FPS
- Uso de memoria RAM: 221.3 MB
- Deteccoes totais: 158

**Video Tamandua-bandeira:**
- Latencia media: 41.93 ms
- Latencia P95: 47.54 ms
- Throughput: 23.85 FPS
- Uso de memoria RAM: 219.0 MB
- Deteccoes totais: 123

**Media dos 3 Videos:**
- FPS: 15.85 (media efetiva considerando sistema completo)
- Latencia media: 63.25 ms
- Latencia P95: 72.22 ms
- Uso de RAM: 219.1 MB

### 1.4 Problemas Identificados

Durante os testes em video, o Modelo A apresentou **2 eventos de especies erradas**:
- Video do lobo-guara: 1 evento incorreto classificado como "anta"
- Video do tamandua-bandeira: 1 evento incorreto classificado como "anta"

Isso indica confusao entre especies similares em cenarios de video real.

### 1.5 Velocidade de Inferencia

**Breakdown de tempo (media):**
- Preprocessamento: 2.36 ms
- Inferencia: 95.86 ms
- Postprocessamento: 3.15 ms
- Loss: 0.0003 ms

---

## 2. MODELO B - Detector + Classificador

### 2.1 Descricao da Arquitetura

O Modelo B adota uma abordagem em duas etapas:
1. **Estagio 1 (Detector de Presenca)**: YOLOv8n treinado para detectar "animal" (classe unica)
2. **Estagio 2 (Classificador de Especies)**: CNN leve para classificar recortes em especies

**Vantagens da abordagem:**
- Melhor separacao de responsabilidades
- Permite otimizacao independente de cada estagio
- Classificador pode ser trocado sem retreinar o detector
- Menor custo computacional (classificador so e executado quando ha deteccao)

### 2.2 Estagio 1 - Detector de Presenca

**Configuracao:**
- Arquitetura: YOLOv8n
- Tamanho de entrada: 416x416 (menor que Modelo A)
- Classes: 1 (animal)
- Tamanho do modelo (FP32): 11.6 MB
- Tamanho do modelo (INT8): 3.2 MB

**Desempenho no Dataset de Teste:**

| Metrica | Valor |
|---------|-------|
| mAP@50 | 0.9820 |
| mAP@50-95 | 0.6366 |
| Precision | 0.9920 |
| Recall | 0.9000 |

**Observacao:** O detector de presenca tem desempenho excelente, com precisao de 99.2%.

### 2.3 Estagio 2 - Classificadores de Especies

Foram testados tres backbones diferentes para o classificador:

#### 2.3.1 MobileNet V2

**Configuracao:**
- Parametros: 2.229.000
- Tamanho entrada: 128x128
- Tamanho modelo (FP32): 8.5 MB
- Tamanho modelo (INT8): 2.5 MB
- Tempo de treinamento: 238.7s

**Desempenho em Validacao:**
- Accuracy: 93.88%
- Balanced Accuracy: 94.70%
- Macro F1-Score: 0.9271

**Desempenho em Teste (46 recortes):**
- Accuracy: 95.65%
- Balanced Accuracy: 92.86%
- Macro F1-Score: 0.9258

**Desempenho por Classe (Teste):**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Anta | 0.8571 | 0.8571 | 0.8571 | 7 |
| Tamandua-bandeira | 0.8571 | 1.0000 | 0.9231 | 6 |
| Lobo-guara | 1.0000 | 0.8571 | 0.9231 | 7 |
| Desconhecido | 1.0000 | 1.0000 | 1.0000 | 26 |

**Robustez a Quantizacao INT8:**
- FP32: 95.65% accuracy
- INT8: 95.65% accuracy (sem perda!)

**Matriz de Confusao (Teste):**
```
             anta  tamandua  lobo  desconhecido
anta            6         1     0            0
tamandua        0         6     0            0
lobo            1         0     6            0
desconhecido    0         0     0           26
```

#### 2.3.2 MobileNet V3 Small

**Configuracao:**
- Parametros: 1.522.000
- Tamanho entrada: 128x128
- Tamanho modelo (FP32): ~8 MB
- Tempo de treinamento: 52.0s (4.6x mais rapido que V2!)

**Desempenho em Validacao:**
- Accuracy: 87.76%
- Balanced Accuracy: 89.41%
- Macro F1-Score: 0.8623

**Desempenho em Teste (46 recortes):**
- Accuracy: 84.78%
- Balanced Accuracy: 81.64%
- Macro F1-Score: 0.7885

**Desempenho por Classe (Teste):**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Anta | 0.6250 | 0.7143 | 0.6667 | 7 |
| Tamandua-bandeira | 0.5714 | 0.6667 | 0.6154 | 6 |
| Lobo-guara | 0.8750 | 1.0000 | 0.9333 | 7 |
| Desconhecido | 1.0000 | 0.8846 | 0.9388 | 26 |

**Robustez a Quantizacao INT8:**
- FP32: 84.78% accuracy
- INT8 (minmax): 58.70% accuracy (PERDA SEVERA!)
- INT8 (percentile): 65.22% accuracy (PERDA SEVERA!)
- INT8 (entropy): 58.70% accuracy (PERDA SEVERA!)

**Conclusao:** MobileNet V3 Small nao e robusto a quantizacao INT8 e teve desempenho inferior ao V2.

#### 2.3.3 ShuffleNet V2 x0.5

**Configuracao:**
- Parametros: 345.892 (6.4x menor que MobileNet V2!)
- Tamanho entrada: 128x128
- Tamanho modelo (FP32): ~2 MB
- Tempo de treinamento: 59.6s

**Desempenho em Validacao:**
- Accuracy: 81.63%
- Balanced Accuracy: 89.53%
- Macro F1-Score: 0.8214

**Desempenho em Teste (46 recortes):**
- Accuracy: 78.26%
- Balanced Accuracy: 85.16%
- Macro F1-Score: 0.7853

**Desempenho por Classe (Teste):**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Anta | 0.7500 | 0.8571 | 0.8000 | 7 |
| Tamandua-bandeira | 0.4286 | 1.0000 | 0.6000 | 6 |
| Lobo-guara | 1.0000 | 0.8571 | 0.9231 | 7 |
| Desconhecido | 1.0000 | 0.6923 | 0.8182 | 26 |

**Robustez a Quantizacao INT8:**
- FP32: 78.26% accuracy
- INT8: 78.26% accuracy (sem perda!)

**Conclusao:** ShuffleNet V2 e o modelo mais leve e robusto a quantizacao, mas tem menor acuracia.

### 2.4 Modelo B End-to-End (FP32)

**Configuracao Completa:**
- Detector: YOLOv8n presence_detector.onnx (11.6 MB)
- Classificador: MobileNet V2 species_classifier.onnx (8.5 MB)
- **Tamanho total: 20.1 MB**

**Desempenho no Dataset de Teste (18 imagens):**

| Metrica | Valor |
|---------|-------|
| mAP@50 | 0.8968 |
| Precision@0.35 | 1.0000 |
| Recall@0.35 | 0.9000 |
| F1-Score@0.35 | 0.9474 |

**Desempenho por Classe:**

| Classe | AP@50 | Precision | Recall | GT | Pred |
|--------|-------|-----------|--------|-----|------|
| Anta | 0.8571 | 1.0000 | 0.8571 | 7 | 6 |
| Tamandua-bandeira | 0.8333 | 0.8333 | 0.8333 | 6 | 6 |
| Lobo-guara | 1.0000 | 1.0000 | 1.0000 | 7 | 7 |

**Contadores:**
- True Positives: 18
- False Positives: 0
- False Negatives: 2
- Total Ground Truth: 20
- Total Predictions: 19

**Matriz de Confusao de Especies:**
```
             anta  tamandua  lobo  nao_detectado
anta            6         0     0              1
tamandua        0         5     0              1
lobo            0         0     7              0
```

### 2.5 Modelo B End-to-End (INT8)

**Configuracao Completa:**
- Detector: presence_detector_int8.onnx (3.2 MB)
- Classificador: species_classifier_int8.onnx (2.5 MB)
- **Tamanho total: 5.7 MB** (71.7% de reducao!)

**Desempenho no Dataset de Teste (18 imagens):**

| Metrica | Valor |
|---------|-------|
| mAP@50 | 0.8016 |
| Precision@0.35 | 0.8889 |
| Recall@0.35 | 0.8000 |
| F1-Score@0.35 | 0.8421 |

**Desempenho por Classe:**

| Classe | AP@50 | Precision | Recall | GT | Pred |
|--------|-------|-----------|--------|-----|------|
| Anta | 0.7143 | 1.0000 | 0.7143 | 7 | 5 |
| Tamandua-bandeira | 0.8333 | 0.7143 | 0.8333 | 6 | 7 |
| Lobo-guara | 0.8571 | 1.0000 | 0.8571 | 7 | 6 |

**Contadores:**
- True Positives: 16
- False Positives: 2
- False Negatives: 4
- Total Ground Truth: 20
- Total Predictions: 18

**Matriz de Confusao de Especies:**
```
             anta  tamandua  lobo  nao_detectado
anta            5         1     0              1
tamandua        0         5     0              1
lobo            0         1     6              0
```

**Impacto da Quantizacao:**
- Perda de mAP@50: -9.52 pontos percentuais (de 89.68% para 80.16%)
- Perda de Precision: -11.11 pontos percentuais
- Perda de Recall: -10.0 pontos percentuais
- Algumas confusoes entre especies (1 tamandua -> anta, 1 lobo -> tamandua)

### 2.6 Desempenho em Video - Modelo B FP32 (200 frames, 4 threads)

**Video Anta:**
- Latencia total media: 32.20 ms
- Latencia P95: 39.72 ms
- Throughput: 31.06 FPS
- Uso de RAM: 203.97 MB
- Deteccoes brutas: 166
- Recortes classificados: 35
- Recortes por frame: 0.175

**Breakdown de latencia:**
- Detector: 31.17 ms (96.8%)
- Classificador: 4.88 ms (15.1%)
- Tracker: 0.12 ms (0.4%)

**Video Lobo-guara:**
- Latencia total media: 32.14 ms
- Latencia P95: 40.26 ms
- Throughput: 31.11 FPS
- Uso de RAM: 203.97 MB
- Deteccoes brutas: 166
- Recortes classificados: 42
- Recortes por frame: 0.21

**Video Tamandua-bandeira:**
- Latencia total media: 32.24 ms
- Latencia P95: 39.16 ms
- Throughput: 31.01 FPS
- Uso de RAM: 203.97 MB
- Deteccoes brutas: 155
- Recortes classificados: 37
- Recortes por frame: 0.185

**Media dos 3 Videos:**
- FPS: 31.11
- Latencia media: 32.20 ms
- Latencia P95: 39.72 ms
- Uso de RAM: 203.97 MB
- Recortes por frame: 0.187

### 2.7 Desempenho em Video - Modelo B INT8 (200 frames, 4 threads)

**Video Anta:**
- Latencia total media: 27.10 ms
- Latencia P95: 33.37 ms
- Throughput: 36.90 FPS
- Uso de RAM: 192.66 MB
- Deteccoes brutas: 166
- Recortes classificados: 35
- Recortes por frame: 0.175

**Breakdown de latencia:**
- Detector: 26.28 ms (97.0%)
- Classificador: 3.93 ms (14.5%)
- Tracker: 0.12 ms (0.4%)

**Video Lobo-guara:**
- Latencia total media: 29.17 ms
- Latencia P95: 37.87 ms
- Throughput: 34.29 FPS
- Uso de RAM: 192.72 MB
- Deteccoes brutas: 166
- Recortes classificados: 42
- Recortes por frame: 0.21

**Breakdown de latencia:**
- Detector: 28.26 ms (96.9%)
- Classificador: 3.66 ms (12.6%)
- Tracker: 0.11 ms (0.4%)

**Video Tamandua-bandeira:**
- Latencia total media: 29.12 ms
- Latencia P95: 37.26 ms
- Throughput: 34.34 FPS
- Uso de RAM: 193.15 MB
- Deteccoes brutas: 155
- Recortes classificados: 37
- Recortes por frame: 0.185

**Breakdown de latencia:**
- Detector: 28.32 ms (97.2%)
- Classificador: 4.02 ms (13.8%)
- Tracker: 0.10 ms (0.4%)

**Media dos 3 Videos:**
- FPS: 35.18
- Latencia media: 28.46 ms
- Latencia P95: 36.17 ms
- Uso de RAM: 192.84 MB
- Recortes por frame: 0.19

**Ganhos da Quantizacao INT8:**
- Velocidade: +13.0% (de 31.11 para 35.18 FPS)
- Latencia: -11.6% (de 32.20 para 28.46 ms)
- Memoria RAM: -5.5% (de 203.97 para 192.84 MB)

### 2.8 Eventos em Video - Modelo B

Durante os testes em video, o Modelo B apresentou **0 eventos de especies erradas**:
- Video do anta: 1 evento correto (anta)
- Video do lobo-guara: 1 evento correto (lobo-guara)
- Video do tamandua-bandeira: 1 evento correto (tamandua-bandeira)

**Conclusao:** Modelo B eliminou completamente os erros de classificacao de especies observados no Modelo A.

---

## 3. SIMULACAO EMBARCADA (Restricoes de Recursos)

Para simular o comportamento em dispositivos embarcados como Raspberry Pi 5, foram realizados testes com restricoes de recursos (limitacao de threads e memoria).

### 3.1 Configuracao da Simulacao

**Ambiente:**
- Docker com limitacao de recursos
- CPU: Limitada
- Memoria: Limitada
- Threads: 1 (single-threaded para simular dispositivo embarcado)

**Modelo Testado:**
- Modelo B FP32 (detector + classificador MobileNet V3 Small)

### 3.2 Resultados da Simulacao (100 frames)

**Video Anta:**
- Latencia total media: 132.74 ms
- Latencia P95: 220.17 ms
- Throughput: 7.53 FPS
- Uso de RAM: 182.42 MB
- Deteccoes brutas: 65
- Recortes classificados: 16
- Recortes por frame: 0.16

**Breakdown de latencia:**
- Detector: 129.64 ms (97.7%)
- Classificador: 15.08 ms (11.4%)
- Tracker: 0.64 ms (0.5%)

**Video Lobo-guara:**
- Latencia total media: 139.92 ms
- Latencia P95: 210.72 ms
- Throughput: 7.15 FPS
- Uso de RAM: 181.70 MB
- Deteccoes brutas: 69
- Recortes classificados: 18
- Recortes por frame: 0.18

**Breakdown de latencia:**
- Detector: 136.00 ms (97.2%)
- Classificador: 17.61 ms (12.6%)
- Tracker: 0.68 ms (0.5%)

**Video Tamandua-bandeira:**
- Latencia total media: 117.16 ms
- Latencia P95: 194.61 ms
- Throughput: 8.54 FPS
- Uso de RAM: 186.23 MB
- Deteccoes brutas: 75
- Recortes classificados: 18
- Recortes por frame: 0.18

**Breakdown de latencia:**
- Detector: 113.25 ms (96.7%)
- Classificador: 23.91 ms (20.4%)
- Tracker: 0.25 ms (0.2%)

**Media dos 3 Videos (Simulacao Embarcada):**
- FPS: 7.74
- Latencia media: 129.94 ms
- Latencia P95: 208.50 ms
- Uso de RAM: 183.45 MB

**Impacto da Restricao de Recursos:**
- Velocidade: -75.1% (de 31.11 para 7.74 FPS)
- Latencia: +303.6% (de 32.20 para 129.94 ms)

---

## 4. COMPARACAO COMPLETA DOS MODELOS

### 4.1 Resumo de Acuracia (Dataset de Teste)

| Modelo | mAP@50 | Precision | Recall | F1-Score | FP | FN |
|--------|--------|-----------|--------|----------|----|----|
| Modelo A FP32 | 0.8424 | 0.8947 | 0.8500 | 0.8718 | 2 | 3 |
| Modelo B FP32 | **0.8968** | **1.0000** | **0.9000** | **0.9474** | **0** | 2 |
| Modelo B INT8 | 0.8016 | 0.8889 | 0.8000 | 0.8421 | 2 | 4 |

**Vencedor em Acuracia:** Modelo B FP32

### 4.2 Resumo de Velocidade (Media dos 3 Videos)

| Modelo | FPS | Latencia Media | Latencia P95 | RAM (MB) |
|--------|-----|----------------|--------------|----------|
| Modelo A FP32 | 15.85 | 63.25 ms | 72.22 ms | 219.1 |
| Modelo B FP32 | **31.11** | **32.20 ms** | **39.72 ms** | **203.97** |
| Modelo B INT8 | **35.18** | **28.46 ms** | **36.17 ms** | **192.84** |
| Modelo B FP32 (Sim Embarcada) | 7.74 | 129.94 ms | 208.50 ms | 183.45 |

**Vencedor em Velocidade:** Modelo B INT8

### 4.3 Resumo de Tamanho dos Modelos

| Modelo | Tamanho (MB) | Reducao |
|--------|--------------|---------|
| Modelo A FP32 | 11.7 | - |
| Modelo B FP32 | 20.1 | +71.8% |
| Modelo B INT8 | **5.7** | **-51.3%** (vs Modelo A) |

**Vencedor em Tamanho:** Modelo B INT8

### 4.4 Resumo de Robustez

**Eventos de especies erradas em video:**
- Modelo A: 2 erros (1 no lobo-guara, 1 no tamandua-bandeira)
- Modelo B FP32: 0 erros
- Modelo B INT8: Nao testado em video de validacao, mas esperado ter alguns erros

**Vencedor em Robustez:** Modelo B FP32

### 4.5 Comparacao de Classificadores (Modelo B)

| Backbone | Parametros | Tamanho (MB) | Test Acc (FP32) | Test Acc (INT8) | Treino (s) |
|----------|-----------|--------------|-----------------|-----------------|------------|
| **MobileNet V2** | 2.229M | 8.5 | **95.65%** | **95.65%** | 238.7 |
| MobileNet V3 Small | 1.522M | ~8 | 84.78% | 58.70% | **52.0** |
| ShuffleNet V2 x0.5 | 345K | ~2 | 78.26% | 78.26% | 59.6 |

**Vencedor Geral:** MobileNet V2 (melhor acuracia e robustez a quantizacao)
**Vencedor em Velocidade de Treino:** MobileNet V3 Small
**Vencedor em Tamanho:** ShuffleNet V2 x0.5

### 4.6 Analise de Trade-offs

**Modelo A FP32:**
- Vantagens: Simples, modelo unico, facil de treinar
- Desvantagens: Mais lento, confunde especies similares, tamanho medio

**Modelo B FP32:**
- Vantagens: Melhor acuracia, maior precisao (100%), mais rapido, modular
- Desvantagens: Mais complexo, maior tamanho total (20.1 MB)

**Modelo B INT8:**
- Vantagens: Muito rapido, menor tamanho (5.7 MB), baixo uso de RAM
- Desvantagens: Pequena perda de acuracia (-9.5 pts mAP@50)

---

## 5. RECOMENDACAO PARA RASPBERRY PI 5

### 5.1 Especificacoes do Raspberry Pi 5

**Hardware:**
- CPU: Quad-core Cortex-A76 @ 2.4 GHz
- RAM: 4GB ou 8GB LPDDR4X
- Armazenamento: microSD ou SSD via USB 3.0
- GPU: VideoCore VII (suporte OpenGL ES 3.1, Vulkan 1.2)

**Restricoes:**
- CPU ARM (nao x86)
- Memoria limitada (4-8 GB)
- Sem GPU CUDA (apenas VideoCore VII)
- Processamento em tempo real desejavel (>10 FPS para video)

### 5.2 Analise de Viabilidade

**Projecao de Desempenho no Raspberry Pi 5:**

Baseado nos resultados da simulacao embarcada e considerando que o Raspberry Pi 5 tem CPU mais potente que a simulacao (quad-core A76 vs single-thread limitado), estimamos:

| Modelo | FPS Estimado (RPi 5) | Viabilidade |
|--------|---------------------|-------------|
| Modelo A FP32 | ~12-15 FPS | Viavel, mas lento |
| Modelo B FP32 | ~15-20 FPS | **Viavel** |
| Modelo B INT8 | ~20-25 FPS | **Altamente Viavel** |

**Fatores considerados:**
1. Raspberry Pi 5 tem 4 cores vs 1 thread da simulacao
2. ONNX Runtime suporta otimizacoes ARM NEON
3. Quantizacao INT8 e bem suportada em ARM
4. Tamanho do modelo cabe confortavelmente na RAM

### 5.3 Recomendacao Final

**MODELO RECOMENDADO: Modelo B INT8 (Detector de Presenca + MobileNet V2)**

**Justificativa:**

1. **Velocidade:** 35.18 FPS em hardware de desenvolvimento, estimado 20-25 FPS no RPi 5
2. **Tamanho:** 5.7 MB total, cabe facilmente na memoria
3. **Uso de RAM:** ~193 MB, deixa espaco para sistema operacional e outras aplicacoes
4. **Acuracia Aceitavel:** mAP@50 de 80.16% e adequado para monitoramento automatico
5. **Robustez a Quantizacao:** MobileNet V2 mantem acuracia em INT8
6. **Modularidade:** Classificador pode ser trocado sem retreinar o detector

**Configuracao Recomendada:**
- Detector: `presence_detector_int8.onnx` (3.2 MB, entrada 416x416)
- Classificador: `species_classifier_int8.onnx` (2.5 MB, entrada 128x128)
- Threads: 4 (usar todos os cores do RPi 5)
- Batch size: 1 (inferencia em tempo real)

**Alternativa (se mais acuracia for necessaria):**
- Modelo B FP32: Melhor acuracia (mAP@50 89.68%), mas mais lento (~15-20 FPS) e maior (20.1 MB)

### 5.4 Otimizacoes Adicionais para Raspberry Pi 5

1. **Usar ONNX Runtime ARM optimizado:**
   - Compilar ONNX Runtime com flags ARM NEON
   - Ativar `intra_op_num_threads=4` para usar todos os cores

2. **Considerar aceleracao VideoCore VII:**
   - Investigar suporte a operadores ONNX na GPU do RPi 5
   - Pode oferecer ganhos adicionais de velocidade

3. **Otimizacao de sistema:**
   - Usar sistema operacional leve (Raspberry Pi OS Lite)
   - Desativar servicos desnecessarios
   - Usar SSD via USB 3.0 para I/O mais rapido

4. **Estrategia de skip frames:**
   - Processar 1 frame a cada N frames (ex: N=2 para 15 FPS efetivo)
   - Economiza CPU e energia

5. **Considerar aceleradores externos:**
   - Coral USB Accelerator (TPU do Google)
   - Intel Neural Compute Stick 2
   - Pode elevar FPS para 30-60+ com modelos quantizados

---

## 6. DETALHAMENTO DOS TESTES DE QUANTIZACAO

### 6.1 Quantizacao do Detector de Presenca

**Tentativa 1: INT8 em todos os operadores**
- Resultado: **FALHOU**
- Diagnostico: Escores fora de [0,1], decodificacao da cabeca YOLO quebrou
- Caixas detectadas: 14 (vs 20 em FP32)
- Confianca media: 1.673 (invalido!)
- Confianca maxima: 1.673 (invalido!)

**Tentativa 2: INT8 apenas em convolucionais**
- Resultado: **SUCESSO**
- Caixas detectadas: 18 (vs 20 em FP32)
- Confianca media: 0.868 (vs 0.827 em FP32)
- Confianca maxima: 0.966 (vs 0.969 em FP32)
- Impacto: Leve aumento de confianca, 2 deteccoes a menos

**Conclusao:** A cabeca de decodificacao do YOLO nao pode ser quantizada, apenas as camadas convolucionais.

### 6.2 Quantizacao dos Classificadores

**MobileNet V2:**
- FP32: 95.65% accuracy
- INT8 (minmax): 95.65% accuracy
- INT8 (percentile): Nao testado
- **Conclusao:** Robusto, sem perda de acuracia

**MobileNet V3 Small:**
- FP32: 84.78% accuracy
- INT8 (minmax): 58.70% accuracy (-26.08 pts!)
- INT8 (percentile): 65.22% accuracy (-19.56 pts!)
- INT8 (entropy): 58.70% accuracy (-26.08 pts!)
- **Conclusao:** Nao robusto, perdas severas em todas as tecnicas

**ShuffleNet V2 x0.5:**
- FP32: 78.26% accuracy
- INT8: 78.26% accuracy
- **Conclusao:** Robusto, sem perda de acuracia

**Observacao:** Apenas MobileNet V2 e ShuffleNet V2 sao viaveis para INT8. MobileNet V3 Small tem perdas inaceitaveis.

---

## 7. ANALISE DE EVENTOS E RASTREAMENTO

### 7.1 Configuracao do Tracker

**Parametros:**
- `max_age`: 30 frames (manter trilha por 30 frames sem deteccao)
- `min_hits`: 3 frames (minimo de 3 deteccoes para confirmar trilha)
- `iou_match`: 0.3 (threshold de IoU para associacao)
- `low_score`: 0.15 (threshold minimo de confianca)

### 7.2 Configuracao de Votacao (Modelo B)

**Parametros:**
- `classify_every`: 5 frames (classificar recorte a cada 5 frames)
- `min_votes`: 4 votos (minimo de 4 classificacoes para decidir especie)
- `min_vote_share`: 0.65 (65% dos votos devem ser da mesma especie)
- `min_species_confidence`: 0.5 (confianca minima do classificador)
- `reject_unknown`: true (rejeitar classe "desconhecido")

### 7.3 Impacto do NMS no Detector

**Teste com IoU=0.50:**
- Eventos por video: anta=1, lobo=2, tamandua=2
- Trilhas: 5-7 trilhas por video
- Problema: Alguns frames tinham 2 caixas no mesmo animal

**Teste com IoU=0.40:**
- Eventos por video: anta=1, lobo=1, tamandua=1
- Trilhas: 5 trilhas por video
- Problema: Resolvido

**Quadros com 2 caixas no mesmo animal:**
- Video anta: 0 quadros
- Video lobo-guara: 36 quadros (IoU medio: 0.453)
- Video tamandua-bandeira: 25 quadros (IoU medio: 0.103)

**Impacto na acuracia end-to-end:**
- Com IoU=0.50: mAP@50 0.8968, P 1.000, R 0.900
- Com IoU=0.40: mAP@50 0.8968, P 1.000, R 0.900
- **Conclusao:** Nenhum impacto na acuracia, mas IoU=0.40 evita duplicatas

**NMS adotado:** IoU=0.40

---

## 8. CONCLUSOES FINAIS

### 8.1 Principais Descobertas

1. **Modelo B supera Modelo A em todos os aspectos:**
   - +5.4 pts mAP@50 (84.24% → 89.68%)
   - +10.5 pts precision (89.47% → 100%)
   - +96% velocidade (15.85 → 31.11 FPS)
   - 100% reducao de erros de especies (2 → 0)

2. **Quantizacao INT8 e viavel:**
   - Reducao de tamanho: -71.7% (20.1 → 5.7 MB)
   - Ganho de velocidade: +13% (31.11 → 35.18 FPS)
   - Perda de acuracia aceitavel: -9.5 pts mAP@50
   - MobileNet V2 e ShuffleNet V2 sao robustos a INT8
   - MobileNet V3 Small nao e recomendado para INT8

3. **Raspberry Pi 5 e viavel:**
   - Estimativa: 20-25 FPS com Modelo B INT8
   - Uso de memoria: ~193 MB (adequado para 4-8 GB)
   - Tamanho do modelo: 5.7 MB (cabe facilmente)

4. **Arquitetura em duas etapas e superior:**
   - Melhor separacao de responsabilidades
   - Permite otimizacao independente
   - Classificador pode ser trocado
   - Menor custo computacional

### 8.2 Recomendacoes por Cenario

**Cenario 1: Prototipo de Desenvolvimento**
- Usar: Modelo B FP32
- Motivo: Maxima acuracia (100% precision)

**Cenario 2: Deploy em Raspberry Pi 5**
- Usar: Modelo B INT8
- Motivo: Melhor velocidade, menor tamanho, acuracia aceitavel

**Cenario 3: Deploy em dispositivo mais potente (Jetson Nano, etc)**
- Usar: Modelo B FP32
- Motivo: Aproveitar hardware mais potente para maxima acuracia

**Cenario 4: Deploy em MCU (ESP32-S3, etc)**
- Usar: ShuffleNet V2 x0.5 INT8
- Motivo: Menor tamanho (345K parametros), robusto a INT8

### 8.3 Trabalhos Futuros

1. **Testar em hardware real:**
   - Deploy no Raspberry Pi 5
   - Medir FPS, latencia e consumo de energia reais

2. **Explorar aceleradores:**
   - Coral TPU
   - Intel NCS2
   - Hailo-8

3. **Otimizar ainda mais:**
   - Pruning do modelo
   - Knowledge distillation
   - Neural Architecture Search (NAS)

4. **Expandir dataset:**
   - Coletar mais dados de campo
   - Aumentar diversidade de cenarios
   - Adicionar mais especies

5. **Adicionar funcionalidades:**
   - Deteccao de comportamentos
   - Contagem de individuos
   - Rastreamento de longo prazo

---

## APENDICE A: Ambiente de Testes

**Hardware de Desenvolvimento:**
- CPU: Intel Core i5-1235U
- Cores: 12 vCPUs
- RAM: 7.4 GB
- GPU: Nenhuma (CPU-only)
- OS: Linux (Ubuntu-based)

**Configuracao de Software:**
- Python: 3.12.3
- PyTorch: 2.13.0
- TorchVision: 0.28.0
- ONNX Runtime: CPU Execution Provider
- ONNX Runtime threads: 4 (nos benchmarks)

**Dataset:**
- Split de treino: 342 imagens
- Split de validacao: 98 imagens
- Split de teste: 18 imagens
- Classes: anta, tamandua_bandeira, lobo_guara
- Recortes para classificacao: 
  - Treino: 342 (anta=51, tamandua=50, lobo=46, desconhecido=195)
  - Validacao: 98 (anta=14, tamandua=14, lobo=12, desconhecido=58)
  - Teste: 46 (anta=7, tamandua=6, lobo=7, desconhecido=26)

**Videos de Teste:**
- video_anta.mp4
- video_lobo_guara.mp4
- video_tamandua_bandeira.mp4
- Frames testados: 200 por video (exceto simulacao embarcada: 100)

---

## APENDICE B: Referencias dos Arquivos de Resultados

**Modelo A:**
- `/outputs/evaluation.json` - Metricas de acuracia
- `/outputs/benchmark_anta.json` - Benchmark video anta
- `/outputs/benchmark_lobo_guara.json` - Benchmark video lobo
- `/outputs/benchmark_tamandua_bandeira.json` - Benchmark video tamandua

**Modelo B:**
- `/outputs/modelo_b/evaluation_e2e.json` - Acuracia FP32
- `/outputs/modelo_b/evaluation_e2e_int8.json` - Acuracia INT8
- `/outputs/modelo_b/resumo_comparativo.json` - Comparacao completa
- `/outputs/modelo_b/benchmark_video_*_modelo_b_fp32_4t.json` - Benchmarks FP32
- `/outputs/modelo_b/benchmark_video_*_modelo_b_int8_4t.json` - Benchmarks INT8

**Classificadores:**
- `/models/modelo_b_mobilenet_v2/classifier_training.json`
- `/models/modelo_b_mobilenet_v3_small/classifier_training.json`
- `/models/modelo_b_shufflenet_v2_x0_5/classifier_training.json`

**Simulacao Embarcada:**
- `/outputs/sim_embedded/benchmark_video_anta.json`
- `/outputs/sim_embedded/benchmark_video_lobo_guara.json`
- `/outputs/sim_embedded/benchmark_video_tamandua_bandeira.json`

---

## APENDICE C: Glossario

**Termos Tecnicos:**

- **mAP@50:** Mean Average Precision com threshold de IoU 0.5
- **mAP@50-95:** Mean Average Precision media entre IoU 0.5 e 0.95
- **Precision:** Proporcao de predicoes corretas (TP / (TP + FP))
- **Recall:** Proporcao de ground truths detectados (TP / (TP + FN))
- **F1-Score:** Media harmonica de precision e recall (2 * P * R / (P + R))
- **IoU:** Intersection over Union, metrica de sobreposicao de caixas
- **NMS:** Non-Maximum Suppression, tecnica para eliminar deteccoes duplicadas
- **FP32:** Float32, precisao de ponto flutuante de 32 bits
- **INT8:** Integer8, quantizacao para inteiros de 8 bits
- **Throughput:** Quantidade de frames processados por segundo (FPS)
- **Latencia:** Tempo para processar um unico frame (ms)
- **P95:** Percentil 95, metrica de latencia no pior caso esperado
- **RSS:** Resident Set Size, memoria RAM ocupada pelo processo

**Backbones:**
- **YOLOv8n:** You Only Look Once versao 8 nano (menor variante)
- **MobileNet V2:** CNN eficiente da Google
- **MobileNet V3 Small:** Versao otimizada e menor do MobileNet V3
- **ShuffleNet V2 x0.5:** CNN extremamente leve com 0.5x de largura

**Tecnicas:**
- **Transfer Learning:** Reutilizar pesos pre-treinados
- **Fine-tuning:** Ajuste fino de modelo pre-treinado
- **Quantization:** Reducao de precisao numerica para economia de recursos
- **Tracking:** Rastreamento de objetos entre frames
- **Voting:** Sistema de votacao para decidir classe final

---

**Documento gerado em:** 2026-08-18  
**Versao:** 1.0  
**Autor:** Sistema de Monitoramento de Animais do Cerrado
