# Comparação entre Modelo A e Modelo B

Este documento apresenta a comparação detalhada entre as duas arquiteturas de detecção implementadas neste projeto de TCC.

---

## Sobre os Modelos

### Modelo A: Detector Multiclasse (Baseline)

O Modelo A é a arquitetura baseline do projeto, baseada em um **YOLOv8n** que resolve detecção e classificação em um único estágio. O modelo recebe imagens de 640×640 pixels e produz diretamente as caixas delimitadoras com a espécie detectada (3 classes: anta, tamanduá-bandeira, lobo-guará).

**Características:**
- Arquitetura: YOLOv8n (3 classes)
- Resolução de entrada: 640×640 px
- Formato: ONNX FP32
- Tamanho: 11.7 MB
- Detecção e classificação embutidas na mesma cabeça YOLO
- Confirmação de eventos por 3 quadros consecutivos

**Motivação:**
É a abordagem padrão para detecção de objetos: simples, direta e eficiente quando há dados suficientes para todas as classes.

---

### Modelo B: Detector de Presença + Classificador (Arquitetura em Dois Estágios)

O Modelo B separa a tarefa em duas etapas especializadas:

1. **Estágio 1 (Detector de Presença):** Um YOLOv8n simplificado detecta apenas "animal" (1 classe), em resolução menor (416×416 px). Com todas as 147 caixas de treino trabalhando no mesmo objetivo, o detector alcança mAP50 = 0.982.

2. **Estágio 2 (Classificador de Espécie):** Um MobileNetV2 processa recortes de 128×128 px e classifica em **4 classes**: as 3 espécies conhecidas + `desconhecido` (classe de rejeição). O classificador foi treinado com 486 recortes, incluindo 279 exemplos negativos gerados automaticamente (fundo e caixas desalinhadas).

3. **Rastreamento e Votação:** O sistema usa ByteTrack-lite para rastrear indivíduos ao longo do tempo. A espécie final sai de uma votação ponderada (mínimo 4 votos, 65% de margem), evitando decisões apressadas.

**Características:**
- Detector: YOLOv8n (1 classe, 416 px)
- Classificador: MobileNetV2 (128 px, 4 classes)
- Formato: ONNX FP32 e INT8
- Tamanho FP32: 20.1 MB (11.6 MB detector + 8.5 MB classificador)
- Tamanho INT8: 5.7 MB (3.2 MB detector + 2.5 MB classificador)
- Rastreamento por indivíduo (ByteTrack-lite)
- Decisão por votação com quórum e margem

**Motivação:**
O dataset é pequeno (147 caixas de treino). Dividir essas caixas em 3 classes gasta o pouco dado disponível no problema errado. Detectar "animal" é fácil (mAP50 = 0.982); classificar a espécie é o desafio real. O Modelo B concentra o esforço onde importa e adiciona mecanismo de rejeição explícito.

---

## Comparação de Acurácia

Medido no **split test** (18 imagens, 20 caixas), usando o mesmo código de métrica para ambos os modelos.

| Modelo | mAP50 | Precisão | Recall | F1-Score | Falsos Positivos | Falsos Negativos |
|--------|-------|----------|--------|----------|------------------|------------------|
| **Modelo A FP32** (640, 3 classes) | 0.842 | 0.895 | 0.850 | 0.872 | 2 | 3 |
| **Modelo B FP32** (416 + MobileNetV2) | **0.897** | **1.000** | **0.900** | **0.947** | **0** | 2 |
| **Modelo B INT8** | 0.802 | 0.889 | 0.800 | 0.842 | 2 | 4 |

### AP50 por Classe

| Classe | Modelo A FP32 | Modelo B FP32 | Diferença |
|--------|---------------|---------------|-----------|
| anta | 0.837 | 0.857 | +0.020 |
| tamandua_bandeira | 0.833 | 0.833 | 0.000 |
| lobo_guara | 0.857 | **1.000** | **+0.143** |

**Destaque:** O Modelo B alcançou **precisão perfeita (1.000)** - zero falsos positivos de espécie. Isso ocorre porque o classificador pode recusar detecções ruins através da classe `desconhecido`, algo que o Modelo A não consegue fazer.

---

## Comparação de Desempenho

Medido em **Intel Core i5-1235U** (x86_64, 12 vCPUs, 7.4 GB RAM, sem GPU), processando 200 quadros dos 3 vídeos de amostra, com `intra_op_threads = 4` para todos os modelos.

| Modelo | FPS | Latência Média | p95 | RAM de Pico | Tamanho Total |
|--------|-----|----------------|-----|-------------|---------------|
| **Modelo A FP32** | 15.9 | 63.2 ms | 72.2 ms | 219 MB | 11.7 MB |
| **Modelo B FP32** | **31.1** | **32.2 ms** | 39.7 ms | 204 MB | 20.1 MB |
| **Modelo B INT8** | **35.2** | **28.5 ms** | 36.2 ms | **193 MB** | **5.7 MB** |

### Ganhos do Modelo B

- **Modelo B FP32:** 1.96× mais rápido que o Modelo A
- **Modelo B INT8:** 2.22× mais rápido que o Modelo A
- **Modelo B INT8:** Ocupa menos da metade do espaço (5.7 MB vs 11.7 MB)

### Latência por Estágio (Modelo B)

| Versão | Detector (ms) | Classificador (ms) | Recortes/Quadro |
|--------|---------------|--------------------|--------------------|
| FP32 | 31.2 | 4.9 | 0.187 |
| INT8 | 27.6 | 3.9 | 0.190 |

O classificador roda apenas a cada 5 quadros **por trilha** (~19% dos quadros nos vídeos de teste). Em campo, onde a maioria dos quadros está vazia, esse custo tende a zero.

---

## Comparação em Vídeos Reais

Três vídeos de amostra foram processados (1280×720, ~240 quadros cada). Os vídeos contêm cortes de cena (7-12 cortes por vídeo) e sobreposições gráficas.

| Vídeo | Modelo A | Modelo B | Observação |
|-------|----------|----------|------------|
| `video_anta` | anta (0.98) | anta | Correto em ambos |
| `video_lobo_guara` | lobo_guara (0.96)<br/>+ **anta FALSA (0.73)** | lobo_guara | Modelo A confundiu |
| `video_tamandua_bandeira` | tamandua_bandeira (0.86)<br/>+ **anta FALSA (0.42)** | tamandua_bandeira | Modelo A confundiu |

**Total de eventos de espécie errada:**
- Modelo A: **2 eventos falsos**
- Modelo B: **0 eventos falsos**

Todos os 5 avistamentos do Modelo B saíram de votação **unânime** (4 votos, 100% de consenso na mesma espécie).

---

## Quantização INT8: Achados Importantes

### Achado 1: A Cabeça YOLO Não Sobrevive à Quantização Total

| Detector | Caixas (de 20 reais) | Confiança Média | Confiança Máxima |
|----------|----------------------|-----------------|------------------|
| FP32 | 20 | 0.827 | 0.969 |
| INT8 (todos operadores) | 14 | **1.673** | **1.673** |
| INT8 (apenas Conv) | 18 | 0.868 | 0.966 |

Confiança 1.673 é impossível (deveria estar em [0, 1]). A decodificação da cabeça YOLO (sigmoide, concatenação, DFL) não sobrevive à quantização de ativações. A solução foi restringir a quantização apenas aos operadores `Conv` (`--op-types Conv`).

### Achado 2: MobileNetV2 Sobrevive à Quantização; MobileNetV3 Não

Acurácia no conjunto de teste (46 recortes):

| Backbone | FP32 | INT8 MinMax | INT8 Percentil | INT8 Entropia |
|----------|------|-------------|----------------|---------------|
| **MobileNetV2** | 0.957 | **0.957** | 0.957 | — |
| MobileNetV3-Small | 0.848 | 0.587 | 0.652 | 0.587 |
| ShuffleNetV2 x0.5 | 0.783 | 0.783 | — | — |

**Razão:** O hard-swish e os blocos squeeze-excite do MobileNetV3 têm faixa dinâmica larga e colapsam na quantização pós-treino. MobileNetV2 e ShuffleNetV2 usam ReLU/ReLU6 e atravessam sem perda.

**Consequência prática:** Em dispositivos embarcados de baixo custo, o modelo roda em INT8. Logo, o MobileNetV2 foi escolhido para o Modelo B (0.957 em INT8 vs 0.587 do MobileNetV3).

---

## Resumo Executivo

### Vencedor em Acurácia: **Modelo B FP32**
- mAP50: 0.897 (vs 0.842 do Modelo A)
- Precisão: 1.000 (zero falsos positivos)
- Recall: 0.900
- AP50 para lobo-guará: 1.000 (perfeito)

### Vencedor em Desempenho: **Modelo B INT8**
- 35.2 FPS (2.22× mais rápido que Modelo A)
- 28.5 ms de latência (vs 63.2 ms)
- 5.7 MB de tamanho (vs 11.7 MB)
- Acurácia aceitável (0.802 mAP50)

### Vencedor em Robustez: **Modelo B**
- 0 eventos de espécie errada (vs 2 do Modelo A)
- Classe `desconhecido` permite rejeitar detecções ruins
- Rastreamento por indivíduo evita contagem duplicada
- Votação com quórum e margem previne decisões apressadas

---

## Limitações e Contexto

1. **O dataset não representa armadilha fotográfica real:** Área mediana das caixas = 0.47 (animal ocupa metade do quadro). Em campo real, o animal ocupa poucos pixels, está distante, à noite, em infravermelho.

2. **Split de teste pequeno:** 20 caixas. Uma caixa vale 5 pontos percentuais de recall. Todas as diferenças estão dentro da margem estatística.

3. **14 imagens são geradas por IA:** Imagens `Gemini_Generated_Image_*` no dataset. Isso contamina afirmações sobre desempenho em campo.

4. **Medições de FPS são de x86_64:** A razão entre modelos é estável, mas os valores absolutos não se transferem para Raspberry Pi 5 ou MaixCAM. Medição final precisa ser feita na placa real.

5. **INT8 não validado em NPU:** Foi validado em ONNX Runtime CPU. NPUs como a do SG2002 (MaixCAM) podem introduzir perdas adicionais na conversão.

---

## Recomendação para o TCC

Para o **ambiente embarcado** alvo deste projeto:

**Use o Modelo B INT8** se:
- A prioridade é desempenho (FPS, latência, tamanho)
- O dispositivo tem recursos limitados (RAM, armazenamento)
- Pode tolerar perda leve de acurácia (0.802 vs 0.897 mAP50)

**Use o Modelo B FP32** se:
- A prioridade é máxima acurácia
- O dispositivo suporta modelos maiores (20 MB)
- Precisa de precisão perfeita (zero falsos positivos no teste)

**Evite o Modelo A** porque:
- É mais lento (15.9 FPS vs 35.2)
- Tem menor acurácia (0.842 mAP50)
- Produz falsos positivos de espécie (2 eventos errados nos vídeos)
- Não pode rejeitar detecções ruins (sem classe "desconhecido")

---

## Arquivos de Referência

- **Dados completos:** `outputs/modelo_b/resumo_comparativo.json`
- **Documentação do Modelo B:** `docs/MODELO_B.md`
- **Relatório de validação:** `outputs/relatorio_validacao.md`
- **Código do Modelo A:** `src/animal_monitor/`
- **Código do Modelo B:** `src/animal_monitor_b/`

---

## Como Reproduzir

### Modelo A
```bash
source .venv/bin/activate
python -m pip install -e ".[train]"
animal-monitor train --config config/train.yaml
animal-monitor evaluate --weights runs/train/baseline_yolov8n/weights/best.pt \
  --data config/dataset.yaml --split test --output outputs/evaluation.json
```

### Modelo B
```bash
source .venv/bin/activate
python -m pip install -e ".[modelo-b]"
./scripts/modelo_b_pipeline.sh
```

Ou veja o passo a passo detalhado na seção 6 de `docs/MODELO_B.md`.
