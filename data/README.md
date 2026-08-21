# Dataset

O código espera o padrão YOLO de detecção:

```text
animals/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Cada imagem possui um `.txt` de mesmo nome. Cada linha do rótulo contém:

```text
class_id x_center y_center width height
```

As quatro coordenadas são normalizadas entre 0 e 1. Um arquivo vazio representa uma
imagem negativa, sem animais de interesse.

## Dataset atual

O dataset consolidado contém 180 imagens divididas em:
- **Train**: 126 imagens (70%)
- **Val**: 36 imagens (20%)
- **Test**: 18 imagens (10%)

### Classes

- **Classe 0**: anta (72 instâncias, 34.8%)
- **Classe 1**: tamandua_bandeira (70 instâncias, 33.8%)
- **Classe 2**: lobo_guara (65 instâncias, 31.4%)

Total: 207 instâncias anotadas

### Origem dos dados

1. **Anta** (60 imagens): Dataset pré-dividido em train/val/test
   - Origem: dataset_anta_yolo
   
2. **Tamanduá-bandeira** (60 imagens): Dividido automaticamente em 70/20/10
   - Origem: dataset_tamandua_bandeira_60
   - Classe original: 2 → ajustada para classe 1
   
3. **Lobo-guará** (60 imagens): Dividido automaticamente em 70/20/10
   - Origem: lobo_guara_yolo_labels_60 + imagens de /home/losawa/Imagens/TCC/loboGuara
   - Classe original: 1 → ajustada para classe 2
   - 60 imagens selecionadas por qualidade e adequação ao treinamento

O dataset está balanceado entre as três classes e pronto para treinamento.
