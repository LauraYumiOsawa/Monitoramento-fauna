# 01c — Raspberry Pi 5 · MegaDetector + SpeciesNet · PyTorch CPU, em lote (arquitetura C)

> Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e versoes/gerador/especificacoes.yaml. Edite as entradas e regenere.

**Cruzamento:** Raspberry Pi 5 × MegaDetector + SpeciesNet — **faz sentido — gerado**. Arquitetura de energia: C — captura independente e classificação em lote.

## Por que este cruzamento faz sentido

Modelos grandes demais para tempo real, mas viáveis em lote no Pi 5 de 8 GB: classificação posterior das fotos de um subsistema de captura (arquitetura C) [4; 7; 8; 30].

Papel do hardware na pesquisa: Referência de sistema Linux flexível (arquitetura de referência da pesquisa).

## Ficha técnica

| Item | Valor |
|---|---|
| Hardware | BCM2712, 4× Cortex-A76 a 2,4 GHz, 4/8 GB; conectores de câmera (CSI) e PCIe [12] |
| Câmera | — (processa as fotos de outro subsistema de captura) |
| Modelo | MegaDetector (animal/pessoa/veículo) + SpeciesNet (espécie), em lote |
| Runtime / artefato | PyTorch CPU, em lote (arquitetura C) |
| Disparo | fotos de um subsistema de captura (ex.: 09a) → lote MegaDetector + SpeciesNet → Camtrap DP |
| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |

## Dados para o ranking

Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** (proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita em `versoes/ranking/candidatos.yaml`).

| Grandeza | Valor | Tipo | Fonte/premissa |
|---|---|---|---|
| Custo da estação de lote (R$) | R$ 2.655,90 | documentado/estimado | candidatos.yaml |
| Tempo por imagem (s) | 15,0 s | estimado | 2,3 s/imagem medidos em x86 (i5-1235U, 12 threads) × ~6,5 (Pi 5, premissa) |

**Fora dos rankings de armadilha completa:** estação de classificação em lote (arquitetura C): complementa uma armadilha, não é uma.

## Passo a passo

### 1. Instalar

```bash
./instalar_lote.sh   # venv + PyTorch (cpu) + speciesnet 5.0.6
```

### 2. Executar

```bash
./executar_lote.sh /media/cartao/DCIM          # MegaDetector v5a + SpeciesNet v4.0.3a, --country BRA
./executar_lote.sh /media/cartao/DCIM --detector   # só o filtro de vazias
.venv/bin/python importar_resultados.py dados/speciesnet_*.json --latitude <lat> --longitude <lon>
```

Alternativa mais leve para o Pi: os MegaDetector MDv1000-spruce (YOLOv5s, 640 px, 16,5 GFLOPs) e -sorrel (YOLOv11s, 960 px) do pacote `megadetector`.

## Limitações e o que não foi validado

- **Medido no projeto (x86, i5-1235U):** nas 18 imagens de teste o MegaDetector localizou os 20 animais (presença 20/20) e o SpeciesNet acertou a espécie exata em 13 (F1 0,65); os erros foram arredondamentos para família ("tapiridae", "canine") ou rótulos sem sentido, nunca troca entre as três espécies. Lote de 18 imagens em ~42 s com 12 threads.
- As imagens de teste não são de armadilha fotográfica (animal em close, 14 geradas por IA): o resultado não estima o desempenho de campo do SpeciesNet.
- Pesos baixados do Kaggle na primeira execução (~500 MB). Licenças: SpeciesNet Apache-2.0; inferência do MDv5 via pacote yolov5 (GPL).

## Arquivos desta pasta

- `README.md`
- `executar_lote.sh`
- `importar_resultados.py`
- `instalar_lote.sh`
