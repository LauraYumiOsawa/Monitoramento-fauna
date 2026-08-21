# Melhorias recomendadas

## Prioridade alta

1. **Fechar as classes somente depois de auditar os dados.** As três classes do exemplo
   (`anta`, `tamandua_bandeira` e `onca_pintada`) são configuráveis. A escolha final deve
   considerar quantidade de caixas, diversidade de cenários e licença das imagens.
2. **Separar por câmera, local, vídeo e data.** Quadros vizinhos do mesmo vídeo não podem
   cair em treino e teste. Use `split-dataset --group-regex` e depois rode
   `validate-dataset`. O código detecta cópias idênticas entre os splits.
3. **Revisar manualmente os rótulos.** Uma caixa errada pesa muito em uma base pequena.
   Faça amostragem visual de todas as classes e registre quem revisou cada lote.
4. **Adicionar exemplos negativos.** Inclua vegetação, sombras, pessoas, gado e animais
   não-alvo para medir e reduzir falsos positivos.
5. **Testar no hardware real.** QEMU valida arquitetura, dependências e execução, mas não
   representa FPS, RAM, energia, temperatura nem a câmera da Raspberry Pi ou MaixCAM.

## Experimentos mínimos para o TCC

| Experimento | Modelo | Entrada | Formato | Objetivo | Situação |
|---|---|---:|---|---|---|
| Baseline | YOLOv8n | 640 | PyTorch FP32 | Acurácia inicial | feito (`outputs/evaluation.json`) |
| Leve | YOLOv8n | 512 | ONNX FP32 | Reduzir latência | pendente |
| Comparativo | YOLOv8s | 640 | ONNX FP32 | Medir ganho de acurácia vs. custo | pendente |
| Otimizado | Melhor modelo | 512/640 | ONNX INT8 | Testar implantação embarcada | pendente para o Modelo A |
| Dois estágios | YOLOv8n 1 classe + MobileNetV2 | 416 / 128 | ONNX FP32 e INT8 | Separar localização de identificação | feito (`docs/MODELO_B.md`) |

Em todos os casos, registrar mAP50-95, precisão, recall, matriz de confusão, latência
p50/p95, FPS, tamanho do modelo e pico de RAM. Em hardware real, acrescentar consumo e
temperatura.

## Melhorias de produto

- ~~Trocar o bloqueio simples de alertas por rastreamento (ByteTrack) para contar
  indivíduos e reduzir duplicidade entre quadros.~~ Implementado no Modelo B
  (`src/animal_monitor_b/runtime/tracker.py`).
- Calibrar limiar de confiança por classe; uma única confiança global pode favorecer a
  classe mais fácil. No Modelo B isso virou votação por indivíduo com quórum e margem.
- Guardar `camera_id`, localização aproximada, versão do modelo e hash do arquivo em cada
  evento para garantir rastreabilidade científica.
- Criar uma tela Qt/QML que consulte o SQLite e mostre avistamentos, evidências e filtros.
- Adicionar watchdog, rotação de logs e limite de armazenamento antes de uso contínuo.
- Quantizar somente depois de comparar a queda de mAP. Um modelo menor não compensa se
  perder os animais mais raros. Medido no Modelo B: quantizar a cabeça do YOLO produz
  escores fora de [0, 1] (é preciso restringir a `Conv`), e o MobileNetV3-Small cai de
  0.848 para 0.587 de acurácia em INT8 enquanto o MobileNetV2 não perde nada. Detalhes em
  `docs/MODELO_B.md`, seção 4.3.

## Cuidados metodológicos

- O aviso de 25 caixas por classe no validador é apenas uma triagem estrutural, não uma
  quantidade suficiente para treinar um detector confiável. Busque centenas de exemplos
  variados por classe quando possível.
- Não use o conjunto de teste para escolher hiperparâmetros. Ajuste no `val` e abra o
  `test` apenas para a avaliação final.
- Congele versões, seed, YAML de treino, origem do dataset e pesos. O código já grava um
  `experiment_manifest.json` em cada treinamento.
- Confira as licenças do dataset, dos pesos e do Ultralytics antes de publicar ou usar o
  sistema fora do contexto acadêmico.
