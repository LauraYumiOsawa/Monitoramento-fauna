#!/usr/bin/env bash
# ============================================================================
# demo_qemu_docker.sh
# Demonstração passo a passo de QEMU (ARM64 emulado) e Docker (limites de CPU/RAM),
# feita para captura de tela: cada passo limpa a tela, mostra o comando executado,
# a saída e espera Enter antes do próximo.
#
# Uso:
#   ./scripts/demo_qemu_docker.sh              # todos os passos, com pausa
#   ./scripts/demo_qemu_docker.sh 1.2 2.3      # só os passos indicados
#   ./scripts/demo_qemu_docker.sh 2            # todos os passos do grupo 2
#   ./scripts/demo_qemu_docker.sh --lista      # lista os passos
#   ./scripts/demo_qemu_docker.sh --sem-pausa  # roda direto, sem limpar a tela
#
# Variáveis opcionais:
#   ANIMAL_MONITOR_QEMU_BIN    binário estático qemu-aarch64
#                              (padrão: ../../avaliacao_embarcada/qemu-aarch64)
#   ANIMAL_MONITOR_ARM64_IMAGE imagem ARM64 (padrão: animal-monitor:arm64)
#
# Não usa contêiner privilegiado nem registra binfmt no host: o qemu-aarch64 é
# montado no contêiner e usado como entrypoint. Tempos medidos sob emulação não
# representam a placa física.
# ============================================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU_BIN="${ANIMAL_MONITOR_QEMU_BIN:-$(cd "$PROJECT_DIR/../.." && pwd)/avaliacao_embarcada/qemu-aarch64}"
IMAGEM="${ANIMAL_MONITOR_ARM64_IMAGE:-animal-monitor:arm64}"
SAIDA="$PROJECT_DIR/outputs/demo"
APP=/opt/animal-monitor
PY=(/usr/local/bin/python python)  # este qemu-aarch64 espera: <executável> <argv0> <args...>
PAUSAR=1

if [[ -t 1 ]]; then
  NEGRITO=$'\033[1m'; FRACO=$'\033[2m'; VERDE=$'\033[32m'; VERMELHO=$'\033[31m'
  AMARELO=$'\033[33m'; CIANO=$'\033[36m'; AZUL=$'\033[34m'; RESET=$'\033[0m'
else
  NEGRITO=""; FRACO=""; VERDE=""; VERMELHO=""; AMARELO=""; CIANO=""; AZUL=""; RESET=""
fi

PASSOS=(0 1.1 1.2 1.3 1.4 2.1 2.2 2.3)
declare -A TITULOS=(
  [0]="Ambiente: PC x86_64, QEMU e imagem ARM64"
  [1.1]="QEMU: o contêiner roda em ARM64 (aarch64)"
  [1.2]="QEMU: núcleo de cada placa emulado (/proc/cpuinfo)"
  [1.3]="QEMU: sistema do TCC (Modelo A) iniciando em ARM64 — Raspberry Pi 5"
  [1.4]="QEMU: sistema do TCC (Modelo B) iniciando em ARM64 — Raspberry Pi Zero 2 W"
  [2.1]="Docker: benchmark no perfil Raspberry Pi 5 (4 CPUs, 4 GB) — Modelo A"
  [2.2]="Docker: limites em ação no perfil Pi Zero 2 W (1 CPU, 512 MB) — Modelo B INT8"
  [2.3]="Docker: limite de memória — Modelo A com 192 MB × 256 MB"
)

# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------

limpar_conteineres() {
  local nomes
  nomes="$(docker ps -aq --filter 'name=^tcc-demo-' 2>/dev/null)"
  [[ -n "$nomes" ]] && docker rm -f $nomes >/dev/null 2>&1
  return 0
}
trap 'printf "%s\n" "$RESET"; limpar_conteineres' EXIT
trap 'exit 130' INT TERM

cabecalho() { # $1 = id do passo
  [[ $PAUSAR == 1 ]] && clear
  local titulo="${TITULOS[$1]}"
  printf '%s%s━━ %s ━━ %s%s\n' "$NEGRITO" "$AZUL" "$1" "$titulo" "$RESET"
  printf '%sTCC · Monitoramento de fauna com visão computacional embarcada · %s%s\n\n' \
    "$FRACO" "$(date '+%d/%m/%Y %H:%M')" "$RESET"
}

explica() { printf '%s%s%s\n\n' "$FRACO" "$*" "$RESET"; }

# Mostra um comando legível: junta opção + valor, abrevia caminhos e quebra em ~90 colunas.
mostrar_cmd() { # $@ = comando; use SUFIXO="| head -9" para acrescentar texto
  local -a args=("$@")
  local n=${#args[@]} i=0 token atual='$' saida=""
  while (( i < n )); do
    token="${args[i]}"
    if [[ $token == -* && $((i + 1)) -lt $n && ${args[i + 1]} != -* ]]; then
      token="$token ${args[i + 1]}"
      (( i++ ))
    fi
    token="${token//$QEMU_BIN/\$Q}"
    token="${token//$PROJECT_DIR/\$PWD}"
    if (( ${#atual} + ${#token} + 1 > 90 )); then
      saida+="$atual \\"$'\n'
      atual="    $token"
    else
      atual+=" $token"
    fi
    (( i++ ))
  done
  [[ -n "${SUFIXO:-}" ]] && atual+=" $SUFIXO"
  printf '%s%s%s\n\n' "$CIANO" "$saida$atual" "$RESET"
}

# Monta o comando docker em CMD.  uso: montar_cmd <cpu-emulada> [opções docker...] -- <comando>
montar_cmd() {
  local cpu="$1"
  shift
  local -a extra=()
  while [[ $# -gt 0 && $1 != "--" ]]; do
    extra+=("$1")
    shift
  done
  shift
  CMD=(docker run --platform linux/arm64 --network none "${extra[@]}"
    -v "$QEMU_BIN:/qemu-aarch64:ro" --entrypoint /qemu-aarch64
    "$IMAGEM" -cpu "$cpu" "$@")
}

# Roda CMD em segundo plano com cronômetro; saída em $1. Define DURACAO.
executar_com_tempo() {
  local log="$1" inicio=$SECONDS pid status
  "${CMD[@]}" >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    [[ -t 1 ]] && printf '\r  %s⏳ executando em ARM64 emulado... %3d s%s' "$AMARELO" $((SECONDS - inicio)) "$RESET"
    sleep 1
  done
  wait "$pid"
  status=$?
  [[ -t 1 ]] && printf '\r\033[K'
  DURACAO=$((SECONDS - inicio))
  return "$status"
}

# Arquivos runtime.yaml com intra_op_threads = CPUs do perfil (como na placa).
preparar_runtime() { # $1 = threads
  mkdir -p "$SAIDA"
  sed "s/^intra_op_threads:.*/intra_op_threads: $1/" "$PROJECT_DIR/config/runtime.yaml" \
    >"$SAIDA/runtime_a_${1}threads.yaml"
  sed "s/^intra_op_threads:.*/intra_op_threads: $1/" "$PROJECT_DIR/config/modelo_b/runtime.yaml" \
    >"$SAIDA/runtime_b_${1}threads.yaml"
}

# Resumo legível da saída JSON do benchmark (lida de $1).
resumir_benchmark() {
  python3 - "$1" "${2:-}" <<'EOF'
import json, sys
texto = open(sys.argv[1], encoding="utf-8", errors="replace").read()
inicio = texto.find("{")
try:
    dados, _ = json.JSONDecoder().raw_decode(texto[inicio:]) if inicio >= 0 else ({}, 0)
except json.JSONDecodeError:
    dados = {}
if not dados:
    print("  (saída sem JSON de benchmark)")
    sys.exit(0)
lat = dados.get("latency_ms", {})
lat = lat.get("total", lat)
br = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")
det = dados.get("detections", dados.get("raw_detections"))
linhas = [
    ("Imagem / vídeo", str(dados.get("source", "")).split("/")[-1]),
    ("Quadros medidos", dados.get("measured_frames")),
    ("Detecções", det),
    ("Latência por quadro (p50)", f"{br(lat.get('p50', 0) / 1000)} s"),
    ("Vazão", f"{br(dados.get('throughput_fps', 0), 2)} FPS"),
    ("Pico de RAM do processo", f"{dados.get('peak_process_rss_mb', 0):.0f} MB"),
]
for rotulo, valor in linhas:
    print(f"  {rotulo:.<30} {valor}")
EOF
  printf '\n  %sJSON completo: %s%s\n' "$FRACO" "${1/#$PROJECT_DIR/.}" "$RESET"
}

ok() { printf '  %s✓ %s%s\n' "$VERDE" "$*" "$RESET"; }
erro() { printf '  %s✗ %s%s\n' "$VERMELHO" "$*" "$RESET"; }

pausa() {
  [[ $PAUSAR == 1 ]] || { echo; return 0; }
  printf '\n%s  ↵ Enter para o próximo passo (Ctrl+C para sair) %s' "$FRACO" "$RESET"
  read -r _ </dev/tty
}

# ----------------------------------------------------------------------------
# Passos
# ----------------------------------------------------------------------------

passo_0() {
  cabecalho 0
  explica "O computador é x86_64. O QEMU traduz instruções ARM64 para que a imagem Linux ARM64 do projeto rode nele."
  printf '%s$ uname -m%s\n' "$CIANO" "$RESET"
  uname -m
  printf '\n%s$ grep -m1 "model name" /proc/cpuinfo%s\n' "$CIANO" "$RESET"
  grep -m1 "model name" /proc/cpuinfo
  printf '\n%s$ $Q --version | head -1%s\n' "$CIANO" "$RESET"
  "$QEMU_BIN" --version | head -1
  printf '\n%s$ docker images %s%s\n' "$CIANO" "$IMAGEM" "$RESET"
  docker images "$IMAGEM"
  printf '\n%sQ = %s%s\n' "$FRACO" "$QEMU_BIN" "$RESET"
}

passo_1_1() {
  cabecalho 1.1
  explica "Mesmo comando 'uname -m' do passo anterior, agora dentro do contêiner ARM64 emulado como Cortex-A76 (Raspberry Pi 5). A primeira execução pode levar ~30 s."
  montar_cmd cortex-a76 --rm --name tcc-demo-uname -- /bin/uname uname -m
  mostrar_cmd "${CMD[@]}"
  "${CMD[@]}"
}

passo_1_2() {
  cabecalho 1.2
  explica "O QEMU emula o tipo de núcleo de cada placa. O campo 'CPU part' identifica o núcleo ARM (código do fabricante)."
  montar_cmd cortex-a76 --rm --name tcc-demo-cpuinfo -- /bin/cat cat /proc/cpuinfo
  SUFIXO="| head -9" mostrar_cmd "${CMD[@]}"
  "${CMD[@]}" | head -9
  printf '\n%sMesmo comando para cada perfil de placa:%s\n\n' "$NEGRITO" "$RESET"
  # "Núcleo" tem um caractere de 2 bytes: largura 15 no printf (que conta bytes) = 14 colunas.
  printf '  %s%-14s %-10s %-15s %s%s\n' "$NEGRITO" "-cpu" "CPU part" "Núcleo" "Placa(s)" "$RESET"
  local linha cpu nucleo placas parte
  for linha in "cortex-a76|Cortex-A76|Raspberry Pi 5, Orange Pi 5" \
    "cortex-a53|Cortex-A53|Raspberry Pi Zero 2 W" \
    "cortex-a78ae|Cortex-A78AE|NVIDIA Jetson Orin Nano"; do
    IFS='|' read -r cpu nucleo placas <<<"$linha"
    montar_cmd "$cpu" --rm --name "tcc-demo-cpuinfo-$cpu" -- /bin/cat cat /proc/cpuinfo
    parte="$("${CMD[@]}" 2>/dev/null | grep -m1 'CPU part' | awk '{print $NF}')"
    printf '  %-14s %-10s %-14s %s\n' "$cpu" "${parte:-?}" "$nucleo" "$placas"
  done
}

passo_1_3() {
  cabecalho 1.3
  explica "A CLI do Modelo A inicia dentro do Linux ARM64 e lista as dependências: NumPy, OpenCV e ONNX Runtime precisam estar 'true'."
  montar_cmd cortex-a76 --rm --name tcc-demo-info-a -- "${PY[@]}" -m animal_monitor info
  mostrar_cmd "${CMD[@]}"
  "${CMD[@]}"
}

passo_1_4() {
  cabecalho 1.4
  explica "A CLI do Modelo B (detector + classificador) no núcleo Cortex-A53, o mesmo do Raspberry Pi Zero 2 W."
  montar_cmd cortex-a53 --rm --name tcc-demo-info-b -- "${PY[@]}" -m animal_monitor_b info
  mostrar_cmd "${CMD[@]}"
  "${CMD[@]}"
}

passo_2_1() {
  cabecalho 2.1
  explica "Benchmark do Modelo A em uma imagem real do conjunto de teste, com o Docker limitando o contêiner a 4 CPUs e 4 GB de RAM (sem swap), como no Raspberry Pi 5."
  preparar_runtime 4
  montar_cmd cortex-a76 --rm --name tcc-demo-pi5 --cpus 4 --memory 4g --memory-swap 4g \
    -v "$PROJECT_DIR/data/animals/images/test:/dados:ro" \
    -v "$SAIDA/runtime_a_4threads.yaml:$APP/config/runtime.yaml:ro" \
    -- "${PY[@]}" -m animal_monitor benchmark --model models/animal_detector.onnx \
    --source /dados/anta00000005.jpg --frames 1 --warmup 0 --output /tmp/resultado.json
  mostrar_cmd "${CMD[@]}"
  local log="$SAIDA/2.1_pi5_modelo_a.json"
  if executar_com_tempo "$log"; then
    ok "Benchmark concluído em ${DURACAO} s dentro do limite de 4 CPUs / 4 GB"
    echo
    resumir_benchmark "$log"
  else
    erro "Falhou (veja $log)"
  fi
}

passo_2_2() {
  cabecalho 2.2
  explica "Modelo B INT8 em 40 quadros de vídeo com 1 CPU e 512 MB. A tabela mostra o uso medido pelo Docker durante a execução (~1 min); a coluna LIMITE confirma o teto aplicado."
  preparar_runtime 1
  local nome=tcc-demo-pizero
  montar_cmd cortex-a53 -d --name "$nome" --cpus 1 --memory 512m --memory-swap 512m \
    -v "$PROJECT_DIR/samples:/videos:ro" \
    -v "$SAIDA/runtime_b_1threads.yaml:$APP/config/modelo_b/runtime.yaml:ro" \
    -- "${PY[@]}" -m animal_monitor_b benchmark \
    --detector models/modelo_b/presence_detector_int8.onnx \
    --classifier models/modelo_b/species_classifier_int8.onnx \
    --source /videos/video_anta.mp4 --frames 40 --warmup 1 --output /tmp/resultado.json
  mostrar_cmd "${CMD[@]}"
  docker rm -f "$nome" >/dev/null 2>&1
  if ! "${CMD[@]}" >/dev/null; then
    erro "Não foi possível iniciar o contêiner"
    return
  fi
  printf '%s$ docker stats %s%s   %s(amostra a cada ~5 s)%s\n\n' "$CIANO" "$nome" "$RESET" "$FRACO" "$RESET"
  printf '  %s%-8s %-9s %-24s %-8s%s\n' "$NEGRITO" "TEMPO" "CPU %" "RAM USO / LIMITE" "RAM %" "$RESET"
  local inicio=$SECONDS linha cpu mem pct
  while [[ "$(docker inspect -f '{{.State.Running}}' "$nome" 2>/dev/null)" == "true" ]]; do
    linha="$(docker stats --no-stream --format '{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}' "$nome" 2>/dev/null)"
    if [[ -n "$linha" && "$(docker inspect -f '{{.State.Running}}' "$nome" 2>/dev/null)" == "true" ]]; then
      IFS='|' read -r cpu mem pct <<<"$linha"
      printf '  %-8s %-9s %-24s %-8s\n' "$(printf '%02d:%02d' $(((SECONDS - inicio) / 60)) $(((SECONDS - inicio) % 60)))" "$cpu" "$mem" "$pct"
    fi
    sleep 3
  done
  local log="$SAIDA/2.2_pizero_modelo_b_int8.json" status oom
  docker logs "$nome" >"$log" 2>&1
  status="$(docker inspect -f '{{.State.ExitCode}}' "$nome")"
  oom="$(docker inspect -f '{{.State.OOMKilled}}' "$nome")"
  docker rm "$nome" >/dev/null
  echo
  if [[ "$status" == 0 ]]; then
    ok "Concluiu em $((SECONDS - inicio)) s sem passar de 1 CPU / 512 MB"
    echo
    resumir_benchmark "$log"
  else
    erro "Terminou com exit $status (OOMKilled=$oom)"
  fi
}

passo_2_3() {
  cabecalho 2.3
  explica "O mesmo benchmark do Modelo A com 1 CPU e dois limites de RAM. Se o processo passa do limite, o kernel o encerra (exit 137, OOMKilled=true)."
  preparar_runtime 1
  local mem nome log status oom
  for mem in 192m 256m; do
    nome="tcc-demo-mem-$mem"
    log="$SAIDA/2.3_modelo_a_${mem}.json"
    printf '%s▶ Limite de %s MB%s\n\n' "$NEGRITO" "${mem%m}" "$RESET"
    montar_cmd max --name "$nome" --cpus 1 --memory "$mem" --memory-swap "$mem" \
      -v "$PROJECT_DIR/data/animals/images/test:/dados:ro" \
      -v "$SAIDA/runtime_a_1threads.yaml:$APP/config/runtime.yaml:ro" \
      -- "${PY[@]}" -m animal_monitor benchmark --model models/animal_detector.onnx \
      --source /dados/anta00000005.jpg --frames 1 --warmup 0 --output /tmp/resultado.json
    mostrar_cmd "${CMD[@]}"
    docker rm -f "$nome" >/dev/null 2>&1
    executar_com_tempo "$log"
    status="$(docker inspect -f '{{.State.ExitCode}}' "$nome" 2>/dev/null)"
    oom="$(docker inspect -f '{{.State.OOMKilled}}' "$nome" 2>/dev/null)"
    docker rm -f "$nome" >/dev/null 2>&1
    if [[ "$oom" == "true" || "$status" == 137 ]]; then
      erro "${mem%m} MB: processo encerrado por falta de memória (exit $status, OOMKilled=$oom) após ${DURACAO} s"
    elif [[ "$status" == 0 ]]; then
      ok "${mem%m} MB: benchmark concluído em ${DURACAO} s (exit 0, OOMKilled=$oom)"
      echo
      resumir_benchmark "$log"
    else
      erro "${mem%m} MB: terminou com exit $status"
    fi
    echo
  done
  explica "Conclusão: nesta configuração o Modelo A FP32 precisa de mais de 192 MB e cabe em 256 MB."
}

# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------

uso() {
  sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
}

listar() {
  local p
  for p in "${PASSOS[@]}"; do printf '  %-4s %s\n' "$p" "${TITULOS[$p]}"; done
}

checar_ambiente() {
  command -v docker >/dev/null || { erro "Docker não encontrado."; exit 2; }
  command -v python3 >/dev/null || { erro "python3 não encontrado (usado nos resumos)."; exit 2; }
  if [[ ! -x "$QEMU_BIN" ]]; then
    erro "qemu-aarch64 não encontrado em $QEMU_BIN"
    echo "  Defina ANIMAL_MONITOR_QEMU_BIN ou extraia de tonistiigi/binfmt:"
    echo "    docker create --name binfmt-tmp tonistiigi/binfmt"
    echo "    docker cp binfmt-tmp:/usr/bin/qemu-aarch64 ./qemu-aarch64 && docker rm binfmt-tmp"
    exit 2
  fi
  if ! docker image inspect "$IMAGEM" >/dev/null 2>&1; then
    erro "Imagem $IMAGEM não encontrada (construa com scripts/qemu_arm64_smoke.sh)."
    exit 2
  fi
}

main() {
  local -a pedidos=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help) uso; exit 0 ;;
      --lista) listar; exit 0 ;;
      --sem-pausa) PAUSAR=0 ;;
      *) pedidos+=("$1") ;;
    esac
    shift
  done
  [[ -t 0 ]] || PAUSAR=0

  local -a selecionados=()
  local pedido p
  if [[ ${#pedidos[@]} -eq 0 ]]; then
    selecionados=("${PASSOS[@]}")
  else
    for pedido in "${pedidos[@]}"; do
      local achou=0
      for p in "${PASSOS[@]}"; do
        if [[ $p == "$pedido" || $p == "$pedido".* ]]; then
          selecionados+=("$p")
          achou=1
        fi
      done
      if [[ $achou == 0 ]]; then
        erro "Passo desconhecido: $pedido"
        listar
        exit 2
      fi
    done
  fi

  checar_ambiente
  cd "$PROJECT_DIR" || exit 2
  limpar_conteineres
  local total=${#selecionados[@]} i=0
  for p in "${selecionados[@]}"; do
    (( i++ ))
    "passo_${p//./_}"
    (( i < total )) && pausa
  done
  printf '\n%sFim da demonstração.%s\n' "$NEGRITO" "$RESET"
}

main "$@"
