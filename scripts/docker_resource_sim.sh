#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# docker_resource_sim.sh
# Simulação de ambiente embarcado via limites de recursos Docker (CPU/RAM)
# ============================================================================
# PROPÓSITO:
#   Testar o modelo em um ambiente nativo (sem emulação de ISA) mas com
#   CPU e RAM artificialmente limitados via cgroups do Docker, aproximando
#   as restrições de um dispositivo embarcado como Raspberry Pi 5.
#
# DIFERENÇAS em relação ao qemu_arm64_smoke.sh:
#   - QEMU: simula ISA ARM64 (compatibilidade de arquitetura), mas é lento
#   - Este script: roda nativamente (rápido), mas limita recursos (CPU/RAM)
#
# LIMITAÇÕES:
#   - NÃO simula: ISA ARM, cache L1/L2/L3, largura de banda de memória,
#     throttling térmico, NPU, consumo de energia, câmera real
#   - Os valores de FPS/latência são indicativos de "o modelo funciona com
#     N CPUs e M GB de RAM?", mas NÃO substituem medição em hardware real
# ============================================================================

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${SIM_IMAGE:-animal-monitor-sim:latest}"

# Defaults aproximando Raspberry Pi 5 (4GB RAM, 4-core Cortex-A76)
sim_cpus="${SIM_CPUS:-4.0}"
sim_memory="${SIM_MEMORY:-4g}"
sim_threads="${SIM_THREADS:-4}"
sim_model="${SIM_MODEL:-modelo-b}"  # "modelo-a" ou "modelo-b"

output_dir="$project_dir/outputs/sim_embedded"
mkdir -p "$output_dir"

# ============================================================================
# Validações
# ============================================================================
if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: Docker não encontrado." >&2
  exit 2
fi

if [[ "$sim_model" != "modelo-a" && "$sim_model" != "modelo-b" ]]; then
  echo "Erro: SIM_MODEL deve ser 'modelo-a' ou 'modelo-b' (atual: '$sim_model')" >&2
  exit 2
fi

# ============================================================================
# Build da imagem (nativa, sem --platform fixo)
# ============================================================================
echo "==> Construindo imagem $image_name (nativa, sem emulação QEMU)..."
docker build \
  --file "$project_dir/deploy/sim-embedded/Dockerfile" \
  --tag "$image_name" \
  "$project_dir"

# ============================================================================
# Criar runtime.yaml customizado em /tmp (disco cheio no projeto)
# ============================================================================
tmp_runtime="/tmp/runtime_sim_$$.yaml"
cat > "$tmp_runtime" <<EOF
providers:
  - CPUExecutionProvider
intra_op_threads: $sim_threads
EOF

# Cleanup ao sair (remove arquivo temporário)
cleanup() {
  rm -f "$tmp_runtime"
}
trap cleanup EXIT

# ============================================================================
# Execução do benchmark com limites de CPU/RAM
# ============================================================================
echo "==> Rodando benchmark com limites: --cpus=$sim_cpus --memory=$sim_memory (intra_op_threads=$sim_threads)"
echo "==> Modelo: $sim_model"
echo "==> Outputs serão salvos em $output_dir/"
echo ""

for video in "$project_dir/samples"/*.mp4; do
  video_name="$(basename "$video" .mp4)"
  output_json="$output_dir/benchmark_${video_name}.json"
  
  echo "[$(date +%T)] Processando $video_name..."
  
  if [[ "$sim_model" == "modelo-b" ]]; then
    docker run --rm \
      --cpus="$sim_cpus" \
      --memory="$sim_memory" \
      -v "$project_dir/samples:/opt/animal-monitor/samples:ro" \
      -v "$project_dir/config:/opt/animal-monitor/config:ro" \
      -v "$tmp_runtime:/opt/animal-monitor/config/modelo_b/runtime.yaml:ro" \
      -v "$project_dir/models:/opt/animal-monitor/models:ro" \
      -v "$output_dir:/opt/animal-monitor/outputs" \
      "$image_name" \
      benchmark \
        --detector models/modelo_b/presence_detector.onnx \
        --classifier models/modelo_b/species_classifier.onnx \
        --source "samples/$(basename "$video")" \
        --runtime config/modelo_b/runtime.yaml \
        --output "outputs/$(basename "$output_json")"
  else
    # Modelo A (fallback, caso usuário especifique)
    docker run --rm \
      --cpus="$sim_cpus" \
      --memory="$sim_memory" \
      -v "$project_dir/samples:/opt/animal-monitor/samples:ro" \
      -v "$project_dir/config:/opt/animal-monitor/config:ro" \
      -v "$tmp_runtime:/opt/animal-monitor/config/runtime.yaml:ro" \
      -v "$project_dir/models:/opt/animal-monitor/models:ro" \
      -v "$output_dir:/opt/animal-monitor/outputs" \
      "$image_name" \
      benchmark \
        --model models/animal_detector.onnx \
        --source "samples/$(basename "$video")" \
        --data config/dataset.yaml \
        --runtime config/runtime.yaml \
        --output "outputs/$(basename "$output_json")"
  fi
  
  echo "[$(date +%T)] ✓ $video_name concluído → $output_json"
  echo ""
done

# ============================================================================
# Resumo
# ============================================================================
echo "==> Simulação concluída!"
echo ""
echo "Configuração usada:"
echo "  CPUs: $sim_cpus"
echo "  RAM: $sim_memory"
echo "  intra_op_threads: $sim_threads"
echo "  Modelo: $sim_model"
echo ""
echo "Resultados salvos em: $output_dir/"
echo ""
echo "IMPORTANTE: Estes resultados simulam restrições de CPU/RAM, mas NÃO"
echo "substituem medição em hardware real (Raspberry Pi 5, MaixCAM)."
echo "Ver README.md seção 'Limitações' para detalhes."
