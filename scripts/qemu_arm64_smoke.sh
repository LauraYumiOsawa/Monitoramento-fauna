#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${ANIMAL_MONITOR_ARM64_IMAGE:-animal-monitor:arm64}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: Docker não encontrado." >&2
  exit 2
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "Erro: Docker Buildx não está disponível." >&2
  exit 2
fi

docker buildx build \
  --platform linux/arm64 \
  --load \
  --file "$project_dir/deploy/arm64/Dockerfile" \
  --tag "$image_name" \
  "$project_dir"

docker run --rm --platform linux/arm64 "$image_name" info

echo "Smoke test ARM64 concluído. Isso valida compatibilidade de software, não FPS do hardware real."
