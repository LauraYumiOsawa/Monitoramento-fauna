#!/usr/bin/env bash
set -euo pipefail

image_path="${ANIMAL_MONITOR_QEMU_IMAGE:-}"
firmware_path="${ANIMAL_MONITOR_QEMU_EFI:-}"
image_format="${ANIMAL_MONITOR_QEMU_FORMAT:-qcow2}"

if ! command -v qemu-system-aarch64 >/dev/null 2>&1; then
  echo "Erro: qemu-system-aarch64 não encontrado." >&2
  exit 2
fi

if [[ -z "$image_path" || ! -f "$image_path" ]]; then
  echo "Erro: defina ANIMAL_MONITOR_QEMU_IMAGE com uma imagem Linux ARM64 existente." >&2
  exit 2
fi

if [[ -z "$firmware_path" ]]; then
  for candidate in \
    /usr/share/AAVMF/AAVMF_CODE.fd \
    /usr/share/qemu-efi-aarch64/QEMU_EFI.fd; do
    if [[ -f "$candidate" ]]; then
      firmware_path="$candidate"
      break
    fi
  done
fi

if [[ -z "$firmware_path" || ! -f "$firmware_path" ]]; then
  echo "Erro: firmware UEFI ARM64 não encontrado; defina ANIMAL_MONITOR_QEMU_EFI." >&2
  exit 2
fi

qemu-system-aarch64 \
  -machine virt \
  -cpu cortex-a72 \
  -smp 4 \
  -m 2048 \
  -bios "$firmware_path" \
  -drive "if=virtio,file=$image_path,format=$image_format" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-device,netdev=net0 \
  -device virtio-rng-pci \
  -nographic
