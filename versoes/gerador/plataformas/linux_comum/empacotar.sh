#!/usr/bin/env bash
# Gera dist/@@ID@@.tar.gz: pasta da versão + modelos + bibliotecas Python do projeto.
# Rode na máquina de desenvolvimento, depois de preparar_modelo.sh.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../../.." && pwd)"
SAIDA="$RAIZ/versoes/dist"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT

DESTINO="$TEMP/@@ID@@"
mkdir -p "$DESTINO/lib"
cp -a "$AQUI/." "$DESTINO/"
rm -rf "$DESTINO/.venv" "$DESTINO/dados"
cp -a "$RAIZ/src/animal_monitor" "$RAIZ/src/animal_monitor_b" "$DESTINO/lib/"
cp -a "$RAIZ/versoes/comum/animal_monitor_hw" "$DESTINO/lib/"
find "$DESTINO" -name "__pycache__" -type d -prune -exec rm -rf {} +

faltando=0
while IFS= read -r arquivo; do
  if [[ ! -f "$DESTINO/$arquivo" ]]; then
    echo "AVISO: artefato ausente: $arquivo (rode ./preparar_modelo.sh)" >&2
    faltando=1
  fi
done < <(sed -n 's/^ *arquivo: *//p' "$AQUI/versao.yaml")

mkdir -p "$SAIDA"
tar -C "$TEMP" -czf "$SAIDA/@@ID@@.tar.gz" "@@ID@@"
echo "Pacote: $SAIDA/@@ID@@.tar.gz"
[[ $faltando -eq 0 ]] || echo "Pacote gerado SEM todos os modelos." >&2
