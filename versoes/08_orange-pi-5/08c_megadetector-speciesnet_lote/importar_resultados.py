"""Converte as predições do SpeciesNet (formato 20261015) em um pacote Camtrap DP.

Cada foto vira uma mídia; cada detecção do MegaDetector com confiança >= --limiar vira
uma observação no nível da mídia (`observationLevel = media`). A espécie é a
`processed_classification` (já filtrada pelo país): espécie exata das três do projeto →
`animal` com nome científico; outro táxon → `animal` com o nome do táxon no comentário;
pessoa/veículo → `human`/`vehicle`; nenhuma detecção → `blank`.

    python importar_resultados.py dados/speciesnet_*.json [--pasta-saida dados/camtrap-dp]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
for candidato in (AQUI / "lib", AQUI.parents[2] / "versoes" / "comum", AQUI.parents[2] / "src"):
    if candidato.is_dir():
        sys.path.insert(0, str(candidato))

from animal_monitor_hw.camtrap_dp import (  # noqa: E402
    Implantacao,
    RegistroCamtrap,
    agora_iso,
    novo_id,
)

PROJETO = {
    ("tapirus", "terrestris"): "anta",
    ("myrmecophaga", "tridactyla"): "tamandua_bandeira",
    ("chrysocyon", "brachyurus"): "lobo_guara",
}
TIPOS = {"1": "animal", "2": "human", "3": "vehicle"}


def momento(caminho: Path) -> str:
    try:
        return datetime.fromtimestamp(caminho.stat().st_mtime).astimezone().isoformat(
            timespec="seconds")
    except OSError:
        return agora_iso()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predicoes")
    parser.add_argument("--limiar", type=float, default=0.2, help="MegaDetector (MDv5: 0,2).")
    parser.add_argument("--registro", default=str(AQUI / "dados" / "camtrap.sqlite3"))
    parser.add_argument("--pasta-saida", default=str(AQUI / "dados" / "camtrap-dp"))
    parser.add_argument("--deployment", default="dep-08c")
    parser.add_argument("--latitude", type=float, default=0.0)
    parser.add_argument("--longitude", type=float, default=0.0)
    args = parser.parse_args()

    dados = json.loads(Path(args.predicoes).read_text(encoding="utf-8"))
    versao = sorted({str(p.get("model_version")) for p in dados["predictions"]})
    classificado_por = f"MegaDetector v5a + SpeciesNet {','.join(versao)} (versão 08c)"
    with RegistroCamtrap(args.registro) as registro:
        registro.registrar_implantacao(Implantacao(
            args.deployment, args.latitude, args.longitude, agora_iso(), "08c",
            "subsistema de captura + Orange Pi 5 de 8 ou 16 GB",
            comentarios="lote MegaDetector+SpeciesNet"))
        for predicao in dados["predictions"]:
            arquivo = Path(predicao["filepath"])
            quando = momento(arquivo)
            midia = registro.registrar_midia(args.deployment, arquivo, quando)
            evento = novo_id("e")
            deteccoes = [d for d in predicao.get("detections") or []
                         if float(d.get("conf", 0)) >= args.limiar and d.get("category") in TIPOS]
            if not deteccoes:
                registro.registrar_observacao(args.deployment, evento, quando, quando, "blank",
                                              classificado_por, media_id=midia)
                continue
            for deteccao in deteccoes:
                tipo = TIPOS[deteccao["category"]]
                processada = deteccao.get("processed_classification") or {}
                partes = str(processada.get("class", "")).split(";")
                rotulo, comentario = None, None
                if tipo == "animal" and len(partes) >= 7:
                    rotulo = PROJETO.get((partes[4], partes[5]))
                    if rotulo is None:
                        comentario = f"SpeciesNet: {partes[6] or 'sem nome'}"
                x, y, w, h = (float(v) for v in deteccao["bbox"])
                registro.registrar_observacao(
                    args.deployment, evento, quando, quando,
                    tipo,
                    classificado_por, rotulo=rotulo,
                    probabilidade=float(processada.get("score", deteccao.get("conf", 0))),
                    quantidade=1, caixa_normalizada=(x, y, w, h), media_id=midia,
                    comentario=comentario,
                )
        pacote = registro.exportar(args.pasta_saida, "Lote MegaDetector + SpeciesNet",
                                   "Laura Yumi Osawa", descricao="Versão 08c")
    print(f"Camtrap DP: {pacote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
