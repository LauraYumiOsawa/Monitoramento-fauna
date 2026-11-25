"""Mede MegaDetector + SpeciesNet no split `test` com o mesmo código de métrica dos Modelos A/B.

Entrada: o JSON de predições do SpeciesNet (`python -m speciesnet.scripts.run_model
--folders data/animals/images/test --predictions_json saida.json --country BRA`).
Cada detecção do MegaDetector vira uma `Detection` cuja espécie é a
`processed_classification` do SpeciesNet, mapeada para os rótulos do projeto; espécie
fora das três vira `outro` (erro de espécie, não fundo). Escore = confiança do detector ×
escore da classificação, como no Modelo B.

    python versoes/comum/ferramentas/avaliar_speciesnet.py saida.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

from animal_monitor.types import BoundingBox, Detection  # noqa: E402
from animal_monitor_b.evaluation import evaluate_species_predictor  # noqa: E402

DATASET = ROOT / "config" / "dataset.yaml"
# gênero;espécie da taxonomia do SpeciesNet -> rótulo do projeto
ESPECIES = {
    ("tapirus", "terrestris"): "anta",
    ("myrmecophaga", "tridactyla"): "tamandua_bandeira",
    ("chrysocyon", "brachyurus"): "lobo_guara",
}
CLASSES = {"anta": 0, "tamandua_bandeira": 1, "lobo_guara": 2, "outro": 3}


def rotulo(classe: str) -> tuple[str, str]:
    """('anta', 'espécie') ou ('outro', nível taxonômico atingido)."""
    partes = (classe or "").split(";")
    if len(partes) < 7:
        return "outro", "desconhecido"
    genero, especie, nome = partes[4], partes[5], partes[6]
    if (genero, especie) in ESPECIES:
        return ESPECIES[(genero, especie)], "espécie"
    if genero in {"tapirus", "myrmecophaga", "chrysocyon"} and not especie:
        return "outro", f"gênero ({nome})"
    return "outro", nome or "sem nome"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predicoes", help="JSON do SpeciesNet")
    parser.add_argument("--limiar-md", type=float, default=0.2,
                        help="Confiança mínima do MegaDetector (típica do MDv5: 0,2).")
    parser.add_argument("--saida", default=str(ROOT / "versoes" / "ranking" / "medicoes" /
                                                "speciesnet_x86.json"))
    args = parser.parse_args()
    dados = json.loads(Path(args.predicoes).read_text(encoding="utf-8"))
    por_arquivo: dict[str, list[Detection]] = {}
    niveis: Counter[str] = Counter()
    for predicao in dados["predictions"]:
        deteccoes = []
        for item in predicao.get("detections") or []:
            if item.get("category") != "1" or float(item.get("conf", 0)) < args.limiar_md:
                continue
            processada = item.get("processed_classification") or {}
            nome, nivel = rotulo(str(processada.get("class", "")))
            niveis[nivel] += 1
            # bbox do MegaDetector: [x_min, y_min, largura, altura] normalizados.
            x, y, w, h = (float(valor) for valor in item["bbox"])
            escore = float(item["conf"]) * float(processada.get("score", 0))
            caixa = BoundingBox(x, y, x + w, y + h)
            deteccoes.append(Detection(CLASSES[nome], nome, escore, caixa))
        por_arquivo[Path(predicao["filepath"]).name] = deteccoes

    from animal_monitor_b.evaluation import _split_images

    # evaluate_species_predictor percorre exatamente esta lista, na mesma ordem.
    nomes = iter(caminho.name for caminho in _split_images(DATASET, "test"))

    def preditor(quadro):
        altura, largura = quadro.shape[:2]
        return [
            Detection(d.class_id, d.class_name, d.confidence,
                      BoundingBox(d.box.x1 * largura, d.box.y1 * altura, d.box.x2 * largura,
                                  d.box.y2 * altura))
            for d in por_arquivo.get(next(nomes), [])
        ]

    relatorio = evaluate_species_predictor(preditor, DATASET, split="test",
                                           label="megadetector_v5a+speciesnet_4.0.3a")
    relatorio["niveis_taxonomicos"] = dict(niveis)
    relatorio["limiar_megadetector"] = args.limiar_md
    relatorio["model_version"] = sorted(
        {str(p.get("model_version")) for p in dados["predictions"]}
    )
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    limiar = relatorio["score_threshold"]
    print(json.dumps({
        "map50": round(relatorio["map50"], 4),
        "precisao": round(relatorio[f"precision@{limiar}"], 4),
        "recall": round(relatorio[f"recall@{limiar}"], 4),
        "f1": round(relatorio[f"f1@{limiar}"], 4),
        "contagens": relatorio["counts"],
        "niveis": dict(niveis),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
