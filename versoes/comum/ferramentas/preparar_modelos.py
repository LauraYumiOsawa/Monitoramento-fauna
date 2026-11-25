"""Coloca em `<versão>/modelos/` os artefatos ONNX que a versão declara em `artefatos:`.

Roda na máquina de desenvolvimento (precisa do extra de treino: ultralytics, torch, onnx).
Cada item de `artefatos` tem um destino e uma de três origens:

    - destino: modelos/a_640.onnx
      origem: models/animal_detector.onnx          # copia (caminho relativo à raiz)
    - destino: modelos/b_416_split.onnx
      exportar: {pesos: runs/.../best.pt, entrada: [416, 416], cabeca: dividida}
    - destino: modelos/b_416_split_int8full.onnx
      quantizar: {de: modelos/b_416_split.onnx, apenas_conv: false}

Artefatos idênticos já gerados em `versoes/comum/modelos/` são reaproveitados pelo nome.
O passo seguinte (HEF, RKNN, cvimodel, rpk, TFLite) é feito pelo script da própria versão,
na toolchain do fabricante.

    python versoes/comum/ferramentas/preparar_modelos.py versoes/01_raspberry-pi-5/01a_...
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

from animal_monitor.config import load_yaml  # noqa: E402

CACHE = ROOT / "versoes" / "comum" / "modelos"
CALIBRACAO_PADRAO = ROOT / "data" / "animals" / "images" / "val"


def _silencioso(funcao, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return funcao(*args, **kwargs)


def _exportar(destino: Path, pesos: Path, entrada: list[int], cabeca: str) -> None:
    if cabeca == "dividida":
        from animal_monitor_hw.yolo_heads import export_split_head_onnx

        _silencioso(export_split_head_onnx, pesos, destino, imgsz=(entrada[0], entrada[1]))
        return
    from ultralytics import YOLO

    trabalho = destino.parent / f".exportando_{destino.stem}"
    trabalho.mkdir(parents=True, exist_ok=True)
    copia = trabalho / "best.pt"
    shutil.copy2(pesos, copia)  # o Ultralytics grava o .onnx ao lado do .pt
    gerado = _silencioso(
        YOLO(str(copia)).export, format="onnx", imgsz=list(entrada), batch=1, dynamic=False,
        simplify=True, opset=17, nms=False,
    )
    shutil.move(str(gerado), destino)
    shutil.rmtree(trabalho, ignore_errors=True)


def preparar(pasta_versao: Path, forcar: bool = False) -> list[dict[str, str]]:
    from animal_monitor_hw.quantizacao import quantize_detector_int8

    versao = load_yaml(pasta_versao / "versao.yaml")
    relatorio = []
    for item in versao.get("artefatos") or []:
        destino = (pasta_versao / item["destino"]).resolve()
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.is_file() and not forcar:
            relatorio.append({"artefato": item["destino"], "acao": "já existe"})
            continue
        em_cache = CACHE / destino.name
        if "origem" in item:
            shutil.copy2(ROOT / item["origem"], destino)
            acao = f"copiado de {item['origem']}"
        elif em_cache.is_file() and not forcar:
            shutil.copy2(em_cache, destino)
            acao = "copiado do cache versoes/comum/modelos"
        elif "exportar" in item:
            cfg = item["exportar"]
            cabeca = cfg.get("cabeca", "padrao")
            _exportar(destino, ROOT / cfg["pesos"], list(cfg["entrada"]), cabeca)
            acao = f"exportado ({cfg.get('cabeca', 'padrao')}, {cfg['entrada']})"
        elif "quantizar" in item:
            cfg = item["quantizar"]
            origem = (pasta_versao / cfg["de"]).resolve()
            calibracao = ROOT / cfg.get("calibracao", CALIBRACAO_PADRAO)
            _silencioso(
                quantize_detector_int8, origem, calibracao, destino,
                amostras=int(cfg.get("amostras", 36)),
                apenas_conv=bool(cfg.get("apenas_conv", True)),
            )
            escopo = "só Conv" if cfg.get("apenas_conv", True) else "todos"
            acao = f"quantizado INT8 ({escopo})"
        else:
            raise SystemExit(f"Artefato sem origem: {item}")
        relatorio.append({"artefato": item["destino"], "acao": acao})
    return relatorio


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara os artefatos ONNX de uma versão.")
    parser.add_argument("versoes", nargs="+", help="Pastas de versão (com versao.yaml).")
    parser.add_argument("--forcar", action="store_true", help="Refaz mesmo se já existir.")
    args = parser.parse_args()
    for pasta in args.versoes:
        caminho = Path(pasta).expanduser().resolve()
        print(f"== {caminho.name}")
        for linha in preparar(caminho, args.forcar):
            print(f"   {linha['artefato']}: {linha['acao']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
