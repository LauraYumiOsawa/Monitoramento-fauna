#!/usr/bin/env python3
"""Reexecuta as partes centrais do TCC e confronta cada número com o relatório final.

Cada seção vira uma "evidência": um bloco de texto com o comando, a máquina, a tabela
obtida agora e a coluna do valor impresso no relatório (✓ igual, ✗ diferente, ≈ medida
de tempo, que depende da máquina). Os blocos ficam em ``outputs/evidencias/NN_secao.txt``
e, se o Google Chrome estiver instalado, também em PNG, prontos para o relatório.

Uso (na raiz do projeto):

    .venv/bin/python scripts/evidencias_relatorio.py --listar
    .venv/bin/python scripts/evidencias_relatorio.py                # todas as seções
    .venv/bin/python scripts/evidencias_relatorio.py dataset eficacia
    .venv/bin/python scripts/evidencias_relatorio.py --so-png       # só refaz os PNG

Nada aqui altera modelos, dados ou os JSON de resultados originais: tudo o que é gerado
vai para ``outputs/evidencias/``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
AVAL = ROOT.parents[1] / "avaliacao_embarcada"
OUT = ROOT / "outputs" / "evidencias"
PY = sys.executable
QEMU = Path(os.environ.get("ANIMAL_MONITOR_QEMU_BIN", AVAL / "qemu-aarch64"))
IMAGEM_ARM64 = "animal-monitor:arm64"
MEDICOES = ROOT / "versoes" / "ranking" / "medicoes"
VIDEOS = {
    "anta": ROOT / "samples" / "video_anta.mp4",
    "lobo_guara": ROOT / "samples" / "video_lobo_guara.mp4",
    "tamandua_bandeira": ROOT / "samples" / "video_tamandua_bandeira.mp4",
}
NOME = {"anta": "anta", "lobo_guara": "lobo-guará", "tamandua_bandeira": "tamanduá-bandeira"}

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]

LARGURA = 96
OK, ERRO, APROX = "✓", "✗", "≈"


# --------------------------------------------------------------------------- formatação


def br(valor: float | None, casas: int = 3) -> str:
    if valor is None:
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor: float, casas: int = 2) -> str:
    return br(100 * valor, casas) + "%"


def marca(condicao: bool) -> str:
    return OK if condicao else ERRO


def igual(obtido: float, esperado: float, casas: int = 3) -> bool:
    return round(obtido, casas) == round(esperado, casas)


class Figura:
    """Acumula as linhas de uma figura e as grava em texto."""

    def __init__(self, numero: int, chave: str, titulo: str, secao_relatorio: str) -> None:
        self.numero, self.chave, self.titulo = numero, chave, titulo
        self.secao_relatorio = secao_relatorio
        self.linhas: list[str] = []
        self.verificacoes: list[bool] = []
        self.dados: dict[str, object] = {}

    def __call__(self, texto: str = "") -> None:
        self.linhas.append(texto)

    def regua(self, caractere: str = "─") -> None:
        self.linhas.append(caractere * LARGURA)

    def tabela(self, cabecalho: list[str], linhas: list[list[str]], direita: set[int] | None = None) -> None:
        direita = direita if direita is not None else set(range(1, len(cabecalho)))
        larguras = [max(len(str(c)) for c in coluna) for coluna in zip(cabecalho, *linhas)]

        def formata(celulas: list[str]) -> str:
            partes = []
            for indice, (celula, largura) in enumerate(zip(celulas, larguras)):
                celula = str(celula)
                partes.append(celula.rjust(largura) if indice in direita else celula.ljust(largura))
            return "  ".join(partes).rstrip()

        self(formata(cabecalho))
        self("  ".join("─" * largura for largura in larguras))
        for linha in linhas:
            self(formata(linha))

    def confere(self, condicao: bool) -> str:
        self.verificacoes.append(bool(condicao))
        return marca(condicao)

    def texto(self, comando: str, ambiente: str) -> str:
        topo = [
            "═" * LARGURA,
            f"EVIDÊNCIA {self.numero:02d} · {self.titulo}",
            f"Relatório: {self.secao_relatorio}",
            f"$ {comando}",
            ambiente,
            "═" * LARGURA,
        ]
        rodape = []
        if self.verificacoes:
            total, certos = len(self.verificacoes), sum(self.verificacoes)
            simbolo = OK if certos == total else ERRO
            rodape = [
                "─" * LARGURA,
                f"CONFERÊNCIA COM O RELATÓRIO: {certos}/{total} valores iguais  {simbolo}",
            ]
        return "\n".join(topo + self.linhas + rodape) + "\n"


def cpu_modelo() -> str:
    with contextlib.suppress(OSError):
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name"):
                nome = linha.split(":", 1)[1].strip()
                return nome.replace("(R)", "").replace("(TM)", "").replace("  ", " ")
    return platform.processor() or platform.machine()


def versao(modulo: str) -> str:
    try:
        return __import__(modulo).__version__
    except Exception:  # pragma: no cover
        return "ausente"


def linha_ambiente() -> str:
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    return (
        f"Executado em {agora} · {cpu_modelo()} · {os.cpu_count()} CPUs · "
        f"Python {platform.python_version()} · ORT {versao('onnxruntime')}"
    )


def rodar(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None,
          cwd: Path = ROOT) -> subprocess.CompletedProcess:
    ambiente = dict(os.environ)
    ambiente.update(env or {})
    return subprocess.run(cmd, cwd=cwd, env=ambiente, capture_output=True, text=True,
                          timeout=timeout)


def json_da_saida(texto: str) -> dict:
    """Pega o último objeto JSON impresso por uma CLI (que pode logar antes)."""
    inicio = texto.find("{")
    while inicio != -1:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(texto[inicio:])
        inicio = texto.find("{", inicio + 1)
    raise ValueError(f"Saída sem JSON:\n{texto[-2000:]}")


def sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def silencioso(funcao: Callable, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return funcao(*args, **kwargs)


def runtime_temporario(base: Path, destino: Path, trocas: dict[str, str]) -> Path:
    """Copia um YAML de runtime trocando chaves de primeiro nível (saídas, threads, NMS)."""
    linhas = []
    for linha in base.read_text(encoding="utf-8").splitlines():
        chave = linha.split(":", 1)[0].strip()
        if chave in trocas and not linha.startswith((" ", "#")):
            linha = f"{chave}: {trocas[chave]}"
        linhas.append(linha)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return destino


def docker_arm64(argumentos: list[str], *, cpu: str, montagens: list[str], env: dict[str, str],
                 cpus: str | None = None, memoria: str | None = None, somente_leitura: bool = True,
                 timeout: float = 1800) -> subprocess.CompletedProcess:
    """Executa Python ARM64 com o qemu-aarch64 estático como entrypoint (sem binfmt)."""
    cmd = ["docker", "run", "--rm", "--platform", "linux/arm64", "--network", "none",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    if somente_leitura:
        cmd += ["--read-only", "--tmpfs", "/tmp"]
    if cpus:
        cmd += ["--cpus", cpus]
    if memoria:
        cmd += ["--memory", memoria, "--memory-swap", memoria]
    for chave, valor in env.items():
        cmd += ["-e", f"{chave}={valor}"]
    for montagem in montagens:
        cmd += ["-v", montagem]
    cmd += ["-v", f"{QEMU}:/qemu-aarch64:ro", "--entrypoint", "/qemu-aarch64", IMAGEM_ARM64,
            "-cpu", cpu, "/usr/local/bin/python", "python", *argumentos]
    return rodar(cmd, timeout=timeout)


def importar_avaliacao():
    """Importa o avaliar_modelos.py usado no relatório (mesmas variantes e benchmark)."""
    sys.path.insert(0, str(AVAL))
    import avaliar_modelos  # type: ignore

    return avaliar_modelos


# --------------------------------------------------------------------------- seções


def sec_ambiente(fig: Figura) -> None:
    import cv2
    import numpy

    memoria = "?"
    with contextlib.suppress(OSError):
        for linha in Path("/proc/meminfo").read_text().splitlines():
            if linha.startswith("MemTotal"):
                memoria = f"{int(linha.split()[1]) / 1024**2:.1f} GiB"
    docker = rodar(["docker", "version", "--format", "{{.Server.Version}}"]).stdout.strip()
    qemu = rodar([str(QEMU), "--version"]).stdout.splitlines()[0] if QEMU.is_file() else "ausente"
    imagem = rodar(["docker", "image", "inspect", IMAGEM_ARM64, "--format",
                    "{{.Architecture}} · {{.Id}}"]).stdout.strip()[:40]
    fig.tabela(
        ["Item", "Valor"],
        [
            ["CPU", cpu_modelo()],
            ["CPUs lógicas / RAM", f"{os.cpu_count()} / {memoria}"],
            ["Sistema", platform.platform()],
            ["Python", platform.python_version()],
            ["ONNX Runtime / OpenCV / NumPy", f"{versao('onnxruntime')} / {cv2.__version__} / {numpy.__version__}"],
            ["Docker (servidor)", docker or "ausente"],
            ["Emulador ARM64", qemu],
            [f"Imagem {IMAGEM_ARM64}", imagem or "ausente"],
        ],
        direita=set(),
    )
    fig()
    fig("Máquinas citadas no relatório:")
    fig("  · vídeos (15,9 / 31,1 / 35,2 FPS): Intel Core i5-1235U, 12 threads (ago/2026);")
    registro = json.loads((AVAL / "resultados_cpu_host.json").read_text(encoding="utf-8"))
    afinidade = registro["host"]["affinity"]
    fig(f"  · Quadro 4 (benchmark de 18 imagens, 11/11/2026): {len(afinidade)} CPUs lógicas "
        f"(afinidade {afinidade[0]}–{afinidade[-1]}),")
    fig("    o que não corresponde ao i5-1235U (12 threads) e sim a esta máquina de 16 CPUs.")
    fig("Métricas de eficácia não dependem da máquina; tempos dependem.")


def sec_dataset(fig: Figura) -> None:
    saida = OUT / "dados" / "dataset_report.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    proc = rodar([PY, "-m", "animal_monitor", "validate-dataset", "--data", "config/dataset.yaml",
                  "--json-output", str(saida)], env={"PYTHONPATH": "src"})
    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    veredito = [linha for linha in proc.stdout.splitlines() if linha.startswith("Resultado")]
    splits = relatorio["splits"]
    linhas = []
    for split in ("train", "val", "test"):
        dados = splits[split]
        caixas = dados["class_boxes"]
        linhas.append([split, dados["images"], dados["boxes"], caixas["anta"],
                       caixas["tamandua_bandeira"], caixas["lobo_guara"],
                       dados["missing_labels"], dados["empty_labels"]])
    total_img = sum(splits[s]["images"] for s in splits)
    total_cx = sum(splits[s]["boxes"] for s in splits)
    linhas.append(["TOTAL", total_img, total_cx, "", "", "", "", ""])
    fig.tabela(["Split", "Imagens", "Caixas", "anta", "tamanduá", "lobo-guará", "s/ rótulo",
                "vazios"], [[str(c) for c in linha] for linha in linhas])
    fig()
    fig(f"Saída do validador: {veredito[-1] if veredito else proc.stdout.strip()[-80:]}")
    fig(f"Problemas listados: {len(relatorio['issues'])}")
    fig()
    erros = sum(1 for item in relatorio["issues"] if item.get("severity") == "error")
    esperado = [
        ("Imagens no total", total_img, 180),
        ("Caixas anotadas", total_cx, 207),
        ("Imagens de treino (70%)", splits["train"]["images"], 126),
        ("Imagens de validação (20%)", splits["val"]["images"], 36),
        ("Imagens de teste (10%)", splits["test"]["images"], 18),
        ("Caixas no teste", splits["test"]["boxes"], 20),
        ("Erros do validador", erros, 0),
    ]
    fig.tabela(["Conferência", "Obtido", "Relatório", ""],
               [[n, str(o), str(e), fig.confere(o == e)] for n, o, e in esperado])
    sinteticas = {}
    for split in ("train", "val", "test"):
        pasta = ROOT / "data" / "animals" / "images" / split
        sinteticas[split] = sum(1 for p in pasta.iterdir()
                                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                                and "gemini_generated" in p.name.lower())
    fig()
    fig(f"Observação: {sum(sinteticas.values())} imagens geradas por IA (Gemini) no conjunto — "
        f"treino {sinteticas['train']}, validação {sinteticas['val']}, teste {sinteticas['test']}.")
    fig("Esse dado não aparece no relatório.")
    fig.dados.update(relatorio=relatorio, sinteticas=sinteticas)


def _resultado_testes(pasta: Path) -> dict:
    return json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))


def sec_testes(fig: Figura) -> None:
    iniciais = rodar([PY, "-m", "unittest", "tests.test_config", "tests.test_dataset",
                      "tests.test_events", "tests.test_prepost", "tests.test_splitting"],
                     env={"PYTHONPATH": "src"})
    texto = iniciais.stderr
    rodados = next((int(l.split()[1]) for l in texto.splitlines() if l.startswith("Ran ")), 0)
    ok_iniciais = texto.strip().endswith("OK")

    pasta_x86 = OUT / "dados" / "testes_x86"
    rodar([PY, "scripts/testar_tudo.py", "--output-dir", str(pasta_x86)])
    x86 = _resultado_testes(pasta_x86)

    pasta_arm = OUT / "dados" / "testes_arm64"
    pasta_arm.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    proc = docker_arm64(["/work/scripts/testar_tudo.py", "--output-dir", "/out"], cpu="cortex-a76",
                        montagens=[f"{ROOT}:/work:ro", f"{pasta_arm}:/out"],
                        env={"PYTHONPYCACHEPREFIX": "/tmp/pyc"}, timeout=1800)
    duracao_arm = time.perf_counter() - inicio
    if not (pasta_arm / "resultado.json").is_file():
        raise RuntimeError(f"Testes ARM64 não geraram resultado:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    arm = _resultado_testes(pasta_arm)

    def linha(nome: str, ambiente: str, t: dict, compila: bool | None) -> list[str]:
        return [nome, ambiente, str(t["total"]), str(t["passed"]), str(t["failed"]),
                str(t["errors"]), str(t["skipped"]), "sim" if compila else ("—" if compila is None else "NÃO")]

    fig.tabela(
        ["Conjunto", "Ambiente", "Total", "Passou", "Falhou", "Erros", "Ignor.", "Compila"],
        [
            linha("Iniciais (Modelo A)", "x86_64", {"total": rodados, "passed": rodados if ok_iniciais else 0,
                  "failed": 0 if ok_iniciais else "?", "errors": 0, "skipped": 0}, None),
            linha("Ampliado (núcleo + versões)", "x86_64", x86["tests"], x86["compile_ok"]),
            linha("Ampliado (núcleo + versões)", "ARM64 · QEMU cortex-a76", arm["tests"], arm["compile_ok"]),
        ],
        direita={2, 3, 4, 5, 6},
    )
    fig()
    fig(f"ARM64: {arm['platform']} · Python {arm['python'].split()[0]} · "
        f"{duracao_arm:.0f} s sob emulação")
    fig()
    t = arm["tests"]
    esperado = [
        ("Testes iniciais aprovados", f"{rodados}/{rodados}" if ok_iniciais else "falhou", "12/12"),
        ("Conjunto ampliado em ARM64", f"{t['passed']}/{t['total']}", "93/93"),
        ("Falhas + erros + ignorados (ARM64)", str(t["failed"] + t["errors"] + t["skipped"]), "0"),
        ("Compilação dos módulos (ARM64)", "aprovada" if arm["compile_ok"] else "falhou", "aprovada"),
    ]
    fig.tabela(["Conferência", "Obtido", "Relatório", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in esperado])
    fig.dados.update(x86=x86["tests"], arm64=t, iniciais=rodados)


def sec_integridade(fig: Figura) -> None:
    registro = json.loads((AVAL / "resultados_cpu_host.json").read_text(encoding="utf-8"))
    registrados: dict[str, str] = {}
    for variante in registro["variants"].values():
        for artefato in variante["artifacts"].values():
            registrados[artefato["path"]] = artefato["sha256"]
    metadados: dict[str, str] = {}
    for arquivo in (ROOT / "models").rglob("*.json"):
        with contextlib.suppress(Exception):
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
            if "model_file" in dados and "sha256" in dados:
                metadados[str((arquivo.parent / dados["model_file"]).relative_to(ROOT / "models"))] = dados["sha256"]
            if "sha256_int8" in dados:
                alvo = arquivo.with_suffix(".onnx")
                if alvo.is_file():
                    metadados[str(alvo.relative_to(ROOT / "models"))] = dados["sha256_int8"]
    linhas = []
    hashes: dict[str, str] = {}
    for caminho in sorted(registrados):
        atual = sha256(ROOT / "models" / caminho)
        copia = sha256(AVAL / "artefatos" / "models" / caminho)
        hashes[caminho] = atual
        meta = metadados.get(caminho)
        confere = atual == registrados[caminho] == copia and (meta is None or meta == atual)
        linhas.append([caminho, atual[:12], marca(atual == registrados[caminho]),
                       marca(atual == copia), "—" if meta is None else marca(meta == atual),
                       fig.confere(confere)])
    fig("SHA-256 calculado agora × hash registrado na avaliação de 11/11 × cópia em")
    fig("avaliacao_embarcada/artefatos × metadados gravados na exportação/quantização:")
    fig()
    fig.tabela(["Artefato ONNX (models/)", "SHA-256", "11/11", "cópia", "metad.", ""], linhas,
               direita=set())
    repetidos = Counter(hashes.values())
    duplicados = [c for c, h in hashes.items() if repetidos[h] > 1]
    if duplicados:
        fig()
        fig("Atenção: arquivos com conteúdo idêntico (mesmo SHA-256):")
        for caminho in duplicados:
            fig(f"  · {caminho}")
        fig("  As linhas 'MinMax' e 'Entropy' do Quadro 3 usam o MESMO arquivo; os resultados")
        fig("  iguais dessas duas linhas não são dois experimentos independentes.")
    fig.dados.update(hashes=hashes, duplicados=duplicados)


QUADRO3 = {
    # chave em avaliar_modelos.VARIANTS: (rótulo, mAP50, precisão, recall, FP, FN, MiB)
    "A_YOLOv8n_FP32": ("A - YOLOv8n FP32", 0.842, 0.895, 0.850, 2, 3, 11.70),
    "B_MobileNetV2_FP32": ("B - MobileNetV2 FP32", 0.897, 1.000, 0.900, 0, 2, 20.08),
    "B_MobileNetV2_INT8": ("B - MobileNetV2 INT8", 0.802, 0.889, 0.800, 2, 4, 5.68),
    "B_MobileNetV3_FP32": ("B - MobileNetV3 S. FP32", 0.745, 0.882, 0.750, 2, 5, 17.42),
    "B_MobileNetV3_INT8": ("B - MNV3 S. INT8 MinMax", 0.048, 1.000, 0.050, 0, 19, 4.98),
    "B_MobileNetV3_INT8_percentile": ("B - MNV3 S. INT8 Percentile", 0.095, 0.667, 0.100, 1, 18, 4.98),
    "B_MobileNetV3_INT8_entropy": ("B - MNV3 S. INT8 Entropy", 0.048, 1.000, 0.050, 0, 19, 4.98),
    "B_MobileNetV3_INT8_conv": ("B - MNV3 S. INT8 convoluções", 0.048, 1.000, 0.050, 0, 19, 6.64),
    "B_ShuffleNetV2_FP32": ("B - ShuffleNetV2 x0,5 FP32", 0.727, 0.882, 0.750, 2, 5, 13.03),
    "B_ShuffleNetV2_INT8": ("B - ShuffleNetV2 x0,5 INT8", 0.722, 0.882, 0.750, 2, 5, 3.83),
}


def sec_eficacia(fig: Figura) -> None:
    from animal_monitor_b.evaluation import evaluate_species_predictor

    aval = importar_avaliacao()
    resultados = {}
    linhas = []
    for chave, (rotulo, e_map, e_p, e_r, e_fp, e_fn, e_mib) in QUADRO3.items():
        tipo, det, cls = aval.VARIANTS[chave]
        preditor, _ = aval.make_predictor(tipo, det, cls, 1)
        rel = silencioso(evaluate_species_predictor, preditor, aval.DATA, label=chave)
        cont = rel["counts"]
        mib = sum((aval.MODELS / p).stat().st_size for p in (det, cls) if p) / 1024**2
        obtido = (rel["map50"], rel["precision@0.35"], rel["recall@0.35"],
                  cont["false_positive"], cont["false_negative"], mib)
        bate = (igual(obtido[0], e_map) and igual(obtido[1], e_p) and igual(obtido[2], e_r)
                and obtido[3] == e_fp and obtido[4] == e_fn and igual(mib, e_mib, 2))
        resultados[chave] = rel
        linhas.append([rotulo, br(rel["map50"]), br(rel["precision@0.35"]), br(rel["recall@0.35"]),
                       br(rel["f1@0.35"]), str(cont["true_positive"]), str(cont["false_positive"]),
                       str(cont["false_negative"]), br(mib, 2), fig.confere(bate)])
    fig("18 imagens de teste, 20 caixas · IoU 0,50 · escore 0,35 · ONNX Runtime CPU, 1 thread")
    fig()
    fig.tabela(["Configuração", "mAP@50", "Prec.", "Recall", "F1", "VP", "FP", "FN", "MiB", "Q3"],
               linhas)
    fig()
    b = resultados["B_MobileNetV2_FP32"]
    ap = b["ap50_by_class"]
    extra = [
        ("F1 Modelo A FP32", br(resultados["A_YOLOv8n_FP32"]["f1@0.35"]), "0,872"),
        ("F1 Modelo B FP32", br(b["f1@0.35"]), "0,947"),
        ("F1 Modelo B INT8", br(resultados["B_MobileNetV2_INT8"]["f1@0.35"]), "0,842"),
        ("B FP32: VP / FP / FN", "{true_positive} / {false_positive} / {false_negative}".format(**b["counts"]), "18 / 0 / 2"),
        ("B FP32: AP@50 anta", br(ap["anta"]["ap50"]), "0,857"),
        ("B FP32: AP@50 tamanduá-bandeira", br(ap["tamandua_bandeira"]["ap50"]), "0,833"),
        ("B FP32: AP@50 lobo-guará", br(ap["lobo_guara"]["ap50"]), "1,000"),
    ]
    fig.tabela(["Conferência do texto", "Obtido", "Relatório", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in extra])
    fig.dados["resultados"] = {k: {c: v[c] for c in ("map50", "precision@0.35", "recall@0.35",
                                                       "f1@0.35", "counts", "ap50_by_class")}
                               for k, v in resultados.items()}


CLASSIFICADORES = [
    ("MobileNetV2 FP32", "modelo_b/species_classifier.onnx", 0.9565),
    ("MobileNetV2 INT8", "modelo_b/species_classifier_int8.onnx", 0.9565),
    ("MobileNetV3 Small FP32", "modelo_b_mobilenet_v3_small/species_classifier.onnx", 0.8478),
    ("MobileNetV3 Small INT8 MinMax", "modelo_b_mobilenet_v3_small/species_classifier_int8.onnx", 0.5870),
    ("MobileNetV3 Small INT8 Percentile", "modelo_b_mobilenet_v3_small/species_classifier_int8_percentile.onnx", 0.6522),
    ("MobileNetV3 Small INT8 Entropy", "modelo_b_mobilenet_v3_small/species_classifier_int8_entropy.onnx", 0.5870),
    ("ShuffleNetV2 x0,5 FP32", "modelo_b_shufflenet_v2_x0_5/species_classifier.onnx", 0.7826),
    ("ShuffleNetV2 x0,5 INT8", "modelo_b_shufflenet_v2_x0_5/species_classifier_int8.onnx", 0.7826),
]


def sec_classificadores(fig: Figura) -> None:
    import cv2
    import numpy as np
    import onnxruntime as ort

    from animal_monitor_b.augment import resize_square, to_tensor
    from animal_monitor_b.classifier import CropCache
    from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner
    from animal_monitor_hw.models import YoloDetector

    linhas = []
    contagem = None
    for rotulo, relativo, esperado in CLASSIFICADORES:
        caminho = ROOT / "models" / relativo
        meta = json.loads((caminho.parent / "classifier_metadata.json").read_text(encoding="utf-8"))
        rotulos = meta["labels"]
        cache = CropCache(ROOT / "data" / "crops", "test", rotulos)
        contagem = cache.counts()
        sessao = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
        entrada = sessao.get_inputs()[0]
        tamanho = int(meta.get("input_size", 128))
        acertos = 0
        for imagem, alvo in zip(cache.images, cache.targets):
            tensor = to_tensor(resize_square(imagem, tamanho))[None]
            saida = sessao.run(None, {entrada.name: tensor})[0]
            acertos += int(int(np.argmax(saida[0])) == int(alvo))
        acuracia = acertos / len(cache)
        linhas.append([rotulo, f"{acertos}/{len(cache)}", pct(acuracia), pct(esperado),
                       fig.confere(igual(acuracia, esperado, 4))])
    fig(f"Recortes de teste: {sum(contagem.values())} → "
        + ", ".join(f"{NOME.get(k, k)} {v}" for k, v in contagem.items()))
    fig("Pré-processamento igual ao do treino (resize quadrado 128 + normalização ImageNet)")
    fig()
    fig.tabela(["Classificador (ONNX)", "Acertos", "Acurácia", "Relatório", ""], linhas)

    fig()
    fig("Quantização do detector de presença (18 imagens de teste, limiar 0,30, NMS 0,40):")
    variantes = [
        ("FP32", "modelo_b/presence_detector.onnx"),
        ("INT8 só convoluções (adotado)", "modelo_b/presence_detector_int8.onnx"),
        ("INT8 todos os operadores", "modelo_b/presence_detector_int8_todos_ops.onnx"),
    ]
    from animal_monitor_b.evaluation import _split_images

    imagens = [cv2.imread(str(p)) for p in _split_images(ROOT / "config" / "dataset.yaml", "test")]
    det_linhas = []
    for rotulo, relativo in variantes:
        caminho = ROOT / "models" / relativo
        detector = YoloDetector(OnnxRuntimeRunner(caminho, intra_op_threads=4), {0: "animal"},
                                0.30, 0.40, max_detections=20, output_format="standard",
                                model_path=caminho)
        escores = []
        for quadro in imagens:
            _, s, _ = detector.detect_all(quadro)
            escores.extend(float(x) for x in s)
        maximo = max(escores) if escores else 0.0
        det_linhas.append([rotulo, str(len(escores)), br(statistics.mean(escores)) if escores else "—",
                           br(maximo), "inválido (>1)" if maximo > 1 else "válido [0,1]"])
    fig.tabela(["Detector", "Caixas", "Escore médio", "Escore máx.", "Escores"], det_linhas,
               direita={1, 2, 3})
    invalido = det_linhas[2][3]
    fig()
    fig(f"Relatório: 'a quantização de todos os operadores produziu escores inválidos, "
        f"superiores a 1' {fig.confere(det_linhas[2][4].startswith('inválido') and det_linhas[1][4].startswith('válido'))}")
    fig.dados["detector_int8_todos_ops_max"] = invalido


def sec_resolucao(fig: Figura) -> None:
    import medir_proxies as mp  # type: ignore  # versoes/comum/ferramentas

    esperado = [("640 × 640", "a_640", 0.872), ("512 × 512", "a_512", 0.872),
                ("320 × 320", "a_320", 0.833), ("224 × 320", "a_224x320", 0.857),
                ("256 × 256", "a_256", 0.727), ("224 × 224", "a_224", 0.667),
                ("192 × 192", "a_192", 0.429)]
    linhas = []
    for rotulo, nome, f1_esperado in esperado:
        caminho = mp.REFERENCIA.get(nome, mp.MODELOS / f"{nome}.onnx")
        rel = silencioso(mp.avaliar_a, caminho, "standard")
        r = mp.resumo(rel)
        linhas.append([rotulo, br(r["map50"]), br(r["precisao"]), br(r["recall"]), br(r["f1"]),
                       br(f1_esperado), fig.confere(igual(r["f1"], f1_esperado))])
    fig("Modelo A (YOLOv8n) FP32 exportado em várias entradas · 18 imagens de teste")
    fig()
    fig.tabela(["Entrada", "mAP@50", "Prec.", "Recall", "F1", "F1 relatório", ""], linhas)
    fig()
    fig("Saída YOLO dividida (6 tensores box/cls + DFL e NMS em NumPy) × export padrão:")
    nomes_a = mp.load_class_names(mp.DATASET)
    eq_linhas = []
    for rotulo, padrao, dividida, nomes, esperado_px in [
        ("Modelo A 640", mp.REFERENCIA["a_640"], mp.MODELOS / "a_640_split.onnx", nomes_a, None),
        ("Modelo B 416 (presença)", mp.REFERENCIA["b_416"], mp.MODELOS / "b_416_split.onnx", {0: "animal"}, 0.0004),
    ]:
        eq = silencioso(mp.equivalencia_dividida, padrao, dividida, nomes)
        bate = eq["imagens_com_conjunto_diferente"] == 0 and (
            esperado_px is None or eq["maior_diferenca_caixa_px"] == esperado_px)
        eq_linhas.append([rotulo, str(eq["imagens"]), str(eq["imagens_com_conjunto_diferente"]),
                          br(eq["maior_diferenca_caixa_px"], 4) + " px",
                          br(eq["maior_diferenca_escore"], 6),
                          "0,0004 px" if esperado_px else "—", fig.confere(bate)])
    fig.tabela(["Detector", "Imagens", "Conj. diferente", "Máx. Δ caixa", "Máx. Δ escore",
                "Relatório", ""], eq_linhas)


def sec_rajadas(fig: Figura) -> None:
    import simular_rajadas as sr  # type: ignore

    salvo = json.loads((MEDICOES / "rajadas_x86.json").read_text(encoding="utf-8"))["proxies"]
    linhas = []
    resultados = {}
    for proxy in sr.PROXIES:
        r = silencioso(sr.simular, proxy)
        resultados[proxy] = r
        d = Counter(r["desfechos"])
        igual_salvo = Counter(salvo[proxy]["desfechos"]) == d
        linhas.append([proxy, str(r["eventos"]), str(d["correto"]), str(d["errado"]),
                       str(d["inconclusivo"]), str(d["vazio"]), fig.confere(igual_salvo)])
    fig("90 eventos por configuração (30 por vídeo) · rajadas de 8 quadros · mesmo código das versões")
    fig()
    fig.tabela(["Configuração", "Eventos", "Correto", "Errado", "Inconcl.", "Vazio", "= JSON"],
               linhas)
    errados_a = [Counter(v["desfechos"])["errado"] for k, v in resultados.items() if k.startswith("a_")]
    errados_b = [Counter(v["desfechos"])["errado"] for k, v in resultados.items() if k.startswith("b_")]
    inconc_b = [Counter(v["desfechos"])["inconclusivo"] for k, v in resultados.items() if k.startswith("b_")]
    fig()
    conferencia = [
        ("Eventos por configuração × quadros", f"{resultados['a_640']['eventos']} × {resultados['a_640']['quadros_por_evento']}", "90 × 8"),
        ("Modelo A: eventos com espécie errada", f"{min(errados_a)} a {max(errados_a)}", "7 a 14"),
        ("Modelo B: eventos com espécie errada", f"{max(errados_b)}" if max(errados_b) == 0 else f"{min(errados_b)} a {max(errados_b)}", "0"),
        ("Modelo B: eventos inconclusivos", f"{min(inconc_b)} a {max(inconc_b)}", "27 a 33"),
    ]
    fig.tabela(["Conferência", "Obtido", "Relatório", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in conferencia])
    fig()
    fig("'= JSON': desfechos idênticos aos de versoes/ranking/medicoes/rajadas_x86.json (21/10/2026).")
    fig.dados["rajadas"] = {k: v["desfechos"] for k, v in resultados.items()}


def _infer_a(video: Path, pasta: Path) -> dict:
    runtime = runtime_temporario(ROOT / "config" / "runtime.yaml", pasta / "runtime.yaml",
                                 {"output_dir": str(pasta), "database": str(pasta / "eventos.sqlite3")})
    proc = rodar([PY, "-m", "animal_monitor", "infer", "--model", "models/animal_detector.onnx",
                  "--source", str(video.relative_to(ROOT)), "--data", "config/dataset.yaml",
                  "--runtime", str(runtime)], env={"PYTHONPATH": "src"})
    return json_da_saida(proc.stdout)


def _infer_b(video: Path, pasta: Path, iou: str = "0.40") -> dict:
    runtime = runtime_temporario(ROOT / "config" / "modelo_b" / "runtime.yaml", pasta / "runtime.yaml",
                                 {"output_dir": str(pasta), "database": str(pasta / "eventos.sqlite3"),
                                  "detector_iou": iou})
    proc = rodar([PY, "-m", "animal_monitor_b", "infer",
                  "--detector", "models/modelo_b/presence_detector.onnx",
                  "--classifier", "models/modelo_b/species_classifier.onnx",
                  "--source", str(video.relative_to(ROOT)), "--runtime", str(runtime)],
                 env={"PYTHONPATH": "src"})
    return json_da_saida(proc.stdout)


def _fmt_eventos(eventos: dict[str, int]) -> str:
    return ", ".join(f"{NOME.get(k, k)} {v}" for k, v in sorted(eventos.items())) or "nenhum"


def sec_videos(fig: Figura) -> None:
    base = OUT / "dados" / "videos"
    if base.exists():
        shutil.rmtree(base)
    linhas = []
    errados = {"A": 0, "B": 0}
    errados_por_video_a = {}
    eventos_b_050 = {}
    for especie, video in VIDEOS.items():
        a = _infer_a(video, base / "modelo_a" / especie)["new_events"]
        b = _infer_b(video, base / "modelo_b_nms040" / especie)["new_sightings"]
        b50 = _infer_b(video, base / "modelo_b_nms050" / especie, "0.50")["new_sightings"]
        eventos_b_050[especie] = sum(b50.values())
        err_a = sum(v for k, v in a.items() if k != especie)
        err_b = sum(v for k, v in b.items() if k != especie)
        errados["A"] += err_a
        errados["B"] += err_b
        errados_por_video_a[especie] = {k: v for k, v in a.items() if k != especie}
        linhas.append([NOME[especie], _fmt_eventos(a), _fmt_eventos(b), _fmt_eventos(b50)])
    fig("3 vídeos de amostra (240 quadros, 24 fps) · eventos novos gravados no SQLite")
    fig("Modelo A: confirmação em 3 quadros seguidos · Modelo B: rastreamento + votação por trilha")
    fig()
    fig.tabela(["Vídeo", "Modelo A (NMS 0,45)", "Modelo B (NMS 0,40)", "Modelo B (NMS 0,50)"],
               linhas, direita=set())

    from animal_monitor_b.config import RuntimeConfigB
    from animal_monitor_b.evaluation import evaluate_species_predictor
    from animal_monitor_b.runtime.pipeline import TwoStagePipeline
    from dataclasses import replace

    maps = {}
    for iou in (0.40, 0.50):
        cfg = replace(RuntimeConfigB.from_yaml(ROOT / "config" / "modelo_b" / "runtime.yaml"),
                      detector_iou=iou)
        pipe = TwoStagePipeline.from_models(ROOT / "models/modelo_b/presence_detector.onnx",
                                            ROOT / "models/modelo_b/species_classifier.onnx", cfg)
        maps[iou] = silencioso(evaluate_species_predictor, pipe.detect_species,
                               ROOT / "config" / "dataset.yaml")["map50"]
    fig()
    esperado_a = {"lobo_guara": {"anta": 1}, "tamandua_bandeira": {"anta": 1}}
    conferencia = [
        ("Modelo A: eventos de espécie errada", str(errados["A"]), "2"),
        ("Modelo A: errado no vídeo do lobo-guará", _fmt_eventos(errados_por_video_a["lobo_guara"]),
         _fmt_eventos(esperado_a["lobo_guara"])),
        ("Modelo A: errado no vídeo do tamanduá", _fmt_eventos(errados_por_video_a["tamandua_bandeira"]),
         _fmt_eventos(esperado_a["tamandua_bandeira"])),
        ("Modelo B: eventos de espécie errada", str(errados["B"]), "0"),
        ("Modelo B NMS 0,50: eventos (anta/lobo/tamanduá)",
         "/".join(str(eventos_b_050[k]) for k in VIDEOS), "1/2/2"),
        ("Modelo B: mAP@50 com NMS 0,40 e 0,50", f"{br(maps[0.40])} e {br(maps[0.50])}", "0,897 e 0,897"),
    ]
    fig.tabela(["Conferência", "Obtido", "Relatório / projeto", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in conferencia])
    fig()
    fig("Duplicações com NMS 0,50 (1/2/2) e o mAP inalterado vêm de config/modelo_b/runtime.yaml")
    fig("e RESULTADOS_COMPLETOS.md §7.3; o relatório cita só 'eliminou duplicações'.")


def sec_speciesnet(fig: Figura) -> None:
    import numpy as np

    from animal_monitor_b.data_prep import parse_label_file
    from animal_monitor_b.evaluation import _labels_for, _split_images

    predicoes = MEDICOES / "speciesnet_predicoes_test.json"
    saida = OUT / "dados" / "speciesnet.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    proc = rodar([PY, "versoes/comum/ferramentas/avaliar_speciesnet.py", str(predicoes),
                  "--saida", str(saida)])
    resumo = json_da_saida(proc.stdout)
    rel = json.loads(saida.read_text(encoding="utf-8"))

    # Presença: caixa do MegaDetector (animal, conf ≥ 0,2) com IoU ≥ 0,5 em cada animal anotado.
    dados = json.loads(predicoes.read_text(encoding="utf-8"))
    por_arquivo = {Path(p["filepath"]).name: p.get("detections") or [] for p in dados["predictions"]}
    encontrados = total = 0
    for caminho in _split_images(ROOT / "config" / "dataset.yaml", "test"):
        caixas_md = [d["bbox"] for d in por_arquivo.get(caminho.name, [])
                     if d.get("category") == "1" and float(d.get("conf", 0)) >= 0.2]
        livres = list(range(len(caixas_md)))
        for gt in parse_label_file(_labels_for(caminho)):
            total += 1
            g = np.array([gt.x_center - gt.width / 2, gt.y_center - gt.height / 2,
                          gt.x_center + gt.width / 2, gt.y_center + gt.height / 2])
            melhor, indice = 0.0, None
            for i in livres:
                x, y, w, h = caixas_md[i]
                p = np.array([x, y, x + w, y + h])
                inter = max(0, min(g[2], p[2]) - max(g[0], p[0])) * max(0, min(g[3], p[3]) - max(g[1], p[1]))
                uniao = (g[2] - g[0]) * (g[3] - g[1]) + w * h - inter
                iou = inter / uniao if uniao > 0 else 0
                if iou > melhor:
                    melhor, indice = iou, i
            if indice is not None and melhor >= 0.5:
                encontrados += 1
                livres.remove(indice)
    c = resumo["contagens"]
    fig("Reprocessa as predições salvas do MegaDetector v5a + SpeciesNet 4.0.3a (--country BRA)")
    fig("com o mesmo código de métrica dos Modelos A/B. A inferência do SpeciesNet em si")
    fig("(PyTorch) não foi refeita aqui; o arquivo de predições é a evidência de entrada.")
    fig()
    fig.tabela(["Métrica", "Valor"], [
        ["mAP@50", br(resumo["map50"])], ["Precisão", br(resumo["precisao"])],
        ["Recall", br(resumo["recall"])], ["F1", br(resumo["f1"])],
        ["VP / FP / FN", f"{c['true_positive']} / {c['false_positive']} / {c['false_negative']}"],
    ], direita=set())
    fig("Níveis taxonômicos devolvidos pelo SpeciesNet: "
        + ", ".join(f"{k} {v}" for k, v in list(resumo["niveis"].items())[:4]))
    fig("  " + ", ".join(f"{k} {v}" for k, v in list(resumo["niveis"].items())[4:]))
    fig()
    conferencia = [
        ("Animais com presença detectada (IoU ≥ 0,5)", f"{encontrados}/{total}", "20/20"),
        ("Espécie exata atribuída", f"{c['true_positive']}/{c['ground_truth_total']}", "13/20"),
        ("F1", br(resumo["f1"], 2), "0,65"),
        ("Versão do modelo", ", ".join(rel["model_version"]), "4.0.3a"),
    ]
    fig.tabela(["Conferência", "Obtido", "Relatório", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in conferencia])


QUADRO4 = {
    "A_YOLOv8n_FP32": ("A - YOLOv8n FP32", 51.21, 50.28, 57.90, 19.53),
    "B_MobileNetV2_FP32": ("B - MobileNetV2 FP32", 41.92, 38.85, 61.04, 23.86),
    "B_MobileNetV2_INT8": ("B - MobileNetV2 INT8", 33.93, 31.86, 51.67, 29.47),
}


def sec_benchmark(fig: Figura) -> None:
    import cv2

    aval = importar_avaliacao()
    caminhos = sorted((aval.ART / "data/animals/images/test").iterdir())
    imagens = [cv2.imread(str(p)) for p in caminhos]
    linhas = []
    medidas = {}
    for chave, (rotulo, e_med, e_p50, e_p95, e_thr) in QUADRO4.items():
        tipo, det, cls = aval.VARIANTS[chave]
        preditor, _ = aval.make_predictor(tipo, det, cls, 4)
        b = aval.benchmark(preditor, imagens)
        medidas[chave] = b
        linhas.append([rotulo, "agora", br(b["mean_ms"], 2), br(b["p50_ms"], 2), br(b["p95_ms"], 2),
                       br(b["throughput_sequential_fps"], 2)])
        linhas.append(["", "Quadro 4", br(e_med, 2), br(e_p50, 2), br(e_p95, 2), br(e_thr, 2)])
    fig("Mesmo procedimento do avaliar_modelos.py: 2 imagens de aquecimento, 18 imagens × 2 =")
    fig("36 chamadas medidas, 4 threads, pré + inferência + pós-processamento por imagem.")
    fig()
    fig.tabela(["Configuração", "Origem", "Média ms", "p50 ms", "p95 ms", "img/s"], linhas,
               direita={2, 3, 4, 5})
    p50 = {k: v["p50_ms"] for k, v in medidas.items()}
    ordem = p50["B_MobileNetV2_INT8"] < p50["B_MobileNetV2_FP32"] < p50["A_YOLOv8n_FP32"]
    fig()
    fig(f"{APROX} Tempos variam com a carga da máquina; o relatório afirma a ORDEM, que se confere:")
    fig(f"  p50: B INT8 < B FP32 < A FP32  →  {br(p50['B_MobileNetV2_INT8'], 1)} < "
        f"{br(p50['B_MobileNetV2_FP32'], 1)} < {br(p50['A_YOLOv8n_FP32'], 1)} ms  {fig.confere(ordem)}")
    fig.dados["medidas"] = medidas


def _media_videos(arquivos: list[Path], modelo_b: bool) -> tuple[float, float]:
    fps, lat = [], []
    for arquivo in arquivos:
        d = json.loads(arquivo.read_text(encoding="utf-8"))
        fps.append(d["throughput_fps"])
        lat.append(d["latency_ms"]["total"]["mean"] if modelo_b else d["latency_ms"]["mean"])
    return statistics.mean(fps), statistics.mean(lat)


def sec_video_fps(fig: Figura) -> None:
    pasta = OUT / "dados" / "video_fps"
    pasta.mkdir(parents=True, exist_ok=True)
    runtime_a = runtime_temporario(ROOT / "config" / "runtime.yaml", pasta / "runtime_a.yaml",
                                   {"intra_op_threads": "4", "output_dir": str(pasta),
                                    "database": str(pasta / "a.sqlite3")})
    runtime_b = runtime_temporario(ROOT / "config" / "modelo_b" / "runtime.yaml", pasta / "runtime_b.yaml",
                                   {"intra_op_threads": "4", "output_dir": str(pasta),
                                    "database": str(pasta / "b.sqlite3")})
    configs = {
        "Modelo A FP32": ("a", None, None, "modelo_a_4t", 15.9, 63.2),
        "Modelo B FP32": ("b", "presence_detector.onnx", "species_classifier.onnx", "modelo_b_fp32_4t", 31.1, 32.2),
        "Modelo B INT8": ("b", "presence_detector_int8.onnx", "species_classifier_int8.onnx", "modelo_b_int8_4t", 35.2, 28.5),
    }
    linhas = []
    agora = {}
    salvos_fps = {}
    for rotulo, (tipo, det, cls, sufixo, e_fps, e_lat) in configs.items():
        arquivos_agora = []
        for especie, video in VIDEOS.items():
            saida = pasta / f"{especie}_{sufixo}.json"
            if tipo == "a":
                cmd = [PY, "-m", "animal_monitor", "benchmark", "--model", "models/animal_detector.onnx",
                       "--runtime", str(runtime_a)]
            else:
                cmd = [PY, "-m", "animal_monitor_b", "benchmark",
                       "--detector", f"models/modelo_b/{det}", "--classifier", f"models/modelo_b/{cls}",
                       "--runtime", str(runtime_b)]
            cmd += ["--source", str(video.relative_to(ROOT)), "--frames", "200", "--output", str(saida)]
            rodar(cmd, env={"PYTHONPATH": "src"})
            arquivos_agora.append(saida)
        salvos = [ROOT / "outputs" / "modelo_b" / f"benchmark_video_{e}_{sufixo}.json" for e in VIDEOS]
        fps_s, lat_s = _media_videos(salvos, tipo == "b")
        fps_a, lat_a = _media_videos(arquivos_agora, tipo == "b")
        agora[rotulo] = fps_a
        salvos_fps[rotulo] = fps_s
        linhas.append([rotulo, br(e_fps, 1), br(e_lat, 1), br(fps_s, 2), br(lat_s, 2),
                       fig.confere(igual(fps_s, e_fps, 1) and igual(lat_s, e_lat, 1)),
                       br(fps_a, 1), br(lat_a, 1)])
    fig("Vídeos de amostra, 200 quadros medidos por vídeo, 4 threads, média dos 3 vídeos.")
    fig("'JSON salvo' = outputs/modelo_b/benchmark_video_*_4t.json (i5-1235U, ago/2026).")
    fig()
    fig.tabela(["Configuração", "FPS rel.", "ms rel.", "FPS JSON", "ms JSON", "",
                "FPS agora", "ms agora"], linhas, direita={1, 2, 3, 4, 6, 7})
    ordem = agora["Modelo B INT8"] > agora["Modelo B FP32"] > agora["Modelo A FP32"]
    fig()
    fig("✓ na 6ª coluna: o número do relatório é a média dos JSON salvos, arredondada a 1 casa.")
    if not igual(salvos_fps["Modelo A FP32"], 15.9, 1):
        fig(f"{ERRO} Modelo A: média {br(salvos_fps['Modelo A FP32'], 3)} FPS arredonda para "
            f"{br(salvos_fps['Modelo A FP32'], 1)}; o texto diz 15,9 (arredondamento duplo 15,846 → 15,85 → 15,9).")
    fig(f"{APROX} 'agora' roda em outra CPU ({cpu_modelo()}); vale a ordem relativa:")
    fig(f"  FPS: B INT8 > B FP32 > A FP32  {fig.confere(ordem)}")
    sim = [ROOT / "outputs" / "sim_embedded" / f"benchmark_video_{e}.json" for e in VIDEOS]
    fps_sim = [json.loads(p.read_text(encoding="utf-8"))["throughput_fps"] for p in sim]
    fig()
    fig("Contêiner nativo com limites de CPU/RAM (outputs/sim_embedded, 100 quadros por vídeo):")
    fig(f"  {' + '.join(br(v, 2) for v in fps_sim)} → média {br(statistics.mean(fps_sim), 2)} FPS"
        f" · relatório 7,74  {fig.confere(igual(statistics.mean(fps_sim), 7.74, 2))}")
    fig("  O JSON não registra o classificador; RESULTADOS_COMPLETOS.md indica MobileNetV3 Small.")


PERFIS = [
    ("Raspberry Pi 5", "cortex-a76", "4", "4096m",
     ["animal_detector.onnx", "modelo_b/presence_detector.onnx", "modelo_b/species_classifier.onnx"],
     ["A", "B"]),
    ("Raspberry Pi Zero 2 W", "cortex-a53", "4", "512m",
     ["modelo_b/presence_detector_int8.onnx", "modelo_b/species_classifier_int8.onnx"], ["B_INT8"]),
    ("Orange Pi 5", "cortex-a76", "4", "4096m",
     ["modelo_b/presence_detector_int8.onnx", "modelo_b/species_classifier_int8.onnx"], ["B_INT8"]),
    ("Jetson Orin Nano Super", "cortex-a78ae", "6", "8192m",
     ["modelo_b/presence_detector.onnx", "modelo_b/species_classifier.onnx"], ["B"]),
]
PIPELINE_ESPERADO = {
    "A": ("anta", 0.99163, [41.5, 0.0, 799.0, 471.0]),
    "B": ("anta", 0.86023, [14.5, 0.0, 799.0, 471.0]),
    "B_INT8": ("anta", 0.79998, [13.8, 0.0, 799.0, 470.0]),
}
NOME_VARIANTE = {"A": "Modelo A FP32", "B": "Modelo B FP32", "B_INT8": "Modelo B INT8"}


def _linhas_json(texto: str) -> list[dict]:
    resultado = []
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                resultado.append(json.loads(linha))
    return resultado


def _perfil_arm64(perfil: tuple) -> dict:
    nome, cpu, cpus, memoria, modelos, variantes = perfil
    comum = dict(cpu=cpu, cpus=cpus, memoria=memoria, montagens=[f"{AVAL}:/work:ro"])
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/work/artefatos/src"}
    inicio = time.perf_counter()
    smoke = docker_arm64(["/work/arm64_smoke.py"], env={**env, "TCC_MODELS": ",".join(modelos)}, **comum)
    pipelines = {}
    codigos = [smoke.returncode]
    for variante in variantes:
        proc = docker_arm64(["/work/arm64_pipeline.py"], env={**env, "TCC_VARIANT": variante}, **comum)
        codigos.append(proc.returncode)
        saida = _linhas_json(proc.stdout)
        pipelines[variante] = saida[-1] if saida else {"erro": (proc.stdout + proc.stderr)[-600:]}
    return {"nome": nome, "cpu": cpu, "cpus": cpus, "memoria": memoria, "codigos": codigos,
            "smoke": _linhas_json(smoke.stdout), "smoke_erro": smoke.stderr[-600:],
            "pipelines": pipelines, "duracao_s": time.perf_counter() - inicio}


def sec_arm64(fig: Figura) -> None:
    with ThreadPoolExecutor(max_workers=len(PERFIS)) as pool:
        perfis = list(pool.map(_perfil_arm64, PERFIS))
    host = {}
    for variante in PIPELINE_ESPERADO:
        proc = rodar([PY, str(AVAL / "arm64_pipeline.py")],
                     env={"TCC_VARIANT": variante, "PYTHONPATH": str(AVAL / "artefatos" / "src")})
        host[variante] = _linhas_json(proc.stdout)[-1]

    fig("Imagem Docker linux/arm64 · qemu-aarch64 (user-mode) · ONNX Runtime CPU, 1 thread ·")
    fig("rede desligada · contêiner somente leitura · --cpus/--memory/--memory-swap por perfil")
    fig()
    fig("1) Smoke test (arm64_smoke.py): carrega cada ONNX, roda tensor zerado, confere saída")
    smoke_linhas = []
    for p in perfis:
        for item in p["smoke"]:
            smoke_linhas.append([p["nome"], p["cpu"], f"{p['cpus']} CPU · {p['memoria'][:-1]} MiB",
                                 item["model"].replace("modelo_b/", "B/"),
                                 "×".join(str(v) for v in item["output_shapes"][0]),
                                 item["architecture"], fig.confere(item["finite"] and item["architecture"] == "aarch64")])
        if not p["smoke"]:
            smoke_linhas.append([p["nome"], p["cpu"], "", "FALHOU", "", "", fig.confere(False)])
    fig.tabela(["Perfil", "CPU emulada", "Limites", "Modelo", "Saída", "Arq.", "finita"],
               smoke_linhas, direita=set())
    fig()
    fig("2) Pipeline completo (arm64_pipeline.py) na imagem real anta00000005.jpg")
    pipe_linhas = []
    for p in perfis:
        for variante, r in p["pipelines"].items():
            if "predictions" not in r or not r["predictions"]:
                pipe_linhas.append([p["nome"], NOME_VARIANTE[variante], "ERRO", "", "", fig.confere(False)])
                continue
            pred = r["predictions"][0]
            classe, conf, caixa = PIPELINE_ESPERADO[variante]
            bate = pred["class"] == classe and pred["confidence"] == conf and pred["box"] == caixa
            pipe_linhas.append([p["nome"], NOME_VARIANTE[variante], pred["class"],
                                br(pred["confidence"], 5),
                                "[" + "; ".join(br(v, 1) for v in pred["box"]) + "]", fig.confere(bate)])
    fig.tabela(["Perfil", "Modelo", "Classe", "Confiança", "Caixa (x1; y1; x2; y2)", ""], pipe_linhas,
               direita={3})
    fig()
    fig("3) Verificação cruzada x86_64 (host) × ARM64 (emulado) na mesma imagem")
    cruz = []
    esperado_host = {"A": 0.99163, "B": 0.86023, "B_INT8": 0.80757}
    for variante, (classe, conf_arm, caixa) in PIPELINE_ESPERADO.items():
        h = host[variante]["predictions"][0]
        a = next(p["pipelines"][variante]["predictions"][0] for p in perfis if variante in p["pipelines"]
                 and p["pipelines"][variante].get("predictions"))
        bate = (h["confidence"] == esperado_host[variante] and a["confidence"] == conf_arm
                and h["box"] == a["box"] == caixa)
        cruz.append([NOME_VARIANTE[variante], br(h["confidence"], 5), br(a["confidence"], 5),
                     "igual" if h["box"] == a["box"] else "DIFERENTE",
                     f"{br(esperado_host[variante], 5)} / {br(conf_arm, 5)}", fig.confere(bate)])
    fig.tabela(["Modelo", "Conf. x86_64", "Conf. ARM64", "Caixa", "Relatório (x86/ARM)", ""], cruz)
    fig()
    codigos = {p["nome"]: p["codigos"] for p in perfis}
    fig("Código de saída dos contêineres (0 = concluiu; 137 = encerrado por falta de memória):")
    fig("  " + "; ".join(f"{n} {c}" for n, c in codigos.items()))
    fig.dados["perfis"] = perfis
    fig.dados["host"] = host


def sec_matriz(fig: Figura) -> None:
    pasta = OUT / "dados" / "matriz"
    proc = rodar([PY, "scripts/testar_matriz.py", "--etapas", "qemu,docker", "--output-dir", str(pasta)],
                 timeout=3600)
    qemu = json.loads((pasta / "qemu-matriz" / "resultado.json").read_text(encoding="utf-8"))
    docker = json.loads((pasta / "docker-recursos" / "resultado.json").read_text(encoding="utf-8"))
    fig(f"Etapa QEMU: CLI de cada versão com 'info' em aarch64 · {qemu['qemu']}")
    fig(f"Etapa Docker: benchmark em 1 imagem de teste com --cpus/--memory da placa (timeout {docker['timeout_s']} s)")
    fig()
    linhas = []
    por_id = {c["id"]: c for c in docker["cases"]}
    for caso in qemu["cases"]:
        d = por_id.get(caso["id"])
        limite = f"{d['cpus']} CPU · {d['memoria']}" if d else "—"
        estado_d = ("—" if d is None else (OK if d["resultado"] == "passed" else f"{ERRO} {d['motivo']}"))
        linhas.append([caso["id"], caso["placa"].replace("Raspberry Pi 5", "Pi 5").replace("NVIDIA ", ""),
                       caso["qemu_cpu"],
                       OK if caso["resultado"] == "passed" else ERRO, limite, estado_d])
    fig.tabela(["Versão", "Placa", "CPU QEMU", "QEMU", "Limites Docker", "Docker"], linhas,
               direita=set())
    fig()
    fig(f"QEMU: {qemu['passed']}/{qemu['total']} versões iniciaram · Docker: "
        f"{docker['passed']}/{docker['total']} benchmarks concluídos, "
        f"{len(docker.get('nao_aplicavel', []))} versões sem pipeline ONNX/CPU (firmware/MCU)")
    conferencia = [("Combinações placa × modelo × ambiente", str(qemu["total"]), "25"),
                   ("Versões que iniciam em ARM64 (QEMU)", f"{qemu['passed']}/{qemu['total']}", "25/25")]
    fig()
    fig.tabela(["Conferência", "Obtido", "Relatório / projeto", ""],
               [[n, o, e, fig.confere(o == e)] for n, o, e in conferencia])
    fig()
    genericos = [c["id"] for c in docker["cases"] if "genérico" in (c.get("nota_modelo") or "")]
    if genericos:
        fig(f"Em {', '.join(genericos)} a variante específica não existe no acervo: o benchmark usou o")
        fig("ONNX genérico do Modelo A (640). Isso mostra que o software roda sob o limite, não a variante.")
    fig("Rodada anterior (04/11/2026, timeout 120 s): 16/20 concluídos (05a e 06a por timeout,")
    fig("07a e 10a encerrados por memória). O relatório não informa o resultado desses testes Docker.")
    if proc.returncode not in (0, 1):
        fig(f"(testar_matriz.py terminou com código {proc.returncode})")


SECOES: dict[str, tuple[int, str, str, Callable[[Figura], None]]] = {
    "ambiente": (1, "Ambiente de execução das evidências", "seções 5.2 e 5.3", sec_ambiente),
    "dataset": (2, "Validação do conjunto de dados", "seções 4 e 5 (180 imagens, 207 caixas)", sec_dataset),
    "testes": (3, "Testes automatizados em x86_64 e ARM64 emulado", "seção 5 (12 testes; 93/93 em ARM64)", sec_testes),
    "integridade": (4, "Integridade SHA-256 dos artefatos ONNX", "seção 6 (verificação de integridade)", sec_integridade),
    "eficacia": (5, "Eficácia comparável das 10 configurações", "Quadro 3, seções 5 e 6, resumo", sec_eficacia),
    "classificadores": (6, "Classificadores do Modelo B nos 46 recortes e quantização", "seção 5", sec_classificadores),
    "resolucao": (7, "Resolução de entrada e saída YOLO dividida", "seção 5", sec_resolucao),
    "rajadas": (8, "Simulação de rajadas da armadilha", "seção 5 (90 eventos de 8 quadros)", sec_rajadas),
    "videos": (9, "Eventos nos vídeos de amostra e ajuste do NMS", "seção 5", sec_videos),
    "speciesnet": (10, "MegaDetector v5a + SpeciesNet como referência", "seção 5", sec_speciesnet),
    "benchmark": (11, "Benchmark em CPU x86_64 (18 imagens)", "Quadro 4, seção 5.2", sec_benchmark),
    "video_fps": (12, "Velocidade nos vídeos (200 quadros, 4 threads)", "seção 5", sec_video_fps),
    "arm64": (13, "Compatibilidade ARM64 em QEMU por perfil de placa", "seções 5.3 e 5.4, Quadro 6", sec_arm64),
    "matriz": (14, "Matriz das 25 versões em QEMU e Docker limitado", "seção 5", sec_matriz),
}


# --------------------------------------------------------------------------- PNG


def png(txt: Path, chrome: str) -> Path:
    linhas = txt.read_text(encoding="utf-8").rstrip("\n").splitlines()

    def colore(linha: str) -> str:
        texto = html.escape(linha)
        for simbolo, classe in ((OK, "ok"), (ERRO, "erro"), (APROX, "aprox")):
            texto = texto.replace(simbolo, f'<span class="{classe}">{simbolo}</span>')
        if linha.startswith("EVIDÊNCIA"):
            return f'<span class="titulo">{texto}</span>'
        if linha.startswith("$ "):
            return f'<span class="cmd">{texto}</span>'
        if linha.startswith(("Relatório:", "Executado em")):
            return f'<span class="meta">{texto}</span>'
        if linha.startswith("CONFERÊNCIA"):
            return f'<span class="total">{texto}</span>'
        return texto

    corpo = "\n".join(colore(linha) for linha in linhas)
    colunas = max(len(linha) for linha in linhas)
    largura = int(colunas * 8.43 + 48)
    altura = int(len(linhas) * 18.9 + 44)
    pagina = f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:#16181d}}
pre{{margin:0;padding:22px 24px;font:14px/18.9px "DejaVu Sans Mono","Ubuntu Mono",monospace;
color:#d7dae0;white-space:pre}}
.titulo{{color:#7cc4ff;font-weight:bold}}.cmd{{color:#f0c674}}.meta{{color:#8b93a1}}
.ok{{color:#5fd068;font-weight:bold}}.erro{{color:#ff6b6b;font-weight:bold}}
.aprox{{color:#f0c674;font-weight:bold}}.total{{color:#ffffff;font-weight:bold}}
</style><pre>{corpo}</pre>"""
    pagina_html = txt.with_suffix(".html")
    pagina_html.write_text(pagina, encoding="utf-8")
    destino = txt.with_suffix(".png")
    perfil = OUT / "dados" / ".chrome"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={perfil}", "--force-device-scale-factor=2",
                    f"--window-size={largura},{altura}", f"--screenshot={destino}",
                    pagina_html.as_uri()], capture_output=True, timeout=120)
    pagina_html.unlink(missing_ok=True)
    return destino


def gerar_pngs() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        print("Chrome/Chromium não encontrado: PNG não gerados (os .txt servem para captura manual).")
        return
    for txt in sorted(OUT.glob("[0-9][0-9]_*.txt")):
        print(f"PNG: {png(txt, chrome).relative_to(ROOT)}")


# --------------------------------------------------------------------------- main


def executar(chave: str) -> bool:
    numero, titulo, secao, funcao = SECOES[chave]
    fig = Figura(numero, chave, titulo, secao)
    comando = f".venv/bin/python scripts/evidencias_relatorio.py {chave}"
    ambiente = linha_ambiente()
    inicio = time.perf_counter()
    print(f"\n>>> [{numero:02d}] {titulo} ...", flush=True)
    try:
        funcao(fig)
    except Exception as exc:  # registra a falha na própria figura
        fig()
        fig(f"{ERRO} FALHA AO EXECUTAR: {type(exc).__name__}: {exc}")
        fig.verificacoes.append(False)
    texto = fig.texto(comando, ambiente)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{numero:02d}_{chave}.txt").write_text(texto, encoding="utf-8")
    dados = OUT / "dados"
    dados.mkdir(exist_ok=True)
    (dados / f"{numero:02d}_{chave}.json").write_text(
        json.dumps({"titulo": titulo, "verificacoes": fig.verificacoes, "dados": fig.dados,
                    "duracao_s": round(time.perf_counter() - inicio, 1)},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(texto, flush=True)
    return all(fig.verificacoes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("secoes", nargs="*", help="seções a executar (padrão: todas)")
    parser.add_argument("--listar", action="store_true", help="lista as seções e sai")
    parser.add_argument("--sem-png", action="store_true", help="não gera os PNG")
    parser.add_argument("--so-png", action="store_true", help="só gera PNG dos .txt existentes")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "versoes" / "comum" / "ferramentas"))
    if args.listar:
        for chave, (numero, titulo, secao, _) in SECOES.items():
            print(f"{numero:02d}  {chave:<16} {titulo}  [{secao}]")
        return 0
    if args.so_png:
        gerar_pngs()
        return 0
    desconhecidas = set(args.secoes) - set(SECOES)
    if desconhecidas:
        parser.error(f"seções desconhecidas: {', '.join(sorted(desconhecidas))}")
    escolhidas = args.secoes or list(SECOES)
    resultado = {chave: executar(chave) for chave in escolhidas}
    if not args.sem_png:
        gerar_pngs()
    print("\nResumo:", ", ".join(f"{k} {marca(v)}" for k, v in resultado.items()))
    return 0 if all(resultado.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
