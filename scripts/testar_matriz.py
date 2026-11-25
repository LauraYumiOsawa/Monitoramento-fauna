#!/usr/bin/env python3
"""Roda a matriz de versões em ARM64 emulado (QEMU) e com limites de CPU/RAM (Docker).

Etapa "qemu":   para cada uma das 25 versões em versoes/, inicia a CLI correspondente
                (animal-monitor ou animal-monitor-b) com o comando `info` dentro da imagem
                ARM64, emulando o tipo de núcleo da placa quando ela é ARM64.
Etapa "docker": para cada versão com pipeline ONNX/CPU, roda `benchmark` em uma imagem
                real do conjunto de teste, com --cpus e --memory do perfil da placa.

A emulação usa o binário estático qemu-aarch64 montado como entrypoint do contêiner, sem
contêiner privilegiado e sem registrar binfmt no host. Os tempos medidos incluem o custo da
emulação: não são FPS nem latência das placas físicas.

Resultados:
    outputs/testes/qemu-matriz/{resultado.json, casos.csv, <id>.json, <id>.log}
    outputs/testes/docker-recursos/{resultado.json, casos.csv, <id>.json, <id>.log}

Exemplos:
    python3 scripts/testar_matriz.py                      # as duas etapas
    python3 scripts/testar_matriz.py --etapas qemu        # só QEMU (~1 min)
    python3 scripts/testar_matriz.py --casos 01a,05b      # só algumas versões
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSOES = ROOT / "versoes"
OUTPUTS = ROOT / "outputs" / "testes"
DEFAULT_IMAGE = "animal-monitor:arm64"
DEFAULT_QEMU = ROOT.parents[1] / "avaliacao_embarcada" / "qemu-aarch64"
TEST_IMAGE = ROOT / "data" / "animals" / "images" / "test" / "anta00000005.jpg"
APP = "/opt/animal-monitor"


@dataclass(frozen=True)
class Placa:
    nome: str
    qemu_cpu: str
    cpus: str | None = None
    memoria: str | None = None
    nota: str = ""


# Perfis por placa. CPU/RAM repetem os limites da rodada de 04/11/2026; o tipo de núcleo
# segue a avaliação de 11/11/2026. "max" = CPU ARM64 genérica do QEMU.
PLACAS: dict[str, Placa] = {
    "01": Placa("Raspberry Pi 5", "cortex-a76", "4", "4g"),
    "02": Placa("Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L)", "cortex-a76", "2", "2g"),
    "03": Placa("Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8)", "cortex-a76", "4", "4g"),
    "04": Placa("Raspberry Pi 5 + AI Camera (IMX500)", "cortex-a76", "2", "2g"),
    "05": Placa("Raspberry Pi Zero 2 W", "cortex-a53", "1", "512m"),
    "06": Placa(
        "Sipeed MaixCAM", "max", "1", "512m", "Linux da placa roda em RISC-V (C906), não em ARM64"
    ),
    "07": Placa("Seeed Grove Vision AI V2", "max", "1", "256m", "Cortex-M55 sem Linux"),
    "08": Placa("Orange Pi 5", "cortex-a76", "4", "4g"),
    "09": Placa("Espressif ESP32-S3-EYE", "max", nota="ESP32-S3 (Xtensa) sem Linux"),
    "10": Placa("Luckfox Pico (RV1106)", "max", "1", "256m", "RV1106 usa Cortex-A7 de 32 bits"),
    "11": Placa("Arduino Nicla Vision", "max", nota="Cortex-M7 sem Linux"),
    "12": Placa("Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)", "cortex-a76", "4", "4g"),
    "13": Placa("NVIDIA Jetson Orin Nano Super", "cortex-a78ae", "4", "8g"),
}

TOKENS = {
    "yolov8n": "YOLOv8n",
    "onnxruntime": "ONNX Runtime",
    "hailort": "HailoRT",
    "hailort5": "HailoRT 5",
    "imx500": "IMX500",
    "maixpy": "MaixPy",
    "sscma": "SSCMA",
    "rknn": "RKNN",
    "rv1106": "RV1106",
    "tensorrt": "TensorRT",
    "tinyml": "TinyML",
    "esp": "ESP",
    "dl": "DL",
    "openmv": "OpenMV",
    "gpu": "GPU",
    "int8": "INT8",
    "megadetector": "MegaDetector",
    "speciesnet": "SpeciesNet",
    "modelo": "Modelo",
    "b": "B",
}


@dataclass(frozen=True)
class Versao:
    id: str
    placa_id: str
    pasta: str

    @property
    def placa(self) -> Placa:
        return PLACAS[self.placa_id]

    @property
    def slug(self) -> str:
        return self.pasta.split("_", 1)[1] if "_" in self.pasta else self.pasta

    @property
    def nome(self) -> str:
        partes = []
        for parte in self.slug.split("_"):
            palavras = [TOKENS.get(p, p) for p in parte.split("-")]
            texto = " ".join(palavras)
            texto = texto.replace("ESP DL", "ESP-DL").replace(
                "MegaDetector SpeciesNet", "MegaDetector + SpeciesNet"
            )
            texto = texto.replace("RKNN RV1106", "RKNN (RV1106)")
            partes.append(texto)
        return " · ".join(partes)

    @property
    def modelo_b(self) -> bool:
        return "modelo-b" in self.slug

    @property
    def cli(self) -> str:
        return "animal_monitor_b" if self.modelo_b else "animal_monitor"

    @property
    def motivo_sem_benchmark(self) -> str | None:
        if "megadetector" in self.slug:
            return "MegaDetector + SpeciesNet: pesos não fazem parte da imagem"
        if self.placa.cpus is None:
            return f"{self.placa.nota}: sem ONNX Runtime"
        return None


def descobrir_versoes() -> list[Versao]:
    versoes = []
    for pasta in sorted(VERSOES.glob("[0-9][0-9]_*/[0-9][0-9][a-z]_*")):
        if not pasta.is_dir():
            continue
        versao_id = pasta.name.split("_", 1)[0]
        placa_id = versao_id[:2]
        if placa_id in PLACAS:
            versoes.append(Versao(versao_id, placa_id, pasta.name))
    return versoes


# ---------------------------------------------------------------------------
# Execução de contêineres
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _fmt_s(segundos: float) -> str:
    return f"{segundos:.1f} s".replace(".", ",")


@dataclass
class Execucao:
    exit_code: int | None
    oom: bool
    timeout: bool
    duracao_s: float
    saida: str


def rodar_conteiner(
    nome: str,
    imagem: str,
    qemu: Path,
    qemu_cpu: str,
    modulo: str,
    args: list[str],
    *,
    montagens: list[tuple[Path, str, str]] | None = None,
    cpus: str | None = None,
    memoria: str | None = None,
    timeout_s: float | None = None,
) -> Execucao:
    cmd = ["docker", "run", "--name", nome, "--platform", "linux/arm64", "--network", "none"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    if cpus:
        cmd += ["--cpus", cpus]
    if memoria:
        cmd += ["--memory", memoria, "--memory-swap", memoria]
    cmd += ["-e", f"PYTHONPATH={APP}/src", "-v", f"{ROOT / 'src'}:{APP}/src:ro"]
    cmd += ["-v", f"{qemu}:/qemu-aarch64:ro"]
    for origem, destino, modo in montagens or []:
        cmd += ["-v", f"{origem}:{destino}:{modo}"]
    # Este qemu-aarch64 espera o argv[0] do programa logo após o caminho do executável.
    cmd += ["--entrypoint", "/qemu-aarch64", imagem, "-cpu", qemu_cpu]
    cmd += ["/usr/local/bin/python", "python", "-m", modulo, *args]

    subprocess.run(["docker", "rm", "-f", nome], capture_output=True)
    inicio = time.perf_counter()
    expirou = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        saida = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        expirou = True
        subprocess.run(["docker", "kill", nome], capture_output=True)
        saida = _texto(exc.stdout) + _texto(exc.stderr)
    duracao = time.perf_counter() - inicio
    estado = subprocess.run(
        ["docker", "inspect", nome, "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}"],
        capture_output=True,
        text=True,
    )
    subprocess.run(["docker", "rm", "-f", nome], capture_output=True)
    exit_code, oom = None, False
    if estado.returncode == 0 and estado.stdout.strip():
        codigo, oom_txt = estado.stdout.split()
        exit_code, oom = int(codigo), oom_txt == "true"
    return Execucao(exit_code, oom, expirou, duracao, saida)


def _texto(valor: bytes | str | None) -> str:
    if valor is None:
        return ""
    return valor.decode(errors="replace") if isinstance(valor, bytes) else valor


def latencia_media(bench: dict | None) -> float | None:
    """Latência média por quadro (ms): A grava latency_ms.mean; B, latency_ms.total.mean."""
    latencia = (bench or {}).get("latency_ms", {})
    if isinstance(latencia.get("total"), dict):
        latencia = latencia["total"]
    valor = latencia.get("mean")
    return float(valor) if valor is not None else None


def _json_da_saida(texto: str) -> dict | None:
    inicio = texto.find("{")
    if inicio < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(texto[inicio:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Etapa QEMU
# ---------------------------------------------------------------------------


def caso_qemu(v: Versao, imagem: str, qemu: Path, pasta: Path, sufixo: str) -> dict:
    for antigo in (pasta / f"{v.id}.json", pasta / f"{v.id}.log"):
        antigo.unlink(missing_ok=True)
    exe = rodar_conteiner(
        f"tcc-qemu-{v.id}-{sufixo}", imagem, qemu, v.placa.qemu_cpu, v.cli, ["info"], timeout_s=300
    )
    (pasta / f"{v.id}.log").write_text(exe.saida, encoding="utf-8")
    info = _json_da_saida(exe.saida)
    if info is not None:
        (pasta / f"{v.id}.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    modulos = (info or {}).get("modules", {})
    arquitetura = str((info or {}).get("platform", ""))
    obrigatorios = ("numpy", "cv2", "onnxruntime")
    faltando = [m for m in obrigatorios if modulos.get(m) is not True]
    ok = exe.exit_code == 0 and "aarch64" in arquitetura and not faltando
    if ok:
        observacao = f"CLI {v.cli.replace('_', '-')} info em aarch64 ({v.placa.qemu_cpu})"
    elif exe.timeout:
        observacao = "timeout"
    elif faltando:
        observacao = f"módulos ausentes: {', '.join(faltando)}"
    else:
        observacao = f"exit {exe.exit_code}"
    pacotes = (info or {}).get("packages", {})
    return {
        "id": v.id,
        "placa": v.placa.nome,
        "versao": v.nome,
        "pasta": f"versoes/{v.placa_id}_*/{v.pasta}",
        "cli": v.cli.replace("_", "-"),
        "qemu_cpu": v.placa.qemu_cpu,
        "nota_cpu": v.placa.nota,
        "resultado": "passed" if ok else "failed",
        "observacao": observacao,
        "duracao_s": round(exe.duracao_s, 2),
        "python": (info or {}).get("python"),
        "onnxruntime": pacotes.get("onnxruntime"),
    }


# ---------------------------------------------------------------------------
# Etapa Docker com limites
# ---------------------------------------------------------------------------


def _modelo(v: Versao) -> tuple[str, list[str], str]:
    """Devolve (rótulo do modelo, argumentos do benchmark, observação)."""
    if v.modelo_b:
        sufixo = "_int8" if "int8" in v.slug else ""
        rotulo = "Modelo B INT8" if sufixo else "Modelo B FP32"
        args = [
            "--detector",
            f"models/modelo_b/presence_detector{sufixo}.onnx",
            "--classifier",
            f"models/modelo_b/species_classifier{sufixo}.onnx",
            "--runtime",
            "config/modelo_b/runtime.yaml",
        ]
        return rotulo, args, ""
    args = [
        "--model",
        "models/animal_detector.onnx",
        "--data",
        "config/dataset.yaml",
        "--runtime",
        "config/runtime.yaml",
    ]
    especifico = re.search(r"int8|reduzido|320|512", v.slug)
    nota = "variante específica não existe no acervo; usado o ONNX genérico" if especifico else ""
    return "Modelo A FP32 640", args, nota


def caso_docker(
    v: Versao, imagem: str, qemu: Path, pasta: Path, sufixo: str, timeout_s: float
) -> dict:
    for antigo in (pasta / f"{v.id}.json", pasta / f"{v.id}.log"):
        antigo.unlink(missing_ok=True)
    rotulo, args, nota = _modelo(v)
    threads = max(1, int(float(v.placa.cpus or "1")))
    runtime_origem = ROOT / (
        "config/modelo_b/runtime.yaml" if v.modelo_b else "config/runtime.yaml"
    )
    runtime_destino = (
        f"{APP}/{'config/modelo_b/runtime.yaml' if v.modelo_b else 'config/runtime.yaml'}"
    )
    with tempfile.TemporaryDirectory(prefix=f"tcc-{v.id}-") as tmp:
        tmp_dir = Path(tmp)
        os.chmod(tmp_dir, 0o777)
        runtime = tmp_dir / "runtime.yaml"
        texto = runtime_origem.read_text(encoding="utf-8")
        texto = re.sub(r"(?m)^intra_op_threads:.*$", f"intra_op_threads: {threads}", texto)
        runtime.write_text(texto, encoding="utf-8")
        saida_dir = tmp_dir / "saida"
        saida_dir.mkdir()
        os.chmod(saida_dir, 0o777)
        montagens = [
            (ROOT / "models", f"{APP}/models", "ro"),
            (ROOT / "config", f"{APP}/config", "ro"),
            (runtime, runtime_destino, "ro"),
            (TEST_IMAGE.parent, "/dados", "ro"),
            (saida_dir, "/saida", "rw"),
        ]
        bench_args = [
            "benchmark",
            *args,
            "--source",
            f"/dados/{TEST_IMAGE.name}",
            "--frames",
            "1",
            "--warmup",
            "0",
            "--output",
            "/saida/benchmark.json",
        ]
        exe = rodar_conteiner(
            f"tcc-docker-{v.id}-{sufixo}",
            imagem,
            qemu,
            v.placa.qemu_cpu,
            v.cli,
            bench_args,
            montagens=montagens,
            cpus=v.placa.cpus,
            memoria=v.placa.memoria,
            timeout_s=timeout_s,
        )
        bench_file = saida_dir / "benchmark.json"
        bench = json.loads(bench_file.read_text(encoding="utf-8")) if bench_file.exists() else None
    (pasta / f"{v.id}.log").write_text(exe.saida, encoding="utf-8")
    if bench is not None:
        (pasta / f"{v.id}.json").write_text(
            json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if exe.timeout:
        motivo, observacao = "timeout", f"não terminou em {timeout_s:.0f} s"
    elif exe.oom:
        motivo, observacao = "oom", f"encerrado por falta de memória (limite {v.placa.memoria})"
    elif exe.exit_code == 137:
        motivo = "oom"
        observacao = f"SIGKILL (exit 137), provável falta de memória (limite {v.placa.memoria})"
    elif exe.exit_code == 0 and bench is not None:
        motivo, observacao = "ok", "benchmark concluído"
    else:
        motivo, observacao = "erro", f"exit {exe.exit_code}"
    latencia = latencia_media(bench)
    return {
        "id": v.id,
        "placa": v.placa.nome,
        "versao": v.nome,
        "modelo": "modelo-b" if v.modelo_b else "modelo-a",
        "modelo_usado": rotulo,
        "nota_modelo": nota,
        "cpus": v.placa.cpus,
        "memoria": v.placa.memoria,
        "qemu_cpu": v.placa.qemu_cpu,
        "nota_cpu": v.placa.nota,
        "resultado": "passed" if motivo == "ok" else "failed",
        "motivo": motivo,
        "observacao": observacao,
        "exit_code": exe.exit_code,
        "oom_killed": exe.oom,
        "duracao_s": round(exe.duracao_s, 2),
        "latencia_ms": round(latencia, 1) if latencia is not None else None,
        "fps": (bench or {}).get("throughput_fps"),
        "pico_rss_mb": (bench or {}).get("peak_process_rss_mb"),
        "deteccoes": (bench or {}).get("detections", (bench or {}).get("raw_detections")),
    }


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------


def salvar(pasta: Path, resumo: dict, campos: tuple[str, ...]) -> None:
    (pasta / "resultado.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (pasta / "casos.csv").open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resumo["cases"])


def executar(
    etapa: str,
    versoes: list[Versao],
    funcao,
    paralelo: int,
    descricao,
) -> tuple[list[dict], float]:
    inicio = time.perf_counter()
    resultados: dict[str, dict] = {}
    total = len(versoes)

    def tarefa(v: Versao) -> None:
        caso = funcao(v)
        resultados[v.id] = caso
        ok = caso["resultado"] == "passed"
        marca = "✓" if ok else "✗"
        log(
            f"  {marca} [{len(resultados):2d}/{total}] {v.id}  {v.placa.nome} · {v.nome}"
            f"  — {descricao(caso)}  ({_fmt_s(caso['duracao_s'])})"
        )

    with ThreadPoolExecutor(max_workers=max(1, paralelo)) as pool:
        list(pool.map(tarefa, versoes))
    return [resultados[v.id] for v in versoes], time.perf_counter() - inicio


def checar_ambiente(imagem: str, qemu: Path) -> None:
    if shutil.which("docker") is None:
        sys.exit("Erro: Docker não encontrado.")
    if subprocess.run(["docker", "image", "inspect", imagem], capture_output=True).returncode:
        sys.exit(
            f"Erro: imagem {imagem} não encontrada. Construa com scripts/qemu_arm64_smoke.sh "
            "(exige binfmt/QEMU registrado no host)."
        )
    if not qemu.is_file():
        sys.exit(
            f"Erro: qemu-aarch64 não encontrado em {qemu}.\n"
            "Extraia da imagem tonistiigi/binfmt (sem privilégios):\n"
            "  docker create --name binfmt-tmp tonistiigi/binfmt\n"
            "  docker cp binfmt-tmp:/usr/bin/qemu-aarch64 ./qemu-aarch64\n"
            "  docker rm binfmt-tmp\n"
            "e passe --qemu ./qemu-aarch64"
        )


def _qemu_versao(qemu: Path) -> str:
    proc = subprocess.run([str(qemu), "--version"], capture_output=True, text=True)
    return proc.stdout.splitlines()[0] if proc.stdout else "desconhecida"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--etapas", default="qemu,docker", help="qemu, docker ou qemu,docker")
    parser.add_argument("--casos", default="", help="IDs separados por vírgula (ex.: 01a,05b)")
    parser.add_argument("--imagem", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--qemu",
        type=Path,
        default=Path(os.environ.get("ANIMAL_MONITOR_QEMU_BIN", DEFAULT_QEMU)),
        help="binário estático qemu-aarch64",
    )
    parser.add_argument("--timeout", type=float, default=300, help="limite por benchmark (s)")
    parser.add_argument("--paralelo-qemu", type=int, default=4)
    parser.add_argument("--paralelo-docker", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    etapas = {e.strip() for e in args.etapas.split(",") if e.strip()}
    if not etapas <= {"qemu", "docker"}:
        parser.error("--etapas aceita apenas qemu e docker")
    qemu = args.qemu.resolve()
    checar_ambiente(args.imagem, qemu)
    if "docker" in etapas and not TEST_IMAGE.is_file():
        sys.exit(f"Erro: imagem de teste não encontrada: {TEST_IMAGE}")

    versoes = descobrir_versoes()
    if args.casos:
        pedidos = {c.strip() for c in args.casos.split(",") if c.strip()}
        desconhecidos = pedidos - {v.id for v in versoes}
        if desconhecidos:
            parser.error(f"IDs desconhecidos: {', '.join(sorted(desconhecidos))}")
        versoes = [v for v in versoes if v.id in pedidos]

    sufixo = str(os.getpid())
    comum = {
        "host": platform.platform(),
        "imagem": args.imagem,
        "qemu": _qemu_versao(qemu),
        "modo_emulacao": "qemu-aarch64 estático como entrypoint (sem binfmt no host)",
    }
    codigo = 0

    if "qemu" in etapas:
        pasta = args.output_dir / "qemu-matriz"
        pasta.mkdir(parents=True, exist_ok=True)
        log(f"\n=== QEMU ARM64: {len(versoes)} versões ({args.paralelo_qemu} em paralelo) ===")
        inicio = datetime.now(timezone.utc).isoformat()
        casos, duracao = executar(
            "qemu",
            versoes,
            lambda v: caso_qemu(v, args.imagem, qemu, pasta, sufixo),
            args.paralelo_qemu,
            lambda c: c["observacao"],
        )
        passed = sum(c["resultado"] == "passed" for c in casos)
        salvar(
            pasta,
            {
                "tipo": "qemu-matriz",
                "started_at": inicio,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(duracao, 1),
                **comum,
                "total": len(casos),
                "passed": passed,
                "failed": len(casos) - passed,
                "cases": casos,
            },
            ("id", "placa", "versao", "cli", "qemu_cpu", "resultado", "observacao", "duracao_s"),
        )
        log(f"  → {passed}/{len(casos)} passaram em {_fmt_s(duracao)}. Resultado: {pasta}/")
        codigo |= int(passed != len(casos))

    if "docker" in etapas:
        pasta = args.output_dir / "docker-recursos"
        pasta.mkdir(parents=True, exist_ok=True)
        executaveis = [v for v in versoes if v.motivo_sem_benchmark is None]
        ignoradas = [
            {"id": v.id, "placa": v.placa.nome, "versao": v.nome, "motivo": v.motivo_sem_benchmark}
            for v in versoes
            if v.motivo_sem_benchmark is not None
        ]
        log(
            f"\n=== Docker com limites de CPU/RAM: {len(executaveis)} pipelines "
            f"(timeout {args.timeout:.0f} s; alguns minutos) ==="
        )
        inicio = datetime.now(timezone.utc).isoformat()

        def descreve(c: dict) -> str:
            limite = f"{c['cpus']} CPU, {c['memoria']}"
            if c["motivo"] == "ok":
                rss = f"{c['pico_rss_mb']:.0f} MB" if c["pico_rss_mb"] else "?"
                return f"{c['modelo_usado']} [{limite}] RAM pico {rss}"
            return f"{c['modelo_usado']} [{limite}] {c['observacao']}"

        casos, duracao = executar(
            "docker",
            executaveis,
            lambda v: caso_docker(v, args.imagem, qemu, pasta, sufixo, args.timeout),
            args.paralelo_docker,
            descreve,
        )
        passed = sum(c["resultado"] == "passed" for c in casos)
        salvar(
            pasta,
            {
                "tipo": "docker-recursos",
                "started_at": inicio,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(duracao, 1),
                **comum,
                "timeout_s": args.timeout,
                "imagem_teste": str(TEST_IMAGE.relative_to(ROOT)),
                "quadros_medidos": 1,
                "total": len(casos),
                "passed": passed,
                "failed": len(casos) - passed,
                "nao_aplicavel": ignoradas,
                "cases": casos,
            },
            (
                "id",
                "placa",
                "modelo",
                "modelo_usado",
                "cpus",
                "memoria",
                "qemu_cpu",
                "resultado",
                "motivo",
                "observacao",
                "latencia_ms",
                "pico_rss_mb",
                "duracao_s",
            ),
        )
        log(f"  → {passed}/{len(casos)} concluíram em {_fmt_s(duracao)}. Resultado: {pasta}/")
        if ignoradas:
            log(f"  Sem benchmark: {', '.join(i['id'] for i in ignoradas)}")
        codigo |= int(passed != len(casos))

    log("\nPara o relatório visual: python3 scripts/relatorio_testes_html.py --open")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
