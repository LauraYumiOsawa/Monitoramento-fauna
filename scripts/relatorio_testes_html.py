#!/usr/bin/env python3
# ruff: noqa: E501  (CSS e HTML embutidos)
"""Gera um relatório HTML, pronto para captura de tela, a partir dos resultados dos testes.

Lê outputs/testes/resultado.json (testes unitários, de scripts/testar_tudo.py) e, quando
existirem, outputs/testes/qemu-matriz/resultado.json e outputs/testes/docker-recursos/
resultado.json (de scripts/testar_matriz.py).

Uso:
    python3 scripts/relatorio_testes_html.py            # gera outputs/testes/relatorio.html
    python3 scripts/relatorio_testes_html.py --open     # e abre no navegador
    python3 scripts/relatorio_testes_html.py --input outputs/testes/resultado_arm64_2026-11-04.json \\
        --qemu-dir outputs/testes/qemu-matriz_2026-11-04 \\
        --docker-dir outputs/testes/docker-recursos_2026-11-04

Depende apenas da biblioteca padrão.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testar_matriz import Versao, descobrir_versoes, latencia_media  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMON_TESTS = ROOT / "versoes" / "comum" / "tests"

GROUPS = {
    "nucleo": ("Núcleo (Modelo A)", "tests/"),
    "modelo_b": ("Modelo B (dois estágios)", "tests/modelo_b/"),
    "hardware": ("Versões embarcadas (camada de hardware)", "versoes/comum/tests/"),
}

MODULE_LABELS = {
    "test_config": "Configuração YAML",
    "test_dataset": "Validação do dataset",
    "test_events": "Eventos e confirmação de avistamentos",
    "test_prepost": "Pré e pós-processamento (letterbox, NMS)",
    "test_splitting": "Divisão treino / validação / teste",
    "modelo_b.test_config_and_events": "Configuração, aumento de dados e avistamentos",
    "modelo_b.test_data_prep": "Preparação de recortes e rótulos",
    "modelo_b.test_metrics": "Métricas (matriz de confusão, AP, associação)",
    "modelo_b.test_pipeline": "Pipeline detector + classificador e votação",
    "modelo_b.test_tracker": "Rastreador ByteTrack-lite",
    "test_analise": "Análise dos modelos A e B",
    "test_backends_falsos": "Backends Hailo, RKNN e IMX500 (SDK simulado)",
    "test_camtrap_dp": "Exportação Camtrap DP",
    "test_dispositivos": "Importação CSV e app MaixPy",
    "test_energia": "Autonomia e energia",
    "test_models": "Tensores, detector YOLO e classificador",
    "test_yolo_heads": "Decodificação das saídas YOLO de aceleradores",
}

STATUS = {
    "passed": ("✓", "Passou", "good"),
    "failed": ("✗", "Falhou", "critical"),
    "error": ("!", "Erro", "serious"),
    "skipped": ("–", "Ignorado", "muted"),
}

DOCKER_STATUS = {
    "ok": ("✓", "Concluiu", "good"),
    "timeout": ("!", "Tempo esgotado", "serious"),
    "oom": ("✗", "Sem memória", "critical"),
    "erro": ("✗", "Erro", "critical"),
}

CSS = """
:root {
  --surface: #fcfcfb; --page: #f4f3ef; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --good-text: #006300; --good-wash: #e7f5e7;
  --critical: #d03b3b; --critical-wash: #fbeaea;
  --serious: #ec835a; --serious-text: #a2451f; --serious-wash: #fdf0ea;
  --track: #e9e8e2;
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Ubuntu, Roboto, sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 48px; }
.eyebrow { color: var(--ink-2); font-size: 13px; letter-spacing: .04em; text-transform: uppercase; margin: 0 0 6px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 8px; font-weight: 650; }
h2 { font-size: 20px; margin: 44px 0 6px; font-weight: 650; }
h3 { font-size: 16px; margin: 28px 0 12px; font-weight: 600; }
.lead { color: var(--ink-2); margin: 0 0 14px; max-width: 860px; }
.meta { color: var(--ink-2); margin: 0; display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 14px; }
.meta b { color: var(--ink); font-weight: 600; }
.section-meta { margin-bottom: 14px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
.hero { margin-top: 24px; padding: 24px 28px; display: grid; grid-template-columns: auto 1fr; gap: 8px 32px; align-items: center; }
.hero-figure { font-size: 64px; line-height: 1; font-weight: 650; letter-spacing: -.02em; }
.hero-figure small { font-size: 28px; color: var(--ink-2); font-weight: 500; }
.hero-label { color: var(--ink-2); margin-top: 6px; }
.verdict { display: inline-flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 600; padding: 10px 16px; border-radius: 10px; }
.verdict.ok { background: var(--good-wash); color: var(--good-text); }
.verdict.fail { background: var(--critical-wash); color: var(--critical); }
.verdict.warn { background: var(--serious-wash); color: var(--serious-text); }
.verdict .icon { display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; color: #fff; font-size: 18px; }
.verdict.ok .icon { background: var(--good); }
.verdict.fail .icon { background: var(--critical); }
.verdict.warn .icon { background: var(--serious); }
.verdict.small { font-size: 14px; padding: 5px 10px; gap: 8px; border-radius: 8px; }
.verdict.small .icon { width: 20px; height: 20px; font-size: 12px; }
.hero-side p { margin: 10px 0 0; color: var(--ink-2); }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 24px; }
.summary .card { padding: 20px 22px; display: flex; flex-direction: column; gap: 8px; }
.summary .label { font-weight: 600; }
.summary .figure { font-size: 44px; line-height: 1.05; font-weight: 650; letter-spacing: -.02em; }
.summary .figure small { font-size: 20px; color: var(--ink-2); font-weight: 500; letter-spacing: 0; }
.summary p { margin: 0; color: var(--ink-2); font-size: 14px; }
.summary .when { color: var(--muted); font-size: 12px; margin-top: auto; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; margin-top: 12px; }
.tile { padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .value .unit { font-size: 15px; color: var(--ink-2); font-weight: 500; margin-left: 2px; }
.groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.group { padding: 18px 20px; break-inside: avoid; }
.group-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.group-title { font-weight: 600; font-size: 16px; }
.group-path { color: var(--muted); font-size: 12px; font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }
.group-count { font-size: 22px; font-weight: 600; white-space: nowrap; }
.group-count small { font-size: 14px; color: var(--ink-2); font-weight: 500; }
.bar { display: flex; gap: 2px; height: 8px; margin: 12px 0 14px; }
.bar span { border-radius: 4px; }
.bar .good { background: var(--good); } .bar .critical { background: var(--critical); }
.bar .serious { background: var(--serious); } .bar .muted { background: var(--track); }
.modules { list-style: none; margin: 0; padding: 0; }
.modules li { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-top: 1px solid var(--grid); font-size: 14px; }
.modules li span:last-child { color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.modules li.has-problem span:last-child { color: var(--critical); font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
thead th { text-align: left; color: var(--ink-2); font-weight: 500; font-size: 13px; padding: 10px 14px; border-bottom: 1px solid var(--grid); }
tbody td { padding: 8px 14px; border-top: 1px solid var(--grid); vertical-align: top; }
tr.module-row td { background: var(--page); font-weight: 600; padding-top: 10px; color: var(--ink); }
tr.module-row td .group-path { font-weight: 400; margin-left: 8px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.case-class, .sub { color: var(--muted); font-size: 12px; }
.id { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-weight: 600; font-size: 13px; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; font-size: 13px; white-space: nowrap; }
.badge .dot { display: inline-grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; color: #fff; font-size: 11px; flex: none; }
.badge.good { color: var(--good-text); } .badge.good .dot { background: var(--good); }
.badge.critical { color: var(--critical); } .badge.critical .dot { background: var(--critical); }
.badge.serious { color: var(--serious-text); } .badge.serious .dot { background: var(--serious); }
.badge.muted { color: var(--ink-2); } .badge.muted .dot { background: var(--muted); }
.versions { display: flex; flex-direction: column; gap: 4px; }
.version { display: flex; align-items: baseline; gap: 8px; }
.version .badge { font-weight: 400; color: var(--ink); gap: 8px; white-space: normal; }
.version .badge.critical { color: var(--critical); }
.cpu { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 13px; white-space: nowrap; }
.details { margin-top: 4px; color: var(--critical); font-family: ui-monospace, Menlo, monospace; font-size: 12px; white-space: pre-wrap; }
.details.muted { color: var(--ink-2); } .details.serious { color: var(--serious-text); }
.notes { color: var(--ink-2); font-size: 13px; margin: 10px 2px 0; padding-left: 18px; }
.notes li { margin: 3px 0; }
.warn-note { color: var(--serious-text); }
.skip { margin-top: 12px; padding: 12px 16px; font-size: 14px; color: var(--ink-2); }
.skip b { color: var(--ink); }
footer { margin-top: 28px; color: var(--ink-2); font-size: 13px; }
footer code { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 2px 6px; font-size: 12px; }
@media (max-width: 640px) {
  .hero { grid-template-columns: 1fr; }
  .hero-figure { font-size: 48px; }
  .case-col-time, .col-hide-sm { display: none; }
  table { font-size: 13px; }
  tbody td, thead th { padding: 7px 8px; }
}
@media print {
  @page { size: A4; margin: 12mm; }
  body { background: #fff; }
  main { padding: 0; max-width: none; }
  .card { break-inside: avoid; }
  tr { break-inside: avoid; }
  h2 { break-before: page; }
  .summary-section h2, h2.first { break-before: auto; }
}
"""


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _fmt_decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _fmt_duration(ms: float) -> tuple[str, str]:
    if ms >= 60_000:
        return _fmt_decimal(ms / 60_000, 1), "min"
    if ms >= 1000:
        return _fmt_decimal(ms / 1000, 2 if ms < 10_000 else 1), "s"
    return _fmt_decimal(ms, 0 if ms >= 100 else 1), "ms"


def _fmt_case_ms(ms: float) -> str:
    return _fmt_decimal(ms, 1 if ms < 100 else 0)


def _fmt_mem(value: str | None) -> str:
    if not value:
        return "—"
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg]?)", str(value).lower())
    if not match:
        return str(value)
    number, unit = match.groups()
    return f"{number.replace('.', ',')} {({'k': 'KB', 'm': 'MB', 'g': 'GB'}).get(unit, 'B')}"


def _fmt_cpus(value: str | None) -> str:
    if not value:
        return "—"
    number = float(value)
    text = _fmt_decimal(number, 0 if number.is_integer() else 1)
    return f"{text} CPU" if number == 1 else f"{text} CPUs"


def _arch(platform: str) -> str:
    lowered = platform.lower()
    if "aarch64" in lowered or "arm64" in lowered:
        return "ARM64 (aarch64)"
    if "x86_64" in lowered or "amd64" in lowered:
        return "x86_64"
    return platform


def _when(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso


def _humanize(method: str) -> str:
    text = method.removeprefix("test_").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else method


def _split_name(name: str) -> tuple[str, str, str]:
    parts = name.split(".")
    if len(parts) < 3:
        return name, "", name
    return ".".join(parts[:-2]), parts[-2], parts[-1]


def _group_of(module: str) -> str:
    if module.startswith("modelo_b."):
        return "modelo_b"
    if (COMMON_TESTS / f"{module}.py").exists():
        return "hardware"
    return "nucleo"


def _badge(icon: str, label: str, tone: str) -> str:
    return f'<span class="badge {tone}"><span class="dot">{icon}</span>{html.escape(label)}</span>'


def _status_badge(status: str) -> str:
    icon, label, tone = STATUS.get(status, ("?", status, "muted"))
    return _badge(icon, label, tone)


def _bar(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return '<div class="bar"><span class="muted" style="flex:1"></span></div>'
    spans = []
    for status in ("passed", "failed", "error", "skipped"):
        if counts.get(status):
            tone = STATUS[status][2]
            spans.append(f'<span class="{tone}" style="flex:{counts[status]}"></span>')
    label = ", ".join(f"{counts[s]} {STATUS[s][1].lower()}" for s in STATUS if counts.get(s))
    return f'<div class="bar" role="img" aria-label="{html.escape(label)}">{"".join(spans)}</div>'


def _tiles(items: list[tuple[str, str, str]]) -> str:
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="card tile"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}<span class="unit">{unit}</span></div></div>'
            for label, value, unit in items
        )
        + "</div>"
    )


def _verdict(ok: bool, text: str, *, small: bool = False, warn: bool = False) -> str:
    tone = "ok" if ok else ("warn" if warn else "fail")
    icon = "✓" if ok else ("!" if warn else "✗")
    size = " small" if small else ""
    return f'<div class="verdict {tone}{size}"><span class="icon">{icon}</span>{html.escape(text)}</div>'


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Leitura da matriz (formato novo de testar_matriz.py e formato manual de 04/11/2026)
# ---------------------------------------------------------------------------


def _versoes_por_id() -> dict[str, Versao]:
    return {v.id: v for v in descobrir_versoes()}


def _quando_matriz(data: dict, arquivo: Path) -> str:
    if data.get("started_at"):
        return f"Executado em <b>{_when(data['started_at'])}</b>"
    gravado = datetime.fromtimestamp(arquivo.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
    return f"Resultado gravado em <b>{gravado}</b>"


def load_qemu(pasta: Path | None) -> dict | None:
    if pasta is None or not (pasta / "resultado.json").is_file():
        return None
    arquivo = pasta / "resultado.json"
    data = json.loads(arquivo.read_text(encoding="utf-8"))
    versoes = _versoes_por_id()
    for caso in data.get("cases", []):
        v = versoes.get(caso["id"])
        if v is not None:
            caso.setdefault("placa", v.placa.nome)
            caso.setdefault("versao", v.nome)
            caso.setdefault("nota_cpu", v.placa.nota)
        caso.setdefault("qemu_cpu", None)
    data["_quando"] = _quando_matriz(data, arquivo)
    data["_arquivo"] = arquivo
    return data


def load_docker(pasta: Path | None) -> dict | None:
    if pasta is None or not (pasta / "resultado.json").is_file():
        return None
    arquivo = pasta / "resultado.json"
    data = json.loads(arquivo.read_text(encoding="utf-8"))
    versoes = _versoes_por_id()
    for caso in data.get("cases", []):
        v = versoes.get(caso["id"])
        if v is not None:
            caso.setdefault("placa", v.placa.nome)
            caso.setdefault("versao", v.nome)
            caso.setdefault("nota_cpu", v.placa.nota)
        if "motivo" not in caso:  # formato de 04/11: só resultado + observação
            obs = str(caso.get("observacao", ""))
            if caso.get("resultado") == "passed":
                caso["motivo"] = "ok"
            elif "exit_124" in obs:
                caso["motivo"] = "timeout"
            elif "exit_137" in obs:
                caso["motivo"] = "oom"
            else:
                caso["motivo"] = "erro"
        if "modelo_usado" not in caso:
            caso["modelo_usado"] = {"modelo-a": "Modelo A", "modelo-b": "Modelo B"}.get(
                caso.get("modelo", ""), caso.get("modelo", "")
            )
        if caso.get("resultado") == "passed" and caso.get("latencia_ms") is None:
            bench_file = pasta / f"{caso['id']}.json"
            if bench_file.is_file():
                bench = json.loads(bench_file.read_text(encoding="utf-8"))
                caso["latencia_ms"] = latencia_media(bench)
                caso["pico_rss_mb"] = bench.get("peak_process_rss_mb")
    if "nao_aplicavel" not in data:
        rodados = {c["id"] for c in data.get("cases", [])}
        data["nao_aplicavel"] = [
            {"id": v.id, "placa": v.placa.nome, "versao": v.nome, "motivo": v.motivo_sem_benchmark}
            for v in versoes.values()
            if v.id not in rodados and v.motivo_sem_benchmark
        ]
    data["_quando"] = _quando_matriz(data, arquivo)
    data["_arquivo"] = arquivo
    return data


# ---------------------------------------------------------------------------
# Seções
# ---------------------------------------------------------------------------


def unit_section(data: dict) -> dict[str, str]:
    tests = data.get("tests", {})
    total = int(tests.get("total", 0))
    passed = int(tests.get("passed", 0))
    failed = int(tests.get("failed", 0))
    errors = int(tests.get("errors", 0))
    skipped = int(tests.get("skipped", 0))
    compile_ok = bool(data.get("compile_ok", False))
    all_ok = data.get("exit_code", 1) == 0 and failed == 0 and errors == 0 and compile_ok
    platform = str(data.get("platform", ""))

    grouped: dict[str, dict[str, list[dict]]] = {key: {} for key in GROUPS}
    for case in data.get("cases", []):
        module, _, _ = _split_name(case["name"])
        grouped[_group_of(module)].setdefault(module, []).append(case)

    group_cards = []
    for key, (title, path) in GROUPS.items():
        modules = grouped[key]
        cases = [c for cs in modules.values() for c in cs]
        if not cases:
            continue
        counts = {s: sum(c["status"] == s for c in cases) for s in STATUS}
        rows = []
        for module, mod_cases in modules.items():
            ok = sum(c["status"] == "passed" for c in mod_cases)
            problem = any(c["status"] in ("failed", "error") for c in mod_cases)
            label = html.escape(MODULE_LABELS.get(module, module))
            rows.append(
                f'<li class="{"has-problem" if problem else ""}"><span>{label}</span>'
                f"<span>{ok}/{len(mod_cases)}</span></li>"
            )
        group_cards.append(
            f"""<section class="card group">
  <div class="group-head">
    <div><div class="group-title">{html.escape(title)}</div><div class="group-path">{path}</div></div>
    <div class="group-count">{counts["passed"]}<small>/{len(cases)}</small></div>
  </div>
  {_bar(counts, len(cases))}
  <ul class="modules">{"".join(rows)}</ul>
</section>"""
        )

    table_rows = []
    for key, (_title, path) in GROUPS.items():
        for module, mod_cases in grouped[key].items():
            label = html.escape(MODULE_LABELS.get(module, module))
            table_rows.append(
                f'<tr class="module-row"><td colspan="3">{label}'
                f'<span class="group-path">{html.escape(path + module.split(".")[-1])}.py</span></td></tr>'
            )
            for case in mod_cases:
                _, cls, method = _split_name(case["name"])
                details = ""
                if case.get("details") and case["status"] != "passed":
                    tone = STATUS.get(case["status"], ("", "", "muted"))[2]
                    details = f'<div class="details {tone}">{html.escape(case["details"])}</div>'
                table_rows.append(
                    f"<tr><td>{html.escape(_humanize(method))}"
                    f'<div class="case-class">{html.escape(cls)}</div>{details}</td>'
                    f"<td>{_status_badge(case['status'])}</td>"
                    f'<td class="num case-col-time">{_fmt_case_ms(float(case.get("duration_ms", 0)))}</td></tr>'
                )

    dur_value, dur_unit = _fmt_duration(float(data.get("duration_ms", 0)))
    tiles = [
        ("Aprovados", _fmt_int(passed), ""),
        ("Falhas", _fmt_int(failed), ""),
        ("Erros", _fmt_int(errors), ""),
        ("Ignorados", _fmt_int(skipped), ""),
        ("Tempo total", dur_value, dur_unit),
        ("Compilação", "OK" if compile_ok else "Falhou", ""),
    ]
    # Em Linux, testar_tudo.py grava ru_maxrss (KiB) / 1024, ou seja, MiB, apesar do nome "_kb".
    memory = data.get("memory_peak_kb")
    if memory and platform.startswith("Linux"):
        tiles.append(("Pico de memória", _fmt_int(int(memory)), "MiB"))

    problems = failed + errors
    verdict_text = (
        "Todos os testes passaram"
        if all_ok
        else (f"{problems} teste(s) com problema" if problems else "Falha na compilação")
    )
    meta = (
        f'<p class="meta section-meta"><span>Executado em <b>{_when(str(data.get("started_at", "")))}</b></span>'
        f"<span>Python <b>{html.escape(str(data.get('python', '')).split(' ')[0])}</b></span>"
        f"<span>Arquitetura <b>{html.escape(_arch(platform))}</b></span></p>"
    )
    return {
        "ok": all_ok,
        "figure": f"{passed}<small> de {total}</small>",
        "verdict": verdict_text,
        "percent": _fmt_decimal((passed / total * 100) if total else 0.0, 0),
        "meta": meta,
        "when": f"Executado em {_when(str(data.get('started_at', '')))} · {_arch(platform)}",
        "tiles": _tiles(tiles),
        "groups": f'<div class="groups">{"".join(group_cards)}</div>',
        "table": (
            '<div class="card"><table><thead><tr><th>Caso</th><th>Status</th>'
            '<th class="num case-col-time">Tempo (ms)</th></tr></thead>'
            f"<tbody>{''.join(table_rows)}</tbody></table></div>"
        ),
        "platform": platform,
    }


def qemu_section(data: dict) -> dict[str, str]:
    cases = data.get("cases", [])
    total = len(cases)
    passed = sum(c.get("resultado") == "passed" for c in cases)
    placas: dict[str, list[dict]] = {}
    for caso in cases:
        placas.setdefault(caso.get("placa", caso["id"][:2]), []).append(caso)

    rows = []
    for placa, casos in placas.items():
        ok = sum(c.get("resultado") == "passed" for c in casos)
        cpus = sorted({c.get("qemu_cpu") or "padrão" for c in casos})
        nota = next((c.get("nota_cpu") for c in casos if c.get("nota_cpu")), "")
        versoes = []
        for c in casos:
            passou = c.get("resultado") == "passed"
            tone = "good" if passou else "critical"
            extra = (
                ""
                if passou
                else f' <span class="sub">({html.escape(str(c.get("observacao", "")))})</span>'
            )
            versoes.append(
                f'<div class="version"><span class="badge {tone}"><span class="dot">{"✓" if passou else "✗"}</span>'
                f'<span><span class="id">{c["id"]}</span>&nbsp; {html.escape(str(c.get("versao", "")))}{extra}</span></span></div>'
            )
        nota_html = f'<div class="sub">{html.escape(nota)}</div>' if nota else ""
        rows.append(
            f"<tr><td><b>{html.escape(placa)}</b>{nota_html}</td>"
            f'<td class="cpu col-hide-sm">{html.escape(", ".join(cpus))}</td>'
            f'<td><div class="versions">{"".join(versoes)}</div></td>'
            f'<td class="num"><b>{ok}</b>/{len(casos)}</td></tr>'
        )

    tiles = [
        ("Versões testadas", _fmt_int(total), ""),
        ("Passaram", _fmt_int(passed), ""),
        ("Falharam", _fmt_int(total - passed), ""),
        ("Placas", _fmt_int(len(placas)), ""),
    ]
    if data.get("duration_s") is not None:
        tiles.append(("Tempo total", *_fmt_duration(float(data["duration_s"]) * 1000)))
    ort = next((c.get("onnxruntime") for c in cases if c.get("onnxruntime")), None)
    if ort:
        tiles.append(("ONNX Runtime", html.escape(str(ort)), ""))

    meta_items = [data["_quando"]]
    if data.get("qemu"):
        meta_items.append(f"<b>{html.escape(str(data['qemu']).replace(' version ', ' '))}</b>")
    if data.get("imagem"):
        meta_items.append(f"imagem <b>{html.escape(data['imagem'])}</b>")
    meta = (
        '<p class="meta section-meta">' + "".join(f"<span>{m}</span>" for m in meta_items) + "</p>"
    )

    return {
        "ok": passed == total and total > 0,
        "figure": f"{passed}<small> de {total}</small>",
        "verdict": "Todas as versões iniciaram em ARM64"
        if passed == total
        else f"{total - passed} versão(ões) falharam",
        "when": re.sub(r"</?b>", "", data["_quando"]),
        "meta": meta,
        "tiles": _tiles(tiles),
        "table": (
            '<div class="card"><table><thead><tr><th>Placa</th><th class="col-hide-sm">CPU emulada</th>'
            '<th>Versões (CLI <code>info</code> em ARM64)</th><th class="num">Passou</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        ),
        "arquivo": _display_path(data["_arquivo"]),
    }


def docker_section(data: dict) -> dict[str, str]:
    cases = data.get("cases", [])
    total = len(cases)
    passed = sum(c.get("motivo") == "ok" for c in cases)
    timeouts = sum(c.get("motivo") == "timeout" for c in cases)
    ooms = sum(c.get("motivo") == "oom" for c in cases)
    erros = sum(c.get("motivo") == "erro" for c in cases)
    ignoradas = data.get("nao_aplicavel", [])

    rows = []
    tem_nota = False
    for c in cases:
        icon, label, tone = DOCKER_STATUS.get(c.get("motivo", "erro"), DOCKER_STATUS["erro"])
        modelo = html.escape(str(c.get("modelo_usado", "")))
        if c.get("nota_modelo"):
            modelo += "&nbsp;*"
            tem_nota = True
        latencia = c.get("latencia_ms")
        rss = c.get("pico_rss_mb")
        nota_placa = ""
        if c.get("nota_cpu"):
            nota_placa = f'<div class="sub warn-note">⚠ {html.escape(str(c["nota_cpu"]))}: perfil é só referência</div>'
        obs = ""
        if c.get("motivo") != "ok":
            obs = f'<div class="sub">{html.escape(str(c.get("observacao", "")))}</div>'
        rows.append(
            f'<tr><td><span class="id">{c["id"]}</span></td>'
            f'<td>{html.escape(str(c.get("placa", "")))}<div class="sub">{html.escape(str(c.get("versao", "")))}</div>{nota_placa}</td>'
            f'<td class="col-hide-sm">{modelo}</td>'
            f'<td class="num">{_fmt_cpus(c.get("cpus"))} · {_fmt_mem(c.get("memoria"))}</td>'
            f"<td>{_badge(icon, label, tone)}{obs}</td>"
            f'<td class="num">{_fmt_decimal(latencia / 1000, 1) + " s" if latencia else "—"}</td>'
            f'<td class="num col-hide-sm">{_fmt_int(round(rss)) + " MB" if rss else "—"}</td></tr>'
        )

    tiles = [
        ("Pipelines executados", _fmt_int(total), ""),
        ("Concluíram", _fmt_int(passed), ""),
        ("Tempo esgotado", _fmt_int(timeouts), ""),
        ("Sem memória", _fmt_int(ooms), ""),
    ]
    if erros:
        tiles.append(("Erros", _fmt_int(erros), ""))
    tiles.append(("Sem benchmark", _fmt_int(len(ignoradas)), ""))
    if data.get("timeout_s"):
        tiles.append(("Limite por caso", _fmt_int(int(data["timeout_s"])), "s"))

    skip_html = ""
    if ignoradas:
        itens = "; ".join(
            f'<span class="id">{html.escape(i["id"])}</span> {html.escape(str(i.get("placa", "")))} '
            f"({html.escape(str(i.get('motivo', '')))})"
            for i in ignoradas
        )
        skip_html = f'<div class="card skip"><b>Sem benchmark nesta etapa:</b> {itens}.</div>'

    notas = [
        "Os tempos incluem o custo da emulação ARM64 pelo QEMU: servem para comparar casos entre si, não são FPS nem latência das placas.",
        "Pico de RAM informado pelo próprio processo emulado; pode incluir páginas compartilhadas (bibliotecas, modelo mapeado) e passar do limite. O critério de memória é o processo ter sido ou não encerrado pelo cgroup.",
    ]
    if tem_nota:
        notas.append(
            "* Variante específica (INT8, 512, 320 ou reduzida) ainda não existe no acervo: foi usado o ONNX genérico do Modelo A."
        )
    meta_items = [data["_quando"]]
    if data.get("imagem_teste"):
        meta_items.append(
            f"entrada <b>{html.escape(Path(data['imagem_teste']).name)}</b> (1 quadro)"
        )
    meta = (
        '<p class="meta section-meta">' + "".join(f"<span>{m}</span>" for m in meta_items) + "</p>"
    )

    return {
        "ok": passed == total and total > 0,
        "warn": passed > 0,
        "figure": f"{passed}<small> de {total}</small>",
        "verdict": "Todos os pipelines concluíram"
        if passed == total
        else f"{total - passed} não concluíram",
        "when": re.sub(r"</?b>", "", data["_quando"]),
        "meta": meta,
        "tiles": _tiles(tiles),
        "table": (
            '<div class="card"><table><thead><tr><th>ID</th><th>Placa / versão</th>'
            '<th class="col-hide-sm">Modelo executado</th><th class="num">Limite</th><th>Resultado</th>'
            '<th class="num">Latência/quadro</th><th class="num col-hide-sm">Pico RAM</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        ),
        "skip": skip_html,
        "notes": '<ul class="notes">'
        + "".join(f"<li>{html.escape(n)}</li>" for n in notas)
        + "</ul>",
        "arquivo": _display_path(data["_arquivo"]),
    }


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def _summary_card(label: str, sec: dict, desc: str, warn: bool = False) -> str:
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="figure">{sec["figure"]}</div>'
        f"<div>{_verdict(sec['ok'], sec['verdict'], small=True, warn=warn and not sec['ok'])}</div>"
        f'<p>{desc}</p><div class="when">{html.escape(sec["when"])}</div></div>'
    )


def build_html(
    data: dict, source: Path, qemu: dict | None = None, docker: dict | None = None
) -> str:
    unit = unit_section(data)
    q = qemu_section(qemu) if qemu else None
    d = docker_section(docker) if docker else None

    if q or d:
        cards = [
            _summary_card(
                "Testes unitários",
                unit,
                "Núcleo (Modelo A), Modelo B e camada de hardware das versões embarcadas.",
            )
        ]
        if q:
            cards.append(
                _summary_card(
                    "Matriz em ARM64 (QEMU)",
                    q,
                    "Cada versão da matriz inicia a CLI em Linux ARM64 emulado, com o núcleo da placa.",
                )
            )
        if d:
            cards.append(
                _summary_card(
                    "Docker com limites de CPU/RAM",
                    d,
                    "Benchmark em ARM64 emulado com a CPU e a RAM do perfil de cada placa.",
                    warn=d["warn"],
                )
            )
        topo = f"""
  <section class="summary-section">
    <h2 class="first">Visão geral</h2>
    <div class="summary">{"".join(cards)}</div>
  </section>"""
        unit_head = f"""
  <h2>1. Testes unitários</h2>
  <p class="lead">Testes automatizados sem dataset nem pesos treinados: validação de dados, pré e
  pós-processamento, métricas, rastreamento, eventos e a camada de hardware de cada versão.</p>
  {unit["meta"]}"""
    else:
        topo = ""
        unit_head = f"""
  {unit["meta"]}
  <section class="card hero">
    <div>
      <div class="hero-figure">{unit["figure"]}</div>
      <div class="hero-label">testes aprovados ({unit["percent"]}%)</div>
    </div>
    <div class="hero-side">
      {_verdict(unit["ok"], unit["verdict"])}
      <p>Testes unitários do núcleo (Modelo A), do Modelo B e da camada de hardware das versões
      embarcadas. Não dependem de dataset nem de pesos treinados.</p>
    </div>
  </section>"""

    q_html = ""
    if q:
        q_html = f"""
  <h2>2. Matriz de versões em ARM64 (QEMU)</h2>
  <p class="lead">Para cada versão, a CLI correspondente (<code>animal-monitor</code> ou
  <code>animal-monitor-b</code>) executa <code>info</code> dentro da imagem Linux ARM64, com o tipo
  de núcleo da placa emulado pelo QEMU quando a placa é ARM64. Confirma arquitetura e dependências
  (NumPy, OpenCV, ONNX Runtime). Não emula NPU, GPU, câmera, energia nem desempenho.</p>
  {q["meta"]}
  {q["tiles"]}
  <h3>Resultado por placa</h3>
  {q["table"]}"""

    d_html = ""
    if d:
        d_html = f"""
  <h2>{"3" if q else "2"}. Docker com limites de CPU e RAM</h2>
  <p class="lead">Benchmark do pipeline em uma imagem real do conjunto de teste, em ARM64 emulado,
  com <code>--cpus</code> e <code>--memory</code> (sem swap) no perfil de cada placa. Mostra se o
  pipeline cabe no limite de memória; os tempos não representam a placa física.</p>
  {d["meta"]}
  {d["tiles"]}
  <h3>Resultado por pipeline</h3>
  {d["table"]}
  {d["skip"]}
  {d["notes"]}"""

    n_casos = 1 + bool(q) + bool(d)
    fontes = [f"<code>{html.escape(_display_path(source))}</code>"]
    if q:
        fontes.append(f"<code>{html.escape(q['arquivo'])}</code>")
    if d:
        fontes.append(f"<code>{html.escape(d['arquivo'])}</code>")
    comandos = "<code>python scripts/testar_tudo.py</code>"
    if q or d:
        comandos += " · <code>python scripts/testar_matriz.py</code>"
    comandos += " · <code>python scripts/relatorio_testes_html.py</code>"

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de testes</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <p class="eyebrow">TCC · Monitoramento de fauna com visão computacional embarcada</p>
  <h1>Relatório de testes{" e validação embarcada" if q or d else " automatizados"}</h1>
  {topo}
  {unit_head}
  {unit["tiles"]}

  <h3>Resultado por componente</h3>
  {unit["groups"]}
  {q_html}
  {d_html}

  <h2>{n_casos + 1}. Casos de teste unitários</h2>
  {unit["table"]}

  <footer>
    Fontes: {", ".join(fontes)}
    · ambiente dos testes unitários: <code>{html.escape(unit["platform"])}</code>
    · reprodução: {comandos}
  </footer>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "testes" / "resultado.json",
        help="resultado.json gerado por scripts/testar_tudo.py",
    )
    parser.add_argument("--qemu-dir", type=Path, default=None, help="pasta da matriz QEMU")
    parser.add_argument("--docker-dir", type=Path, default=None, help="pasta do Docker limitado")
    parser.add_argument("--sem-matriz", action="store_true", help="só os testes unitários")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="arquivo HTML de saída (padrão: relatorio*.html ao lado do JSON)",
    )
    parser.add_argument("--open", action="store_true", help="abre o relatório no navegador")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    qemu_dir, docker_dir = args.qemu_dir, args.docker_dir
    # As pastas vizinhas só valem para o resultado.json corrente; um resultado arquivado
    # (ex.: resultado_arm64_2026-11-04.json) precisa indicar as pastas da mesma rodada.
    if args.input.stem == "resultado":
        qemu_dir = qemu_dir or args.input.parent / "qemu-matriz"
        docker_dir = docker_dir or args.input.parent / "docker-recursos"
    qemu = docker = None
    if not args.sem_matriz:
        qemu, docker = load_qemu(qemu_dir), load_docker(docker_dir)

    # resultado.json -> relatorio.html; resultado_arm64.json -> relatorio_arm64.html
    default_name = "relatorio" + args.input.stem.removeprefix("resultado") + ".html"
    output = args.output or args.input.with_name(default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(data, args.input, qemu, docker), encoding="utf-8")
    print(f"Relatório: {output.resolve()}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
