#!/usr/bin/env python3
"""Executa todos os testes e coleta métricas reproduzíveis da execução."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import unittest
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

try:
    import resource
except ImportError:  # pragma: no cover - suporte a plataformas sem resource
    resource = None


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TestRecord:
    name: str
    status: str
    duration_ms: float
    details: str = ""


class MetricsResult(unittest.TextTestResult):
    """Resultado unittest que preserva duração e estado de cada teste."""

    def __init__(self, stream: TextIO, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.records: list[TestRecord] = []
        self._started_at = 0.0
        self._current_name = ""

    def startTest(self, test: unittest.case.TestCase) -> None:
        self._started_at = time.perf_counter()
        self._current_name = test.id()
        super().startTest(test)

    def _record(self, status: str, details: str = "") -> None:
        self.records.append(
            TestRecord(
                self._current_name,
                status,
                round((time.perf_counter() - self._started_at) * 1000, 3),
                details,
            )
        )

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        self._record("passed")
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err: object) -> None:
        self._record("failed", self._exception_text(err))
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err: object) -> None:
        self._record("error", self._exception_text(err))
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        self._record("skipped", reason)
        super().addSkip(test, reason)

    @staticmethod
    def _exception_text(error: object) -> str:
        if isinstance(error, tuple) and len(error) == 3:
            return str(error[1])
        return str(error)


class MetricsRunner(unittest.TextTestRunner):
    resultclass = MetricsResult


def _memory_kb() -> int | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value / 1024) if sys.platform == "linux" else int(value)


def _load_suite(include_common: bool) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover("tests", top_level_dir="tests"))
    if include_common:
        suite.addTests(loader.discover("versoes/comum/tests", top_level_dir="versoes/comum/tests"))
    return suite


def _compile_sources() -> tuple[bool, str]:
    import compileall
    import io

    output = io.StringIO()
    ok = compileall.compile_dir(str(ROOT / "src"), quiet=1, force=True)
    ok = compileall.compile_dir(str(ROOT / "versoes"), quiet=1, force=True) and ok
    return ok, output.getvalue()


def _write_reports(
    output_dir: Path,
    records: list[TestRecord],
    started_at: str,
    duration_ms: float,
    compile_ok: bool,
    include_common: bool,
    exit_code: int,
    memory_peak_kb: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 3),
        "exit_code": exit_code,
        "compile_ok": compile_ok,
        "include_common_tests": include_common,
        "python": sys.version,
        "platform": platform.platform(),
        "memory_peak_kb": memory_peak_kb,
        "tests": {
            "total": len(records),
            "passed": sum(record.status == "passed" for record in records),
            "failed": sum(record.status == "failed" for record in records),
            "errors": sum(record.status == "error" for record in records),
            "skipped": sum(record.status == "skipped" for record in records),
        },
        "cases": [asdict(record) for record in records],
    }
    (output_dir / "resultado.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "casos.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("name", "status", "duration_ms", "details"))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "testes",
        help="diretório dos relatórios JSON e CSV",
    )
    parser.add_argument(
        "--skip-common",
        action="store_true",
        help="não executar os testes de versoes/comum",
    )
    args = parser.parse_args()

    import os

    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "versoes" / "comum")]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    compile_ok, compile_output = _compile_sources()
    suite = _load_suite(not args.skip_common)
    runner = MetricsRunner(verbosity=2)
    result = runner.run(suite)
    duration_ms = (time.perf_counter() - started) * 1000
    exit_code = 0 if compile_ok and result.wasSuccessful() else 1
    _write_reports(
        args.output_dir,
        result.records,
        started_at,
        duration_ms,
        compile_ok,
        not args.skip_common,
        exit_code,
        _memory_kb(),
    )
    print(f"Relatórios: {args.output_dir / 'resultado.json'} e {args.output_dir / 'casos.csv'}")
    if compile_output:
        print(compile_output, file=sys.stderr, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
