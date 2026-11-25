"""CLI comum das versões (`python -m animal_monitor_hw`)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from animal_monitor.errors import AnimalMonitorError
from animal_monitor_hw import __version__
from animal_monitor_hw.armadilha import Armadilha, processar_lote
from animal_monitor_hw.camtrap_dp import RegistroCamtrap
from animal_monitor_hw.captura import CameraArquivos, criar_camera
from animal_monitor_hw.energia import (
    Bateria,
    CicloEvento,
    autonomia_dias,
    tabela_sensibilidade,
)
from animal_monitor_hw.gatilho import GatilhoSimulado, criar_gatilho, criar_saida_ir
from animal_monitor_hw.versao import Versao


def _json(valor: object) -> str:
    return json.dumps(valor, ensure_ascii=False, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animal-monitor-hw",
        description="Versões embarcadas da armadilha fotográfica com IA local.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    trap = commands.add_parser("trap", help="Executa a armadilha (PIR → rajada → análise).")
    trap.add_argument("--versao", required=True, help="Pasta da versão ou versao.yaml.")
    trap.add_argument(
        "--simular", help="Pasta de imagens ou vídeo no lugar da câmera (bancada sem hardware)."
    )
    trap.add_argument("--eventos", type=int, help="Número máximo de disparos.")
    trap.add_argument("--intervalo", type=float, default=2.0,
                      help="Com --simular: segundos entre disparos simulados.")

    batch = commands.add_parser("batch", help="Arquitetura C: classifica fotos já gravadas.")
    batch.add_argument("--versao", required=True)
    batch.add_argument("--entrada", required=True, help="Pasta com as fotos (ex.: cartão do MCU).")
    batch.add_argument("--csv-mcu", help="Log gravado pelo MCU (evento,timestamp_iso,arquivo,...).")
    batch.add_argument("--janela", type=float, default=30.0,
                       help="Sem CSV: fotos separadas por mais que isto (s) são eventos distintos.")

    export = commands.add_parser("export-camtrapdp", help="Gera o pacote Camtrap DP.")
    export.add_argument("--versao", required=True)
    export.add_argument("--saida", help="Pasta do pacote (padrão: <dados>/camtrap-dp).")
    export.add_argument("--titulo", default="Monitoramento de fauna com IA embarcada")
    export.add_argument("--contato", default="Laura Yumi Osawa")

    autonomy = commands.add_parser("autonomy", help="Autonomia pelo balanço da seção 7.")
    autonomy.add_argument("--bateria-wh", type=float, required=True)
    autonomy.add_argument("--fator-utilizavel", type=float, default=1.0)
    autonomy.add_argument("--eficiencia", type=float, default=0.85)
    autonomy.add_argument("--espera-w", type=float, nargs="+", required=True)
    autonomy.add_argument("--ciclo-j", type=float, required=True, help="Energia do ciclo (J).")
    autonomy.add_argument("--ciclo-s", type=float, required=True, help="Duração do ciclo (s).")
    autonomy.add_argument("--eventos", type=float, nargs="+", default=[10, 100, 1000])

    bench = commands.add_parser("benchmark", help="Latência por estágio em rajadas simuladas.")
    bench.add_argument("--versao", required=True)
    bench.add_argument("--fonte", required=True, help="Pasta de imagens ou vídeo.")
    bench.add_argument("--eventos", type=int, default=10)
    bench.add_argument("--saida", help="JSON de saída.")

    info = commands.add_parser("info", help="Mostra a versão carregada e o ambiente.")
    info.add_argument("--versao", required=True)

    importar = commands.add_parser(
        "import-csv", help="Importa o registros.csv de MaixCAM/Grove/ESP32/Nicla/Luckfox."
    )
    importar.add_argument("--csv", required=True, help="registros.csv copiado do cartão.")
    importar.add_argument("--midias", required=True, help="Pasta base dos arquivos do CSV.")
    importar.add_argument("--versao-id", required=True, help="Ex.: 09a (vai para classifiedBy).")
    importar.add_argument("--registro", default="dados/camtrap.sqlite3")
    importar.add_argument("--saida", default="dados/camtrap-dp")
    importar.add_argument("--latitude", type=float, default=0.0)
    importar.add_argument("--longitude", type=float, default=0.0)
    importar.add_argument("--camera", default="armadilha embarcada")
    importar.add_argument("--contato", default="Laura Yumi Osawa")
    return parser


def _registro(versao: Versao) -> RegistroCamtrap:
    caminho = versao.secao("armazenamento").get("registro", "dados/camtrap.sqlite3")
    return RegistroCamtrap(versao.caminho(caminho))


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "autonomy":
        bateria = Bateria(args.bateria_wh, args.fator_utilizavel, args.eficiencia)
        ciclo = CicloEvento(args.ciclo_j, args.ciclo_s)
        print(_json({
            "energia_util_wh": round(bateria.energia_util_wh, 3),
            "dias": tabela_sensibilidade(bateria, args.espera_w, ciclo, args.eventos),
        }))
        return 0

    if args.command == "import-csv":
        from animal_monitor_hw.camtrap_dp import Implantacao, agora_iso, importar_csv_mcu

        deployment = f"dep-{args.versao_id}"
        with RegistroCamtrap(args.registro) as registro:
            registro.registrar_implantacao(Implantacao(
                deployment, args.latitude, args.longitude, agora_iso(), args.versao_id,
                args.camera, comentarios=f"importado de {Path(args.csv).name}"))
            contagem = importar_csv_mcu(registro, args.csv, deployment, Path(args.midias),
                                        f"animal-monitor-hw {args.versao_id} (no dispositivo)")
            pacote = registro.exportar(args.saida, "Monitoramento de fauna com IA embarcada",
                                       args.contato, descricao=f"Versão {args.versao_id}")
        print(_json({"eventos_por_tipo": contagem, "datapackage": str(pacote)}))
        return 0

    versao = Versao.carregar(args.versao)
    if args.command == "info":
        modulos = {}
        for nome in ("onnxruntime", "cv2", "picamera2", "gpiozero", "gpiod", "hailo_platform",
                     "rknnlite"):
            try:
                __import__(nome)
                modulos[nome] = True
            except ImportError:
                modulos[nome] = False
        print(_json({"versao": versao.resumo(), "python": platform.python_version(),
                     "plataforma": platform.platform(), "modulos": modulos}))
        return 0

    if args.command == "trap":
        detector = versao.construir_detector()
        analisador = versao.construir_analisador(detector)
        if getattr(detector, "camera", None) is not None and not args.simular:
            camera = detector.camera  # IMX500: a câmera é o próprio acelerador
        else:
            camera = criar_camera(versao.secao("captura"), args.simular)
        gatilho = (
            GatilhoSimulado(args.intervalo, args.eventos)
            if args.simular
            else criar_gatilho(versao.secao("gatilho"))
        )
        iluminador = criar_saida_ir(None if args.simular else versao.secao("iluminacao_ir"))
        with _registro(versao) as registro:
            armadilha = Armadilha(versao, camera, gatilho, iluminador, analisador, registro)
            total = armadilha.executar(args.eventos, pausa_s=0.0 if args.simular else None)
        print(_json({"eventos_processados": total}))
        return 0

    if args.command == "batch":
        print(_json(processar_lote(versao, args.entrada, args.csv_mcu, args.janela)))
        return 0

    if args.command == "export-camtrapdp":
        pasta = args.saida or versao.caminho(
            versao.secao("armazenamento").get("pasta", "dados")
        ) / "camtrap-dp"
        with _registro(versao) as registro:
            caminho = registro.exportar(pasta, args.titulo, args.contato,
                                        descricao=f"Versão {versao.identificador}")
        print(_json({"datapackage": str(caminho)}))
        return 0

    if args.command == "benchmark":
        analisador = versao.construir_analisador()
        camera = CameraArquivos(args.fonte)
        quadros_por_evento = int(versao.secao("evento").get("quadros", 8))
        camera.rajada(quadros_por_evento, 0.0)  # aquecimento
        tempos, medidas = [], []
        for _ in range(args.eventos):
            quadros = camera.rajada(quadros_por_evento, 0.0)
            inicio = time.perf_counter()
            resultado = analisador.analisar(quadros)
            tempos.append((time.perf_counter() - inicio) * 1000.0)
            medidas.append(resultado.latencia_ms)
        valores = np.asarray(tempos)
        relatorio = {
            "versao": versao.identificador,
            "quadros_por_evento": quadros_por_evento,
            "eventos": args.eventos,
            "analise_por_evento_ms": {
                "media": float(valores.mean()),
                "p50": float(np.percentile(valores, 50)),
                "p95": float(np.percentile(valores, 95)),
            },
            "por_quadro_ms": float(valores.mean() / quadros_por_evento),
            "estagios_medios_ms": {
                chave: float(np.mean([medida.get(chave, 0.0) for medida in medidas]))
                for chave in medidas[0]
            },
            "plataforma": platform.platform(),
        }
        if args.saida:
            Path(args.saida).expanduser().write_text(_json(relatorio), encoding="utf-8")
        print(_json(relatorio))
        return 0
    raise AssertionError(f"Comando não tratado: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except (AnimalMonitorError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


def autonomia_simples(bateria_wh: float, espera_w: float, ciclo_j: float, ciclo_s: float,
                      eventos: float) -> float:
    """Atalho usado pelo cálculo de ranking."""
    return autonomia_dias(Bateria(bateria_wh), espera_w, CicloEvento(ciclo_j, ciclo_s), eventos)
