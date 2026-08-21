from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from animal_monitor import __version__
from animal_monitor.benchmarking import benchmark_model
from animal_monitor.config import RuntimeConfig, load_class_names
from animal_monitor.dataset import summarize_issue_counts, validate_dataset
from animal_monitor.errors import AnimalMonitorError
from animal_monitor.exporting import export_onnx
from animal_monitor.inference import OnnxYoloDetector
from animal_monitor.monitoring import run_monitoring
from animal_monitor.splitting import split_dataset
from animal_monitor.training import evaluate_model, train_model


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animal-monitor",
        description="Treino e inferência local para monitoramento de animais.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-dataset", help="Verifica estrutura e rótulos YOLO.")
    validate.add_argument("--data", default="config/dataset.yaml", help="YAML do dataset.")
    validate.add_argument("--minimum-boxes", type=int, default=25)
    validate.add_argument("--json-output", help="Arquivo opcional para salvar o relatório.")

    split = commands.add_parser(
        "split-dataset", help="Separa dados brutos sem misturar grupos entre os splits."
    )
    split.add_argument("--images", required=True, help="Diretório com imagens brutas.")
    split.add_argument("--labels", required=True, help="Diretório com rótulos brutos.")
    split.add_argument("--output", default="data/animals")
    split.add_argument("--ratios", default="0.70,0.20,0.10", help="train,val,test")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--group-regex",
        help="Regex aplicada ao nome; o primeiro grupo identifica câmera/local/vídeo.",
    )
    split.add_argument("--mode", choices=("copy", "hardlink"), default="copy")

    train = commands.add_parser("train", help="Treina o baseline configurado.")
    train.add_argument("--config", default="config/train.yaml")

    evaluate = commands.add_parser("evaluate", help="Calcula métricas no split val ou test.")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", default="config/dataset.yaml")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--output", default="outputs/evaluation.json")

    export = commands.add_parser("export", help="Exporta os pesos treinados para ONNX.")
    export.add_argument("--weights", required=True)
    export.add_argument("--data", default="config/dataset.yaml")
    export.add_argument("--output-dir", default="models")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--dynamic", action="store_true")

    infer = commands.add_parser("infer", help="Monitora câmera, vídeo, imagem ou diretório.")
    infer.add_argument("--model", required=True)
    infer.add_argument(
        "--source", required=True, help="Caminho, URL de stream ou índice da câmera."
    )
    infer.add_argument("--data", default="config/dataset.yaml")
    infer.add_argument("--runtime", default="config/runtime.yaml")
    infer.add_argument("--output-video")
    infer.add_argument("--show", action="store_true")
    infer.add_argument("--max-frames", type=int)

    benchmark = commands.add_parser("benchmark", help="Mede latência e FPS do modelo ONNX.")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--data", default="config/dataset.yaml")
    benchmark.add_argument("--runtime", default="config/runtime.yaml")
    benchmark.add_argument("--frames", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--output", default="outputs/benchmark.json")

    commands.add_parser("info", help="Mostra dependências disponíveis no ambiente.")
    return parser


def _detector(args: argparse.Namespace) -> OnnxYoloDetector:
    runtime = RuntimeConfig.from_yaml(args.runtime)
    names = load_class_names(args.data)
    return OnnxYoloDetector(args.model, names, runtime)


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "validate-dataset":
        report = validate_dataset(args.data, minimum_boxes_per_class=args.minimum_boxes)
        if args.json_output:
            output = Path(args.json_output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.to_json(), encoding="utf-8")
        counts = summarize_issue_counts(report.issues)
        print(report.to_json())
        print(
            f"Resultado: {'APROVADO' if report.ok else 'REPROVADO'} "
            f"({counts['errors']} erro(s), {counts['warnings']} aviso(s))"
        )
        return 0 if report.ok else 1

    if args.command == "split-dataset":
        ratios = tuple(float(value.strip()) for value in args.ratios.split(","))
        if len(ratios) != 3:
            raise ValueError("--ratios precisa conter três números: train,val,test.")
        result = split_dataset(
            args.images,
            args.labels,
            args.output,
            ratios=ratios,
            seed=args.seed,
            group_regex=args.group_regex,
            mode=args.mode,
        )
        print(_json(result))
        return 0

    if args.command == "train":
        print(_json({"best_weights": str(train_model(args.config))}))
        return 0

    if args.command == "evaluate":
        result = evaluate_model(args.weights, args.data, args.split, args.imgsz, args.output)
        print(_json(result))
        return 0

    if args.command == "export":
        exported = export_onnx(
            args.weights, args.data, args.output_dir, args.imgsz, args.opset, args.dynamic
        )
        print(_json({"onnx_model": str(exported)}))
        return 0

    if args.command == "infer":
        detector = _detector(args)
        runtime = detector.config
        result = run_monitoring(
            detector,
            args.source,
            runtime,
            output_video=args.output_video,
            show=args.show,
            max_frames=args.max_frames,
        )
        print(_json(result))
        return 0

    if args.command == "benchmark":
        detector = _detector(args)
        result = benchmark_model(
            detector,
            args.source,
            frame_limit=args.frames,
            warmup_frames=args.warmup,
            output_json=args.output,
        )
        print(_json(result))
        return 0

    if args.command == "info":
        packages = ["numpy", "PyYAML", "opencv-python", "onnxruntime", "ultralytics"]
        result = {
            "animal_monitor": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in packages},
            "modules": {
                name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "yaml", "cv2", "onnxruntime", "ultralytics")
            },
        }
        print(_json(result))
        return 0
    raise AssertionError(f"Comando não tratado: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (AnimalMonitorError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
