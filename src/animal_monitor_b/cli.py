"""Interface de linha de comando do Modelo B (`animal-monitor-b`)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from animal_monitor.config import load_class_names
from animal_monitor.errors import AnimalMonitorError
from animal_monitor_b import __version__
from animal_monitor_b.benchmarking import benchmark_pipeline
from animal_monitor_b.classifier import train_classifier
from animal_monitor_b.config import RuntimeConfigB
from animal_monitor_b.data_prep import prepare_model_b_data
from animal_monitor_b.detector_train import evaluate_presence_detector, train_presence_detector
from animal_monitor_b.evaluation import compare_reports, evaluate_species_predictor
from animal_monitor_b.exporting import (
    export_presence_detector,
    export_species_classifier,
    quantize_int8,
)
from animal_monitor_b.monitoring import run_monitoring_b
from animal_monitor_b.runtime.pipeline import TwoStagePipeline


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animal-monitor-b",
        description=(
            "Modelo B: detecção de presença + classificação de espécie + rastreamento, "
            "voltado a hardware embarcado de baixo custo."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-data",
        help="Deriva o dataset de presença e os recortes de espécie do dataset original.",
    )
    prepare.add_argument("--data", default="config/dataset.yaml")
    prepare.add_argument("--agnostic-dir", default="data/animals_agnostic")
    prepare.add_argument("--crops-dir", default="data/crops")
    prepare.add_argument("--crop-size", type=int, default=160)
    prepare.add_argument("--crop-margin", type=float, default=0.12)
    prepare.add_argument("--negatives-per-image", type=float, default=1.0)
    prepare.add_argument("--misaligned-per-box", type=float, default=0.5)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--json-output", default="outputs/modelo_b/data_prep.json")

    train_detector = commands.add_parser(
        "train-detector", help="Treina o detector de presença (uma classe)."
    )
    train_detector.add_argument("--config", default="config/modelo_b/detector.yaml")

    train_cls = commands.add_parser(
        "train-classifier", help="Treina o classificador de espécie sobre recortes."
    )
    train_cls.add_argument("--config", default="config/modelo_b/classifier.yaml")

    export = commands.add_parser("export", help="Exporta os dois modelos para ONNX.")
    export.add_argument("--detector-weights", required=True)
    export.add_argument("--classifier-checkpoint", required=True)
    export.add_argument("--output-dir", default="models/modelo_b")
    export.add_argument("--imgsz", type=int, default=416)
    export.add_argument("--opset", type=int, default=17)

    quantize = commands.add_parser(
        "quantize", help="Quantiza um modelo ONNX para INT8 com calibração real."
    )
    quantize.add_argument("--model", required=True)
    quantize.add_argument("--kind", choices=("detector", "classifier"), required=True)
    quantize.add_argument("--calibration-dir", required=True)
    quantize.add_argument("--output")
    quantize.add_argument("--samples", type=int, default=64)
    quantize.add_argument(
        "--calibrate-method",
        choices=("minmax", "entropy", "percentile"),
        default="minmax",
        help="Método de calibração das ativações.",
    )
    quantize.add_argument(
        "--op-types",
        nargs="*",
        help="Restringe a quantização a estes operadores (ex.: Conv). Vazio = todos.",
    )

    eval_detector = commands.add_parser(
        "evaluate-detector", help="Mede o detector de presença isoladamente."
    )
    eval_detector.add_argument("--weights", required=True)
    eval_detector.add_argument("--data", default="data/animals_agnostic/dataset_agnostic.yaml")
    eval_detector.add_argument("--split", choices=("val", "test"), default="test")
    eval_detector.add_argument("--imgsz", type=int, default=416)
    eval_detector.add_argument("--output", default="outputs/modelo_b/evaluation_detector.json")

    evaluate = commands.add_parser(
        "evaluate", help="Avalia o pipeline ponta a ponta por espécie no split escolhido."
    )
    evaluate.add_argument("--detector", required=True, help="ONNX do detector de presença.")
    evaluate.add_argument("--classifier", required=True, help="ONNX do classificador.")
    evaluate.add_argument("--data", default="config/dataset.yaml")
    evaluate.add_argument("--runtime", default="config/modelo_b/runtime.yaml")
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--score-threshold", type=float, default=0.35)
    evaluate.add_argument("--label", default="modelo_b")
    evaluate.add_argument("--output", default="outputs/modelo_b/evaluation_e2e.json")

    baseline = commands.add_parser(
        "evaluate-baseline",
        help="Avalia o Modelo A com o MESMO código de métrica, para comparação honesta.",
    )
    baseline.add_argument("--model", required=True, help="ONNX multiclasse do Modelo A.")
    baseline.add_argument("--data", default="config/dataset.yaml")
    baseline.add_argument("--runtime", default="config/runtime.yaml")
    baseline.add_argument("--split", choices=("train", "val", "test"), default="test")
    baseline.add_argument("--score-threshold", type=float, default=0.35)
    baseline.add_argument("--label", default="modelo_a")
    baseline.add_argument("--output", default="outputs/modelo_b/evaluation_baseline.json")

    compare = commands.add_parser("compare", help="Monta a tabela comparativa dos relatórios.")
    compare.add_argument("--reports", nargs="+", required=True)
    compare.add_argument("--output", default="outputs/modelo_b/comparison.json")

    infer = commands.add_parser("infer", help="Monitora vídeo, câmera, imagem ou diretório.")
    infer.add_argument("--detector", required=True)
    infer.add_argument("--classifier", required=True)
    infer.add_argument("--source", required=True)
    infer.add_argument("--runtime", default="config/modelo_b/runtime.yaml")
    infer.add_argument("--camera-id")
    infer.add_argument("--output-video")
    infer.add_argument("--show", action="store_true")
    infer.add_argument("--max-frames", type=int)

    benchmark = commands.add_parser(
        "benchmark", help="Mede latência por estágio, FPS e pico de memória."
    )
    benchmark.add_argument("--detector", required=True)
    benchmark.add_argument("--classifier", required=True)
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--runtime", default="config/modelo_b/runtime.yaml")
    benchmark.add_argument("--frames", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--output", default="outputs/modelo_b/benchmark.json")

    commands.add_parser("info", help="Mostra dependências e versões do ambiente.")
    return parser


def _pipeline(args: argparse.Namespace) -> tuple[TwoStagePipeline, RuntimeConfigB]:
    config = RuntimeConfigB.from_yaml(args.runtime)
    pipeline = TwoStagePipeline.from_models(args.detector, args.classifier, config)
    return pipeline, config


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "prepare-data":
        report = prepare_model_b_data(
            args.data,
            agnostic_dir=args.agnostic_dir,
            crops_dir=args.crops_dir,
            crop_size=args.crop_size,
            crop_margin=args.crop_margin,
            negatives_per_image=args.negatives_per_image,
            misaligned_per_box=args.misaligned_per_box,
            seed=args.seed,
            report_json=args.json_output,
        )
        print(_json(report))
        return 0

    if args.command == "train-detector":
        print(_json({"best_weights": str(train_presence_detector(args.config))}))
        return 0

    if args.command == "train-classifier":
        result = train_classifier(args.config)
        print(
            _json(
                {
                    "checkpoint": result["checkpoint"],
                    "best_epoch": result["best_epoch"],
                    "duration_s": result["duration_s"],
                    "val_report": result["val_report"],
                    "test_report": result.get("test_report"),
                }
            )
        )
        return 0

    if args.command == "export":
        detector = export_presence_detector(
            args.detector_weights, args.output_dir, image_size=args.imgsz, opset=args.opset
        )
        classifier = export_species_classifier(
            args.classifier_checkpoint, args.output_dir, opset=args.opset
        )
        print(_json({"presence_detector": str(detector), "species_classifier": str(classifier)}))
        return 0

    if args.command == "quantize":
        result = quantize_int8(
            args.model,
            args.calibration_dir,
            kind=args.kind,
            output_path=args.output,
            samples=args.samples,
            calibrate_method=args.calibrate_method,
            op_types=tuple(args.op_types) if args.op_types else None,
        )
        print(_json(result))
        return 0

    if args.command == "evaluate-detector":
        result = evaluate_presence_detector(
            args.weights, args.data, args.split, args.imgsz, args.output
        )
        print(_json(result))
        return 0

    if args.command == "evaluate":
        pipeline, _ = _pipeline(args)
        result = evaluate_species_predictor(
            pipeline.detect_species,
            args.data,
            split=args.split,
            score_threshold=args.score_threshold,
            label=args.label,
            output_json=args.output,
        )
        print(_json(result))
        return 0

    if args.command == "evaluate-baseline":
        from animal_monitor.config import RuntimeConfig
        from animal_monitor.inference import OnnxYoloDetector

        runtime = RuntimeConfig.from_yaml(args.runtime)
        names = load_class_names(args.data)
        detector = OnnxYoloDetector(args.model, names, runtime)
        result = evaluate_species_predictor(
            detector.infer,
            args.data,
            split=args.split,
            score_threshold=args.score_threshold,
            label=args.label,
            output_json=args.output,
        )
        print(_json(result))
        return 0

    if args.command == "compare":
        reports = [
            json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
            for path in args.reports
        ]
        table = compare_reports(reports)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_json(table), encoding="utf-8")
        print(_json(table))
        return 0

    if args.command == "infer":
        pipeline, config = _pipeline(args)
        result = run_monitoring_b(
            pipeline,
            args.source,
            config,
            camera_id=args.camera_id,
            output_video=args.output_video,
            show=args.show,
            max_frames=args.max_frames,
        )
        print(_json(result))
        return 0

    if args.command == "benchmark":
        pipeline, _ = _pipeline(args)
        result = benchmark_pipeline(
            pipeline,
            args.source,
            frame_limit=args.frames,
            warmup_frames=args.warmup,
            output_json=args.output,
        )
        print(_json(result))
        return 0

    if args.command == "info":
        packages = [
            "numpy",
            "PyYAML",
            "opencv-python",
            "onnxruntime",
            "onnx",
            "torch",
            "torchvision",
            "ultralytics",
        ]
        print(
            _json(
                {
                    "animal_monitor_b": __version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {name: _package_version(name) for name in packages},
                    "modules": {
                        name: importlib.util.find_spec(name) is not None
                        for name in (
                            "numpy",
                            "yaml",
                            "cv2",
                            "onnxruntime",
                            "torch",
                            "torchvision",
                            "ultralytics",
                        )
                    },
                }
            )
        )
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
