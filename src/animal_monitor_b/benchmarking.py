"""Benchmark com latência separada por estágio.

Um número único de FPS esconde a informação que importa para dimensionar o embarcado:
quanto custa o detector (roda sempre) e quanto custa o classificador (roda só quando há
animal e apenas a cada `classify_every` quadros por trilha). Sem essa separação não é
possível estimar o consumo médio em campo, onde a maioria dos quadros está vazia.
"""

from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np

from animal_monitor.sources import FrameStream
from animal_monitor_b.runtime.pipeline import TwoStagePipeline


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "calls": int(array.size),
    }


def benchmark_pipeline(
    pipeline: TwoStagePipeline,
    source: str | int,
    frame_limit: int = 100,
    warmup_frames: int = 3,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    """Mede o pipeline completo e cada estágio isoladamente."""
    total_ms: list[float] = []
    detector_ms: list[float] = []
    classifier_ms: list[float] = []
    tracker_ms: list[float] = []
    detections_total = 0
    crops_total = 0
    frames_with_animal = 0
    seen = 0

    with FrameStream(source) as stream:
        for packet in stream:
            if seen >= frame_limit + warmup_frames:
                break
            measure = seen >= warmup_frames

            start_total = perf_counter()
            start = perf_counter()
            boxes, scores = pipeline.detector.detect(packet.image)
            detector_elapsed = (perf_counter() - start) * 1000.0

            start = perf_counter()
            active, finished = pipeline.tracker.update(
                boxes, scores, packet.index, packet.source_time_s
            )
            tracker_elapsed = (perf_counter() - start) * 1000.0
            pipeline.finished_tracks.extend(finished)

            pending = [
                track
                for track in active
                if packet.index - track.last_classified_frame >= pipeline.config.classify_every
            ]
            pending.sort(key=lambda track: track.score, reverse=True)
            pending = pending[: pipeline.config.max_crops_per_frame]
            classifier_elapsed = 0.0
            if pending:
                crop_boxes = np.stack([track.box for track in pending], axis=0)
                start = perf_counter()
                probabilities = pipeline.classifier.classify(packet.image, crop_boxes)
                classifier_elapsed = (perf_counter() - start) * 1000.0
                for track, row in zip(pending, probabilities, strict=True):
                    best = int(np.argmax(row))
                    track.add_vote(
                        pipeline.classifier.labels[best],
                        pipeline._vote_weight(track.score, float(row[best])),
                    )
                    track.last_classified_frame = packet.index
            for track in active:
                pipeline._maybe_emit(track)
            total_elapsed = (perf_counter() - start_total) * 1000.0

            if measure:
                total_ms.append(total_elapsed)
                detector_ms.append(detector_elapsed)
                tracker_ms.append(tracker_elapsed)
                detections_total += int(boxes.shape[0])
                if boxes.shape[0]:
                    frames_with_animal += 1
                if pending:
                    classifier_ms.append(classifier_elapsed)
                    crops_total += len(pending)
            seen += 1

    if not total_ms:
        raise RuntimeError("Não houve quadros suficientes para executar o benchmark.")
    mean_total = float(np.mean(total_ms))
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_mb = max_rss / (1024.0 if max_rss < 10**9 else 1024.0**2)
    frames = len(total_ms)

    result: dict[str, object] = {
        "source": str(source),
        "measured_frames": frames,
        "warmup_frames": min(warmup_frames, seen),
        "raw_detections": detections_total,
        "frames_with_animal": frames_with_animal,
        "classified_crops": crops_total,
        "crops_per_frame": crops_total / frames if frames else 0.0,
        "latency_ms": {
            "total": _summary(total_ms),
            "presence_detector": _summary(detector_ms),
            "species_classifier": _summary(classifier_ms),
            "tracker": _summary(tracker_ms),
        },
        "throughput_fps": 1000.0 / mean_total if mean_total > 0 else None,
        "peak_process_rss_mb": rss_mb,
        "pipeline": pipeline.info(),
    }
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
