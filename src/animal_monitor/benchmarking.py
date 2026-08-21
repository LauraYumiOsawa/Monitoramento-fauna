from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np

from animal_monitor.sources import FrameStream


def benchmark_model(
    detector,
    source: str | int,
    frame_limit: int = 100,
    warmup_frames: int = 3,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    latencies_ms: list[float] = []
    total_detections = 0
    seen_frames = 0
    with FrameStream(source) as stream:
        for packet in stream:
            if seen_frames >= frame_limit + warmup_frames:
                break
            start = perf_counter()
            detections = detector.infer(packet.image)
            elapsed_ms = (perf_counter() - start) * 1000.0
            if seen_frames >= warmup_frames:
                latencies_ms.append(elapsed_ms)
                total_detections += len(detections)
            seen_frames += 1

    if not latencies_ms:
        raise RuntimeError("Não houve quadros suficientes para executar o benchmark.")
    latency = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(latency.mean())
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux informa KiB; macOS informa bytes.
    rss_mb = max_rss / (1024.0 if max_rss < 10**9 else 1024.0**2)
    result: dict[str, object] = {
        "source": str(source),
        "measured_frames": len(latencies_ms),
        "warmup_frames": min(warmup_frames, seen_frames),
        "detections": total_detections,
        "latency_ms": {
            "mean": mean_ms,
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "min": float(latency.min()),
            "max": float(latency.max()),
        },
        "throughput_fps": 1000.0 / mean_ms if mean_ms > 0 else None,
        "peak_process_rss_mb": rss_mb,
        "detector": detector.info(),
    }
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
