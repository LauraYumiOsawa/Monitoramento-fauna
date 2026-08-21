from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from animal_monitor.config import RuntimeConfig
from animal_monitor.events import DetectionGate, EventStore
from animal_monitor.sources import FrameStream
from animal_monitor.types import Detection


def _color(class_id: int) -> tuple[int, int, int]:
    return (
        int((37 * (class_id + 3)) % 205 + 50),
        int((67 * (class_id + 5)) % 205 + 50),
        int((97 * (class_id + 7)) % 205 + 50),
    )


def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado.") from exc
    annotated = frame.copy()
    for detection in detections:
        box = detection.box
        x1, y1, x2, y2 = (int(round(value)) for value in (box.x1, box.y1, box.x2, box.y2))
        color = _color(detection.class_id)
        label = f"{detection.class_name} {detection.confidence:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_y = max(text_height + baseline + 3, y1)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_height - baseline - 4),
            (x1 + text_width + 6, text_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _save_evidence(frame: np.ndarray, output_dir: Path, frame_index: int) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado.") from exc
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = evidence_dir / f"{timestamp}_frame_{frame_index:08d}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Falha ao salvar evidência: {path}")
    return path


def run_monitoring(
    detector,
    source: str | int,
    config: RuntimeConfig,
    output_video: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> dict[str, object]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado.") from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)
    gate = DetectionGate(config.min_hits, config.cooldown_seconds)
    detection_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    processed = 0
    writer = None
    output_path = Path(output_video).expanduser().resolve() if output_video else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with FrameStream(source) as stream, EventStore(config.database) as store:
            for packet in stream:
                if max_frames is not None and processed >= max_frames:
                    break
                detections: list[Detection] = []
                inference_frame = packet.index % config.frame_stride == 0
                if inference_frame:
                    detections = detector.infer(packet.image)
                    detection_counts.update(item.class_name for item in detections)
                annotated = annotate_frame(packet.image, detections)
                emitted = gate.update(detections, packet.source_time_s) if inference_frame else []
                evidence_path = None
                if emitted and config.save_evidence:
                    evidence_path = _save_evidence(annotated, config.output_dir, packet.index)
                for detection in emitted:
                    store.add(
                        detection,
                        source=stream.label,
                        source_time_s=packet.source_time_s,
                        frame_index=packet.index,
                        evidence_path=evidence_path,
                    )
                    event_counts[detection.class_name] += 1

                if output_path:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(1.0, stream.fps),
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Não foi possível criar o vídeo: {output_path}")
                    writer.write(annotated)

                if show:
                    cv2.imshow("Monitoramento de animais", annotated)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                        processed += 1
                        break
                processed += 1

            total_events = store.count()
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return {
        "source": str(source),
        "processed_frames": processed,
        "detections": dict(detection_counts),
        "new_events": dict(event_counts),
        "database_total_events": total_events,
        "database": str(config.database),
        "output_video": str(output_path) if output_path else None,
    }
