"""Laço de monitoramento do Modelo B: vídeo, câmera, imagem ou diretório."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from animal_monitor.sources import FrameStream
from animal_monitor_b import __version__
from animal_monitor_b.config import RuntimeConfigB
from animal_monitor_b.events import SightingStore
from animal_monitor_b.runtime.pipeline import TrackObservation, TwoStagePipeline


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV não está instalado.") from exc
    return cv2


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((53 * (track_id + 2)) % 200 + 55),
        int((97 * (track_id + 5)) % 200 + 55),
        int((151 * (track_id + 7)) % 200 + 55),
    )


def annotate_frame(frame: np.ndarray, observations: list[TrackObservation]) -> np.ndarray:
    cv2 = _cv2()
    annotated = frame.copy()
    for observation in observations:
        box = observation.box
        x1, y1, x2, y2 = (int(round(value)) for value in (box.x1, box.y1, box.x2, box.y2))
        color = _track_color(observation.track_id)
        thickness = 2 if observation.emitted else 1
        label = (
            f"#{observation.track_id} {observation.species} "
            f"{observation.species_confidence:.0%} v{observation.votes}"
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
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
            0.52,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _save_evidence(frame: np.ndarray, output_dir: Path, track_id: int, species: str) -> Path:
    cv2 = _cv2()
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = evidence_dir / f"{timestamp}_track{track_id:04d}_{species}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Falha ao salvar evidência: {path}")
    return path


def _sha256_from_metadata(model_path: Path, metadata_name: str) -> str | None:
    candidate = model_path.with_name(metadata_name)
    if not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("sha256")
    return str(value) if value else None


def run_monitoring_b(
    pipeline: TwoStagePipeline,
    source: str | int,
    config: RuntimeConfigB,
    camera_id: str | None = None,
    output_video: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Processa a fonte, grava avistamentos por indivíduo e devolve o resumo."""
    cv2 = _cv2()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    detector_sha = _sha256_from_metadata(
        pipeline.detector.model_path, "presence_detector_metadata.json"
    )
    classifier_sha = _sha256_from_metadata(
        pipeline.classifier.model_path, "classifier_metadata.json"
    )

    species_counts: Counter[str] = Counter()
    inserted = 0
    duplicates = 0
    processed = 0
    total_detections = 0
    total_crops = 0
    writer = None
    output_path = Path(output_video).expanduser().resolve() if output_video else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with FrameStream(source) as stream, SightingStore(config.database) as store:
            for packet in stream:
                if max_frames is not None and processed >= max_frames:
                    break
                if packet.index % config.frame_stride != 0:
                    processed += 1
                    continue
                result = pipeline.process(packet.image, packet.index, packet.source_time_s)
                total_detections += result.detections
                total_crops += result.classified_crops
                annotated = annotate_frame(packet.image, result.observations)

                for event in result.events:
                    evidence_path = None
                    if config.save_evidence:
                        evidence_path = _save_evidence(
                            annotated, config.output_dir, event.track_id, event.species
                        )
                    row_id = store.add(
                        event,
                        source=stream.label,
                        camera_id=camera_id,
                        frames_tracked=event.frames_tracked,
                        evidence_path=evidence_path,
                        detector_sha256=detector_sha,
                        classifier_sha256=classifier_sha,
                        pipeline_version=__version__,
                    )
                    if row_id is None:
                        duplicates += 1
                    else:
                        inserted += 1
                        species_counts[event.species] += 1

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
                    cv2.imshow("Modelo B - monitoramento", annotated)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                        processed += 1
                        break
                processed += 1
            pipeline.close()
            total_events = store.count()
            by_species = store.count_by_species()
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return {
        "source": str(source),
        "camera_id": camera_id,
        "processed_frames": processed,
        "raw_detections": total_detections,
        "classified_crops": total_crops,
        "tracks_created": pipeline.tracker.created_tracks,
        "new_sightings": dict(species_counts),
        "duplicate_sightings": duplicates,
        "inserted_sightings": inserted,
        "database_total": total_events,
        "database_by_species": by_species,
        "database": str(config.database),
        "output_video": str(output_path) if output_path else None,
    }
