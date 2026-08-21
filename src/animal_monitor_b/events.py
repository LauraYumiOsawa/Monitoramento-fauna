"""Persistência de avistamentos do Modelo B.

Diferenças em relação ao banco do Modelo A (`detections`):

* a linha representa **um indivíduo rastreado**, não um quadro isolado;
* guarda o intervalo de tempo, a contagem de votos e a distribuição completa da votação,
  o que permite auditar por que aquela espécie foi escolhida;
* guarda `camera_id` e o SHA-256 dos dois modelos, atendendo à rastreabilidade científica
  pedida em `MELHORIAS.md`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from animal_monitor_b.runtime.pipeline import SpeciesEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    camera_id TEXT,
    track_id INTEGER NOT NULL,
    species TEXT NOT NULL,
    confidence REAL NOT NULL,
    vote_share REAL NOT NULL,
    votes INTEGER NOT NULL,
    votes_detail TEXT NOT NULL,
    detection_score REAL NOT NULL,
    start_time_s REAL NOT NULL,
    end_time_s REAL NOT NULL,
    frame_index INTEGER NOT NULL,
    frames_tracked INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    evidence_path TEXT,
    detector_sha256 TEXT,
    classifier_sha256 TEXT,
    pipeline_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species);
CREATE INDEX IF NOT EXISTS idx_sightings_observed_at ON sightings(observed_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sightings_track
    ON sightings(source, camera_id, track_id, species);
"""


class SightingStore:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def add(
        self,
        event: SpeciesEvent,
        source: str,
        camera_id: str | None,
        frames_tracked: int,
        evidence_path: str | Path | None,
        detector_sha256: str | None,
        classifier_sha256: str | None,
        pipeline_version: str,
    ) -> int | None:
        """Grava o avistamento. Retorna None se a trilha já estava registrada."""
        try:
            cursor = self.connection.execute(
                """
                INSERT INTO sightings (
                    observed_at, source, camera_id, track_id, species, confidence,
                    vote_share, votes, votes_detail, detection_score, start_time_s,
                    end_time_s, frame_index, frames_tracked, x1, y1, x2, y2,
                    evidence_path, detector_sha256, classifier_sha256, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    source,
                    camera_id,
                    event.track_id,
                    event.species,
                    event.confidence,
                    event.vote_share,
                    event.votes,
                    json.dumps(event.votes_detail, ensure_ascii=False),
                    event.detection_score,
                    event.start_time_s,
                    event.source_time_s,
                    event.frame_index,
                    frames_tracked,
                    event.box.x1,
                    event.box.y1,
                    event.box.x2,
                    event.box.y2,
                    str(evidence_path) if evidence_path else None,
                    detector_sha256,
                    classifier_sha256,
                    pipeline_version,
                ),
            )
        except sqlite3.IntegrityError:
            return None
        self.connection.commit()
        return int(cursor.lastrowid)

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM sightings").fetchone()
        return int(row[0]) if row else 0

    def count_by_species(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT species, COUNT(*) FROM sightings GROUP BY species ORDER BY species"
        ).fetchall()
        return {str(species): int(total) for species, total in rows}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SightingStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
