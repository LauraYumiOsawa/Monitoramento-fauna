from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from animal_monitor.types import Detection


class DetectionGate:
    """Reduz falsos alertas e evita gravar o mesmo animal a cada quadro."""

    def __init__(self, min_hits: int, cooldown_seconds: float) -> None:
        self.min_hits = min_hits
        self.cooldown_seconds = cooldown_seconds
        self._hits: Counter[int] = Counter()
        self._last_emission: dict[int, float] = {}

    def update(self, detections: Iterable[Detection], source_time_s: float) -> list[Detection]:
        best_by_class: dict[int, Detection] = {}
        for detection in detections:
            current = best_by_class.get(detection.class_id)
            if current is None or detection.confidence > current.confidence:
                best_by_class[detection.class_id] = detection

        present = set(best_by_class)
        for class_id in list(self._hits):
            if class_id not in present:
                self._hits[class_id] = 0

        emitted: list[Detection] = []
        for class_id, detection in best_by_class.items():
            self._hits[class_id] += 1
            last = self._last_emission.get(class_id, float("-inf"))
            enough_hits = self._hits[class_id] >= self.min_hits
            cooldown_finished = source_time_s - last >= self.cooldown_seconds
            if enough_hits and cooldown_finished:
                emitted.append(detection)
                self._last_emission[class_id] = source_time_s
        return emitted


class EventStore:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_time_s REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                evidence_path TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_observed_at ON detections(observed_at)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_class_name ON detections(class_name)"
        )
        self.connection.commit()

    def add(
        self,
        detection: Detection,
        source: str,
        source_time_s: float,
        frame_index: int,
        evidence_path: str | Path | None,
    ) -> int:
        observed_at = datetime.now(timezone.utc).isoformat()
        cursor = self.connection.execute(
            """
            INSERT INTO detections (
                observed_at, source, source_time_s, frame_index,
                class_id, class_name, confidence, x1, y1, x2, y2, evidence_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observed_at,
                source,
                source_time_s,
                frame_index,
                detection.class_id,
                detection.class_name,
                detection.confidence,
                detection.box.x1,
                detection.box.y1,
                detection.box.x2,
                detection.box.y2,
                str(evidence_path) if evidence_path else None,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM detections").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
