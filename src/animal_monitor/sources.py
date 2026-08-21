from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    image: np.ndarray
    source_time_s: float


class FrameStream:
    """Fonte uniforme para câmera, vídeo, imagem ou diretório de imagens."""

    def __init__(self, source: str | int) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV não está instalado. Instale o projeto com as dependências de runtime."
            ) from exc
        self.cv2 = cv2
        self.source = source
        self.capture = None
        self.images: list[Path] = []
        self.fps = 0.0
        self.label = str(source)

        candidate = Path(str(source)).expanduser()
        if candidate.is_dir():
            self.images = sorted(
                path
                for path in candidate.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self.images:
                raise FileNotFoundError(f"Nenhuma imagem encontrada em: {candidate}")
            self.label = str(candidate.resolve())
            self.fps = 1.0
        elif candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            self.images = [candidate.resolve()]
            self.label = str(candidate.resolve())
            self.fps = 1.0
        else:
            capture_source: str | int
            if isinstance(source, int) or (str(source).isdigit() and not candidate.exists()):
                capture_source = int(source)
            else:
                capture_source = str(source)
            self.capture = cv2.VideoCapture(capture_source)
            if not self.capture.isOpened():
                raise RuntimeError(f"Não foi possível abrir a fonte: {source}")
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0 or self.fps > 240:
                self.fps = 30.0

    def __iter__(self) -> Iterator[Frame]:
        if self.images:
            for index, path in enumerate(self.images):
                image = self.cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Não foi possível ler a imagem: {path}")
                yield Frame(index=index, image=image, source_time_s=index / self.fps)
            return

        assert self.capture is not None
        index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            position_ms = float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC))
            source_time = position_ms / 1000.0 if position_ms > 0 else index / self.fps
            yield Frame(index=index, image=image, source_time_s=source_time)
            index += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> FrameStream:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
