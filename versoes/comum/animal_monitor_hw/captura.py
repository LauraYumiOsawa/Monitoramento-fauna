"""Captura de rajadas após o disparo: Picamera2, OpenCV/GStreamer ou arquivos (bancada).

Uma armadilha fotográfica não analisa vídeo contínuo: o PIR dispara, a câmera registra
uma rajada curta e o sistema volta a esperar. A rajada é a unidade de análise — é sobre
ela que o Modelo B rastreia e vota, e é ela que vira um `event` no Camtrap DP.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.camtrap_dp import agora_iso


@dataclass(slots=True)
class QuadroCapturado:
    timestamp: str
    instante_s: float
    imagem: np.ndarray
    metadados: dict[str, object] = field(default_factory=dict)


class Camera(Protocol):
    def iniciar(self) -> None: ...

    def rajada(self, quadros: int, intervalo_s: float) -> list[QuadroCapturado]: ...

    def luminosidade(self) -> float | None: ...

    def parar(self) -> None: ...


class CameraPicamera2:
    """Câmeras CSI da Raspberry Pi (Camera Module 3/NoIR, Global Shutter, IMX462...).

    `RGB888` no Picamera2 entrega os bytes na ordem B, G, R — o que o OpenCV espera.
    Foco manual (`posicao_lente` em dioptrias = 1 / distância em metros) evita a busca do
    autofoco a cada disparo, que somaria tempo ao disparo.
    """

    def __init__(
        self,
        largura: int = 1536,
        altura: int = 864,
        fps: float = 10.0,
        indice: int = 0,
        posicao_lente: float | None = None,
        controles: dict[str, object] | None = None,
        manter_ligada: bool = True,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 não instalado (sudo apt install python3-picamera2)."
            ) from exc
        self.picam2 = Picamera2(indice)
        self.tamanho = (int(largura), int(altura))
        self.fps = float(fps)
        self.posicao_lente = posicao_lente
        self.controles = dict(controles or {})
        self.manter_ligada = manter_ligada
        self._ligada = False

    def iniciar(self) -> None:
        if self._ligada:
            return
        configuracao = self.picam2.create_video_configuration(
            main={"size": self.tamanho, "format": "RGB888"},
            controls={"FrameRate": self.fps},
            buffer_count=4,
        )
        self.picam2.configure(configuracao)
        controles = dict(self.controles)
        if self.posicao_lente is not None:
            from libcamera import controls

            controles.update(
                {"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(self.posicao_lente)}
            )
        if controles:
            self.picam2.set_controls(controles)
        self.picam2.start()
        self._ligada = True

    def rajada(self, quadros: int, intervalo_s: float) -> list[QuadroCapturado]:
        self.iniciar()
        capturados = []
        for indice in range(quadros):
            requisicao = self.picam2.capture_request()
            try:
                imagem = requisicao.make_array("main")
                metadados = dict(requisicao.get_metadata())
            finally:
                requisicao.release()
            capturados.append(
                QuadroCapturado(agora_iso(), time.monotonic(), np.ascontiguousarray(imagem),
                                metadados)
            )
            if indice + 1 < quadros and intervalo_s > 1.0 / self.fps:
                time.sleep(intervalo_s - 1.0 / self.fps)
        if not self.manter_ligada:
            self.parar()
        return capturados

    def luminosidade(self) -> float | None:
        """Estimativa de lux do próprio pipeline da câmera (metadado `Lux` do libcamera)."""
        self.iniciar()
        valor = self.picam2.capture_metadata().get("Lux")
        return None if valor is None else float(valor)

    def parar(self) -> None:
        if self._ligada:
            self.picam2.stop()
            self._ligada = False


class CameraOpenCV:
    """USB/UVC, V4L2 ou pipeline GStreamer (CSI no Jetson e no Orange Pi 5)."""

    def __init__(self, dispositivo: int | str = 0, largura: int | None = None,
                 altura: int | None = None) -> None:
        import cv2

        self.cv2 = cv2
        self.dispositivo = dispositivo
        self.largura = largura
        self.altura = altura
        self.captura = None

    def iniciar(self) -> None:
        if self.captura is not None:
            return
        if isinstance(self.dispositivo, str) and "!" in self.dispositivo:
            self.captura = self.cv2.VideoCapture(self.dispositivo, self.cv2.CAP_GSTREAMER)
        else:
            self.captura = self.cv2.VideoCapture(self.dispositivo)
            if self.largura and self.altura:
                self.captura.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.largura)
                self.captura.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.altura)
        if not self.captura.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera: {self.dispositivo}")

    def rajada(self, quadros: int, intervalo_s: float) -> list[QuadroCapturado]:
        self.iniciar()
        assert self.captura is not None
        capturados = []
        for indice in range(quadros):
            ok, imagem = self.captura.read()
            if not ok:
                break
            capturados.append(QuadroCapturado(agora_iso(), time.monotonic(), imagem))
            if indice + 1 < quadros and intervalo_s > 0:
                time.sleep(intervalo_s)
        return capturados

    def luminosidade(self) -> float | None:
        return None

    def parar(self) -> None:
        if self.captura is not None:
            self.captura.release()
            self.captura = None


class CameraArquivos:
    """Simula a câmera com imagens de uma pasta ou quadros de um vídeo (bancada/CI)."""

    def __init__(self, origem: str | Path) -> None:
        import cv2

        self.cv2 = cv2
        self.origem = Path(origem).expanduser().resolve()
        self._iterador: Iterator[np.ndarray] | None = None

    def _quadros(self) -> Iterator[np.ndarray]:
        if self.origem.is_dir():
            arquivos = sorted(
                caminho for caminho in self.origem.rglob("*")
                if caminho.suffix.lower() in IMAGE_SUFFIXES
            )
            if not arquivos:
                raise ConfigurationError(f"Nenhuma imagem em {self.origem}.")
            while True:
                for caminho in arquivos:
                    imagem = self.cv2.imread(str(caminho))
                    if imagem is not None:
                        yield imagem
        captura = self.cv2.VideoCapture(str(self.origem))
        if not captura.isOpened():
            raise RuntimeError(f"Não foi possível abrir: {self.origem}")
        while True:
            ok, imagem = captura.read()
            if not ok:
                captura.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield imagem

    def iniciar(self) -> None:
        if self._iterador is None:
            self._iterador = self._quadros()

    def rajada(self, quadros: int, intervalo_s: float) -> list[QuadroCapturado]:
        self.iniciar()
        assert self._iterador is not None
        return [
            QuadroCapturado(agora_iso(), time.monotonic(), next(self._iterador))
            for _ in range(quadros)
        ]

    def luminosidade(self) -> float | None:
        return None

    def parar(self) -> None:
        self._iterador = None


def criar_camera(config: dict[str, object] | None, origem_simulada: str | None = None) -> Camera:
    config = config or {}
    if origem_simulada:
        return CameraArquivos(origem_simulada)
    tipo = str(config.get("tipo", "picamera2"))
    if tipo == "picamera2":
        return CameraPicamera2(
            largura=int(config.get("largura", 1536)),
            altura=int(config.get("altura", 864)),
            fps=float(config.get("fps", 10.0)),
            indice=int(config.get("indice", 0)),
            posicao_lente=config.get("posicao_lente"),
            controles=config.get("controles"),
            manter_ligada=bool(config.get("manter_ligada", True)),
        )
    if tipo == "opencv":
        dispositivo = config.get("dispositivo", 0)
        return CameraOpenCV(
            int(dispositivo) if str(dispositivo).isdigit() else str(dispositivo),
            config.get("largura"),
            config.get("altura"),
        )
    if tipo == "arquivos":
        return CameraArquivos(str(config["origem"]))
    raise ConfigurationError(f"Tipo de câmera desconhecido: {tipo}")
