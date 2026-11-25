"""Raspberry Pi AI Camera (Sony IMX500): a rede roda no sensor e chega como metadado.

Diferente dos outros backends, aqui câmera e acelerador são o mesmo dispositivo. O
`Imx500Detector` abre a câmera, guarda o resultado da rede junto de cada quadro capturado
e o entrega quando o pipeline pede `detect(quadro)`. Para o restante do código ele se
comporta como qualquer detector, e `detector.camera` substitui a câmera da versão.

Saídas do YOLOv8n exportado pelo Ultralytics (`format="imx"`, NMS embutido): caixas
`(x1, y1, x2, y2)` em pixels do tensor de entrada, escores, classes e número válido
(ordem documentada no pós-processador `pp_od_yolo_ultralytics` da Sony). As caixas são
normalizadas e convertidas para o quadro principal com `convert_inference_coords`, que já
considera o recorte do ISP.

Cuidados documentados pela Raspberry Pi: o firmware da rede pode levar minutos para
carregar na primeira vez (fica em cache), por isso a câmera fica sempre ligada; e a AI
Camera tem filtro IR, sem versão NoIR — não há foto noturna com iluminador IR.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor.types import BoundingBox, Detection
from animal_monitor_hw.camtrap_dp import agora_iso
from animal_monitor_hw.captura import QuadroCapturado


class _CameraImx500:
    """Câmera da versão IMX500: rajadas com o resultado da rede anexado a cada quadro."""

    def __init__(self, detector: Imx500Detector) -> None:
        self.detector = detector

    def iniciar(self) -> None:
        self.detector.iniciar()

    def rajada(self, quadros: int, intervalo_s: float) -> list[QuadroCapturado]:
        self.iniciar()
        picam2 = self.detector.picam2
        capturados = []
        for indice in range(quadros):
            requisicao = picam2.capture_request()
            try:
                imagem = np.ascontiguousarray(requisicao.make_array("main"))
                metadados = dict(requisicao.get_metadata())
            finally:
                requisicao.release()
            self.detector.registrar(imagem, metadados)
            capturados.append(QuadroCapturado(agora_iso(), time.monotonic(), imagem, metadados))
            if indice + 1 < quadros and intervalo_s > 0:
                time.sleep(intervalo_s)
        return capturados

    def luminosidade(self) -> float | None:
        self.iniciar()
        valor = self.detector.picam2.capture_metadata().get("Lux")
        return None if valor is None else float(valor)

    def parar(self) -> None:
        self.detector.parar()


class Imx500Detector:
    def __init__(
        self,
        rpk_path: str | Path,
        names: Mapping[int, str],
        confidence: float = 0.35,
        max_detections: int = 20,
        bbox_normalizada: bool = True,
        ordem_bbox: str = "xy",
        largura: int = 1536,
        altura: int = 864,
        fps: float = 10.0,
        preservar_proporcao: bool = True,
    ) -> None:
        self.model_path = Path(rpk_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ConfigurationError(
                f"Rede IMX500 não encontrada: {self.model_path}. Gere o network.rpk com "
                "`imx500-package` (ver README da versão)."
            )
        if ordem_bbox not in {"xy", "yx"}:
            raise ConfigurationError("'ordem_bbox' deve ser 'xy' ou 'yx'.")
        self.names = dict(names)
        self.confidence = float(confidence)
        self.max_detections = int(max_detections)
        self.bbox_normalizada = bbox_normalizada
        self.ordem_bbox = ordem_bbox
        self.tamanho = (int(largura), int(altura))
        self.fps = float(fps)
        self.preservar_proporcao = preservar_proporcao
        self.label = self.names[0] if len(self.names) == 1 else "multiclasse"
        self.imx500 = None
        self.picam2 = None
        self._resultados: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.camera = _CameraImx500(self)

    def iniciar(self) -> None:
        if self.picam2 is not None:
            return
        try:
            from picamera2 import Picamera2
            from picamera2.devices import IMX500
        except ImportError as exc:
            raise RuntimeError("Instale python3-picamera2 e imx500-all.") from exc
        # O IMX500 precisa ser instanciado antes do Picamera2 (exemplos oficiais).
        self.imx500 = IMX500(str(self.model_path))
        self.picam2 = Picamera2(self.imx500.camera_num)
        configuracao = self.picam2.create_video_configuration(
            main={"size": self.tamanho, "format": "RGB888"},
            controls={"FrameRate": self.fps},
            buffer_count=12,
        )
        self.imx500.show_network_fw_progress_bar()
        self.picam2.configure(configuracao)
        if self.preservar_proporcao:
            # O modelo foi treinado com letterbox: evita esticar o quadro até o quadrado.
            self.imx500.set_auto_aspect_ratio()
        self.picam2.start()

    def parar(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

    def registrar(self, imagem: np.ndarray, metadados: dict[str, object]) -> None:
        """Decodifica a saída da rede do quadro e guarda até o pipeline pedir."""
        assert self.imx500 is not None and self.picam2 is not None
        saidas = self.imx500.get_outputs(metadados, add_batch=True)
        if saidas is None:
            self._resultados[id(imagem)] = _vazio()
            return
        caixas = np.asarray(saidas[0][0], dtype=np.float32).reshape(-1, 4)
        escores = np.asarray(saidas[1][0], dtype=np.float32).reshape(-1)
        classes = np.rint(np.asarray(saidas[2][0], dtype=np.float32).reshape(-1)).astype(np.int64)
        if len(saidas) > 3:
            validos = int(np.asarray(saidas[3]).reshape(-1)[0])
            caixas, escores, classes = caixas[:validos], escores[:validos], classes[:validos]
        largura_entrada, altura_entrada = self.imx500.get_input_size()
        if self.bbox_normalizada:
            caixas = caixas / np.array(
                [largura_entrada, altura_entrada, largura_entrada, altura_entrada], np.float32
            )
        if self.ordem_bbox == "xy":
            caixas = caixas[:, [1, 0, 3, 2]]  # convert_inference_coords espera (y0, x0, y1, x1)
        manter = (escores >= self.confidence) & (classes >= 0) & (classes < len(self.names))
        ordem = np.argsort(escores[manter])[::-1][: self.max_detections]
        caixas, escores, classes = (
            caixas[manter][ordem], escores[manter][ordem], classes[manter][ordem]
        )
        pixels = []
        for caixa in caixas:
            x, y, w, h = self.imx500.convert_inference_coords(tuple(caixa), metadados, self.picam2)
            pixels.append([x, y, x + w, y + h])
        self._resultados[id(imagem)] = (
            np.asarray(pixels, dtype=np.float32).reshape(-1, 4), escores, classes
        )

    def detect_all(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resultado = self._resultados.pop(id(frame), None)
        if resultado is None:
            # Quadro que não veio da câmera IMX500 (ex.: --simular): não há metadado de rede.
            raise RuntimeError(
                "Quadro sem resultado do IMX500. A versão IMX500 não roda com --simular: "
                "use a versão ONNX equivalente para bancada."
            )
        return resultado

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        caixas, escores, _ = self.detect_all(frame)
        return caixas, escores

    def infer(self, frame: np.ndarray) -> list[Detection]:
        caixas, escores, classes = self.detect_all(frame)
        return [
            Detection(int(classe), self.names.get(int(classe), f"classe_{classe}"), float(escore),
                      BoundingBox(*(float(valor) for valor in caixa)))
            for caixa, escore, classe in zip(caixas, escores, classes, strict=True)
        ]

    def info(self) -> dict[str, object]:
        return {"runtime": "imx500", "model": str(self.model_path), "classes": self.names,
                "confidence": self.confidence}


def _vazio() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0,), np.int64))
