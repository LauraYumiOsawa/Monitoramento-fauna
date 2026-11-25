"""Backends de SDK proprietário exercitados com módulos falsos que imitam a API documentada.

Não substituem o teste no hardware: verificam a cola deste projeto (formato de entrada,
layout, decodificação e conversão de coordenadas) contra a forma das saídas descrita na
documentação de cada SDK.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from animal_monitor_hw.models import YoloDetector


@contextmanager
def _modulos(**modulos):
    anteriores = {nome: sys.modules.get(nome) for nome in modulos}
    sys.modules.update(modulos)
    try:
        yield
    finally:
        for nome, anterior in anteriores.items():
            if anterior is None:
                sys.modules.pop(nome, None)
            else:
                sys.modules[nome] = anterior


_PASTA = tempfile.TemporaryDirectory()  # removida ao fim do processo


def _arquivo(sufixo: str) -> Path:
    """Artefato vazio: os runners só conferem a existência antes de chamar o SDK."""
    caminho = Path(_PASTA.name) / f"modelo{sufixo}"
    caminho.touch()
    return caminho


class _Hailo:
    """hailo_platform mínimo: VDevice → InferModel → ConfiguredInferModel → Bindings."""

    def __init__(self, por_classe):
        self.por_classe = por_classe
        self.recebido = None
        modulo = types.ModuleType("hailo_platform")
        modulo.FormatType = types.SimpleNamespace(UINT8="u8", FLOAT32="f32")
        modulo.HailoSchedulingAlgorithm = types.SimpleNamespace(ROUND_ROBIN="rr")
        externo = self

        class Saida:
            name = "yolov8n/yolov8_nms_postprocess"
            shape = (1,)

            def set_format_type(self, tipo):
                self.tipo = tipo

            def set_nms_score_threshold(self, valor):
                externo.limiar = valor

            def set_nms_iou_threshold(self, valor):
                externo.iou = valor

        class Entrada:
            shape = (416, 416, 3)

            def set_format_type(self, tipo):
                self.tipo = tipo

        class Ligacao:
            def __init__(self):
                self._entrada = types.SimpleNamespace(set_buffer=self._guardar)

            def _guardar(self, buffer):
                externo.recebido = buffer

            def input(self):
                return self._entrada

            def output(self, nome=None):
                return types.SimpleNamespace(get_buffer=lambda: externo.por_classe)

        class Configurado:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def create_bindings(self, output_buffers=None):
                return Ligacao()

            def run(self, ligacoes, timeout):
                externo.rodou = True

        class InferModel:
            outputs = [Saida()]

            def set_batch_size(self, tamanho):
                self.lote = tamanho

            def input(self):
                return Entrada()

            def output(self, nome=None):
                return self.outputs[0]

            def configure(self):
                return Configurado()

        class VDevice:
            @staticmethod
            def create_params():
                return types.SimpleNamespace()

            def __init__(self, parametros):
                self.parametros = parametros

            def create_infer_model(self, caminho):
                return InferModel()

            def release(self):
                return None

        modulo.VDevice = VDevice
        self.modulo = modulo


class HailoTest(unittest.TestCase):
    def test_nms_por_classe_vira_caixa_no_quadro(self):
        # Uma detecção da classe 0 cobrindo o centro da entrada: y1 x1 y2 x2 score.
        falso = _Hailo([np.array([[0.25, 0.25, 0.75, 0.75, 0.9]], dtype=np.float32)])
        with _modulos(hailo_platform=falso.modulo):
            from animal_monitor_hw.backends.hailo_backend import HailoRunner

            runner = HailoRunner(_arquivo(".hef"), score_threshold=0.3, iou_threshold=0.4)
            detector = YoloDetector(runner, {0: "animal"}, 0.3, 0.4, output_format="hailo_nms")
            quadro = np.zeros((832, 832, 3), dtype=np.uint8)  # escala 0,5 sem preenchimento
            caixas, escores = detector.detect(quadro)
        self.assertEqual(falso.recebido.shape, (416, 416, 3))
        self.assertEqual(falso.recebido.dtype, np.uint8)
        self.assertAlmostEqual(falso.limiar, 0.3)
        np.testing.assert_allclose(caixas, [[208, 208, 624, 624]], atol=1e-3)
        np.testing.assert_allclose(escores, [0.9])


class RknnTest(unittest.TestCase):
    def test_saidas_divididas_nchw_da_npu(self):
        recebido = {}

        class RKNNLite:
            NPU_CORE_AUTO = 0
            NPU_CORE_0 = 1

            def __init__(self, verbose=False):
                pass

            def load_rknn(self, caminho):
                return 0

            def init_runtime(self, core_mask=0):
                recebido["nucleo"] = core_mask
                return 0

            def inference(self, inputs, data_format=None):
                recebido["entrada"] = inputs[0]
                saidas = []
                for lado in (40, 20, 10):  # entrada 320: passos 8, 16, 32
                    caixa = np.full((1, 64, lado, lado), -20.0, np.float32)
                    cls = np.full((1, 3, lado, lado), 0.01, np.float32)
                    if lado == 20:
                        for indice in range(4):
                            caixa[0, indice * 16 + 2, 10, 10] = 20.0  # distância 2 em cada lado
                        cls[0, 1, 10, 10] = 0.8
                    saidas += [caixa, cls, cls.sum(axis=1, keepdims=True)]
                return saidas

            def release(self):
                return None

        api = types.ModuleType("rknnlite.api")
        api.RKNNLite = RKNNLite
        pacote = types.ModuleType("rknnlite")
        pacote.api = api
        with _modulos(rknnlite=pacote, **{"rknnlite.api": api}):
            from animal_monitor_hw.backends.rknn_backend import RknnLiteRunner

            runner = RknnLiteRunner(_arquivo(".rknn"), nucleos="0", entrada=[320, 320])
            detector = YoloDetector(runner, {0: "anta", 1: "tamandua_bandeira", 2: "lobo_guara"},
                                    0.35, 0.45, output_format="split")
            deteccoes = detector.infer(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(recebido["entrada"].shape, (1, 320, 320, 3))
        self.assertEqual(recebido["nucleo"], 1)
        self.assertEqual([d.class_name for d in deteccoes], ["tamandua_bandeira"])
        caixa = deteccoes[0].box
        # célula (10, 10) na grade 20×20: âncora 10,5 → (8,5 a 12,5) × passo 16
        np.testing.assert_allclose([caixa.x1, caixa.y1, caixa.x2, caixa.y2],
                                   [136, 136, 200, 200], atol=1e-3)


class Imx500Test(unittest.TestCase):
    def test_saida_ultralytics_convertida_para_o_quadro(self):
        chamadas = {}

        class IMX500:
            camera_num = 0

            def __init__(self, caminho):
                pass

            def get_outputs(self, metadados, add_batch=False):
                # NMS embutido: caixas x1 y1 x2 y2 em pixels do tensor 640, escores, classes, n
                return [np.array([[[64, 128, 320, 384], [0, 0, 10, 10]]], np.float32),
                        np.array([[0.8, 0.1]], np.float32), np.array([[2, 0]], np.float32),
                        np.array([2])]

            def get_input_size(self):
                return (640, 640)

            def convert_inference_coords(self, coords, metadados, picam2):
                chamadas["coords"] = coords
                y0, x0, y1, x1 = coords
                return (x0 * 1536, y0 * 864, (x1 - x0) * 1536, (y1 - y0) * 864)

        devices = types.ModuleType("picamera2.devices")
        devices.IMX500 = IMX500
        picamera2 = types.ModuleType("picamera2")
        picamera2.Picamera2 = object
        picamera2.devices = devices
        with _modulos(picamera2=picamera2, **{"picamera2.devices": devices}):
            from animal_monitor_hw.backends.imx500_backend import Imx500Detector

            detector = Imx500Detector(_arquivo(".rpk"), {0: "anta", 1: "tamandua_bandeira",
                                                          2: "lobo_guara"}, confidence=0.35)
            detector.imx500 = IMX500("x")
            detector.picam2 = object()
            quadro = np.zeros((864, 1536, 3), dtype=np.uint8)
            detector.registrar(quadro, {})
            deteccoes = detector.infer(quadro)
        self.assertEqual(len(deteccoes), 1)  # a segunda caixa está abaixo do limiar
        self.assertEqual(deteccoes[0].class_name, "lobo_guara")
        # normalizada e reordenada para (y0, x0, y1, x1) antes da conversão do Picamera2
        np.testing.assert_allclose(chamadas["coords"], (0.2, 0.1, 0.6, 0.5))
        caixa = deteccoes[0].box
        np.testing.assert_allclose([caixa.x1, caixa.y1, caixa.x2, caixa.y2],
                                   [153.6, 172.8, 768.0, 518.4], rtol=1e-5)
        with self.assertRaises(RuntimeError):
            detector.infer(np.zeros((10, 10, 3), dtype=np.uint8))  # quadro sem metadado


if __name__ == "__main__":
    unittest.main()
