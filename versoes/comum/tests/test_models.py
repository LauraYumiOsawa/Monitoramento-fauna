from __future__ import annotations

import unittest

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.backends.base import TensorSpec, rgb_to_tensor
from animal_monitor_hw.models import CropClassifier, YoloDetector


class RunnerFalso:
    """Devolve saídas programadas e guarda o último tensor recebido."""

    def __init__(self, spec: TensorSpec, saidas):
        self.spec = spec
        self.saidas = saidas
        self.ultimo = None

    def run(self, tensor):
        self.ultimo = tensor
        return self.saidas(tensor) if callable(self.saidas) else self.saidas

    def describe(self):
        return {"runtime": "falso", "model": "falso.bin"}


class TensorTest(unittest.TestCase):
    def test_formatos_de_entrada(self):
        rgb = np.full((4, 6, 3), 255, dtype=np.uint8)
        nchw = rgb_to_tensor(rgb, TensorSpec(4, 6))
        self.assertEqual(nchw.shape, (1, 3, 4, 6))
        self.assertAlmostEqual(float(nchw.max()), 1.0)
        nhwc = rgb_to_tensor(rgb, TensorSpec(4, 6, layout="nhwc", dtype="uint8"))
        self.assertEqual(nhwc.shape, (1, 4, 6, 3))
        self.assertEqual(nhwc.dtype, np.uint8)


class YoloDetectorTest(unittest.TestCase):
    def test_saida_padrao_volta_para_pixels_do_quadro(self):
        # Uma caixa centrada em (32, 32), 16x16, na entrada 64x64; classe 1 com 0.9.
        saida = np.zeros((1, 4 + 2, 3), dtype=np.float32)
        saida[0, :, 0] = [32, 32, 16, 16, 0.1, 0.9]
        runner = RunnerFalso(TensorSpec(64, 64), [saida])
        detector = YoloDetector(runner, {0: "anta", 1: "lobo_guara"}, 0.5, 0.45)
        quadro = np.zeros((128, 128, 3), dtype=np.uint8)  # escala 0.5, sem preenchimento
        deteccoes = detector.infer(quadro)
        self.assertEqual(len(deteccoes), 1)
        self.assertEqual(deteccoes[0].class_name, "lobo_guara")
        caixa = deteccoes[0].box
        np.testing.assert_allclose([caixa.x1, caixa.y1, caixa.x2, caixa.y2], [48, 48, 80, 80])
        caixas, escores = detector.detect(quadro)
        self.assertEqual(caixas.shape, (1, 4))

    def test_formato_invalido(self):
        with self.assertRaises(ConfigurationError):
            YoloDetector(RunnerFalso(TensorSpec(64, 64), []), {0: "a"}, 0.5, 0.4,
                         output_format="desconhecido")


class CropClassifierTest(unittest.TestCase):
    def test_softmax_por_recorte_e_uint8_sem_normalizacao(self):
        logits = np.array([[0.0, 5.0, 0.0, 0.0]], dtype=np.float32)
        runner = RunnerFalso(TensorSpec(32, 32, layout="nhwc", dtype="uint8"), [logits])
        classificador = CropClassifier(runner, ["anta", "tamandua_bandeira", "lobo_guara",
                                                "desconhecido"])
        quadro = np.full((100, 100, 3), 200, dtype=np.uint8)
        probabilidades = classificador.classify(quadro, np.array([[10, 10, 60, 60],
                                                                  [0, 0, 99, 99]]))
        self.assertEqual(probabilidades.shape, (2, 4))
        np.testing.assert_allclose(probabilidades.sum(axis=1), [1.0, 1.0], rtol=1e-6)
        self.assertEqual(int(np.argmax(probabilidades[0])), 1)
        self.assertEqual(runner.ultimo.dtype, np.uint8)
        self.assertEqual(runner.ultimo.shape, (1, 32, 32, 3))


if __name__ == "__main__":
    unittest.main()
