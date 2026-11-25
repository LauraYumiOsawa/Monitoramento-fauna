from __future__ import annotations

import unittest

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.yolo_heads import (
    REG_MAX,
    decode_boxes_scores_classes,
    decode_hailo_nms_by_class,
    decode_split_head,
)

INPUT = (64, 64)  # grades 8x8, 4x4 e 2x2 com passos 8, 16 e 32


def _mapas(num_classes: int, celula: tuple[int, int, int], classe: int,
           distancias: tuple[float, float, float, float], escore: float = 0.9):
    """Saídas divididas sintéticas com um único objeto em uma célula de uma escala."""
    saidas = []
    for escala, lado in enumerate((8, 4, 2)):
        caixa = np.full((1, 4 * REG_MAX, lado, lado), -20.0, dtype=np.float32)
        cls = np.full((1, num_classes, lado, lado), 0.01, dtype=np.float32)
        if escala == celula[0]:
            _, linha, coluna = celula
            # Logit alto exatamente no bin da distância: a esperança da DFL vira o próprio bin.
            for lado_caixa, distancia in enumerate(distancias):
                caixa[0, lado_caixa * REG_MAX + int(distancia), linha, coluna] = 20.0
            cls[0, classe, linha, coluna] = escore
        saidas.extend([caixa, cls])
    return saidas


class DecodeSplitHeadTest(unittest.TestCase):
    def test_reconstroi_caixa_na_escala_certa(self):
        saidas = _mapas(3, (1, 2, 1), classe=2, distancias=(1, 2, 3, 1))
        caixas, escores, classes = decode_split_head(saidas, INPUT, 3, 0.25, 0.45)
        self.assertEqual(caixas.shape, (1, 4))
        # escala 1: passo 16; âncora na célula (linha 2, coluna 1) = (1.5, 2.5) em grade.
        np.testing.assert_allclose(caixas[0], [8.0, 8.0, 72.0, 56.0], atol=1e-3)
        self.assertAlmostEqual(float(escores[0]), 0.9, places=5)
        self.assertEqual(int(classes[0]), 2)

    def test_layout_nhwc_equivale_a_nchw(self):
        saidas = _mapas(1, (0, 3, 3), classe=0, distancias=(2, 2, 2, 2))
        nhwc = [np.transpose(saida, (0, 2, 3, 1)) for saida in saidas]
        a = decode_split_head(saidas, INPUT, 1, 0.25, 0.45)
        b = decode_split_head(nhwc, INPUT, 1, 0.25, 0.45, layout="nhwc")
        np.testing.assert_allclose(a[0], b[0])

    def test_aceita_nove_saidas_do_rknn_model_zoo(self):
        saidas = _mapas(3, (2, 0, 0), classe=1, distancias=(0, 0, 1, 1))
        com_soma = []
        for indice in range(0, len(saidas), 2):
            caixa, cls = saidas[indice], saidas[indice + 1]
            com_soma.extend([caixa, cls, np.clip(cls.sum(axis=1, keepdims=True), 0, 1)])
        caixas, _, classes = decode_split_head(com_soma, INPUT, 3, 0.25, 0.45)
        self.assertEqual(caixas.shape[0], 1)
        self.assertEqual(int(classes[0]), 1)

    def test_sem_objeto_devolve_vazio(self):
        saidas = _mapas(3, (0, 0, 0), classe=0, distancias=(1, 1, 1, 1), escore=0.1)
        caixas, escores, classes = decode_split_head(saidas, INPUT, 3, 0.25, 0.45)
        self.assertEqual(caixas.shape, (0, 4))
        self.assertEqual(escores.size, 0)
        self.assertEqual(classes.size, 0)

    def test_layout_invalido(self):
        with self.assertRaises(ConfigurationError):
            decode_split_head(_mapas(1, (0, 0, 0), 0, (1, 1, 1, 1)), INPUT, 1, 0.2, 0.4,
                              layout="hwc")


class DecodeAcceleratorOutputsTest(unittest.TestCase):
    def test_nms_por_classe_do_hailo(self):
        por_classe = [
            np.zeros((0, 5), dtype=np.float32),
            np.array([[0.25, 0.5, 0.75, 1.0, 0.8], [0.0, 0.0, 0.1, 0.1, 0.1]], np.float32),
        ]
        caixas, escores, classes = decode_hailo_nms_by_class(por_classe, (100, 200), 0.3)
        np.testing.assert_allclose(caixas, [[100.0, 25.0, 200.0, 75.0]])
        np.testing.assert_allclose(escores, [0.8])
        self.assertEqual(classes.tolist(), [1])

    def test_caixas_normalizadas_em_ordem_yx(self):
        caixas, escores, classes = decode_boxes_scores_classes(
            np.array([[0.1, 0.2, 0.5, 0.6]]), np.array([0.7]), np.array([2.0]),
            (100, 100), 0.5, normalized=True, order="yx",
        )
        np.testing.assert_allclose(caixas, [[20.0, 10.0, 60.0, 50.0]])
        self.assertEqual(classes.tolist(), [2])
        self.assertAlmostEqual(float(escores[0]), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
