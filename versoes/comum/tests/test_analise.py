from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from animal_monitor.types import BoundingBox, Detection
from animal_monitor_b.config import RuntimeConfigB, TrackerConfig
from animal_monitor_hw.analise import AnalisadorModeloA, AnalisadorModeloB
from animal_monitor_hw.captura import QuadroCapturado

LABELS = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]
CAIXA = np.array([[100.0, 80.0, 300.0, 260.0]], dtype=np.float32)


def _rajada(quadros: int) -> list[QuadroCapturado]:
    imagem = np.zeros((360, 480, 3), dtype=np.uint8)
    return [
        QuadroCapturado(f"2026-10-20T10:00:0{indice}-03:00", 0.1 * indice, imagem)
        for indice in range(quadros)
    ]


class DetectorA:
    def __init__(self, por_quadro: list[list[str]]):
        self.por_quadro = por_quadro
        self.chamadas = 0

    def infer(self, frame):
        nomes = self.por_quadro[min(self.chamadas, len(self.por_quadro) - 1)]
        self.chamadas += 1
        return [
            Detection(LABELS.index(nome), nome, 0.8, BoundingBox(*CAIXA[0].tolist()))
            for nome in nomes
        ]


class DetectorB:
    model_path = Path("fake.onnx")

    def __init__(self, presente: list[bool]):
        self.presente = presente
        self.chamadas = 0

    def detect(self, frame):
        ok = self.presente[min(self.chamadas, len(self.presente) - 1)]
        self.chamadas += 1
        if not ok:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return CAIXA.copy(), np.array([0.9], np.float32)


class ClassificadorB:
    model_path = Path("fake_cls.onnx")
    labels = LABELS

    def __init__(self, rotulo: str):
        self.rotulo = rotulo

    def classify(self, frame, boxes):
        linha = np.full(len(LABELS), 0.02, dtype=np.float32)
        linha[LABELS.index(self.rotulo)] = 0.94
        return np.tile(linha, (np.asarray(boxes).reshape(-1, 4).shape[0], 1))


def _config_armadilha() -> RuntimeConfigB:
    # Valores de rajada usados nas versões: votação possível em 8 quadros.
    return RuntimeConfigB(
        classify_every=1, min_votes=3, min_vote_share=0.65,
        tracker=TrackerConfig(max_age=10, min_hits=2),
    )


class ModeloATest(unittest.TestCase):
    def test_especie_confirmada_apos_min_hits(self):
        resultado = AnalisadorModeloA(DetectorA([["lobo_guara"]] * 5), 3).analisar(_rajada(5))
        self.assertEqual(resultado.tipo, "animal")
        self.assertEqual([item.rotulo for item in resultado.avistamentos], ["lobo_guara"])

    def test_deteccao_sem_confirmacao_e_inconclusiva(self):
        resultado = AnalisadorModeloA(DetectorA([["anta"], [], [], []]), 3).analisar(_rajada(4))
        self.assertEqual(resultado.tipo, "unknown")
        self.assertEqual(resultado.avistamentos[0].tipo, "unknown")

    def test_sem_deteccao_e_vazio(self):
        resultado = AnalisadorModeloA(DetectorA([[]]), 3).analisar(_rajada(4))
        self.assertEqual(resultado.tipo, "blank")


class ModeloBTest(unittest.TestCase):
    def test_votacao_confirma_especie(self):
        analisador = AnalisadorModeloB(DetectorB([True] * 8), ClassificadorB("anta"),
                                       _config_armadilha())
        resultado = analisador.analisar(_rajada(8))
        self.assertEqual(resultado.tipo, "animal")
        self.assertEqual(resultado.avistamentos[0].rotulo, "anta")
        self.assertGreater(resultado.latencia_ms["recortes"], 0)

    def test_classe_desconhecido_nao_vira_especie_nem_vazio(self):
        analisador = AnalisadorModeloB(DetectorB([True] * 8), ClassificadorB("desconhecido"),
                                       _config_armadilha())
        resultado = analisador.analisar(_rajada(8))
        self.assertEqual(resultado.tipo, "unknown")

    def test_rajada_sem_animal_e_vazia(self):
        analisador = AnalisadorModeloB(DetectorB([False]), ClassificadorB("anta"),
                                       _config_armadilha())
        self.assertEqual(analisador.analisar(_rajada(6)).tipo, "blank")

    def test_rastreador_reinicia_a_cada_evento(self):
        analisador = AnalisadorModeloB(DetectorB([True] * 16), ClassificadorB("lobo_guara"),
                                       _config_armadilha())
        primeiro = analisador.analisar(_rajada(8))
        segundo = analisador.analisar(_rajada(8))
        self.assertEqual(len(primeiro.avistamentos), 1)
        self.assertEqual(len(segundo.avistamentos), 1)


if __name__ == "__main__":
    unittest.main()
