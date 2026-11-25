from __future__ import annotations

import unittest

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.energia import (
    Bateria,
    CicloEvento,
    autonomia_dias,
    energia_ciclo_por_etapas,
    tabela_sensibilidade,
)


class AutonomiaTest(unittest.TestCase):
    def test_reproduz_tabela_da_secao_7(self):
        """50 Wh, f = 1, η = 0,85 e 10 J incrementais por evento (exemplo da pesquisa)."""
        bateria = Bateria(50.0, 1.0, 0.85)
        ciclo = CicloEvento(energia_ciclo_j=10.0, duracao_s=0.0)
        tabela = tabela_sensibilidade(bateria, [0.01, 0.5, 2.5], ciclo, [10, 100, 1000])
        esperado = [
            [158.71, 82.08, 14.08],
            [3.53, 3.46, 2.88],
            [0.71, 0.71, 0.68],
        ]
        for linha, valores in zip(tabela, esperado, strict=True):
            obtidos = [linha["10_eventos"], linha["100_eventos"], linha["1000_eventos"]]
            for obtido, alvo in zip(obtidos, valores, strict=True):
                self.assertAlmostEqual(obtido, alvo, places=2)

    def test_energia_incremental_desconta_a_espera(self):
        ciclo = energia_ciclo_por_etapas([(5.0, 2.0), (1.0, 4.0)])  # 14 J em 6 s
        self.assertAlmostEqual(ciclo.incremental_j(1.0), 8.0)

    def test_ciclo_menor_que_espera_e_erro_de_ponto_eletrico(self):
        with self.assertRaises(ConfigurationError):
            CicloEvento(1.0, 10.0).incremental_j(1.0)

    def test_ciclos_que_ocupam_o_dia_exigem_modelo_por_estados(self):
        with self.assertRaises(ConfigurationError):
            autonomia_dias(Bateria(50.0), 0.1, CicloEvento(100.0, 60.0), 1000)


if __name__ == "__main__":
    unittest.main()
