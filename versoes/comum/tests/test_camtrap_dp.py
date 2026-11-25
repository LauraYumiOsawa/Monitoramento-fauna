from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.camtrap_dp import COLUNAS, Implantacao, RegistroCamtrap


class CamtrapDpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _registro(self) -> RegistroCamtrap:
        registro = RegistroCamtrap(self.raiz / "camtrap.sqlite3")
        registro.registrar_implantacao(
            Implantacao("dep-1", -22.2, -49.9, "2026-10-20T10:00:00-03:00", "cam01", "Pi 5")
        )
        return registro

    def test_pacote_tem_tres_tabelas_e_perfil(self):
        with self._registro() as registro:
            midia = registro.registrar_midia("dep-1", "midias/e1/000.jpg",
                                             "2026-10-20T10:05:00-03:00")
            registro.registrar_observacao(
                "dep-1", "e1", "2026-10-20T10:05:00-03:00", "2026-10-20T10:05:02-03:00",
                "animal", "teste", rotulo="lobo_guara", probabilidade=0.91, quantidade=1,
                caixa_normalizada=(0.1, 0.2, 0.3, 0.4),
            )
            registro.registrar_observacao(
                "dep-1", "e2", "2026-10-20T11:00:00-03:00", "2026-10-20T11:00:02-03:00",
                "blank", "teste",
            )
            caminho = registro.exportar(self.raiz / "pacote", "Teste", "Autora")
        pacote = json.loads(caminho.read_text(encoding="utf-8"))
        self.assertIn("camtrap-dp-profile.json", pacote["profile"])
        self.assertEqual([r["name"] for r in pacote["resources"]],
                         ["deployments", "media", "observations"])
        self.assertEqual(pacote["taxonomic"][0]["scientificName"], "Chrysocyon brachyurus")
        with (caminho.parent / "observations.csv").open(encoding="utf-8") as arquivo:
            linhas = list(csv.DictReader(arquivo))
        self.assertEqual(list(linhas[0].keys()), COLUNAS["observations"])
        tipos = {linha["observationType"] for linha in linhas}
        self.assertEqual(tipos, {"animal", "blank"})
        animal = next(linha for linha in linhas if linha["observationType"] == "animal")
        self.assertEqual(animal["scientificName"], "Chrysocyon brachyurus")
        self.assertEqual(animal["classificationMethod"], "machine")
        self.assertEqual(animal["bboxWidth"], "0.3")
        self.assertTrue(midia.startswith("m-"))

    def test_rotulo_fora_do_mapa_vira_unknown(self):
        with self._registro() as registro:
            registro.registrar_observacao(
                "dep-1", "e1", "2026-10-20T10:05:00-03:00", "2026-10-20T10:05:02-03:00",
                "animal", "teste", rotulo="desconhecido",
            )
            linha = registro.linhas("observations")[0]
        self.assertEqual(linha["observationType"], "unknown")
        self.assertIsNone(linha["scientificName"])

    def test_tipo_invalido(self):
        with self._registro() as registro, self.assertRaises(ConfigurationError):
            registro.registrar_observacao("dep-1", "e1", "a", "b", "vazio", "teste")


if __name__ == "__main__":
    unittest.main()
