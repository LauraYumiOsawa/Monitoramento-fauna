"""Testes da lógica que roda fora do Linux: importação do CSV e app MaixPy (com `maix` falso)."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path

from animal_monitor_hw.camtrap_dp import Implantacao, RegistroCamtrap, importar_csv_mcu

APP_MAIXPY = (
    Path(__file__).resolve().parents[2] / "gerador" / "plataformas" / "maixpy" / "app" / "main.py"
)


class ImportarCsvTest(unittest.TestCase):
    def test_um_evento_por_desfecho(self):
        linhas = [
            ("e1", "anta", "0.91"), ("e1", "anta", "0.88"),
            ("e2", "inconclusivo", "0.40"),
            ("e3", "vazio", "0"),
            ("e4", "lobo_guara", "0.70"), ("e4", "desconhecido", "0.55"),
            ("e4", "lobo_guara", "0.80"),
            ("e5", "", ""),
        ]
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "registros.csv"
            with caminho.open("w", newline="", encoding="utf-8") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(["evento", "timestamp_iso", "arquivo", "rotulo", "probabilidade"])
                for indice, (evento, rotulo, prob) in enumerate(linhas):
                    escritor.writerow([evento, "2026-10-20T10:00:00-03:00", f"m/{indice}.jpg",
                                       rotulo, prob])
            with RegistroCamtrap(Path(pasta) / "r.sqlite3") as registro:
                registro.registrar_implantacao(
                    Implantacao("dep-x", 0.0, 0.0, "2026-10-20T09:00:00-03:00", "x", "teste")
                )
                contagem = importar_csv_mcu(registro, caminho, "dep-x", Path(pasta), "teste")
                observacoes = {linha["eventID"]: linha for linha in registro.linhas("observations")}
                midias = registro.linhas("media")
        self.assertEqual(contagem, {"animal": 2, "unknown": 1, "blank": 1, "unclassified": 1})
        self.assertEqual(observacoes["e4"]["scientificName"], "Chrysocyon brachyurus")
        self.assertEqual(observacoes["e4"]["classificationProbability"], "0.8")
        self.assertEqual(observacoes["e2"]["observationType"], "unknown")
        self.assertEqual(len(midias), len(linhas))


class _Imagem:
    def width(self):
        return 640

    def height(self):
        return 480

    def crop(self, x, y, w, h):
        self.recorte = (x, y, w, h)
        return self


class _Objeto:
    def __init__(self, classe, escore, x=100, y=100, w=200, h=150):
        self.class_id, self.score, self.x, self.y, self.w, self.h = classe, escore, x, y, w, h


class _Classificador:
    labels = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]

    def __init__(self, indice, prob):
        self.resposta = [(indice, prob)]

    def classify(self, img):
        return self.resposta


def _carregar_app(modelo: str):
    """Executa o main.py do MaixPy com um pacote `maix` falso (só para a lógica)."""
    falso = types.ModuleType("maix")
    for nome in ("app", "camera", "display", "err", "gpio", "nn", "pinmap", "time"):
        setattr(falso, nome, types.SimpleNamespace())
    sys.modules["maix"] = falso
    codigo = APP_MAIXPY.read_text(encoding="utf-8").replace("@@MODELO@@", modelo)
    modulo = types.ModuleType("app_maixpy")
    exec(compile(codigo, str(APP_MAIXPY), "exec"), modulo.__dict__)  # noqa: S102
    return modulo


class AppMaixpyTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("maix", None)

    def test_modelo_a_confirma_com_tres_quadros(self):
        app = _carregar_app("modelo-a")
        rotulos = ["anta", "tamandua_bandeira", "lobo_guara"]
        deteccoes = [[_Objeto(2, 0.8)], [_Objeto(2, 0.7)], [_Objeto(2, 0.9)], []]
        self.assertEqual(app.analisar_a(deteccoes, rotulos), ("animal", "lobo_guara", 0.9))
        self.assertEqual(app.analisar_a([[_Objeto(0, 0.5)], []], rotulos)[0], "inconclusivo")
        self.assertEqual(app.analisar_a([[], []], rotulos)[0], "vazio")

    def test_modelo_b_vota_e_rejeita_desconhecido(self):
        app = _carregar_app("modelo-b")
        quadros = [_Imagem() for _ in range(4)]
        deteccoes = [[_Objeto(0, 0.9)] for _ in range(4)]
        desfecho, rotulo, _ = app.analisar_b(quadros, deteccoes, _Classificador(1, 0.95))
        self.assertEqual((desfecho, rotulo), ("animal", "tamandua_bandeira"))
        desfecho, _, _ = app.analisar_b(quadros, deteccoes, _Classificador(3, 0.95))
        self.assertEqual(desfecho, "inconclusivo")
        self.assertEqual(app.analisar_b(quadros, [[]] * 4, _Classificador(0, 0.9))[0], "vazio")

    def test_recorte_quadrado_cabe_na_imagem(self):
        app = _carregar_app("modelo-b")
        imagem = _Imagem()
        app.recorte_quadrado(imagem, _Objeto(0, 0.9, x=600, y=400, w=100, h=100))
        x, y, w, h = imagem.recorte
        self.assertEqual(w, h)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 480)


if __name__ == "__main__":
    unittest.main()
