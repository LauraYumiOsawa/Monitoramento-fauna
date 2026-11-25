"""Análise de um evento (rajada de quadros) com o Modelo A ou o Modelo B.

O resultado distingue três situações que a pesquisa pede para não confundir:

* `animal` — espécie confirmada (porta de quadros do Modelo A ou votação do Modelo B);
* `unknown` — houve detecção, mas a espécie não foi confirmada (classe `desconhecido`,
  votação dividida ou poucos quadros). Vai para revisão humana;
* `blank` — nenhuma detecção em nenhum quadro.

"A falha na identificação não deve ser interpretada automaticamente como imagem vazia"
(seção 8 da pesquisa): por isso `unknown` nunca é descartado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from animal_monitor.events import DetectionGate
from animal_monitor.types import BoundingBox
from animal_monitor_b.config import UNKNOWN_LABEL, RuntimeConfigB
from animal_monitor_b.runtime.pipeline import TwoStagePipeline
from animal_monitor_hw.captura import QuadroCapturado


@dataclass(frozen=True, slots=True)
class Avistamento:
    rotulo: str
    tipo: str  # "animal" ou "unknown"
    probabilidade: float
    caixa: BoundingBox
    quadro: int
    votos: int | None = None


@dataclass(slots=True)
class ResultadoEvento:
    tipo: str
    avistamentos: list[Avistamento]
    deteccoes_brutas: int
    quadros: int
    latencia_ms: dict[str, float] = field(default_factory=dict)

    def resumo(self) -> dict[str, object]:
        return {
            "tipo": self.tipo,
            "quadros": self.quadros,
            "deteccoes_brutas": self.deteccoes_brutas,
            "avistamentos": [
                {
                    "rotulo": item.rotulo,
                    "tipo": item.tipo,
                    "probabilidade": round(item.probabilidade, 4),
                    "quadro": item.quadro,
                    "votos": item.votos,
                }
                for item in self.avistamentos
            ],
            "latencia_ms": {chave: round(valor, 2) for chave, valor in self.latencia_ms.items()},
        }


def _tipo_evento(avistamentos: list[Avistamento], deteccoes: int) -> str:
    if any(item.tipo == "animal" for item in avistamentos):
        return "animal"
    return "unknown" if deteccoes else "blank"


class AnalisadorModeloA:
    """YOLOv8n multiclasse: a espécie sai da porta de `min_hits` quadros consecutivos."""

    def __init__(self, detector, min_hits: int = 3) -> None:
        self.detector = detector
        self.min_hits = int(min_hits)

    def analisar(self, quadros: list[QuadroCapturado]) -> ResultadoEvento:
        # Cooldown "infinito": dentro de uma rajada cada espécie gera no máximo um registro.
        porta = DetectionGate(self.min_hits, cooldown_seconds=1e12)
        avistamentos: list[Avistamento] = []
        melhores: dict[str, Avistamento] = {}
        deteccoes = 0
        tempo_detector = 0.0
        for indice, quadro in enumerate(quadros):
            inicio = time.perf_counter()
            resultado = self.detector.infer(quadro.imagem)
            tempo_detector += (time.perf_counter() - inicio) * 1000.0
            deteccoes += len(resultado)
            for item in resultado:
                atual = melhores.get(item.class_name)
                if atual is None or item.confidence > atual.probabilidade:
                    melhores[item.class_name] = Avistamento(
                        item.class_name, "unknown", item.confidence, item.box, indice
                    )
            for item in porta.update(resultado, float(indice)):
                avistamentos.append(
                    Avistamento(item.class_name, "animal", item.confidence, item.box, indice)
                )
        confirmados = {item.rotulo for item in avistamentos}
        # Detecção sem confirmação vira "inconclusivo" para revisão, não fundo.
        avistamentos.extend(
            item for rotulo, item in melhores.items() if rotulo not in confirmados
        )
        return ResultadoEvento(
            _tipo_evento(avistamentos, deteccoes),
            avistamentos,
            deteccoes,
            len(quadros),
            {"detector_total": tempo_detector,
             "detector_medio": tempo_detector / max(1, len(quadros))},
        )


class AnalisadorModeloB:
    """Detector de presença + classificador + rastreamento/votação por indivíduo."""

    def __init__(self, detector, classificador, config: RuntimeConfigB) -> None:
        self.detector = _Cronometrado(detector, "detect")
        self.classificador = _Cronometrado(classificador, "classify")
        self.config = config

    def analisar(self, quadros: list[QuadroCapturado]) -> ResultadoEvento:
        self.detector.zerar()
        self.classificador.zerar()
        # Pipeline novo a cada evento: o rastreador não deve ligar animais de disparos
        # diferentes, separados por minutos ou horas.
        pipeline = TwoStagePipeline(self.detector, self.classificador, self.config)
        avistamentos: list[Avistamento] = []
        deteccoes = 0
        inicio_rajada = quadros[0].instante_s if quadros else 0.0
        for indice, quadro in enumerate(quadros):
            resultado = pipeline.process(quadro.imagem, indice, quadro.instante_s - inicio_rajada)
            deteccoes += resultado.detections
            for evento in resultado.events:
                avistamentos.append(
                    Avistamento(evento.species, "animal", evento.confidence, evento.box,
                                evento.frame_index, evento.votes)
                )
        pipeline.close()
        # Toda trilha sem espécie confirmada vira "inconclusivo", inclusive a vista em um
        # único quadro: um animal rápido pode cruzar a rajada inteira em um ou dois quadros.
        for trilha in pipeline.finished_tracks:
            if trilha.emitted:
                continue
            ranking = trilha.top_votes()
            rotulo = ranking[0][0] if ranking else UNKNOWN_LABEL
            total = sum(peso for _, peso in ranking) or 1.0
            participacao = ranking[0][1] / total if ranking else 0.0
            avistamentos.append(
                Avistamento(rotulo, "unknown", participacao, trilha.bounding_box(),
                            trilha.last_frame, trilha.vote_count)
            )
        return ResultadoEvento(
            _tipo_evento(avistamentos, deteccoes),
            avistamentos,
            deteccoes,
            len(quadros),
            {
                "detector_total": self.detector.total_ms,
                "detector_medio": self.detector.total_ms / max(1, self.detector.chamadas),
                "classificador_total": self.classificador.total_ms,
                "recortes": float(self.classificador.itens),
            },
        )


class _Cronometrado:
    """Mede o tempo gasto em cada estágio sem alterar o pipeline do Modelo B."""

    def __init__(self, alvo, metodo: str) -> None:
        self._alvo = alvo
        self._metodo = metodo
        self.zerar()

    def zerar(self) -> None:
        self.total_ms = 0.0
        self.chamadas = 0
        self.itens = 0

    def __getattr__(self, nome: str):
        atributo = getattr(self._alvo, nome)
        if nome != self._metodo:
            return atributo

        def medido(*args, **kwargs):
            inicio = time.perf_counter()
            resultado = atributo(*args, **kwargs)
            self.total_ms += (time.perf_counter() - inicio) * 1000.0
            self.chamadas += 1
            if isinstance(resultado, np.ndarray):
                self.itens += int(resultado.shape[0])
            return resultado

        return medido
