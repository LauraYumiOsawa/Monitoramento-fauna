"""Gatilhos de disparo (PIR) e saídas digitais (iluminador IR).

* `gpiozero`: Raspberry Pi (no Pi 5 o gpiozero usa o backend lgpio; RPi.GPIO não funciona);
* `gpiod`: qualquer Linux com libgpiod 2.x (Orange Pi 5, Jetson, Luckfox com Ubuntu);
* `simulado`: dispara em intervalo fixo, para bancada sem sensor;
* `nenhum`: arquitetura B — o próprio boot já é o disparo.

O PIR fica fora da medição de inferência: ele decide quando a câmera trabalha, e é por
isso que a pesquisa trata a espera como o estado dominante do orçamento energético.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Protocol

from animal_monitor.errors import ConfigurationError


class Gatilho(Protocol):
    def aguardar(self, timeout_s: float | None = None) -> bool: ...

    def fechar(self) -> None: ...


class PirGpiozero:
    def __init__(self, pino: int, fila: int = 1, limiar: float = 0.5) -> None:
        try:
            from gpiozero import MotionSensor
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero não instalado (sudo apt install python3-gpiozero)."
            ) from exc
        # queue_len=1: sem média móvel; o PIR já entrega o pulso digital filtrado.
        self.sensor = MotionSensor(pino, queue_len=fila, threshold=limiar)

    def aguardar(self, timeout_s: float | None = None) -> bool:
        return bool(self.sensor.wait_for_motion(timeout=timeout_s))

    def fechar(self) -> None:
        self.sensor.close()


class PirGpiod:
    """libgpiod 2.x. `chip` como /dev/gpiochipN; `linha` é o offset no chip."""

    def __init__(self, chip: str, linha: int, pull_down: bool = True) -> None:
        try:
            import gpiod
            from gpiod.line import Bias, Edge
        except ImportError as exc:
            raise RuntimeError("Módulo gpiod (libgpiod 2.x) não instalado.") from exc
        self.linha = int(linha)
        configuracao = gpiod.LineSettings(
            edge_detection=Edge.RISING, bias=Bias.PULL_DOWN if pull_down else Bias.AS_IS
        )
        self.requisicao = gpiod.request_lines(
            chip, consumer="armadilha-pir", config={self.linha: configuracao}
        )

    def aguardar(self, timeout_s: float | None = None) -> bool:
        espera = None if timeout_s is None else timedelta(seconds=timeout_s)
        if not self.requisicao.wait_edge_events(espera):
            return False
        self.requisicao.read_edge_events()
        return True

    def fechar(self) -> None:
        self.requisicao.release()


class GatilhoSimulado:
    """Dispara a cada `intervalo_s`, até `maximo` vezes (None = sem limite)."""

    def __init__(self, intervalo_s: float = 30.0, maximo: int | None = None) -> None:
        self.intervalo_s = float(intervalo_s)
        self.maximo = maximo
        self.disparos = 0

    @property
    def esgotado(self) -> bool:
        return self.maximo is not None and self.disparos >= self.maximo

    def aguardar(self, timeout_s: float | None = None) -> bool:
        if self.esgotado:
            return False
        espera = self.intervalo_s if timeout_s is None else min(self.intervalo_s, timeout_s)
        time.sleep(max(0.0, espera))
        self.disparos += 1
        return True

    def fechar(self) -> None:
        return None


class SemGatilho:
    """Arquitetura B: a placa foi ligada pelo circuito do PIR; dispara uma vez."""

    def __init__(self) -> None:
        self.esgotado = False

    def aguardar(self, timeout_s: float | None = None) -> bool:
        if self.esgotado:
            return False
        self.esgotado = True
        return True

    def fechar(self) -> None:
        return None


class SaidaDigital:
    """Liga/desliga o iluminador IR (via transistor/MOSFET; o GPIO não alimenta LEDs IR)."""

    def __init__(self, tipo: str, pino: int | None = None, chip: str | None = None) -> None:
        self.tipo = tipo
        self._dispositivo = None
        if tipo == "gpiozero" and pino is not None:
            from gpiozero import OutputDevice

            self._dispositivo = OutputDevice(pino, initial_value=False)
        elif tipo == "gpiod" and pino is not None and chip:
            import gpiod
            from gpiod.line import Direction, Value

            self._valor = Value
            self._linha = int(pino)
            self._dispositivo = gpiod.request_lines(
                chip,
                consumer="armadilha-ir",
                config={self._linha: gpiod.LineSettings(direction=Direction.OUTPUT)},
            )

    def definir(self, ligado: bool) -> None:
        if self._dispositivo is None:
            return
        if self.tipo == "gpiozero":
            self._dispositivo.value = bool(ligado)
        else:
            valor = self._valor.ACTIVE if ligado else self._valor.INACTIVE
            self._dispositivo.set_value(self._linha, valor)

    def fechar(self) -> None:
        if self._dispositivo is None:
            return
        self.definir(False)
        if self.tipo == "gpiozero":
            self._dispositivo.close()
        else:
            self._dispositivo.release()


def criar_gatilho(config: dict[str, object] | None) -> Gatilho:
    config = config or {}
    tipo = str(config.get("tipo", "simulado"))
    if tipo == "gpiozero":
        return PirGpiozero(int(config.get("pino", 17)))
    if tipo == "gpiod":
        return PirGpiod(str(config.get("chip", "/dev/gpiochip0")), int(config.get("linha", 0)))
    if tipo == "simulado":
        maximo = config.get("maximo")
        return GatilhoSimulado(
            float(config.get("intervalo_s", 30.0)), None if maximo is None else int(maximo)
        )
    if tipo == "nenhum":
        return SemGatilho()
    raise ConfigurationError(f"Gatilho desconhecido: {tipo}")


def criar_saida_ir(config: dict[str, object] | None) -> SaidaDigital:
    config = config or {}
    tipo = str(config.get("tipo", "nenhum"))
    pino = config.get("pino")
    return SaidaDigital(tipo, None if pino is None else int(pino), config.get("chip"))
