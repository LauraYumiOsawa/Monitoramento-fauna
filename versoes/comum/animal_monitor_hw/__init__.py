"""Camada de hardware das versões embarcadas do monitoramento de fauna.

Reaproveita `animal_monitor` (Modelo A) e `animal_monitor_b` (Modelo B) sem alterá-los e
acrescenta apenas o que muda de uma placa para outra: o runtime de inferência, a captura
acionada por PIR, a exportação Camtrap DP e o modelo analítico de autonomia.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
