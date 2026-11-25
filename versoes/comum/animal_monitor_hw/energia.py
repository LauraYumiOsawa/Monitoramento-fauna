"""Balanço energético da seção 7 da pesquisa, sem números de placa embutidos.

    E_util = C_nominal · f_utilização · η
    E_incremental = E_ciclo − P_espera · t_ciclo
    D = E_util / (24 · P_espera + N · E_incremental / 3600)

C em Wh, P em W, E em J, t em s, N em eventos por dia, D em dias. A forma incremental
evita contar duas vezes a espera dentro do ciclo. Não aplicar η de novo quando a
potência já foi medida na entrada do sistema.

Os valores por plataforma vêm de `versoes/ranking/candidatos.yaml`, cada um com fonte e
tipo (documentado, medido no projeto ou estimado). Este módulo só faz a conta.
"""

from __future__ import annotations

from dataclasses import dataclass

from animal_monitor.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Bateria:
    nominal_wh: float
    fator_utilizavel: float = 1.0
    eficiencia_conversao: float = 0.85

    @property
    def energia_util_wh(self) -> float:
        return self.nominal_wh * self.fator_utilizavel * self.eficiencia_conversao

    def validate(self) -> None:
        if self.nominal_wh <= 0:
            raise ConfigurationError("A capacidade nominal deve ser positiva (Wh).")
        if not 0 < self.fator_utilizavel <= 1 or not 0 < self.eficiencia_conversao <= 1:
            raise ConfigurationError("Fator utilizável e eficiência devem estar em (0, 1].")


@dataclass(frozen=True, slots=True)
class CicloEvento:
    """Um disparo: do despertar (ou do PIR) até voltar à espera."""

    energia_ciclo_j: float
    duracao_s: float

    def incremental_j(self, potencia_espera_w: float) -> float:
        incremental = self.energia_ciclo_j - potencia_espera_w * self.duracao_s
        if incremental < 0:
            raise ConfigurationError(
                "Energia do ciclo menor que a espera no mesmo intervalo: confira se os dois "
                "valores foram medidos no mesmo ponto elétrico."
            )
        return incremental


def energia_ciclo_por_etapas(etapas: list[tuple[float, float]]) -> CicloEvento:
    """Soma etapas (potência W, duração s): boot, captura, IR, inferência, escrita."""
    energia = sum(potencia * duracao for potencia, duracao in etapas)
    duracao = sum(duracao for _, duracao in etapas)
    return CicloEvento(energia_ciclo_j=energia, duracao_s=duracao)


def autonomia_dias(
    bateria: Bateria,
    potencia_espera_w: float,
    ciclo: CicloEvento,
    eventos_por_dia: float,
) -> float:
    bateria.validate()
    if potencia_espera_w < 0 or eventos_por_dia < 0:
        raise ConfigurationError("Potência de espera e eventos por dia não podem ser negativos.")
    ocupacao = eventos_por_dia * ciclo.duracao_s / 86400.0
    if ocupacao > 0.5:
        # A aproximação supõe ciclos curtos e sem sobreposição; acima disso a espera
        # deixa de ser o estado dominante e cada estado precisa ser modelado.
        raise ConfigurationError(
            f"Os ciclos ocupam {ocupacao:.0%} do dia; modele os estados explicitamente."
        )
    consumo_diario_wh = 24.0 * potencia_espera_w + (
        eventos_por_dia * ciclo.incremental_j(potencia_espera_w) / 3600.0
    )
    if consumo_diario_wh <= 0:
        return float("inf")
    return bateria.energia_util_wh / consumo_diario_wh


def consumo_diario_wh(
    potencia_espera_w: float, ciclo: CicloEvento, eventos_por_dia: float
) -> float:
    return 24.0 * potencia_espera_w + eventos_por_dia * ciclo.incremental_j(
        potencia_espera_w
    ) / 3600.0


def tabela_sensibilidade(
    bateria: Bateria,
    potencias_espera_w: list[float],
    ciclo: CicloEvento,
    eventos: list[float],
) -> list[dict[str, float]]:
    """Reproduz a tabela de sensibilidade da seção 7 para quaisquer premissas."""
    linhas = []
    for potencia in potencias_espera_w:
        linha: dict[str, float] = {"espera_w": potencia}
        for quantidade in eventos:
            linha[f"{quantidade:g}_eventos"] = round(
                autonomia_dias(bateria, potencia, ciclo, quantidade), 2
            )
        linhas.append(linha)
    return linhas


def energia_por_imagem_j(potencia_inferencia_w: float, tempo_inferencia_s: float) -> float:
    """E_img = P_inf · t_inf (fórmula do esqueleto). Ambos no mesmo ponto elétrico."""
    if potencia_inferencia_w < 0 or tempo_inferencia_s < 0:
        raise ConfigurationError("Potência e tempo de inferência não podem ser negativos.")
    return potencia_inferencia_w * tempo_inferencia_s
