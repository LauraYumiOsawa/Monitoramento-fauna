"""Escreve versoes/RANKING.md a partir dos números calculados por calcular_ranking.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from animal_monitor_hw.energia import Bateria, CicloEvento, autonomia_dias

CONFIANCA = {
    "Menor custo": ("Documentado (lojas BR, 20–21/10/2026) + importação estimada",
                    "Média: vários itens esgotados; Orange Pi 5 sem preço atual"),
    "Melhor custo-benefício": ("Derivado", "Baixa a média: herda as incertezas de todos os critérios"),
    "Maior eficiência energética": ("Estimado (P·t; t documentado em parte)",
                                    "Baixa: potência de inferência quase toda estimada"),
    "Maior autonomia": ("Estimado (seção 7) com espera documentada em parte",
                        "Média para a ordem de grandeza; baixa para a ordem fina"),
    "Disparo mais rápido": ("Estimado por arquitetura", "Baixa: nenhum tempo medido"),
    "Maior precisão": ("Medido no projeto (proxy x86, 18 imagens)",
                       "Baixa: 1 caixa = 5 pontos de recall; não é imagem de armadilha"),
    "Melhor filtro de vazias": ("Medido no projeto (presença no detector)",
                                "Baixa: sem imagens vazias no teste"),
    "Melhor identificação de espécies": ("Medido no projeto (90 rajadas por proxy)",
                                         "Média para A × B; vídeos editados, não de campo"),
    "Melhor desempenho noturno": ("Documentado (filtro IR da câmera)",
                                  "Média: diz se há como fotografar com IR, não a qualidade"),
    "Maior robustez": ("Documentado parcialmente (temperatura de operação)",
                       "Baixa: várias placas não declaram; não há histórico de falhas"),
    "Menor custo em dois anos": ("Estimado (custo + visitas de troca)",
                                 "Baixa: depende do custo de visita adotado"),
    "Mais fácil de montar": ("Rubrica sobre as versões geradas", "Média"),
}
# Conclusões fixas só onde o texto não depende dos números; as demais vêm de _conclusoes.
CONCLUSAO_FIXA = {
    "Disparo mais rápido": "Câmera sempre ligada dispara em ~0,1 s ao custo de espera alta; desligar a câmera (0,8–1,2 s) ou a placa inteira (arquitetura B, ~20 s) troca atraso por autonomia (QP1).",
    "Melhor desempenho noturno": "Câmeras sensíveis a IR com iluminador (Camera Module 3 NoIR, OV5647 IR, IMX219 NoIR) habilitam a noite; a AI Camera não, por ter filtro IR fixo e não existir versão NoIR.",
    "Maior robustez": "Só se ordena o que está declarado, e com cautela: Grove e MaixCAM informam só o chip; nas versões Raspberry Pi o elo fraco é a câmera (0 a 50 °C), abaixo do que uma caixa ao sol pode atingir.",
    "Mais fácil de montar": "ONNX Runtime direto no Raspberry Pi é o caminho mais curto; firmware em C/C++ e conversão por toolchain externa (TPU-MLIR, Vela, ESP-PPQ, SDK da Luckfox) são os mais longos.",
}


def _conclusoes(linhas: dict[str, Any], por_titulo: dict[str, Any]) -> dict[str, str]:
    def top(titulo: str, n: int = 3) -> list[str]:
        pos = por_titulo[titulo]["posicoes"]
        return [vid for vid in sorted(pos, key=pos.get)][:n]

    textos: dict[str, str] = {}
    baratos = top("Menor custo")
    precos = [linhas[v]["custo_rs"] for v in baratos]
    pi_barato = min((v for v in linhas if v.startswith("05")), key=lambda v: linhas[v]["custo_rs"])
    textos["Menor custo"] = (
        f"As três mais baratas ({', '.join(baratos)}) ficam entre {_fmt_rs(min(precos))} e "
        f"{_fmt_rs(max(precos))} completas; a Raspberry Pi mais barata é a {pi_barato} "
        f"({_fmt_rs(linhas[pi_barato]['custo_rs'])}). Só a placa Pi 5 de 4 GB custa R$ 1.574,90."
    )
    pos_cb = por_titulo["Melhor custo-benefício"]["posicoes"]
    ultimos = sorted(pos_cb, key=pos_cb.get)[-3:]
    textos["Melhor custo-benefício"] = (
        f"Lideram configurações baratas ({', '.join(top('Melhor custo-benefício'))}); no fim "
        f"ficam {', '.join(ultimos)}: o Pi 5 com acelerador paga por vazão que uma rajada de "
        "poucos quadros não usa."
    )
    cpu = linhas.get("01a", {}).get("e_img_j")
    npus = [linhas[v]["e_img_j"] for v in ("02a", "03a", "12a") if v in linhas]
    if cpu and npus:
        textos["Maior eficiência energética"] = (
            f"Com acelerador, a energia por imagem cai de {_fmt(cpu, 2)} J (CPU do Pi 5) para "
            f"{_fmt(min(npus), 2)}–{_fmt(max(npus), 2)} J ({cpu / max(npus):.0f}× menos); mas a "
            "espera continua dominando o dia, então isso quase não muda a autonomia (QP2, QP3)."
        )
    longos = sorted((v for v in linhas if linhas[v]["autonomia_dias"] >= 14),
                    key=lambda v: -linhas[v]["autonomia_dias"])
    horas_pi5 = [linhas[v]["autonomia_dias"] * 24 for v in linhas if v[:2] in {"01", "02", "03", "04", "12"}]
    textos["Maior autonomia"] = (
        f"Só quem dorme entre eventos passa de duas semanas ({', '.join(longos)}). Linux sempre "
        f"ligado fica em dias (Zero 2 W, Luckfox, MaixCAM) ou em {min(horas_pi5):.0f}–"
        f"{max(horas_pi5):.0f} h (Pi 5 com ou sem acelerador)."
    )
    pares = {}
    for vid, linha in linhas.items():
        pares.setdefault(vid[:2], {})[vid[2]] = linha["qualidade"]
    comparaveis = {hw: par for hw, par in pares.items() if "a" in par and "b" in par}
    excecoes = [hw for hw, par in comparaveis.items() if par["b"] <= par["a"]]
    texto = (f"O Modelo B tem o maior F1 em {len(comparaveis) - len(excecoes)} das "
             f"{len(comparaveis)} plataformas que rodam os dois modelos")
    if excecoes:
        texto += " (exceção: " + ", ".join(
            f"{hw}a {_fmt(comparaveis[hw]['a'], 3)} × {hw}b {_fmt(comparaveis[hw]['b'], 3)}"
            for hw in excecoes) + " — diferença de uma caixa)"
    if "07a" in linhas:
        texto += f". Em 256 px o YOLOv8n cai para {_fmt(linhas['07a']['qualidade'], 3)} (Grove)"
    textos["Maior precisão"] = texto + "."
    pres_b = [linhas[v]["presenca_recall"] for v in linhas if v.endswith("b")]
    pres_a = [linhas[v]["presenca_recall"] for v in linhas if v.endswith("a") and v not in {"09a", "11a"}]
    textos["Melhor filtro de vazias"] = (
        f"Os detectores de presença do Modelo B acham {min(pres_b):.0%}–{max(pres_b):.0%} dos "
        f"animais; o YOLOv8n multiclasse, {min(pres_a):.0%}–{max(pres_a):.0%}, caindo com a "
        "resolução. Sem imagens vazias no teste, não há como medir descarte seguro."
    )
    erro_a = [linhas[v]["rajada_erro_confirmado"] for v in linhas
              if v.endswith("a") and linhas[v]["rajada_erro_confirmado"] is not None]
    erro_b = [linhas[v]["rajada_erro_confirmado"] for v in linhas
              if v.endswith("b") and linhas[v]["rajada_erro_confirmado"] is not None]
    inicio_b = ("Nenhuma variante do Modelo B confirmou espécie errada em 90 rajadas"
                if max(erro_b) == 0 else
                f"O Modelo B confirmou espécie errada em até {max(erro_b):.0%} das rajadas")
    textos["Melhor identificação de espécies"] = (
        f"{inicio_b}; todas as do Modelo A confirmaram ({min(erro_a):.0%}–{max(erro_a):.0%} dos "
        "eventos). O preço é ~1/3 de eventos inconclusivos, guardados para revisão."
    )
    curtos = sorted(v for v in linhas if linhas[v]["autonomia_dias"] < 7)
    textos["Menor custo em dois anos"] = (
        f"Com troca paga por visita, {len(curtos)} das {len(linhas)} configurações duram menos "
        "de uma semana por carga e somam dezenas de milhares de reais em visitas: a bateria não "
        "é o caminho para elas — exigem solar ou rede."
    )
    textos.update(CONCLUSAO_FIXA)
    return textos


def _fmt_rs(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt(valor: float | None, casas: int = 2) -> str:
    if valor is None:
        return "—"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _conclusao_geral(linhas: dict[str, Any], titulo: str) -> str:
    perfil = titulo.split("— ", 1)[1]
    ordem = sorted(linhas, key=lambda v: linhas[v][f"geral_{perfil}"], reverse=True)
    diferenca = linhas[ordem[0]][f"geral_{perfil}"] - linhas[ordem[1]][f"geral_{perfil}"]
    texto = f"{ordem[0]} à frente de {ordem[1]} por {_fmt(diferenca, 3)}"
    if diferenca < 0.02:
        texto += ": diferença menor que a incerteza das premissas, a ordem entre os dois não é robusta"
    return texto + ". Ver seções 3 e 5."


def _arquitetura_b(dados: dict[str, Any], linha: dict[str, Any]) -> tuple[float, float]:
    """Pi 5 desligado entre eventos (POWER_OFF_ON_HALT=1) e religado pelo circuito do PIR."""
    cenario = dados["cenario"]
    boot_s, boot_w = 20.0, 4.0  # premissas: ~7 s até o SO (whitepaper) + userland e Python
    espera = 0.01 + 0.001  # Pi desligado (oficial) + controlador do PIR (premissa)
    t_inf = linha["t_inferencia_ms"] / 1000.0
    t_ciclo = boot_s + 8 * 0.15 + cenario["escrita_s"] + 8 * t_inf + 5.0  # + desligamento
    e_ciclo = boot_w * (boot_s + 8 * 0.15 + cenario["escrita_s"] + 5.0) + linha["p_inferencia_w"] * 8 * t_inf
    bateria = Bateria(cenario["bateria_wh"], cenario["fator_utilizavel"], cenario["eficiencia"])
    dias = autonomia_dias(bateria, espera, CicloEvento(e_ciclo, t_ciclo), cenario["eventos_por_dia"])
    return dias, boot_s


def escrever_relatorio(destino: Path, dados, linhas, referencias, lotes, rankings, sensibilidade,
                       nomes, speciesnet, perfis) -> None:
    cenario = dados["cenario"]
    total = len(linhas)
    partes: list[str] = [
        "# Ranking das versões embarcadas (completo e refeito)",
        "",
        f"> Gerado por `versoes/ranking/calcular_ranking.py` em {date.today().strftime('%d/%m/%Y')} "
        "a partir de `versoes/ranking/candidatos.yaml` e `versoes/ranking/medicoes/`. Para "
        "recalcular com outras premissas, edite o YAML e rode o script.",
        "",
        "## O que mudou",
        "",
        "Na pesquisa teórica, a tabela \"O que cada ranking permite concluir\" (seção 7) ficou "
        "quase toda **indeterminada**, porque faltavam custos completos e desempenho comparável. "
        f"Esta versão ordena as **{total} configurações completas de armadilha** que o "
        "cruzamento hardware × software/modelo gerou (ver `MATRIZ_COMPATIBILIDADE.md`), usando "
        "a mesma regra da pesquisa: cada número mantém seu tipo.",
        "",
        "- **documentado**: preço de loja ou fabricante e medição publicada, com fonte e data;",
        "- **medido**: proxy medido neste projeto em x86_64 com os modelos reais (Modelo A e B, "
        "INT8, cabeça dividida, resoluções de 192 a 640 px, MegaDetector + SpeciesNet);",
        "- **estimado**: conta com premissa explícita (consumo de placas sem dado publicado, "
        "tempo de disparo, importação).",
        "",
        "O que continua sem resposta — e o ranking não finge responder: desempenho em campo "
        "nas três espécies, recall noturno, consumo medido e durabilidade. As posições valem para "
        "a configuração descrita, sob estas premissas, e devem ser confirmadas em bancada.",
        "",
        "Fora do ranking de armadilha completa (mas com dados na seção 6): Jetson Orin Nano Super "
        "(13a, 13b — teto de desempenho, não candidato a bateria) e as estações de lote "
        "MegaDetector + SpeciesNet (01c, 08c, 13c — complementam uma armadilha).",
        "",
        "## 1. Tabela de ranking",
        "",
        "| Ranking | Métrica | 1º | 2º | 3º | Base dos dados | Confiança | O que se pode concluir |",
        "|---|---|---|---|---|---|---|---|",
    ]
    por_titulo = {r["titulo"]: r for r in rankings}
    conclusoes = _conclusoes(linhas, por_titulo)
    for ranking in rankings:
        titulo = ranking["titulo"]
        chave = titulo if not titulo.startswith("Melhor geral") else None
        base, confianca = CONFIANCA.get(chave, ("Composição dos critérios acima",
                                                "Baixa a média: ver sensibilidade (seção 5)"))
        conclusao = (conclusoes[chave] if chave in conclusoes
                     else _conclusao_geral(linhas, titulo))
        partes.append(
            f"| {titulo} | {ranking['metrica']} | {ranking['podio'][0]} | {ranking['podio'][1]} | "
            f"{ranking['podio'][2]} | {base} | {confianca} | {conclusao} |"
        )
    partes += [
        "",
        "Na precisão, as versões de microcontrolador sem detector (09a, 11a) entram com a "
        "acurácia do quadro inteiro, não com F1 — a tarefa é outra. A identificação de espécies "
        "só ordena quem tem simulação de rajadas (as duas também ficam de fora). Empates são "
        "mostrados como empates.",
        "",
        "## 2. Resultados (tabela do esqueleto preenchida)",
        "",
    ]
    colunas = ["Menor custo", "Melhor custo-benefício", "Maior eficiência energética",
               "Maior autonomia", "Maior precisão", "Melhor geral — Equilíbrio"]
    cabecalho = ["Mais barato", "Custo-benefício", "Mais eficiente", "Maior autonomia",
                 "Mais preciso", "Melhor geral (Equilíbrio)"]
    partes.append("| Posição | " + " | ".join(cabecalho) + " |")
    partes.append("|---|" + "---|" * len(cabecalho))
    for lugar in (1, 2, 3):
        celulas = []
        for coluna in colunas:
            pos = por_titulo[coluna]["posicoes"]
            ids = [vid for vid, p in pos.items() if p == lugar]
            celulas.append(", ".join(f"{vid} {nomes[vid]}" for vid in ids) or "—")
        partes.append(f"| {lugar}º | " + " | ".join(celulas) + " |")

    partes += ["", "## 3. Melhor geral por perfil de pesos", "",
               "Pesos da seção 7 da pesquisa (cada perfil soma 1). Normalização min-max; "
               "autonomia em log10(dias). Um critério sem variação sairia com redistribuição "
               "dos pesos (não ocorreu).", "",
               "| Critério | " + " | ".join(perfis) + " |", "|---|" + "---|" * len(perfis)]
    for criterio in perfis["Equilíbrio"]:
        rotulo = {"qualidade": "Qualidade de detecção", "autonomia": "Autonomia",
                  "custo": "Custo completo", "disparo": "Tempo de disparo",
                  "noturno": "Capacidade noturna", "integracao": "Integração e documentação"}[criterio]
        partes.append(f"| {rotulo} | " + " | ".join(f"{perfis[p][criterio]:.2f}".replace(".", ",")
                                                     for p in perfis) + " |")
    partes += ["", "| Posição | " + " | ".join(perfis) + " |", "|---|" + "---|" * len(perfis)]
    ordenados = {p: sorted(linhas, key=lambda v: linhas[v][f"geral_{p}"], reverse=True)
                 for p in perfis}
    for indice in range(5):
        celulas = []
        for perfil in perfis:
            vid = ordenados[perfil][indice]
            celulas.append(f"{vid} {nomes[vid]} ({_fmt(linhas[vid][f'geral_{perfil}'], 3)})")
        partes.append(f"| {indice + 1}º | " + " | ".join(celulas) + " |")
    vencedores = {p: ordenados[p][0] for p in perfis}
    if len(set(vencedores.values())) == 1:
        partes += ["", f"O vencedor **não muda** entre os perfis: {next(iter(vencedores.values()))}."]
    else:
        partes += ["", "O vencedor **muda** com o perfil: " + "; ".join(
            f"{p} → {v}" for p, v in vencedores.items()) + "."]

    partes += ["", "## 4. Dados por candidato", "",
               "Tipos: ver `candidatos.yaml` (cada número com fonte ou premissa). Qualidade: F1 "
               "por espécie no teste (09a e 11a: acurácia do quadro inteiro). Rajadas: acerto "
               "entre eventos com espécie confirmada e taxa de eventos com espécie errada "
               "confirmada (90 rajadas de 8 quadros).", "",
               "| Versão | Configuração | Custo | Espera (W) | Inferência (ms) | E_img (J) | "
               "Autonomia (d) | Disparo (s) | Qualidade | Presença | Rajadas: acerto / erro | "
               "Noturno | Integração | TCO 2 anos |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vid in sorted(linhas):
        linha = linhas[vid]
        rajada = ("—" if linha["rajada_acerto_confirmados"] is None else
                  f"{linha['rajada_acerto_confirmados']:.0%} / {linha['rajada_erro_confirmado']:.0%}")
        partes.append(
            f"| {vid} | {nomes[vid]} | {_fmt_rs(linha['custo_rs'])} | {_fmt(linha['espera_w'], 3)} | "
            f"{_fmt(linha['t_inferencia_ms'], 1)} | {_fmt(linha['e_img_j'], 3)} | "
            f"{_fmt(linha['autonomia_dias'], 1)} | {_fmt(linha['disparo_s'], 1)} | "
            f"{_fmt(linha['qualidade'], 3)} | {_fmt(linha['presenca_recall'], 2)} | {rajada} | "
            f"{linha['noturno']} | {linha['integracao']}/6 | {_fmt_rs(linha['tco_2anos_rs'])} |"
        )

    partes += ["", "## 5. Sensibilidade", "", "**Eventos por dia** (vencedor do melhor geral):", "",
               "| Eventos/dia | " + " | ".join(perfis) + " |", "|---|" + "---|" * len(perfis)]
    for eventos, vencedor in sorted(sensibilidade["eventos"].items()):
        partes.append(f"| {eventos} | " + " | ".join(vencedor[p] for p in perfis) + " |")
    partes += ["", "Autonomia (dias) com 10 / 100 / 1000 eventos por dia:", "",
               "| Versão | 10 | 100 | 1000 |", "|---|---|---|---|"]
    for vid in sorted(linhas, key=lambda v: -linhas[v]["autonomia_dias"]):
        linha = linhas[vid]
        partes.append(f"| {vid} | {_fmt(linha['autonomia_10_eventos'], 1)} | "
                      f"{_fmt(linha['autonomia_dias'], 1)} | {_fmt(linha['autonomia_1000_eventos'], 1)} |")
    partes += ["", "**Preço do Orange Pi 5** (sem cotação atual; premissa US$ 100):", "",
               "| Preço FOB | Custo 08a | Posição de 08a em custo | Vencedores (Equilíbrio / Longa / Orçamento) |",
               "|---|---|---|---|"]
    for usd, info in sensibilidade["orangepi"].items():
        venc = info["vencedores"]
        partes.append(f"| US$ {usd:.0f} | {_fmt_rs(info['custo_08a'])} | {info['posicao_custo_08a']}º | "
                      + " / ".join(venc[p] for p in perfis) + " |")
    partes += ["", "**Custo de uma visita de troca de bateria** (três menores TCO em 2 anos):", ""]
    for visita, top in sensibilidade["visita"].items():
        partes.append(f"- R$ {visita:.0f}: " + ", ".join(top))
    if "01b" in linhas:
        dias_b, boot = _arquitetura_b(dados, linhas["01b"])
        partes += [
            "", "**Arquitetura B no Pi 5** (versão 01b desligada entre eventos com "
            "`POWER_OFF_ON_HALT=1`, ~0,01 W oficial, religada pelo circuito do PIR; boot de "
            f"{boot:.0f} s estimado): autonomia de {_fmt(linhas['01b']['autonomia_dias'], 1)} para "
            f"**{_fmt(dias_b, 1)} dias** a 100 eventos/dia, com o tempo até a primeira foto "
            f"passando de {_fmt(linhas['01b']['disparo_s'], 1)} s para ~{boot:.0f} s — o "
            "dilema da QP1/QP2 em números. A versão suporta esse modo (`gatilho.tipo: nenhum`, "
            "`energia.desligar_ao_final: true`).",
        ]

    partes += ["", "## 6. Fora do ranking de armadilha", "",
               "| Versão | Papel | Custo | Dados |", "|---|---|---|---|"]
    for vid, linha in sorted(referencias.items()):
        partes.append(
            f"| {vid} | Teto de desempenho (Jetson) | {_fmt_rs(linha['custo_rs'])} | inferência "
            f"{_fmt(linha['t_inferencia_ms'], 2)} ms; espera {_fmt(linha['espera_w'], 1)} W; "
            f"autonomia {_fmt(linha['autonomia_dias'], 2)} d com 74 Wh; qualidade "
            f"{_fmt(linha['qualidade'], 3)} |")
    for vid, linha in sorted(lotes.items()):
        tempo = linha["lote_s_por_imagem"]
        partes.append(
            f"| {vid} | Estação de lote (MegaDetector + SpeciesNet) | {_fmt_rs(linha['custo_rs'])} | "
            f"~{_fmt(tempo['valor'], 1)} s/imagem ({tempo['tipo']}: {tempo['fonte']}) |")
    partes += ["", "O custo da 08c usa o Orange Pi 5 de 4 GB como piso (preço estimado); o lote pede "
               "8 GB de RAM."]
    if speciesnet:
        contagem = speciesnet.get("counts", {})
        partes += [
            "", "**MegaDetector v5a + SpeciesNet v4.0.3a medidos no projeto** (`--country BRA`, "
            "x86, mesmas 18 imagens): presença 20/20 animais; espécie exata em "
            f"{contagem.get('true_positive')} de {contagem.get('ground_truth_total')} (F1 "
            f"{_fmt(speciesnet.get('f1@0.35'), 2)}). Níveis atribuídos: "
            + ", ".join(f"{k}: {v}" for k, v in (speciesnet.get("niveis_taxonomicos") or {}).items())
            + ". Nenhuma troca entre as três espécies; os erros foram arredondamentos para família "
            "ou rótulos sem sentido. É a referência da área sem treino na fauna local — e fica "
            "abaixo do Modelo B neste conjunto, que não é de armadilha fotográfica.",
        ]

    partes += [
        "", "## 7. Método e limites", "",
        "- **Unidade de comparação:** a configuração completa (placa, acelerador, câmera, modelo, "
        "precisão numérica, runtime e arquitetura de energia), como define a seção 3.",
        f"- **Custo completo:** itens da versão + comuns (PIR, caixa IP66, microSD 32 GB, power "
        f"bank de 74 Wh, diversos). Importação: Remessa Conforme vigente, frete de US$ "
        f"{dados['importacao']['frete_usd']:.0f} por pedido, ICMS {dados['importacao']['icms']:.0%} "
        f"(SP), câmbio {dados['cambio']['usd_brl']} ({dados['cambio']['fonte']}).",
        f"- **Autonomia:** equação da seção 7 com o mesmo pacote para todos ({cenario['bateria_wh']:.0f} Wh, "
        f"f = {cenario['fator_utilizavel']}, η = {cenario['eficiencia']}) e "
        f"{cenario['eventos_por_dia']} eventos/dia; ciclo = partida + rajada + escrita + inferência "
        "de cada quadro. Power banks comuns desligam com carga de µA: nas versões de "
        "microcontrolador use células com regulador de mesma energia.",
        "- **Qualidade:** proxies medidos em x86 com ONNX Runtime sobre o mesmo modelo, na "
        "resolução e precisão da versão (o INT8 de NPUs usa o proxy de cabeça dividida com todos "
        "os operadores em INT8). Não inclui a perda própria de cada compilador.",
        "- **Disparo:** tempo até a primeira captura, separado da latência de inferência e do "
        "alerta (seção 3). Estimado por arquitetura; o atraso do próprio PIR é comum a todos e "
        "fica fora.",
        "- **Noturno (0–2):** 2 = câmera sensível a IR com iluminador; 1 = não documentado ou "
        "exige troca de câmera; 0 = filtro IR fixo documentado.",
        "- **Integração (0–6):** conversão (2 ONNX direto, 1 export de uma linha, 0 toolchain "
        "externa) + API Python no dispositivo + conjunto único + exemplo oficial + caminho "
        "testado neste projeto.",
        "- **Custo-benefício:** benefício só com critérios de desempenho (pesos do perfil "
        "Equilíbrio sem o custo, renormalizados) dividido pelo custo em milhares de reais — o "
        "custo não entra duas vezes.",
        "- **TCO:** custo + (730 / D) visitas × R$ "
        f"{cenario['custo_visita_rs']:.0f}. Para quem dura menos de uma semana, o número só "
        "mostra que a bateria não é o caminho: esses sistemas exigem energia solar ou da rede.",
        "",
        "## 8. Fontes",
        "",
        "- Preços e regra de importação: `versoes/ranking/fontes/precos.md`.",
        "- Consumo e benchmarks: `versoes/ranking/fontes/consumo_desempenho_linux.md` e "
        "`consumo_desempenho_mcu_npu.md`.",
        "- Toolchains e APIs de cada versão: `versoes/ranking/fontes/toolchains_e_apis.md`.",
        "- Dado usado em cada candidato, com fonte ou premissa: `versoes/ranking/candidatos.yaml`.",
        "- Proxies medidos: `versoes/ranking/medicoes/` (`proxies_x86.json`, `rajadas_x86.json`, "
        "`presenca_x86.json`, `speciesnet_x86.json`), gerados pelas ferramentas de "
        "`versoes/comum/ferramentas/`.",
        "- Referências numeradas [n] como na pesquisa teórica.",
    ]
    destino.write_text("\n".join(partes) + "\n", encoding="utf-8")
