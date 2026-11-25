"""Calcula os 13 rankings da pesquisa para as versões geradas e escreve versoes/RANKING.md.

Entradas: candidatos.yaml (dados com tipo e fonte) e medicoes/*.json (proxies medidos).
Saídas: ../RANKING.md, resultados/ranking.json (lido pelo gerador dos READMEs) e
resultados/ranking.csv.

Método (seção 7 da pesquisa e esqueleto):
  * autonomia D = E_util / (24·P_espera + N·E_incremental/3600), E_util = C·f·η;
  * energia por imagem analisada E_img = P_inf · t_inf;
  * melhor geral S_i = Σ w_k · n_ik com min-max (n = (x − mín)/(máx − mín), ou o inverso
    quando menor é melhor); critério sem variação sai e os pesos são redistribuídos;
    autonomia entra como log10(D), porque varia em ordens de grandeza;
  * custo-benefício = benefício (só critérios de desempenho, pesos do perfil Equilíbrio
    renormalizados) ÷ custo completo em milhares de reais — sem contar o custo duas vezes;
  * TCO 2 anos = custo + (730 / D) · custo de uma visita de troca/recarga.

    python versoes/ranking/calcular_ranking.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

AQUI = Path(__file__).resolve().parent
VERSOES = AQUI.parent
sys.path[:0] = [str(VERSOES.parent / "src"), str(VERSOES / "comum")]

from animal_monitor_hw.energia import Bateria, CicloEvento, autonomia_dias  # noqa: E402

PERFIS = {
    "Equilíbrio": {"qualidade": 0.25, "autonomia": 0.20, "custo": 0.20, "disparo": 0.15,
                   "noturno": 0.10, "integracao": 0.10},
    "Longa duração": {"qualidade": 0.20, "autonomia": 0.40, "custo": 0.15, "disparo": 0.10,
                      "noturno": 0.10, "integracao": 0.05},
    "Orçamento": {"qualidade": 0.20, "autonomia": 0.15, "custo": 0.40, "disparo": 0.10,
                  "noturno": 0.10, "integracao": 0.05},
}
MAIOR_MELHOR = {"qualidade": True, "autonomia": True, "custo": False, "disparo": False,
                "noturno": True, "integracao": True}
INTERVALO_PADRAO = {3: 0.3, 5: 0.2}


def carregar() -> tuple[dict[str, Any], ...]:
    dados = yaml.safe_load((AQUI / "candidatos.yaml").read_text(encoding="utf-8"))
    medicoes = AQUI / "medicoes"
    proxies = json.loads((medicoes / "proxies_x86.json").read_text(encoding="utf-8"))
    rajadas = json.loads((medicoes / "rajadas_x86.json").read_text(encoding="utf-8"))
    presenca = json.loads((medicoes / "presenca_x86.json").read_text(encoding="utf-8"))
    speciesnet_json = medicoes / "speciesnet_x86.json"
    speciesnet = (json.loads(speciesnet_json.read_text(encoding="utf-8"))
                  if speciesnet_json.is_file() else {})
    return dados, proxies, rajadas, presenca, speciesnet


def _br(valor: float, casas: int) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def custo(dados: dict[str, Any], ids: list[str], usd_override: dict[str, float] | None = None):
    """Custo em R$ com importação estimada por pedido (loja) para itens em US$."""
    itens = dados["itens"]
    cambio = dados["cambio"]["usd_brl"]
    imp = dados["importacao"]
    total = 0.0
    detalhes = []
    pedidos: dict[str, list[tuple[str, float]]] = {}
    for item_id in ids:
        item = itens[item_id]
        if "preco_rs" in item:
            total += float(item["preco_rs"])
            detalhes.append({"item": item["nome"], "rs": float(item["preco_rs"]),
                             "tipo": item["tipo"], "fonte": item["fonte"]})
        else:
            preco = float((usd_override or {}).get(item_id, item["preco_usd"]))
            pedidos.setdefault(item.get("loja", item_id), []).append((item_id, preco))
    for loja, lista in pedidos.items():
        mercadoria = sum(preco for _, preco in lista)
        aduaneiro = mercadoria + imp["frete_usd"]
        ii = 0.0 if aduaneiro <= 50 else max(0.0, 0.60 * aduaneiro - 30.0)
        com_impostos = (aduaneiro + ii) / (1.0 - imp["icms"])
        reais = com_impostos * cambio
        total += reais
        nomes = " + ".join(itens[item_id]["nome"] for item_id, _ in lista)
        fontes = "; ".join(itens[item_id]["fonte"] for item_id, _ in lista)
        detalhes.append({
            "item": f"{nomes} (importação, pedido {loja})", "rs": round(reais, 2),
            "tipo": "estimado",
            "fonte": f"US$ {_br(mercadoria, 2)} + frete US$ {imp['frete_usd']:.0f}, "
                     f"II US$ {_br(ii, 2)}, ICMS {imp['icms']:.0%} por dentro, câmbio "
                     f"{_br(cambio, 4)}; {fontes}",
        })
    return round(total, 2), detalhes


def energia(dados: dict[str, Any], cand: dict[str, Any], eventos: float) -> dict[str, Any]:
    perfil = dados["energia"][cand["energia"]]
    extra = dados["energia"].get(cand.get("extra", ""), {})
    cenario = dados["cenario"]

    def valor(nome: str, fonte: dict[str, Any] = perfil) -> float:
        return float(fonte[nome]["valor"]) if nome in fonte else 0.0

    espera = valor("espera_w") + valor("extra_espera_w", extra)
    fontes_espera = [perfil["espera_w"]["fonte"]]
    tipos_espera = {perfil["espera_w"]["tipo"]}
    if "extra_espera_w" in extra:
        fontes_espera.append(extra["extra_espera_w"]["fonte"])
        tipos_espera.add(extra["extra_espera_w"]["tipo"])
    leve = valor("leve_w") or valor("ativo_w")
    captura_w = leve + valor("extra_espera_w", extra)
    modo = cand["inferencia_w"]
    if modo == "cpu":
        p_inf = valor("ativo_cpu_w")
    elif modo == "acelerador":
        p_inf = leve + valor("extra_ativo_w", extra)
    elif modo == "sensor":
        p_inf = leve + valor("extra_espera_w", extra)
    else:
        p_inf = valor("ativo_w")
    t_inf = float(cand["inferencia_ms"]["valor"]) / 1000.0
    quadros = int(cand["quadros"])
    intervalo = float(cand.get("intervalo_s", INTERVALO_PADRAO.get(quadros, 0.15)))
    disparo_tipo = cand["disparo"]
    if disparo_tipo == "camera_parada":
        partida = valor("inicio_camera_s")
        disparo = partida
        disparo_fonte = perfil["inicio_camera_s"]["fonte"]
    elif disparo_tipo == "partida":
        partida = valor("partida_s")
        disparo = partida
        disparo_fonte = perfil["partida_s"]["fonte"]
    else:
        partida = 0.0
        disparo = 0.1
        disparo_fonte = "câmera sempre ligada: próximo quadro a 10 fps (premissa)"
    t_captura = partida + quadros * intervalo + float(cenario["escrita_s"])
    t_ciclo = t_captura + quadros * t_inf
    e_ciclo = captura_w * t_captura + p_inf * quadros * t_inf
    bateria = Bateria(float(cenario["bateria_wh"]), float(cenario["fator_utilizavel"]),
                      float(cenario["eficiencia"]))
    ciclo = CicloEvento(max(e_ciclo, espera * t_ciclo), t_ciclo)
    dias = autonomia_dias(bateria, espera, ciclo, eventos)
    return {
        "espera_w": round(espera, 4),
        "espera_fonte": " + ".join(fontes_espera),
        "espera_tipo": "estimado" if "estimado" in tipos_espera else "documentado",
        "p_inferencia_w": round(p_inf, 3),
        "t_inferencia_ms": float(cand["inferencia_ms"]["valor"]),
        "e_img_j": round(p_inf * t_inf, 4),
        "e_evento_j": round(ciclo.incremental_j(espera), 3),
        "t_ciclo_s": round(t_ciclo, 3),
        "autonomia_dias": round(dias, 3),
        "consumo_diario_wh": round(bateria.energia_util_wh / dias, 3),
        "disparo_s": round(disparo, 3),
        "disparo_fonte": disparo_fonte,
    }


def qualidade(cand: dict[str, Any], proxies, rajadas, presenca) -> dict[str, Any]:
    chave = cand["proxy"]
    medido = proxies["acuracia"][chave]
    if chave.startswith("quadro_inteiro"):
        acertos = round(medido["acuracia"] * medido["imagens"])
        confirmados = medido["imagens"] - medido["previsto_desconhecido"]
        return {
            "qualidade": medido["acuracia"],
            "qualidade_metrica": "acurácia do quadro inteiro (18 imagens)",
            "presenca_recall": round(confirmados / medido["imagens"], 4),
            "presenca_precisao": None,
            "especie_entre_confirmados": round(acertos / max(1, confirmados), 4),
            "rajada_erro_confirmado": None,
            "rajada_acerto_confirmados": None,
        }
    matriz = medido["confusao"]["matrix"]
    especies = len(matriz) - 1
    diagonal = sum(matriz[i][i] for i in range(especies))
    cruzados = sum(matriz[i][j] for i in range(especies) for j in range(especies) if i != j)
    rajada = rajadas["proxies"].get(chave, {})
    pres = presenca.get(cand["presenca"], {})
    return {
        "qualidade": medido["f1"],
        "qualidade_metrica": "F1@0,35 por espécie (18 imagens, 20 caixas)",
        "map50": medido["map50"],
        "precisao": medido["precisao"],
        "recall": medido["recall"],
        "presenca_recall": pres.get("recall_presenca"),
        "presenca_precisao": pres.get("precisao_presenca"),
        "especie_entre_confirmados": round(diagonal / max(1, diagonal + cruzados), 4),
        "rajada_erro_confirmado": rajada.get("taxa_erro_confirmado"),
        "rajada_acerto_confirmados": rajada.get("acerto_entre_confirmados"),
        "rajada_desfechos": rajada.get("desfechos"),
    }


def integracao(cand: dict[str, Any]) -> int:
    return int(sum(int(valor) for valor in cand["integracao"].values()))


def normalizar(valores: dict[str, float], maior_melhor: bool) -> dict[str, float] | None:
    minimo, maximo = min(valores.values()), max(valores.values())
    if math.isclose(minimo, maximo):
        return None
    if maior_melhor:
        return {k: (v - minimo) / (maximo - minimo) for k, v in valores.items()}
    return {k: (maximo - v) / (maximo - minimo) for k, v in valores.items()}


def escores(linhas: dict[str, dict[str, Any]], pesos: dict[str, float]) -> dict[str, float]:
    criterios = {
        "qualidade": {k: v["qualidade"] for k, v in linhas.items()},
        "autonomia": {k: math.log10(v["autonomia_dias"]) for k, v in linhas.items()},
        "custo": {k: v["custo_rs"] for k, v in linhas.items()},
        "disparo": {k: v["disparo_s"] for k, v in linhas.items()},
        "noturno": {k: float(v["noturno"]) for k, v in linhas.items()},
        "integracao": {k: float(v["integracao"]) for k, v in linhas.items()},
    }
    normalizados = {}
    for nome, valores in criterios.items():
        if pesos.get(nome, 0) <= 0:
            continue
        resultado = normalizar(valores, MAIOR_MELHOR[nome])
        if resultado is not None:
            normalizados[nome] = resultado
    soma = sum(pesos[nome] for nome in normalizados)
    return {
        k: sum(pesos[nome] / soma * normalizados[nome][k] for nome in normalizados)
        for k in linhas
    }


def posicoes(valores: dict[str, float | None], maior_melhor: bool) -> dict[str, int]:
    """Posição densa com empate (1, 1, 2...); candidatos sem dado ficam fora do ranking."""
    validos = {k: v for k, v in valores.items() if v is not None}
    ordem = sorted(validos, key=lambda k: validos[k], reverse=maior_melhor)
    resultado: dict[str, int] = {}
    lugar = 0
    for indice, chave in enumerate(ordem):
        anterior = ordem[indice - 1] if indice else None
        if anterior is None or not math.isclose(validos[chave], validos[anterior],
                                                rel_tol=1e-9, abs_tol=1e-9):
            lugar += 1
        resultado[chave] = lugar
    return resultado


def podio(valores: dict[str, float | None], maior_melhor: bool, nomes: dict[str, str],
          formato) -> list[str]:
    pos = posicoes(valores, maior_melhor)
    saida = []
    for lugar in (1, 2, 3):
        empatados = [k for k, p in pos.items() if p == lugar]
        if not empatados:
            saida.append("—")
            continue
        if len(empatados) == 1:
            chave = empatados[0]
            saida.append(f"**{chave}** {nomes[chave]} ({formato(valores[chave])})")
        else:
            # Empates longos só com o identificador; os nomes estão na tabela de dados.
            ids = ", ".join(f"**{chave}**" for chave in sorted(empatados))
            saida.append(f"{ids} (empate, {formato(valores[empatados[0]])})")
    return saida


def main() -> int:
    dados, proxies, rajadas, presenca, speciesnet = carregar()
    cenario = dados["cenario"]
    matriz = yaml.safe_load((VERSOES / "matriz.yaml").read_text(encoding="utf-8"))
    nomes_curtos = {}
    for item in matriz["cruzamentos"]:
        if item["veredito"] == "gerar" and not item.get("mesma_versao"):
            hw = matriz["hardware"][item["hardware"]]["nome"]
            sw = {"yolov8n": "YOLOv8n", "modelo_b": "Modelo B",
                  "megadetector_speciesnet": "MD+SpeciesNet", "tinyml": "TinyML"}[item["software"]]
            nomes_curtos[str(item["id"])] = f"{hw} · {sw}"

    linhas: dict[str, dict[str, Any]] = {}
    referencias: dict[str, dict[str, Any]] = {}
    lotes: dict[str, dict[str, Any]] = {}
    for vid, cand in dados["candidatos"].items():
        if cand["papel"] == "lote":
            preco, detalhes = custo(dados, cand["hardware"])
            lotes[vid] = {"custo_rs": preco, "custo_itens": detalhes,
                          "lote_s_por_imagem": cand["lote_s_por_imagem"]}
            continue
        ids = list(cand["itens"]) + list(dados["comuns"])
        preco, detalhes = custo(dados, ids)
        linha = {"custo_rs": preco, "custo_itens": detalhes, "papel": cand["papel"]}
        linha.update(energia(dados, cand, float(cenario["eventos_por_dia"])))
        linha.update(qualidade(cand, proxies, rajadas, presenca))
        linha["noturno"] = int(cand["noturno"])
        linha["integracao"] = integracao(cand)
        linha["robustez_c"] = (cand.get("robustez_c") or {}).get("valor")
        linha["robustez_fonte"] = (cand.get("robustez_c") or {}).get("fonte", "não documentado")
        linha["inferencia_fonte"] = cand["inferencia_ms"]["fonte"]
        linha["inferencia_tipo"] = cand["inferencia_ms"]["tipo"]
        linha["proxy"] = cand["proxy"]
        linha["tco_2anos_rs"] = round(
            preco + float(cenario["horizonte_dias"]) / linha["autonomia_dias"]
            * float(cenario["custo_visita_rs"]), 2)
        for n in cenario["sensibilidade_eventos"]:
            linha[f"autonomia_{n}_eventos"] = round(
                energia(dados, cand, float(n))["autonomia_dias"], 3)
        (linhas if cand["papel"] == "armadilha" else referencias)[vid] = linha

    pesos_beneficio = {k: v for k, v in PERFIS["Equilíbrio"].items() if k != "custo"}
    beneficio = escores(linhas, pesos_beneficio)
    for vid, linha in linhas.items():
        linha["beneficio"] = round(beneficio[vid], 4)
        linha["custo_beneficio"] = round(beneficio[vid] / (linha["custo_rs"] / 1000.0), 4)
    gerais = {perfil: escores(linhas, pesos) for perfil, pesos in PERFIS.items()}
    for perfil, valores in gerais.items():
        for vid, valor in valores.items():
            linhas[vid][f"geral_{perfil}"] = round(valor, 4)

    rankings = [
        ("Menor custo", "custo completo da armadilha (R$)", "custo_rs", False,
         lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")),
        ("Melhor custo-benefício", "benefício de desempenho ÷ custo (por R$ 1.000)",
         "custo_beneficio", True, lambda v: _br(v, 3)),
        ("Maior eficiência energética", "energia por imagem analisada, E_img = P·t (J)", "e_img_j",
         False, lambda v: f"{_br(v, 3)} J"),
        ("Maior autonomia", f"dias com 74 Wh a {cenario['eventos_por_dia']} eventos/dia",
         "autonomia_dias", True, lambda v: f"{_br(v, 1)} d"),
        ("Disparo mais rápido", "tempo até a primeira captura (s)", "disparo_s", False,
         lambda v: f"{v:.1f} s".replace(".", ",")),
        ("Maior precisão", "F1 por espécie no teste (MCU: acurácia do quadro)", "qualidade", True,
         lambda v: f"{v:.3f}".replace(".", ",")),
        ("Melhor filtro de vazias", "recall de presença do detector no teste", "presenca_recall",
         True, lambda v: f"{v:.2f}".replace(".", ",")),
        ("Melhor identificação de espécies", "acerto entre eventos confirmados (rajadas)",
         "rajada_acerto_confirmados", True, lambda v: f"{v:.0%}"),
        ("Melhor desempenho noturno", "nota 0–2 (câmera sensível a IR + iluminador)", "noturno",
         True, lambda v: f"nota {v:.0f}"),
        ("Maior robustez", "limite superior de temperatura declarado do elo mais fraco (°C)",
         "robustez_c", True, lambda v: f"{v:.0f} °C"),
        ("Menor custo em dois anos", f"TCO = custo + (730/D)·R$ {cenario['custo_visita_rs']:.0f}",
         "tco_2anos_rs", False,
         lambda v: f"R$ {v:,.0f}".replace(",", ".")),
        ("Mais fácil de montar", "rubrica de integração e documentação (0–6)", "integracao", True,
         lambda v: f"{v:.0f}/6"),
    ]
    for perfil in PERFIS:
        rankings.append((f"Melhor geral — {perfil}", "escore ponderado min-max (0–1)",
                         f"geral_{perfil}", True, lambda v: f"{v:.3f}".replace(".", ",")))

    resultado_versoes = {}
    for vid in {**linhas, **referencias}:
        resultado_versoes[vid] = {"campos": {}, "rotulos": {}}
    tabela_rankings = []
    for titulo, metrica, campo, maior, formato in rankings:
        valores = {vid: linha.get(campo) for vid, linha in linhas.items()}
        pod = podio(valores, maior, nomes_curtos, formato)
        pos = posicoes(valores, maior)
        tabela_rankings.append({"titulo": titulo, "metrica": metrica, "podio": pod,
                                "posicoes": pos, "campo": campo, "maior": maior})
        for vid, lugar in pos.items():
            resultado_versoes[vid].setdefault("posicoes", {})[titulo] = lugar

    # ------------------------------------------------------------- sensibilidade
    sensibilidade = {"eventos": {}, "orangepi": {}, "visita": {}}
    for n in [*cenario["sensibilidade_eventos"], cenario["eventos_por_dia"]]:
        copia = {vid: dict(linha) for vid, linha in linhas.items()}
        for vid, linha in copia.items():
            cand = dados["candidatos"][vid]
            linha["autonomia_dias"] = energia(dados, cand, float(n))["autonomia_dias"]
        sensibilidade["eventos"][n] = {
            perfil: max(escores(copia, pesos).items(), key=lambda kv: kv[1])[0]
            for perfil, pesos in PERFIS.items()
        }
    for usd in dados["itens"]["orangepi5_4gb"]["faixa_usd"]:
        copia = {vid: dict(linha) for vid, linha in linhas.items()}
        for vid in ("08a", "08b"):
            ids = list(dados["candidatos"][vid]["itens"]) + list(dados["comuns"])
            copia[vid]["custo_rs"] = custo(dados, ids, {"orangepi5_4gb": usd})[0]
        sensibilidade["orangepi"][usd] = {
            "custo_08a": copia["08a"]["custo_rs"],
            "vencedores": {perfil: max(escores(copia, pesos).items(), key=lambda kv: kv[1])[0]
                           for perfil, pesos in PERFIS.items()},
            "posicao_custo_08a": posicoes(
                {k: v["custo_rs"] for k, v in copia.items()}, False
            )["08a"],
        }
    for visita in cenario["sensibilidade_visita_rs"]:
        tco = {vid: linha["custo_rs"] + cenario["horizonte_dias"] / linha["autonomia_dias"] * visita
               for vid, linha in linhas.items()}
        sensibilidade["visita"][visita] = sorted(tco, key=tco.get)[:3]

    # ------------------------------------------------------------- saídas
    rotulos = {
        "custo_rs": "Custo completo (R$)", "espera_w": "Potência em espera (W)",
        "t_inferencia_ms": "Inferência por quadro (ms)", "e_img_j": "Energia por imagem (J)",
        "autonomia_dias": "Autonomia, 74 Wh e 100 eventos/dia (dias)",
        "disparo_s": "Tempo até a 1ª captura (s)", "qualidade": "Qualidade (proxy)",
        "presenca_recall": "Recall de presença (proxy)",
        "rajada_acerto_confirmados": "Acerto entre eventos confirmados (rajadas)",
        "noturno": "Capacidade noturna (0–2)", "integracao": "Integração/documentação (0–6)",
        "tco_2anos_rs": "TCO 2 anos (R$)",
    }
    tipos = {"custo_rs": "documentado + estimado (importação)", "espera_w": None,
             "t_inferencia_ms": None, "e_img_j": "estimado", "autonomia_dias": "estimado",
             "disparo_s": "estimado", "qualidade": "medido (proxy x86)",
             "presenca_recall": "medido (proxy x86)",
             "rajada_acerto_confirmados": "medido (proxy x86)",
             "noturno": "documentado (rubrica)", "integracao": "rubrica",
             "tco_2anos_rs": "estimado"}
    for vid, linha in {**linhas, **referencias}.items():
        saida = resultado_versoes[vid]
        saida["rotulos"] = rotulos
        for campo in rotulos:
            valor = linha.get(campo)
            if valor is None:
                continue
            tipo = tipos[campo] or (linha["espera_tipo"] if campo == "espera_w"
                                    else linha["inferencia_tipo"])
            cand = dados["candidatos"][vid]
            fonte = {
                "custo_rs": "composição na tabela abaixo",
                "espera_w": linha["espera_fonte"],
                "t_inferencia_ms": linha["inferencia_fonte"],
                "e_img_j": "potência de inferência × tempo de inferência",
                "autonomia_dias": "equação da seção 7 (candidatos.yaml: cenario)",
                "disparo_s": linha["disparo_fonte"],
                "qualidade": f"{linha['qualidade_metrica']}; proxy `{linha['proxy']}`",
                "presenca_recall": f"medicoes/presenca_x86.json, detector `{cand['presenca']}`",
                "rajada_acerto_confirmados": f"medicoes/rajadas_x86.json, proxy `{linha['proxy']}`",
                "noturno": "rubrica 0–2 (RANKING.md, seção 7)",
                "integracao": "rubrica 0–6 (RANKING.md, seção 7)",
                "tco_2anos_rs": "custo + (730/D) visitas × R$ 50",
            }.get(campo, "versoes/RANKING.md")
            saida["campos"][campo] = {"valor": valor, "tipo": tipo, "fonte": fonte}
        saida["custo_itens"] = linha["custo_itens"]
        if linha["papel"] != "armadilha":
            saida["fora_do_ranking"] = ("referência de desempenho (Jetson), não candidata a "
                                        "armadilha a bateria [23].")
    for vid, linha in lotes.items():
        resultado_versoes[vid] = {
            "rotulos": {"custo_rs": "Custo da estação de lote (R$)",
                        "lote": "Tempo por imagem (s)"},
            "campos": {"custo_rs": {"valor": linha["custo_rs"], "tipo": "documentado/estimado",
                                    "fonte": "candidatos.yaml"},
                       "lote": {"valor": linha["lote_s_por_imagem"]["valor"],
                                "tipo": linha["lote_s_por_imagem"]["tipo"],
                                "fonte": linha["lote_s_por_imagem"]["fonte"]}},
            "fora_do_ranking": "estação de classificação em lote (arquitetura C): complementa "
                               "uma armadilha, não é uma.",
        }
    (AQUI / "resultados").mkdir(exist_ok=True)
    ranking_json = {
        "gerado_em": date.today().isoformat(),
        "total_candidatos": len(linhas),
        "versoes": resultado_versoes,
        "rankings": [{k: v for k, v in r.items() if k != "posicoes"} | {"posicoes": r["posicoes"]}
                     for r in tabela_rankings],
        "sensibilidade": sensibilidade,
        "linhas": linhas,
        "referencias": referencias,
        "lotes": lotes,
        "speciesnet": {k: speciesnet.get(k) for k in ("map50", "counts", "niveis_taxonomicos")}
        if speciesnet else {},
    }
    (AQUI / "resultados" / "ranking.json").write_text(
        json.dumps(ranking_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    campos_csv = ["versao", "custo_rs", "espera_w", "t_inferencia_ms", "e_img_j",
                  "autonomia_dias", "disparo_s", "qualidade", "presenca_recall",
                  "rajada_acerto_confirmados", "rajada_erro_confirmado", "noturno", "integracao",
                  "robustez_c", "tco_2anos_rs", "custo_beneficio",
                  *[f"geral_{perfil}" for perfil in PERFIS]]
    with (AQUI / "resultados" / "ranking.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos_csv)
        escritor.writeheader()
        for vid, linha in sorted(linhas.items()):
            escritor.writerow({"versao": vid, **{c: linha.get(c) for c in campos_csv[1:]}})

    from relatorio import escrever_relatorio

    escrever_relatorio(VERSOES / "RANKING.md", dados, linhas, referencias, lotes, tabela_rankings,
                       sensibilidade, nomes_curtos, speciesnet, PERFIS)
    print(f"{len(linhas)} candidatos; RANKING.md e resultados/ escritos.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(AQUI))
    raise SystemExit(main())
