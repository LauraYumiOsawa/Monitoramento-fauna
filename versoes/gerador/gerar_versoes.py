"""Gera uma pasta por cruzamento hardware × software/modelo que faz sentido.

Entradas (todas em versoes/):
  matriz.yaml                  — cruzamentos, veredito e justificativa (fonte da matriz);
  gerador/especificacoes.yaml  — como cada versão gerada é configurada;
  gerador/plataformas/<família>/ — arquivos copiados para a versão (@@CHAVE@@ é substituído);
  ranking/resultados/ranking.json — opcional: dados e posições de cada versão no ranking.

Saídas:
  <hardware>/<versão>/          — versao.yaml, config/, scripts, código do dispositivo, README.md;
  <hardware>/README.md, MATRIZ_COMPATIBILIDADE.md e README.md (índice).

Arquivos gerados são sobrescritos: edite as entradas e rode de novo.

    python versoes/gerador/gerar_versoes.py
"""

from __future__ import annotations

import json
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VERSOES = Path(__file__).resolve().parents[1]
RAIZ = VERSOES.parent
PLATAFORMAS = Path(__file__).resolve().parent / "plataformas"
AVISO = (
    "Gerado por versoes/gerador/gerar_versoes.py a partir de versoes/matriz.yaml e "
    "versoes/gerador/especificacoes.yaml. Edite as entradas e regenere."
)
VEREDITOS = {"gerar": "faz sentido — gerado", "condicional": "condicional — não gerado",
             "nao": "não faz sentido"}
ARQUITETURAS = {
    "A": "A — processador ativo: a placa fica ligada aguardando o PIR",
    "B": "B — inicialização por evento: um circuito liga a placa no PIR",
    "C": "C — captura independente e classificação em lote",
}
MODELOS = {
    "modelo-a": "YOLOv8n multiclasse (Modelo A): anta, tamanduá-bandeira, lobo-guará",
    "modelo-b": (
        "Modelo B: YOLOv8n de presença (`animal`) + MobileNetV2 (3 espécies + `desconhecido`)"
    ),
    "lote": "MegaDetector (animal/pessoa/veículo) + SpeciesNet (espécie), em lote",
    "mcu": "Classificador TinyML do quadro inteiro (3 espécies + `desconhecido`), sem detector",
}


@dataclass
class Contexto:
    id: str
    cruzamento: dict[str, Any]
    hardware_chave: str
    hardware: dict[str, Any]
    software: dict[str, Any]
    spec: dict[str, Any]
    padroes: dict[str, Any]
    ranking: dict[str, Any]

    @property
    def pasta(self) -> Path:
        return VERSOES / self.hardware["pasta"] / self.cruzamento["pasta"]

    @property
    def titulo(self) -> str:
        return f"{self.hardware['nome']} · {self.software['nome']} · {self.cruzamento['runtime']}"

    @property
    def modelo(self) -> str:
        return str(self.spec["modelo"])

    @property
    def familia(self) -> str:
        return str(self.cruzamento["familia"])


# --------------------------------------------------------------------------- utilidades
def _yaml(dados: dict[str, Any]) -> str:
    return yaml.safe_dump(dados, sort_keys=False, allow_unicode=True, width=100)


def _escrever(caminho: Path, texto: str, executavel: bool = False) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8")
    if executavel:
        caminho.chmod(caminho.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _substituir(texto: str, valores: dict[str, str]) -> str:
    for chave, valor in valores.items():
        texto = texto.replace(f"@@{chave}@@", valor)
    return texto


def _copiar_plataforma(familia: str, destino: Path, valores: dict[str, str],
                       apenas: list[str] | None = None) -> list[str]:
    """Copia gerador/plataformas/<família>/ substituindo @@CHAVE@@ em arquivos de texto."""
    origem = PLATAFORMAS / familia
    copiados = []
    for arquivo in sorted(origem.rglob("*")):
        if arquivo.is_dir() or "__pycache__" in arquivo.parts:
            continue
        relativo = arquivo.relative_to(origem)
        if apenas is not None and str(relativo) not in apenas:
            continue
        alvo = destino / _substituir(str(relativo), valores)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        try:
            texto = arquivo.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(arquivo, alvo)
        else:
            _escrever(alvo, _substituir(texto, valores),
                      executavel=bool(arquivo.stat().st_mode & stat.S_IXUSR))
        copiados.append(str(alvo.relative_to(destino)))
    return copiados


def _pesos(padroes: dict[str, Any], valor: str) -> str:
    return str(padroes.get(valor, valor))


def _artefatos(ctx: Contexto) -> list[dict[str, Any]]:
    itens = []
    for item in ctx.spec.get("artefatos") or []:
        novo = json.loads(json.dumps(item))
        if "exportar" in novo:
            novo["exportar"]["pesos"] = _pesos(ctx.padroes, novo["exportar"]["pesos"])
        itens.append(novo)
    return itens


# --------------------------------------------------------------------------- versao.yaml
def _versao_linux(ctx: Contexto) -> dict[str, Any]:
    spec, padroes = ctx.spec, ctx.padroes
    detector = dict(spec["detector"])
    detector["classes"] = padroes["classes"] if ctx.modelo == "modelo-a" else {0: "animal"}
    dados: dict[str, Any] = {
        "id": ctx.id,
        "nome": ctx.titulo,
        "hardware": ctx.hardware_chave,
        "modelo": ctx.modelo,
        "arquitetura_energia": spec["arquitetura"],
        "detector": detector,
    }
    if ctx.modelo == "modelo-b":
        classificador = dict(spec["classificador"])
        classificador.update({"rotulos": padroes["rotulos_b"], "margem": 0.12})
        dados["classificador"] = classificador
        dados["runtime_b"] = "config/runtime_b.yaml"
    dados["captura"] = spec.get("captura", padroes["captura_pi"])
    dados["gatilho"] = spec.get("gatilho", padroes["gatilho_pi"])
    dados["iluminacao_ir"] = spec.get("iluminacao_ir", padroes["ir_pi"])
    dados["evento"] = spec.get("evento", padroes["evento"])
    dados["armazenamento"] = padroes["armazenamento"]
    dados["implantacao"] = {
        "deployment_id": f"dep-{ctx.id}",
        "latitude": 0.0,
        "longitude": 0.0,
        "local": "PREENCHER antes do campo",
        "camera_id": ctx.id,
        "altura_m": None,
        "instalado_por": None,
    }
    dados["energia"] = {"desligar_ao_final": spec["arquitetura"] == "B"}
    artefatos = _artefatos(ctx)
    if artefatos:
        dados["artefatos"] = artefatos
    return dados


RUNTIME_B = """\
# {aviso}
# Limiares do Modelo B (config/modelo_b/runtime.yaml) ajustados para a rajada da armadilha:
# com 8 quadros, classify_every 5 e min_votes 4 nunca emitiriam um avistamento.
detector_confidence: {conf}
detector_iou: {iou}
detector_input: {entrada}
classifier_input: 128
crop_margin: 0.12
max_crops_per_frame: 3
max_detections: 20
# Rajada curta: classificar o recorte em todo quadro (a espécie do indivíduo não muda,
# mas aqui há poucos quadros para votar).
classify_every: 1
tracker:
  max_age: 10
  min_hits: 2
  iou_match: 0.30
  low_score: 0.15
  max_tracks: 16
# 3 votos e 65% do peso: ainda impede que 2 votos contra 1 decidam a espécie.
min_votes: 3
min_vote_share: 0.65
min_species_confidence: 0.50
reject_unknown: true
frame_stride: 1
save_evidence: false
output_dir: ../dados
database: ../dados/sightings_b.sqlite3
providers:
  - CPUExecutionProvider
intra_op_threads: 0
"""


# --------------------------------------------------------------------------- famílias
@dataclass
class Familia:
    """Comportamento de geração de uma família de runtime."""

    linux: bool
    apt: str = ""
    pip: str = ""
    extra_instalacao: str = ""
    ambiente_execucao: str = ""
    usuario: str = "pi"
    plataforma: str | None = None  # pasta de gerador/plataformas copiada além de linux_comum


APT_LINUX = "python3-venv python3-numpy python3-opencv python3-yaml python3-libgpiod"
APT_PI = ("python3-venv python3-numpy python3-opencv python3-yaml python3-picamera2 "
          "python3-gpiozero python3-lgpio")
FAMILIAS: dict[str, Familia] = {
    "ort": Familia(linux=True, apt=APT_PI, pip="onnxruntime"),
    "hailo": Familia(linux=True, apt=APT_PI + " hailo-all", pip="onnxruntime",
                     plataforma="hailo"),
    "imx500": Familia(linux=True, apt=APT_PI + " imx500-all", pip="onnxruntime",
                      plataforma="imx500"),
    "rknn": Familia(linux=True, usuario="orangepi",
                    apt=APT_LINUX,
                    pip="onnxruntime rknn-toolkit-lite2", plataforma="rknn"),
    "tensorrt": Familia(linux=True, usuario="jetson",
                        apt=APT_LINUX,
                        pip="-r requisitos_jetson.txt", plataforma="tensorrt",
                        ambiente_execucao='export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1'),
    "lote": Familia(linux=False, plataforma="lote"),
    "maixpy": Familia(linux=False, plataforma="maixpy"),
    "grove": Familia(linux=False, plataforma="grove"),
    "esp": Familia(linux=False, plataforma="esp"),
    "openmv": Familia(linux=False, plataforma="openmv"),
    "luckfox": Familia(linux=False, plataforma="luckfox"),
}


def _valores(ctx: Contexto, familia: Familia) -> dict[str, str]:
    spec = ctx.spec
    hailo = spec.get("hailo", {})
    rknn = spec.get("rknn", {})
    maix = spec.get("maix", {})
    imx = spec.get("imx500", {})
    grove = spec.get("grove", {})
    valores = {
        "ID": ctx.id,
        "NOME": ctx.titulo,
        "ARQUITETURA": spec["arquitetura"],
        "APT": familia.apt,
        "PIP": familia.pip,
        "EXTRA_INSTALACAO": familia.extra_instalacao,
        "AMBIENTE_EXECUCAO": familia.ambiente_execucao,
        "USUARIO": familia.usuario,
        "TIPO_SERVICO": "oneshot" if spec["arquitetura"] == "B" else "simple",
        "REINICIAR": "no" if spec["arquitetura"] == "B" else "on-failure",
        "MODELO": ctx.modelo,
        "HAILO_ARCH": str(hailo.get("arch", "")),
        "HAILO_ENTRADA": str(hailo.get("entrada", "")),
        "HAILO_CLASSES": str(hailo.get("classes", "")),
        "HAILO_ONNX": str(hailo.get("onnx", "")),
        "RKNN_PLATAFORMA": str(rknn.get("plataforma", "")),
        "RKNN_ONNX": str(rknn.get("onnx", "")),
        "RKNN_CLASSIFICADOR_ONNX": str(rknn.get("classificador_onnx", "")),
        "IMX_ENTRADA": str(imx.get("entrada", "")),
        "IMX_PESOS": _pesos(ctx.padroes, str(imx.get("pesos", ""))),
        "IMX_PASTA": Path(
            str(spec.get("detector", {}).get("arquivo", "modelos/x/network.rpk"))
        ).parent.name,
        "MAIX_ALTURA": str((maix.get("entrada") or [0, 0])[0]),
        "MAIX_LARGURA": str((maix.get("entrada") or [0, 0])[1]),
        "MAIX_ONNX": str(maix.get("onnx", "")),
        "MAIX_CVIMODEL": str(maix.get("cvimodel", "")),
        "MAIX_CLASSIFICADOR_ONNX": str(maix.get("classificador_onnx", "")),
        "MAIX_CLASSIFICADOR_CVIMODEL": str(maix.get("classificador_cvimodel", "")),
        "GROVE_ENTRADA": str(grove.get("entrada", "")),
        "GROVE_PESOS": _pesos(ctx.padroes, str(grove.get("pesos", ""))),
        "HARDWARE_LOTE": str(spec.get("hardware_lote", "")),
        "DISPOSITIVO_TORCH": str(spec.get("dispositivo_torch", "cpu")),
        "MAIX_ROTULOS_DETECTOR": (
            ", ".join(ctx.padroes["classes"].values()) if ctx.modelo == "modelo-a" else "animal"
        ),
        "CLASSES_CSV": ",".join(ctx.padroes["classes"].values()),
        "ROTULOS_B_CSV": ",".join(ctx.padroes["rotulos_b"]),
    }
    return valores


# --------------------------------------------------------------------------- README
def _br(valor: float, casas: int) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


FORMATOS = {
    "custo_rs": lambda v: f"R$ {_br(v, 2)}", "tco_2anos_rs": lambda v: f"R$ {_br(v, 2)}",
    "espera_w": lambda v: _br(v, 3), "t_inferencia_ms": lambda v: _br(v, 1),
    "e_img_j": lambda v: _br(v, 3), "autonomia_dias": lambda v: _br(v, 1),
    "disparo_s": lambda v: _br(v, 1), "qualidade": lambda v: _br(v, 3),
    "presenca_recall": lambda v: f"{v:.0%}", "rajada_acerto_confirmados": lambda v: f"{v:.0%}",
    "noturno": lambda v: f"{v:.0f}", "integracao": lambda v: f"{v:.0f}/6",
    "lote": lambda v: f"{_br(v, 1)} s",
}


def _linha_dado(chave: str, nome: str, dado: dict[str, Any] | None) -> str:
    if not dado:
        return f"| {nome} | não disponível nesta revisão | — | — |"
    valor = dado.get("valor")
    formato = FORMATOS.get(chave)
    texto = formato(valor) if formato and isinstance(valor, int | float) else valor
    return f"| {nome} | {texto} | {dado.get('tipo', '—')} | {dado.get('fonte', '—')} |"


def _secao_ranking(ctx: Contexto) -> str:
    dados = ctx.ranking.get("versoes", {}).get(ctx.id)
    if not dados:
        return ("## Dados para o ranking\n\nAinda não calculados: rode "
                "`python versoes/ranking/calcular_ranking.py` e regenere.\n")
    linhas = ["## Dados para o ranking", "",
              "Tipos: **documentado** (fonte do fabricante ou loja, com data), **medido** "
              "(proxy medido no projeto, x86_64), **estimado** (cálculo com premissa explícita "
              "em `versoes/ranking/candidatos.yaml`).", "",
              "| Grandeza | Valor | Tipo | Fonte/premissa |", "|---|---|---|---|"]
    for chave, nome in dados.get("rotulos", {}).items():
        linhas.append(_linha_dado(chave, nome, dados.get("campos", {}).get(chave)))
    itens = dados.get("custo_itens") or []
    if itens:
        linhas += ["", "Composição do custo:", "", "| Item | R$ | Tipo | Fonte |",
                   "|---|---|---|---|"]
        linhas += [f"| {item['item']} | {_br(item['rs'], 2)} | {item['tipo']} | {item['fonte']} |"
                   for item in itens]
    posicoes = dados.get("posicoes")
    if posicoes:
        linhas += ["", "**Posição nos rankings** (entre "
                   f"{ctx.ranking.get('total_candidatos', '?')} configurações completas): "
                   + "; ".join(f"{nome}: {pos}º" for nome, pos in posicoes.items()) + "."]
    elif dados.get("fora_do_ranking"):
        linhas += ["", f"**Fora dos rankings de armadilha completa:** {dados['fora_do_ranking']}"]
    return "\n".join(linhas) + "\n"


def _readme(ctx: Contexto, arquivos: list[str], secoes: dict[str, str]) -> str:
    spec = ctx.spec
    alternativas = spec.get("alternativas") or []
    partes = [
        f"# {ctx.id} — {ctx.titulo}",
        "",
        f"> {AVISO}",
        "",
        f"**Cruzamento:** {ctx.hardware['nome']} × {ctx.software['nome']} — "
        f"**{VEREDITOS['gerar']}**. Arquitetura de energia: "
        f"{ARQUITETURAS[spec['arquitetura']]}.",
        "",
        "## Por que este cruzamento faz sentido",
        "",
        ctx.cruzamento["justificativa"],
        "",
        f"Papel do hardware na pesquisa: {ctx.hardware['papel']}.",
        "",
        "## Ficha técnica",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| Hardware | {ctx.hardware['resumo']} |",
        "| Câmera | " + (
            "— (processa as fotos de outro subsistema de captura)" if ctx.modelo == "lote"
            else ctx.hardware["camera"]
        ) + " |",
        f"| Modelo | {MODELOS[ctx.modelo]} |",
        f"| Runtime / artefato | {ctx.cruzamento['runtime']} |",
        f"| Disparo | {secoes.get('disparo', 'PIR → rajada de quadros → análise → registro')} |",
        "| Registros | SQLite no formato Camtrap DP + JPEG por evento (`dados/`) |",
        "",
    ]
    if alternativas:
        partes += ["**Alternativas na mesma versão:**", ""]
        partes += [f"- {item}" for item in alternativas] + [""]
    partes.append(_secao_ranking(ctx))
    partes += ["## Passo a passo", ""]
    passo = 0
    for titulo in ("preparar", "instalar", "ligacoes", "executar", "exportar"):
        if secoes.get(titulo):
            # Renumera "### N." na ordem em que as seções aparecem nesta versão.
            passo += 1
            texto = re.sub(r"^### \d+\.", f"### {passo}.", secoes[titulo].rstrip(), count=1)
            partes += [texto, ""]
    partes += ["## Limitações e o que não foi validado", "", secoes["limitacoes"].rstrip(), ""]
    partes += ["## Arquivos desta pasta", ""]
    partes += [f"- `{arquivo}`" for arquivo in sorted(arquivos)]
    return "\n".join(partes) + "\n"


LIMITES_COMUNS = (
    "- Nada aqui foi executado no hardware-alvo: a pesquisa é teórica e não há aquisição de "
    "equipamento. O código foi testado em x86_64 (testes unitários e simulação de câmera "
    "com `--simular`).\n"
    "- A qualidade informada é proxy: split `test` do projeto (18 imagens, 20 caixas; uma "
    "caixa = 5 pontos de recall), imagens que não são de armadilha fotográfica (o animal "
    "ocupa ~47% do quadro) e 14 delas geradas por IA (`docs/MODELO_B.md`, seção 5).\n"
    "- Consumo, autonomia e tempo de disparo são estimativas com premissas explícitas "
    "(`versoes/ranking/candidatos.yaml`); não são medição elétrica.\n"
)


def _secoes_linux(ctx: Contexto, familia: Familia) -> dict[str, str]:
    preparar = [
        "### 1. Preparar o modelo (máquina de desenvolvimento, x86_64)",
        "",
        "```bash",
        f"cd versoes/{ctx.hardware['pasta']}/{ctx.cruzamento['pasta']}",
        "./preparar_modelo.sh      # copia/exporta os ONNX declarados em versao.yaml",
    ]
    preparar += ctx.spec.get("_passos_preparar", [])  # compilação para o acelerador
    preparar += ["./empacotar.sh            # gera versoes/dist/" + ctx.id + ".tar.gz", "```"]
    instalar = [
        "### 2. Instalar no dispositivo",
        "",
        "```bash",
        f"scp versoes/dist/{ctx.id}.tar.gz {familia.usuario}@<ip>:/tmp/",
        f"ssh {familia.usuario}@<ip>",
        "sudo mkdir -p /opt/armadilha && sudo tar -C /opt/armadilha -xzf /tmp/"
        f"{ctx.id}.tar.gz",
        f"sudo chown -R $USER /opt/armadilha/{ctx.id} && cd /opt/armadilha/{ctx.id}",
        "./instalar.sh",
        "```",
    ]
    executar = [
        "### 4. Executar",
        "",
        "```bash",
        "# bancada, sem sensor nem câmera: imagens/vídeo simulam os disparos",
        "./executar.sh trap --simular /caminho/imagens_ou_video.mp4 --eventos 5",
        "./executar.sh benchmark --fonte /caminho/video.mp4 --eventos 20",
        "# campo",
        "./executar.sh trap",
        f"sudo cp armadilha.service /etc/systemd/system/armadilha-{ctx.id}.service",
        f"sudo systemctl daemon-reload && sudo systemctl enable --now armadilha-{ctx.id}",
        "```",
    ]
    exportar = [
        "### 5. Exportar os registros",
        "",
        "```bash",
        "./executar.sh export-camtrapdp   # dados/camtrap-dp/{deployments,media,observations}.csv",
        "```",
        "",
        "Antes do campo, preencha `implantacao` em `versao.yaml` (coordenadas, local, altura).",
    ]
    return {
        "preparar": "\n".join(preparar),
        "instalar": "\n".join(instalar),
        "executar": "\n".join(executar),
        "exportar": "\n".join(exportar),
        "limitacoes": LIMITES_COMUNS,
    }


LIGACOES_PI = """### 3. Ligações elétricas

| Sinal | Pino físico | GPIO | Observação |
|---|---|---|---|
| PIR (saída digital) | 11 | GPIO17 | PIR de 3,3 V (ex.: AM312) ou HC-SR501 com saída de 3,3 V |
| Iluminador IR (comando) | 13 | GPIO27 | Via MOSFET/transistor; o GPIO não alimenta LEDs IR |
| Câmera | CSI | — | Pi 5: cabo de 22 para 15 vias; Zero 2 W: cabo de câmera do Zero |

`iluminacao_ir.modo: auto` usa o metadado `Lux` da câmera (limiar em `limiar_lux`). Use
`sempre`/`nunca` para fixar.
"""


def _gerar_linux(ctx: Contexto) -> list[str]:
    familia = FAMILIAS[ctx.familia]
    valores = _valores(ctx, familia)
    destino = ctx.pasta
    destino.mkdir(parents=True, exist_ok=True)
    arquivos = _copiar_plataforma("linux_comum", destino, valores)
    if familia.plataforma:
        arquivos += _copiar_plataforma(familia.plataforma, destino, valores)
    dados = _versao_linux(ctx)
    _escrever(destino / "versao.yaml", f"# {AVISO}\n" + _yaml(dados))
    arquivos.append("versao.yaml")
    if ctx.modelo == "modelo-b":
        detector = ctx.spec["detector"]
        entrada = ctx.spec.get("hailo", {}).get("entrada") or (
            ctx.spec.get("rknn", {}).get("entrada", [416])[0]
        ) or 416
        _escrever(
            destino / "config" / "runtime_b.yaml",
            RUNTIME_B.format(aviso=AVISO, conf=detector.get("confianca", 0.30),
                             iou=detector.get("iou", 0.40), entrada=entrada),
        )
        arquivos.append("config/runtime_b.yaml")
    arquivos.append(_script_preparar(ctx, destino))
    extras = SECOES_EXTRAS.get(ctx.familia, lambda _ctx: {})(ctx)
    ctx.spec["_passos_preparar"] = extras.pop("passos_preparar", [])
    secoes = _secoes_linux(ctx, familia)
    if ctx.hardware_chave.startswith("rpi") or ctx.hardware_chave == "pizero2w":
        secoes["ligacoes"] = LIGACOES_PI
    _mesclar_secoes(secoes, extras)
    _escrever(destino / "README.md", _readme(ctx, arquivos + ["README.md"], secoes))
    return arquivos


def _script_preparar(ctx: Contexto, destino: Path) -> str:
    texto = (
        "#!/usr/bin/env bash\n"
        f"# Prepara os artefatos ONNX da versão {ctx.id} (máquina de desenvolvimento, x86_64).\n"
        "set -euo pipefail\n"
        'AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'RAIZ="$(cd "$AQUI/../../.." && pwd)"\n'
        'PYTHON="${PYTHON:-$RAIZ/.venv/bin/python}"\n'
        '"$PYTHON" "$RAIZ/versoes/comum/ferramentas/preparar_modelos.py" "$AQUI"\n'
    )
    _escrever(destino / "preparar_modelo.sh", texto, executavel=True)
    return "preparar_modelo.sh"


def _mesclar_secoes(secoes: dict[str, str], extras: dict[str, str]) -> None:
    """Chaves `<seção>_extra` completam a seção genérica; as demais a substituem."""
    for chave, texto in extras.items():
        if chave.endswith("_extra"):
            base = chave[: -len("_extra")]
            secoes[base] = (secoes.get(base, "").rstrip() + "\n" + texto).strip("\n")
        else:
            secoes[chave] = texto


def _gerar_dispositivo(ctx: Contexto) -> list[str]:
    """Famílias sem a CLI Linux comum: lote, MaixPy, Grove, ESP-IDF, OpenMV, Luckfox."""
    familia = FAMILIAS[ctx.familia]
    valores = _valores(ctx, familia)
    destino = ctx.pasta
    destino.mkdir(parents=True, exist_ok=True)
    arquivos = _copiar_plataforma(familia.plataforma or ctx.familia, destino, valores)
    if ctx.familia == "luckfox":
        # Mesmo conversor RKNN do Orange Pi 5, com target_platform rv1106.
        arquivos += _copiar_plataforma("rknn", destino, valores, apenas=["converter_rknn.py"])
    if ctx.familia == "maixpy" and ctx.modelo == "modelo-a":
        (destino / "modelos" / "classificador.mud").unlink(missing_ok=True)
        arquivos.remove("modelos/classificador.mud")
    artefatos = _artefatos(ctx)
    if artefatos:
        _escrever(destino / "versao.yaml",
                  f"# {AVISO}\n" + _yaml({"id": ctx.id, "nome": ctx.titulo,
                                          "hardware": ctx.hardware_chave,
                                          "modelo": ctx.modelo, "artefatos": artefatos}))
        arquivos.append("versao.yaml")
        arquivos.append(_script_preparar(ctx, destino))
    secoes = {"limitacoes": LIMITES_COMUNS}
    _mesclar_secoes(secoes, SECOES_EXTRAS.get(ctx.familia, lambda _ctx: {})(ctx))
    _escrever(destino / "README.md", _readme(ctx, arquivos + ["README.md"], secoes))
    return arquivos


# Seções de README específicas de cada família (preenchidas em secoes_familias.py).
SECOES_EXTRAS: dict[str, Any] = {}


# --------------------------------------------------------------------------- documentos
def _matriz_md(matriz: dict[str, Any], gerados: dict[str, str]) -> str:
    hardware, softwares = matriz["hardware"], matriz["softwares"]
    colunas = ["yolov8n", "modelo_b", "megadetector_speciesnet", "tinyml"]
    simbolos = {"gerar": "✅", "condicional": "⚠️", "nao": "—"}
    por_celula: dict[tuple[str, str], dict[str, Any]] = {}
    for item in matriz["cruzamentos"]:
        por_celula[(item["hardware"], item["software"])] = item
    linhas = [
        "# Matriz de compatibilidade hardware × software/modelo",
        "",
        f"> {AVISO}",
        "",
        "Cruzamento dos hardwares candidatos (seção 4 da pesquisa) com os softwares e modelos "
        "(seção 6). ✅ faz sentido e foi gerada uma versão; ⚠️ condicional (possível, com "
        "ressalva documentada, não gerado); — não faz sentido. Clique no identificador para "
        "abrir a versão.",
        "",
        "| Hardware | " + " | ".join(softwares[coluna]["nome"] for coluna in colunas) + " |",
        "|---|" + "---|" * len(colunas),
    ]
    for chave, hw in hardware.items():
        celulas = []
        for coluna in colunas:
            item = por_celula.get((chave, coluna))
            if item is None:
                celulas.append("—")
                continue
            simbolo = simbolos[item["veredito"]]
            if item["veredito"] == "gerar" and item.get("id") in gerados:
                celulas.append(f"{simbolo} [{item['id']}]({gerados[item['id']]})")
            else:
                celulas.append(simbolo)
        linhas.append(f"| {hw['nome']} | " + " | ".join(celulas) + " |")
    linhas += [
        "",
        "## Camadas transversais",
        "",
        "| Software | Onde se aplica | Tratamento |",
        "|---|---|---|",
        "| ONNX + ONNX Runtime [6] | Pi 5, Pi Zero 2 W, Orange Pi 5 (CPU), Jetson (GPU) | "
        "Runtime direto das versões 01a/01b/05a/05b/13a/13b e do classificador nas versões "
        "Hailo/IMX500; alternativa sem NPU nas 08a/08b. Nas NPUs o ONNX é só o formato de "
        "entrada da conversão |",
        "| Conversão para NPU [22; 39] | Hailo, RKNN (RK3588S e RV1106), IMX500, SG2002, "
        "Ethos-U55 | Um script por versão (`compilar_hef.sh`, `converter_rknn.py`, "
        "`exportar_imx500.sh`, `converter_cvimodel.sh`, `exportar_tflite.sh`). INT8 com a "
        "cabeça YOLO dividida (achado medido: INT8 em todos os operadores com a cabeça "
        "inteira quebra os escores) |",
        "| Picamera2 [26; 30] | Todas as versões Raspberry Pi (01–05, 12) | `captura.tipo: "
        "picamera2`; nas demais placas, OpenCV/GStreamer, MaixPy, SSCMA, esp32-camera ou "
        "OpenMV |",
        "| Camtrap DP [11] | Todas | Linux: registro direto (`export-camtrapdp`); MCU: CSV no "
        "cartão importado em lote |",
        "| LILA BC / WCS [9; 10] | Nenhum hardware | Bases para treino e avaliação, uso "
        "offline; não há experimento novo com elas |",
        "| DeepFaune, Zamba, Timelapse, Wildlife Insights, AddaxAI [P0; 7] | Nenhum hardware "
        "embarcado | Ferramentas de desktop/nuvem; consomem os registros exportados (o "
        "AddaxAI é interface do MegaDetector) |",
        "",
        "## Justificativa de cada cruzamento",
        "",
        "| Hardware | Software/modelo | Veredito | Versão | Justificativa |",
        "|---|---|---|---|---|",
    ]
    for item in matriz["cruzamentos"]:
        versao = item.get("id", "")
        if versao and versao in gerados and item["veredito"] == "gerar":
            versao = f"[{versao}]({gerados[versao]})"
        linhas.append(
            f"| {hardware[item['hardware']]['nome']} | {softwares[item['software']]['nome']} | "
            f"{VEREDITOS[item['veredito']]} | {versao} | {item['justificativa']} |"
        )
    return "\n".join(linhas) + "\n"


def _readme_hardware(chave: str, hw: dict[str, Any], itens: list[dict[str, Any]],
                     softwares: dict[str, Any]) -> str:
    linhas = [f"# {hw['nome']}", "", f"> {AVISO}", "", f"{hw['resumo']}.", "",
              f"**Papel na pesquisa:** {hw['papel']}.", "", f"**Câmera:** {hw['camera']}.", "",
              "| Software/modelo | Veredito | Versão | Justificativa |", "|---|---|---|---|"]
    for item in itens:
        versao = item.get("id", "")
        if item["veredito"] == "gerar" and item.get("pasta"):
            versao = f"[{versao}]({item['pasta']}/README.md)"
        linhas.append(f"| {softwares[item['software']]['nome']} | {VEREDITOS[item['veredito']]} "
                      f"| {versao} | {item['justificativa']} |")
    return "\n".join(linhas) + "\n"


def main(familias: set[str] | None = None) -> int:
    matriz = yaml.safe_load((VERSOES / "matriz.yaml").read_text(encoding="utf-8"))
    especificacoes = yaml.safe_load(
        (VERSOES / "gerador" / "especificacoes.yaml").read_text(encoding="utf-8")
    )
    ranking_json = VERSOES / "ranking" / "resultados" / "ranking.json"
    ranking = json.loads(ranking_json.read_text(encoding="utf-8")) if ranking_json.is_file() else {}
    from secoes_familias import registrar  # noqa: PLC0415 - mesmo diretório

    registrar(SECOES_EXTRAS, especificacoes)
    gerados: dict[str, str] = {}
    por_hardware: dict[str, list[dict[str, Any]]] = {}
    for item in matriz["cruzamentos"]:
        por_hardware.setdefault(item["hardware"], []).append(item)
        if item["veredito"] != "gerar" or item.get("mesma_versao"):
            continue
        if familias and item["familia"] not in familias:
            gerados[str(item["id"])] = f"{matriz['hardware'][item['hardware']]['pasta']}/" \
                f"{item['pasta']}/README.md"
            continue
        ctx = Contexto(
            id=str(item["id"]),
            cruzamento=item,
            hardware_chave=item["hardware"],
            hardware=matriz["hardware"][item["hardware"]],
            software=matriz["softwares"][item["software"]],
            spec=especificacoes["versoes"][str(item["id"])],
            padroes=especificacoes["padroes"],
            ranking=ranking,
        )
        gerar = _gerar_linux if FAMILIAS[ctx.familia].linux else _gerar_dispositivo
        arquivos = gerar(ctx)
        gerados[ctx.id] = f"{ctx.hardware['pasta']}/{item['pasta']}/README.md"
        print(f"{ctx.id}: {ctx.pasta.relative_to(VERSOES)} ({len(arquivos)} arquivos)")
    for chave, itens in por_hardware.items():
        hw = matriz["hardware"][chave]
        if hw.get("pasta"):
            _escrever(VERSOES / hw["pasta"] / "README.md",
                      _readme_hardware(chave, hw, itens, matriz["softwares"]))
    _escrever(VERSOES / "MATRIZ_COMPATIBILIDADE.md", _matriz_md(matriz, gerados))
    print(f"{len(gerados)} versões geradas.")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description="Gera as pastas de versão.")
    parser.add_argument("--familias", nargs="*", help="Gera só estas famílias (ort, hailo...).")
    argumentos = parser.parse_args()
    raise SystemExit(main(set(argumentos.familias) if argumentos.familias else None))
