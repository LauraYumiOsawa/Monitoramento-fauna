"""Registro dos disparos no formato Camtrap DP (deployments, media, observations).

Todas as versões gravam aqui, inclusive as de microcontrolador (que escrevem um CSV
mínimo no cartão e são importadas depois por `importar_csv_mcu`). O objetivo é que a
saída de qualquer placa possa ser aberta nas ferramentas da área sem conversão manual, e
que a decisão da máquina fique auditável: `classifiedBy` guarda modelo e versão, e
`classificationProbability` guarda a confiança da votação.

Referência do formato: https://camtrap-dp.tdwg.org/ (versão do perfil em PERFIL).
"""

from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from animal_monitor.errors import ConfigurationError

VERSAO_CAMTRAP_DP = "1.0.1"
PERFIL = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{VERSAO_CAMTRAP_DP}/camtrap-dp-profile.json"
)
ESQUEMAS = {
    tabela: (
        f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{VERSAO_CAMTRAP_DP}/"
        f"{tabela}-table-schema.json"
    )
    for tabela in ("deployments", "media", "observations")
}

# Rótulos do projeto -> nomes científicos (Catalogue of Life / GBIF).
NOMES_CIENTIFICOS = {
    "anta": "Tapirus terrestris",
    "tamandua_bandeira": "Myrmecophaga tridactyla",
    "lobo_guara": "Chrysocyon brachyurus",
}
NOMES_POPULARES = {
    "Tapirus terrestris": "anta",
    "Myrmecophaga tridactyla": "tamanduá-bandeira",
    "Chrysocyon brachyurus": "lobo-guará",
}

COLUNAS = {
    "deployments": [
        "deploymentID", "locationID", "locationName", "latitude", "longitude",
        "coordinateUncertainty", "deploymentStart", "deploymentEnd", "setupBy", "cameraID",
        "cameraModel", "cameraDelay", "cameraHeight", "cameraDepth", "cameraTilt",
        "cameraHeading", "detectionDistance", "timestampIssues", "baitUse", "featureType",
        "habitat", "deploymentGroups", "deploymentTags", "deploymentComments",
    ],
    "media": [
        "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic",
        "fileName", "fileMediatype", "exifData", "favorite", "mediaComments",
    ],
    "observations": [
        "observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd",
        "observationLevel", "observationType", "cameraSetupType", "scientificName", "count",
        "lifeStage", "sex", "behavior", "individualID", "individualPositionRadius",
        "individualPositionAngle", "individualSpeed", "bboxX", "bboxY", "bboxWidth",
        "bboxHeight", "classificationMethod", "classifiedBy", "classificationTimestamp",
        "classificationProbability", "observationTags", "observationComments",
    ],
}
TIPOS_OBSERVACAO = {"animal", "human", "vehicle", "blank", "unknown", "unclassified"}


def agora_iso() -> str:
    """Data e hora local com fuso, exigidas pelo Camtrap DP (ISO 8601)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def novo_id(prefixo: str) -> str:
    return f"{prefixo}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True)
class Implantacao:
    """Um `deployment`: a câmera instalada em um ponto por um período."""

    deployment_id: str
    latitude: float
    longitude: float
    inicio: str
    camera_id: str
    camera_model: str
    location_id: str | None = None
    location_name: str | None = None
    coordinate_uncertainty_m: int | None = None
    setup_by: str | None = None
    camera_height_m: float | None = None
    habitat: str | None = None
    comentarios: str | None = None

    def validate(self) -> None:
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ConfigurationError("Latitude/longitude fora da faixa válida.")


class RegistroCamtrap:
    """Banco SQLite com as três tabelas do Camtrap DP, preenchido durante a operação."""

    def __init__(self, caminho: str | Path) -> None:
        self.caminho = Path(caminho).expanduser().resolve()
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.conexao = sqlite3.connect(self.caminho)
        self.conexao.execute("PRAGMA journal_mode=WAL")
        for tabela, colunas in COLUNAS.items():
            definicao = ", ".join(f'"{coluna}" TEXT' for coluna in colunas)
            self.conexao.execute(f'CREATE TABLE IF NOT EXISTS "{tabela}" ({definicao})')
        self.conexao.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_dep ON "deployments"("deploymentID")'
        )
        self.conexao.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_media ON "media"("mediaID")')
        self.conexao.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_obs ON "observations"("observationID")'
        )
        self.conexao.commit()

    def _inserir(self, tabela: str, valores: dict[str, object], substituir: bool = False) -> None:
        colunas = COLUNAS[tabela]
        desconhecidas = set(valores) - set(colunas)
        if desconhecidas:
            raise ConfigurationError(f"Campos fora do Camtrap DP em {tabela}: {desconhecidas}")
        linha = [
            None if valores.get(coluna) is None else str(valores[coluna]) for coluna in colunas
        ]
        verbo = "INSERT OR REPLACE" if substituir else "INSERT"
        marcadores = ", ".join("?" for _ in colunas)
        self.conexao.execute(f'{verbo} INTO "{tabela}" VALUES ({marcadores})', linha)
        self.conexao.commit()

    def registrar_implantacao(self, implantacao: Implantacao) -> None:
        implantacao.validate()
        self._inserir(
            "deployments",
            {
                "deploymentID": implantacao.deployment_id,
                "locationID": implantacao.location_id,
                "locationName": implantacao.location_name,
                "latitude": implantacao.latitude,
                "longitude": implantacao.longitude,
                "coordinateUncertainty": implantacao.coordinate_uncertainty_m,
                "deploymentStart": implantacao.inicio,
                "deploymentEnd": implantacao.inicio,
                "setupBy": implantacao.setup_by,
                "cameraID": implantacao.camera_id,
                "cameraModel": implantacao.camera_model,
                "cameraHeight": implantacao.camera_height_m,
                "habitat": implantacao.habitat,
                "deploymentComments": implantacao.comentarios,
            },
            substituir=True,
        )

    def encerrar_implantacao(self, deployment_id: str, fim: str | None = None) -> None:
        self.conexao.execute(
            'UPDATE "deployments" SET "deploymentEnd" = ? WHERE "deploymentID" = ?',
            (fim or agora_iso(), deployment_id),
        )
        self.conexao.commit()

    def registrar_midia(
        self,
        deployment_id: str,
        caminho_arquivo: str | Path,
        timestamp: str,
        tipo: str = "image/jpeg",
        metodo: str = "activityDetection",
        media_id: str | None = None,
        comentario: str | None = None,
    ) -> str:
        identificador = media_id or novo_id("m")
        caminho = Path(caminho_arquivo)
        self._inserir(
            "media",
            {
                "mediaID": identificador,
                "deploymentID": deployment_id,
                "captureMethod": metodo,
                "timestamp": timestamp,
                "filePath": str(caminho),
                "filePublic": "false",
                "fileName": caminho.name,
                "fileMediatype": tipo,
                "mediaComments": comentario,
            },
        )
        return identificador

    def registrar_observacao(
        self,
        deployment_id: str,
        evento_id: str,
        inicio: str,
        fim: str,
        tipo: str,
        classificado_por: str,
        rotulo: str | None = None,
        probabilidade: float | None = None,
        quantidade: int | None = None,
        caixa_normalizada: tuple[float, float, float, float] | None = None,
        media_id: str | None = None,
        comentario: str | None = None,
    ) -> str:
        """Observação por evento (padrão) ou por mídia, se `media_id` for informado.

        `rotulo` é o rótulo do projeto (`anta`...) e vira `scientificName`. A classe de
        rejeição do Modelo B (`desconhecido`) vira `observationType = unknown`, sem nome:
        o formato distingue "não sei qual animal" de "não há animal" (`blank`).
        """
        if tipo not in TIPOS_OBSERVACAO:
            raise ConfigurationError(f"observationType inválido: {tipo}")
        nome = NOMES_CIENTIFICOS.get(rotulo or "")
        if tipo == "animal" and rotulo and nome is None:
            tipo = "unknown"
        valores: dict[str, object] = {
            "observationID": novo_id("o"),
            "deploymentID": deployment_id,
            "mediaID": media_id,
            "eventID": evento_id,
            "eventStart": inicio,
            "eventEnd": fim,
            "observationLevel": "media" if media_id else "event",
            "observationType": tipo,
            "cameraSetupType": None,
            "scientificName": nome if tipo == "animal" else None,
            "count": quantidade,
            "classificationMethod": "machine",
            "classifiedBy": classificado_por,
            "classificationTimestamp": agora_iso(),
            "classificationProbability": (
                None if probabilidade is None else round(float(probabilidade), 4)
            ),
            "observationComments": comentario,
        }
        if caixa_normalizada is not None:
            x, y, largura, altura = (min(1.0, max(0.0, float(v))) for v in caixa_normalizada)
            valores.update(
                {"bboxX": round(x, 4), "bboxY": round(y, 4),
                 "bboxWidth": round(largura, 4), "bboxHeight": round(altura, 4)}
            )
        self._inserir("observations", valores)
        return str(valores["observationID"])

    def linhas(self, tabela: str) -> list[dict[str, str | None]]:
        cursor = self.conexao.execute(f'SELECT * FROM "{tabela}"')
        return [dict(zip(COLUNAS[tabela], linha, strict=True)) for linha in cursor.fetchall()]

    def exportar(
        self,
        pasta: str | Path,
        titulo: str,
        contato: str,
        descricao: str | None = None,
        licenca_dados: str = "CC-BY-4.0",
    ) -> Path:
        """Grava os três CSV e o datapackage.json. Devolve o caminho do pacote."""
        destino = Path(pasta).expanduser().resolve()
        destino.mkdir(parents=True, exist_ok=True)
        tabelas = {tabela: self.linhas(tabela) for tabela in COLUNAS}
        for tabela, linhas in tabelas.items():
            with (destino / f"{tabela}.csv").open("w", newline="", encoding="utf-8") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS[tabela])
                escritor.writeheader()
                for linha in linhas:
                    escritor.writerow({chave: valor or "" for chave, valor in linha.items()})
        implantacoes = tabelas["deployments"]
        if not implantacoes:
            raise ConfigurationError(
                "Nenhum deployment registrado; o Camtrap DP exige ao menos um."
            )
        latitudes = [float(linha["latitude"] or 0) for linha in implantacoes]
        longitudes = [float(linha["longitude"] or 0) for linha in implantacoes]
        inicios = sorted(str(linha["deploymentStart"]) for linha in implantacoes)
        fins = sorted(
            str(linha["deploymentEnd"] or linha["deploymentStart"]) for linha in implantacoes
        )
        especies = sorted(
            {
                linha["scientificName"]
                for linha in tabelas["observations"]
                if linha["scientificName"]
            }
        )
        pacote = {
            "profile": PERFIL,
            "name": "monitoramento-fauna-" + datetime.now().strftime("%Y%m%d"),
            "id": str(uuid.uuid4()),
            "created": agora_iso(),
            "title": titulo,
            "description": descricao,
            "contributors": [{"title": contato, "role": "contact"}],
            "licenses": [{"name": licenca_dados, "scope": "data"}],
            "project": {
                "title": titulo,
                "samplingDesign": "opportunistic",
                "captureMethod": ["activityDetection"],
                "individualAnimals": False,
                "observationLevel": ["event"],
            },
            "spatial": {
                "type": "Polygon",
                "bbox": [min(longitudes), min(latitudes), max(longitudes), max(latitudes)],
                "coordinates": [[
                    [min(longitudes), min(latitudes)], [max(longitudes), min(latitudes)],
                    [max(longitudes), max(latitudes)], [min(longitudes), max(latitudes)],
                    [min(longitudes), min(latitudes)],
                ]],
            },
            "temporal": {"start": inicios[0][:10], "end": fins[-1][:10]},
            "taxonomic": [
                {
                    "scientificName": nome,
                    "taxonRank": "species",
                    "vernacularNames": {"por": NOMES_POPULARES.get(nome, "")},
                }
                for nome in especies
            ],
            "resources": [
                {
                    "name": tabela,
                    "path": f"{tabela}.csv",
                    "profile": "tabular-data-resource",
                    "format": "csv",
                    "mediatype": "text/csv",
                    "encoding": "utf-8",
                    "schema": ESQUEMAS[tabela],
                }
                for tabela in COLUNAS
            ],
        }
        caminho = destino / "datapackage.json"
        caminho.write_text(json.dumps(pacote, ensure_ascii=False, indent=2), encoding="utf-8")
        return caminho

    def close(self) -> None:
        self.conexao.close()

    def __enter__(self) -> RegistroCamtrap:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def importar_csv_mcu(
    registro: RegistroCamtrap,
    csv_mcu: str | Path,
    deployment_id: str,
    pasta_midias: Path,
    classificado_por: str,
) -> dict[str, int]:
    """Importa o `registros.csv` gravado pelas versões sem Linux completo.

    Formato (uma linha por foto): `evento,timestamp_iso,arquivo,rotulo,probabilidade`.
    `rotulo`: espécie do projeto (confirmada no dispositivo), `inconclusivo`, `vazio`,
    `desconhecido` ou vazio (sem classificação no dispositivo). Cada evento vira uma
    observação: espécie → `animal`; inconclusivo/desconhecido → `unknown`; vazio → `blank`;
    sem rótulo → `unclassified` (para classificar depois em lote, versões 01c/08c/13c).
    Nas versões de quadro inteiro (ESP32, Nicla) cada foto traz o próprio rótulo; o evento
    fica com o rótulo mais frequente entre as fotos.
    """
    eventos: dict[str, list[dict[str, str]]] = {}
    with Path(csv_mcu).open(newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            eventos.setdefault(linha["evento"], []).append(linha)
    contagem: dict[str, int] = {}
    for evento, linhas in eventos.items():
        for linha in linhas:
            registro.registrar_midia(
                deployment_id, pasta_midias / linha["arquivo"], linha["timestamp_iso"]
            )
        rotulos = [linha.get("rotulo", "") for linha in linhas]
        rotulo = max(set(rotulos), key=rotulos.count)
        probabilidade = max(
            (float(item.get("probabilidade") or 0) for item in linhas if item["rotulo"] == rotulo),
            default=0.0,
        )
        if rotulo in NOMES_CIENTIFICOS:
            tipo = "animal"
        elif rotulo in {"inconclusivo", "desconhecido"}:
            tipo = "unknown"
        elif rotulo == "vazio":
            tipo = "blank"
        else:
            tipo = "unclassified"
        registro.registrar_observacao(
            deployment_id, evento, linhas[0]["timestamp_iso"], linhas[-1]["timestamp_iso"], tipo,
            classificado_por, rotulo=rotulo if tipo == "animal" else None,
            probabilidade=probabilidade if tipo in {"animal", "unknown"} else None,
            quantidade=1 if tipo == "animal" else None,
            comentario=f"rótulos das fotos: {','.join(rotulos)}" if len(set(rotulos)) > 1 else None,
        )
        contagem[tipo] = contagem.get(tipo, 0) + 1
    return contagem


def observacoes_de_eventos(eventos: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Utilitário para relatórios: conta observações por espécie a partir de eventos."""
    contagem: dict[str, int] = {}
    for evento in eventos:
        chave = str(evento.get("especie") or evento.get("tipo") or "unclassified")
        contagem[chave] = contagem.get(chave, 0) + 1
    return [{"rotulo": chave, "eventos": total} for chave, total in sorted(contagem.items())]
