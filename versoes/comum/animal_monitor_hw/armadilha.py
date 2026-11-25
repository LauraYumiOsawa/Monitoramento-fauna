"""Ciclo da armadilha: espera → disparo → rajada → análise → registro → espera.

Implementa as três arquiteturas de energia da seção 3 da pesquisa:

* A — processador ativo: o processo fica aguardando o PIR (`gatilho.tipo` gpiozero/gpiod);
* B — inicialização por evento: um circuito externo liga a placa no PIR; o serviço roda
  um único disparo (`gatilho.tipo: nenhum`) e, se `desligar_ao_final`, desliga a placa;
* C — captura independente: um MCU grava as fotos; `processar_lote` classifica depois.

Os tempos ficam no log (`eventos.jsonl`) com a mesma separação que a pesquisa exige:
tempo até a primeira captura, duração da rajada e latência de inferência não se
substituem umas pelas outras.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor_hw.camtrap_dp import Implantacao, RegistroCamtrap, agora_iso, novo_id
from animal_monitor_hw.captura import Camera, QuadroCapturado
from animal_monitor_hw.gatilho import Gatilho, SaidaDigital
from animal_monitor_hw.versao import Versao


def _salvar_jpeg(imagem: np.ndarray, pasta: Path, nome: str, qualidade: int) -> Path:
    import cv2

    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    if not cv2.imwrite(str(caminho), imagem, [int(cv2.IMWRITE_JPEG_QUALITY), qualidade]):
        raise RuntimeError(f"Falha ao gravar {caminho}")
    return caminho


def _caixa_normalizada(caixa, largura: int, altura: int) -> tuple[float, float, float, float]:
    return (
        caixa.x1 / largura,
        caixa.y1 / altura,
        (caixa.x2 - caixa.x1) / largura,
        (caixa.y2 - caixa.y1) / altura,
    )


class Armadilha:
    def __init__(
        self,
        versao: Versao,
        camera: Camera,
        gatilho: Gatilho,
        iluminador: SaidaDigital,
        analisador,
        registro: RegistroCamtrap,
    ) -> None:
        self.versao = versao
        self.camera = camera
        self.gatilho = gatilho
        self.iluminador = iluminador
        self.analisador = analisador
        self.registro = registro
        self.evento_cfg = versao.secao("evento")
        self.armazenamento = versao.secao("armazenamento")
        self.pasta = versao.caminho(self.armazenamento.get("pasta", "dados"))
        self.log = self.pasta / "eventos.jsonl"
        implantacao = versao.secao("implantacao")
        self.deployment_id = str(implantacao.get("deployment_id", f"dep-{versao.identificador}"))
        self.registro.registrar_implantacao(
            Implantacao(
                deployment_id=self.deployment_id,
                latitude=float(implantacao.get("latitude", 0.0)),
                longitude=float(implantacao.get("longitude", 0.0)),
                inicio=agora_iso(),
                camera_id=str(implantacao.get("camera_id", versao.identificador)),
                camera_model=str(versao.dados.get("nome", versao.dados["hardware"])),
                location_name=implantacao.get("local"),
                coordinate_uncertainty_m=implantacao.get("incerteza_m"),
                setup_by=implantacao.get("instalado_por"),
                camera_height_m=implantacao.get("altura_m"),
                habitat=implantacao.get("habitat"),
                comentarios=f"versão {versao.identificador}; arquitetura "
                f"{versao.dados.get('arquitetura_energia', '?')}",
            )
        )

    def _noite(self) -> bool:
        iluminacao = self.versao.secao("iluminacao_ir")
        modo = str(iluminacao.get("modo", "nunca"))
        if modo in {"sempre", "nunca"}:
            return modo == "sempre"
        lux = self.camera.luminosidade()
        if lux is None:
            hora = datetime.now().hour
            inicio, fim = iluminacao.get("noite_horas", [18, 6])
            return hora >= int(inicio) or hora < int(fim)
        return lux < float(iluminacao.get("limiar_lux", 5.0))

    def disparo(self) -> dict[str, object]:
        inicio = time.monotonic()
        evento_id = novo_id("e")
        noite = self._noite()
        self.iluminador.definir(noite)
        try:
            quadros = self.camera.rajada(
                int(self.evento_cfg.get("quadros", 8)),
                float(self.evento_cfg.get("intervalo_s", 0.15)),
            )
        finally:
            self.iluminador.definir(False)
        fim_captura = time.monotonic()
        if not quadros:
            return {"evento": evento_id, "erro": "câmera não entregou quadros"}
        resultado = self.analisador.analisar(quadros)
        fim_analise = time.monotonic()
        midias = self._registrar(evento_id, quadros, resultado)
        resumo = {
            "evento": evento_id,
            "inicio": quadros[0].timestamp,
            "noite": noite,
            "tempo_ate_primeira_captura_s": round(quadros[0].instante_s - inicio, 3),
            "duracao_rajada_s": round(fim_captura - quadros[0].instante_s, 3),
            "duracao_analise_s": round(fim_analise - fim_captura, 3),
            "midias_gravadas": midias,
            **resultado.resumo(),
        }
        self.pasta.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(resumo, ensure_ascii=False) + "\n")
        return resumo

    def _registrar(self, evento_id: str, quadros: list[QuadroCapturado], resultado) -> int:
        salvar_vazias = bool(self.armazenamento.get("salvar_vazias", True))
        qualidade = int(self.armazenamento.get("qualidade_jpeg", 90))
        pasta_evento = self.pasta / "midias" / quadros[0].timestamp[:10] / evento_id
        media_ids: list[str] = []
        if resultado.tipo != "blank" or salvar_vazias:
            for indice, quadro in enumerate(quadros):
                caminho = _salvar_jpeg(quadro.imagem, pasta_evento, f"{indice:03d}.jpg", qualidade)
                media_ids.append(
                    self.registro.registrar_midia(self.deployment_id, caminho, quadro.timestamp)
                )
        inicio, fim = quadros[0].timestamp, quadros[-1].timestamp
        altura, largura = quadros[0].imagem.shape[:2]
        if resultado.tipo == "blank":
            self.registro.registrar_observacao(
                self.deployment_id, evento_id, inicio, fim, "blank",
                self.versao.classificado_por,
            )
            return len(media_ids)
        contagem: dict[tuple[str, str], int] = defaultdict(int)
        for item in resultado.avistamentos:
            contagem[(item.rotulo, item.tipo)] += 1
        registrados: set[tuple[str, str]] = set()
        for item in resultado.avistamentos:
            chave = (item.rotulo, item.tipo)
            if chave in registrados:
                continue
            registrados.add(chave)
            self.registro.registrar_observacao(
                self.deployment_id,
                evento_id,
                inicio,
                fim,
                item.tipo,
                self.versao.classificado_por,
                rotulo=item.rotulo if item.tipo == "animal" else None,
                probabilidade=item.probabilidade,
                quantidade=contagem[chave],
                caixa_normalizada=_caixa_normalizada(item.caixa, largura, altura),
                comentario=None if item.tipo == "animal" else f"inconclusivo ({item.rotulo})",
            )
        return len(media_ids)

    def executar(self, max_eventos: int | None = None, pausa_s: float | None = None) -> int:
        pausa = float(self.evento_cfg.get("pausa_s", 10.0)) if pausa_s is None else pausa_s
        eventos = 0
        try:
            while max_eventos is None or eventos < max_eventos:
                if not self.gatilho.aguardar(timeout_s=60.0):
                    # Sem disparo no intervalo: PIR segue esperando; simulado/arquitetura B
                    # sinalizam que não haverá mais eventos.
                    if getattr(self.gatilho, "esgotado", False):
                        break
                    continue
                resumo = self.disparo()
                print(json.dumps(resumo, ensure_ascii=False), flush=True)
                eventos += 1
                if pausa > 0:
                    time.sleep(pausa)
        finally:
            self.registro.encerrar_implantacao(self.deployment_id)
            self.camera.parar()
            self.iluminador.fechar()
            self.gatilho.fechar()
        if bool(self.versao.secao("energia").get("desligar_ao_final", False)):
            # Arquitetura B: devolve o controle ao circuito que corta a alimentação.
            subprocess.run(["sudo", "systemctl", "poweroff"], check=False)
        return eventos


def _agrupar_por_tempo(arquivos: list[Path], janela_s: float) -> list[list[Path]]:
    grupos: list[list[Path]] = []
    ultimo = None
    for arquivo in sorted(arquivos, key=lambda caminho: caminho.stat().st_mtime):
        instante = arquivo.stat().st_mtime
        if ultimo is None or instante - ultimo > janela_s:
            grupos.append([])
        grupos[-1].append(arquivo)
        ultimo = instante
    return grupos


def processar_lote(
    versao: Versao,
    entrada: str | Path,
    csv_mcu: str | Path | None = None,
    janela_evento_s: float = 30.0,
) -> dict[str, object]:
    """Arquitetura C: classifica em lote as fotos gravadas por outro subsistema."""
    import cv2

    pasta = Path(entrada).expanduser().resolve()
    if csv_mcu:
        grupos_dict: dict[str, list[Path]] = defaultdict(list)
        with Path(csv_mcu).open(newline="", encoding="utf-8") as arquivo:
            for linha in csv.DictReader(arquivo):
                grupos_dict[linha["evento"]].append(pasta / linha["arquivo"])
        grupos = list(grupos_dict.values())
    else:
        arquivos = [
            caminho for caminho in pasta.rglob("*") if caminho.suffix.lower() in IMAGE_SUFFIXES
        ]
        grupos = _agrupar_por_tempo(arquivos, janela_evento_s)

    analisador = versao.construir_analisador()
    armazenamento = versao.secao("armazenamento")
    registro_caminho = versao.caminho(armazenamento.get("registro", "dados/camtrap.sqlite3"))
    implantacao = versao.secao("implantacao")
    deployment_id = str(implantacao.get("deployment_id", f"dep-{versao.identificador}"))
    contagem: dict[str, int] = defaultdict(int)
    with RegistroCamtrap(registro_caminho) as registro:
        registro.registrar_implantacao(
            Implantacao(
                deployment_id=deployment_id,
                latitude=float(implantacao.get("latitude", 0.0)),
                longitude=float(implantacao.get("longitude", 0.0)),
                inicio=agora_iso(),
                camera_id=str(implantacao.get("camera_id", versao.identificador)),
                camera_model=str(implantacao.get("camera_modelo", "subsistema de captura")),
                comentarios=f"lote processado pela versão {versao.identificador}",
            )
        )
        for grupo in grupos:
            quadros = []
            for caminho in grupo:
                imagem = cv2.imread(str(caminho))
                if imagem is None:
                    continue
                momento = datetime.fromtimestamp(caminho.stat().st_mtime).astimezone()
                quadros.append(
                    QuadroCapturado(momento.isoformat(timespec="seconds"),
                                    caminho.stat().st_mtime, imagem)
                )
            if not quadros:
                continue
            resultado = analisador.analisar(quadros)
            evento_id = novo_id("e")
            for caminho, quadro in zip(grupo, quadros, strict=False):
                registro.registrar_midia(deployment_id, caminho, quadro.timestamp)
            altura, largura = quadros[0].imagem.shape[:2]
            if resultado.tipo == "blank":
                registro.registrar_observacao(
                    deployment_id, evento_id, quadros[0].timestamp, quadros[-1].timestamp,
                    "blank", versao.classificado_por,
                )
            for item in resultado.avistamentos:
                registro.registrar_observacao(
                    deployment_id, evento_id, quadros[0].timestamp, quadros[-1].timestamp,
                    item.tipo, versao.classificado_por,
                    rotulo=item.rotulo if item.tipo == "animal" else None,
                    probabilidade=item.probabilidade, quantidade=1,
                    caixa_normalizada=_caixa_normalizada(item.caixa, largura, altura),
                )
            contagem[resultado.tipo] += 1
    return {"eventos": len(grupos), "por_tipo": dict(contagem), "registro": str(registro_caminho)}
