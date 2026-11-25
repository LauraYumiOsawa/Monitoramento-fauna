"""Carrega `versao.yaml` e monta detector, classificador, câmera e gatilho da versão.

Cada pasta de versão descreve só o que muda: hardware, runtime, artefato do modelo,
câmera e arquitetura de energia. Caminhos relativos são resolvidos a partir da pasta da
versão, para que ela funcione tanto dentro do repositório quanto empacotada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animal_monitor.config import load_class_names, load_yaml
from animal_monitor.errors import ConfigurationError
from animal_monitor_b.config import RuntimeConfigB
from animal_monitor_hw.analise import AnalisadorModeloA, AnalisadorModeloB
from animal_monitor_hw.backends.base import Runner
from animal_monitor_hw.models import CropClassifier, YoloDetector

RUNTIMES = {"onnxruntime", "tensorrt", "hailo", "rknn", "imx500"}


@dataclass(slots=True)
class Versao:
    raiz: Path
    dados: dict[str, Any]

    @classmethod
    def carregar(cls, caminho: str | Path) -> Versao:
        arquivo = Path(caminho).expanduser().resolve()
        if arquivo.is_dir():
            arquivo = arquivo / "versao.yaml"
        dados = load_yaml(arquivo)
        for chave in ("id", "hardware", "modelo", "detector"):
            if chave not in dados:
                raise ConfigurationError(f"'{chave}' ausente em {arquivo}.")
        if dados["modelo"] not in {"modelo-a", "modelo-b"}:
            raise ConfigurationError("'modelo' deve ser 'modelo-a' ou 'modelo-b'.")
        return cls(raiz=arquivo.parent, dados=dados)

    def caminho(self, valor: str | Path) -> Path:
        candidato = Path(str(valor)).expanduser()
        return candidato if candidato.is_absolute() else (self.raiz / candidato).resolve()

    def secao(self, nome: str) -> dict[str, Any]:
        valor = self.dados.get(nome) or {}
        if not isinstance(valor, dict):
            raise ConfigurationError(f"'{nome}' deve ser um objeto em versao.yaml.")
        return valor

    @property
    def identificador(self) -> str:
        return str(self.dados["id"])

    @property
    def classificado_por(self) -> str:
        """Texto de `classifiedBy` no Camtrap DP: versão, modelo e runtime."""
        detector = self.secao("detector")
        partes = [f"animal-monitor-hw {self.identificador}", str(detector.get("arquivo", ""))]
        if self.dados["modelo"] == "modelo-b":
            partes.append(str(self.secao("classificador").get("arquivo", "")))
        return " | ".join(Path(parte).name if "/" in parte else parte for parte in partes)

    def _runner(self, secao: dict[str, Any]) -> Runner:
        runtime = str(secao.get("runtime", "onnxruntime"))
        if runtime not in RUNTIMES - {"imx500"}:
            raise ConfigurationError(f"Runtime sem runner próprio: {runtime}")
        arquivo = self.caminho(secao["arquivo"])
        if runtime in {"onnxruntime", "tensorrt"}:
            from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner

            padrao = (
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                if runtime == "tensorrt"
                else ["CPUExecutionProvider"]
            )
            return OnnxRuntimeRunner(
                arquivo,
                providers=secao.get("providers", padrao),
                provider_options=secao.get("provider_options"),
                intra_op_threads=int(secao.get("threads", 0)),
            )
        if runtime == "hailo":
            from animal_monitor_hw.backends.hailo_backend import HailoRunner

            # Só o detector tem NMS no HEF; os limiares da versão substituem os do export.
            com_nms = bool(secao.get("nms_no_chip", "confianca" in secao))
            return HailoRunner(
                arquivo,
                nms=com_nms,
                score_threshold=secao.get("confianca") if com_nms else None,
                iou_threshold=secao.get("iou") if com_nms else None,
            )
        from animal_monitor_hw.backends.rknn_backend import RknnLiteRunner

        return RknnLiteRunner(arquivo, nucleos=str(secao.get("nucleos_npu", "auto")),
                              entrada=secao.get("entrada"))

    def nomes_classes(self) -> dict[int, str]:
        detector = self.secao("detector")
        if "classes" in detector:
            return {int(chave): str(valor) for chave, valor in detector["classes"].items()}
        dataset = detector.get("dataset")
        if dataset:
            return load_class_names(self.caminho(dataset))
        raise ConfigurationError("Informe 'detector.classes' ou 'detector.dataset'.")

    def construir_detector(self):
        secao = self.secao("detector")
        if secao.get("runtime") == "imx500":
            from animal_monitor_hw.backends.imx500_backend import Imx500Detector

            captura = self.secao("captura")
            return Imx500Detector(
                self.caminho(secao["arquivo"]),
                self.nomes_classes(),
                confidence=float(secao.get("confianca", 0.35)),
                max_detections=int(secao.get("max_deteccoes", 20)),
                bbox_normalizada=bool(secao.get("bbox_normalizada", True)),
                ordem_bbox=str(secao.get("ordem_bbox", "xy")),
                largura=int(captura.get("largura", 1536)),
                altura=int(captura.get("altura", 864)),
                fps=float(captura.get("fps", 10.0)),
                preservar_proporcao=bool(captura.get("preservar_proporcao", True)),
            )
        return YoloDetector(
            self._runner(secao),
            self.nomes_classes(),
            confidence=float(secao.get("confianca", 0.35)),
            iou_threshold=float(secao.get("iou", 0.45)),
            max_detections=int(secao.get("max_deteccoes", 100)),
            output_format=str(secao.get("formato_saida", "standard")),
            split_layout=str(secao.get("layout_saida", "nchw")),
            model_path=self.caminho(secao["arquivo"]),
        )

    def construir_classificador(self) -> CropClassifier | None:
        if self.dados["modelo"] != "modelo-b":
            return None
        secao = self.secao("classificador")
        return CropClassifier(
            self._runner(secao),
            [str(rotulo) for rotulo in secao["rotulos"]],
            crop_margin=float(secao.get("margem", 0.12)),
            model_path=self.caminho(secao["arquivo"]),
        )

    def config_modelo_b(self) -> RuntimeConfigB:
        return RuntimeConfigB.from_yaml(self.caminho(self.dados.get("runtime_b",
                                                                    "config/runtime_b.yaml")))

    def construir_analisador(self, detector=None):
        detector = detector if detector is not None else self.construir_detector()
        if self.dados["modelo"] == "modelo-a":
            return AnalisadorModeloA(detector, int(self.secao("evento").get("min_hits", 3)))
        return AnalisadorModeloB(detector, self.construir_classificador(), self.config_modelo_b())

    def resumo(self) -> dict[str, Any]:
        return {
            "id": self.identificador,
            "nome": self.dados.get("nome"),
            "hardware": self.dados["hardware"],
            "modelo": self.dados["modelo"],
            "arquitetura_energia": self.dados.get("arquitetura_energia"),
            "detector": self.secao("detector"),
            "classificador": self.secao("classificador") or None,
            "captura": self.secao("captura"),
            "gatilho": self.secao("gatilho"),
        }
