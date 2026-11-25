"""Hailo-8L / Hailo-8 (AI HAT+) e Hailo-10H (AI HAT+ 2) via HailoRT (`hailo_platform`).

API usada: `VDevice` → `create_infer_model` → `configure` → `create_bindings` → `run`, a
mesma do tutorial oficial do HailoRT e do `HailoInfer` do hailo-apps. O pacote vem do
sistema: `sudo apt install hailo-all` (Hailo-8/8L, HailoRT 4.x) ou `hailo-h10-all`
(Hailo-10H, HailoRT 5.x) — os dois conjuntos não coexistem na mesma instalação.

Formato:
* entrada: uint8 RGB (H, W, 3) — a divisão por 255 está no HEF (`normalization` do alls);
* saída com NMS (HEF compilado com `nms_postprocess`): `get_buffer()` devolve uma lista com
  um array por classe, linhas `[y_min, x_min, y_max, x_max, score]` normalizadas (ordem
  y-primeiro do `hailo_bbox_float32_t` em hailort.h). O NMS roda no HailoRT
  (`engine=cpu`), com limiares ajustáveis em tempo de execução.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.backends.base import TensorSpec


class HailoRunner:
    def __init__(
        self,
        hef_path: str | Path,
        nms: bool = True,
        score_threshold: float | None = None,
        iou_threshold: float | None = None,
        timeout_ms: int = 10000,
    ) -> None:
        try:
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
        except ImportError as exc:
            raise RuntimeError(
                "HailoRT não encontrado: instale `hailo-all` (Hailo-8/8L) ou `hailo-h10-all` "
                "(Hailo-10H) e crie o venv com --system-site-packages."
            ) from exc
        self.hef_path = Path(hef_path).expanduser().resolve()
        if not self.hef_path.is_file():
            raise ConfigurationError(f"HEF não encontrado: {self.hef_path}")
        self.nms = nms
        self.timeout_ms = int(timeout_ms)

        parametros = VDevice.create_params()
        # ROUND_ROBIN deixa outro processo (ex.: um segundo HEF) dividir o mesmo chip.
        parametros.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(parametros)
        self.infer_model = self.vdevice.create_infer_model(str(self.hef_path))
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.nomes_saida = [saida.name for saida in self.infer_model.outputs]
        for nome in self.nomes_saida:
            saida = self.infer_model.output(nome)
            saida.set_format_type(FormatType.FLOAT32)
            if nms and score_threshold is not None:
                saida.set_nms_score_threshold(float(score_threshold))
            if nms and iou_threshold is not None:
                saida.set_nms_iou_threshold(float(iou_threshold))
        self._configurado = self.infer_model.configure()
        self.configurado = self._configurado.__enter__()
        altura, largura, canais = self.infer_model.input().shape
        if canais != 3:
            raise ConfigurationError(f"Entrada do HEF com {canais} canais; esperado RGB.")
        self.spec = TensorSpec(height=int(altura), width=int(largura), layout="nhwc",
                               dtype="uint8")

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        quadro = np.ascontiguousarray(tensor[0] if tensor.ndim == 4 else tensor, dtype=np.uint8)
        buffers = {
            nome: np.empty(self.infer_model.output(nome).shape, dtype=np.float32)
            for nome in self.nomes_saida
        }
        ligacoes = self.configurado.create_bindings(output_buffers=buffers)
        ligacoes.input().set_buffer(quadro)
        self.configurado.run([ligacoes], self.timeout_ms)
        saidas = [ligacoes.output(nome).get_buffer() for nome in self.nomes_saida]
        if self.nms and len(saidas) == 1:
            # Lista por classe; o decodificador `hailo_nms` espera exatamente isto.
            return [list(saidas[0])]
        return [np.asarray(saida) for saida in saidas]

    def describe(self) -> dict[str, object]:
        return {"runtime": "hailort", "model": str(self.hef_path), "outputs": self.nomes_saida,
                "entrada": [self.spec.height, self.spec.width], "nms_no_hailort": self.nms}

    def close(self) -> None:
        self._configurado.__exit__(None, None, None)
        self.vdevice.release()
