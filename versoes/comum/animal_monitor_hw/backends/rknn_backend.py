"""NPU Rockchip (RK3588S do Orange Pi 5) via rknn-toolkit-lite2.

O artefato `.rknn` é gerado no x86 pelo RKNN-Toolkit2 (`converter_rknn.py` da versão),
já com a normalização embutida (mean 0, std 255 no detector; ImageNet no classificador).
Por isso a entrada aqui é uint8 NHWC, sem divisão por 255 no host.

O RV1106 (Luckfox Pico) não é suportado pelo toolkit-lite2: lá a versão usa a API C.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from animal_monitor.errors import ConfigurationError
from animal_monitor_hw.backends.base import TensorSpec

NUCLEOS = {
    "auto": "NPU_CORE_AUTO",
    "0": "NPU_CORE_0",
    "1": "NPU_CORE_1",
    "2": "NPU_CORE_2",
    "0_1": "NPU_CORE_0_1",
    "0_1_2": "NPU_CORE_0_1_2",
}


class RknnLiteRunner:
    def __init__(
        self,
        model_path: str | Path,
        nucleos: str = "auto",
        entrada: Sequence[int] | None = None,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "rknn-toolkit-lite2 não instalado (pip install rknn-toolkit-lite2 na placa)."
            ) from exc
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ConfigurationError(f"Modelo RKNN não encontrado: {self.model_path}")
        if nucleos not in NUCLEOS:
            raise ConfigurationError(f"'nucleos_npu' deve ser um de {sorted(NUCLEOS)}.")
        if not entrada or len(entrada) != 2:
            # O .rknn não expõe a forma de entrada pela API lite de modo portátil entre
            # versões; a versão declara a mesma entrada usada na conversão.
            raise ConfigurationError("Informe 'entrada: [altura, largura]' do modelo RKNN.")
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(str(self.model_path)) != 0:
            raise RuntimeError(f"Falha ao carregar {self.model_path}")
        mascara = getattr(RKNNLite, NUCLEOS[nucleos])
        if self.rknn.init_runtime(core_mask=mascara) != 0:
            raise RuntimeError("Falha ao iniciar o runtime da NPU (driver RKNPU carregado?).")
        self.nucleos = nucleos
        self.spec = TensorSpec(height=int(entrada[0]), width=int(entrada[1]), layout="nhwc",
                               dtype="uint8")

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        saidas = self.rknn.inference(inputs=[np.ascontiguousarray(tensor)], data_format=["nhwc"])
        if saidas is None:
            raise RuntimeError("A NPU não retornou saídas.")
        return [np.asarray(saida, dtype=np.float32) for saida in saidas]

    def describe(self) -> dict[str, object]:
        return {"runtime": "rknn-toolkit-lite2", "model": str(self.model_path),
                "nucleos_npu": self.nucleos, "entrada": [self.spec.height, self.spec.width]}

    def close(self) -> None:
        self.rknn.release()
