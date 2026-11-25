"""Runners por runtime. Os módulos de SDK proprietário só são importados sob demanda."""

from animal_monitor_hw.backends.base import Runner, TensorSpec, rgb_to_tensor

__all__ = ["Runner", "TensorSpec", "rgb_to_tensor"]
