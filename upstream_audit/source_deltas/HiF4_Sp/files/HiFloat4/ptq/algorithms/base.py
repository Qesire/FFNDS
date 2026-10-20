from __future__ import annotations

from typing import Protocol

import torch

from ..formats.base import WeightFormatBackend
from ..state import CurvatureStats, QuantizedWeightState


class WeightQuantizationAlgorithm(Protocol):
    name: str

    def quantize(
        self,
        weight: torch.Tensor,
        *,
        format_backend: WeightFormatBackend,
        curvature: CurvatureStats | None = None,
    ) -> QuantizedWeightState:
        ...
