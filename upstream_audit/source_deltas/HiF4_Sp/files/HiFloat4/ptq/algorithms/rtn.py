from __future__ import annotations

from dataclasses import dataclass

import torch

from ..formats.base import WeightFormatBackend
from ..state import QuantizedWeightState


@dataclass(frozen=True)
class RTNConfig:
    quant_device: str = "cuda"


@torch.no_grad()
def quantize_rtn(
    weight: torch.Tensor,
    format_backend: WeightFormatBackend,
    config: RTNConfig,
) -> QuantizedWeightState:
    quantized = format_backend.quantize_weight(
        weight,
        quant_device=config.quant_device,
        return_device=weight.device,
    )
    difference = weight.detach().float() - quantized.float()
    rmse = difference.pow(2).mean().sqrt().item()
    denominator = weight.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)
    rel_rmse = float(rmse / denominator.item())
    return QuantizedWeightState(
        dequantized_weight=quantized,
        format_name=format_backend.name,
        algorithm="rtn",
        metadata={**format_backend.metadata(), "quant_device": config.quant_device},
        report={"weight_rmse": float(rmse), "weight_rel_rmse": rel_rmse},
    )
