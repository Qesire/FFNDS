from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..formats.hifx4 import HIFX4Format
from ..state import QuantizedWeightState


class HIFX4Linear(nn.Module):
    """通用 hifx4 fake-QDQ Linear，可承载 RTN 或后续 GPTQ 权重。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        weight_qdq: torch.Tensor,
        bias_tensor: torch.Tensor | None = None,
        smooth_scale: torch.Tensor | None = None,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
        algorithm: str = "unknown",
    ) -> None:
        super().__init__()
        if tuple(weight_qdq.shape) != (out_features, in_features):
            raise ValueError("weight shape mismatch")
        if smooth_scale is not None:
            if smooth_scale.ndim != 1 or smooth_scale.numel() != in_features:
                raise ValueError("smooth_scale shape mismatch")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation_qdq = bool(activation_qdq)
        self.source_name = source_name
        self.algorithm = algorithm
        self.format_backend = HIFX4Format(force_fp32=force_fp32)
        self.weight = nn.Parameter(weight_qdq.detach().clone(), requires_grad=False)
        if bias:
            if bias_tensor is None:
                bias_tensor = torch.zeros(
                    out_features, dtype=weight_qdq.dtype, device=weight_qdq.device
                )
            self.bias = nn.Parameter(bias_tensor.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if smooth_scale is None:
            self.register_buffer("smooth_scale", None, persistent=True)
        else:
            self.register_buffer(
                "smooth_scale",
                smooth_scale.detach().cpu().float().contiguous(),
                persistent=True,
            )

    @classmethod
    def from_state(
        cls,
        module: nn.Linear,
        state: QuantizedWeightState,
        *,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
    ) -> "HIFX4Linear":
        state.validate()
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            weight_qdq=state.dequantized_weight.to(
                device=module.weight.device, dtype=module.weight.dtype
            ),
            bias_tensor=None if module.bias is None else module.bias.detach(),
            smooth_scale=state.smooth_scale,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=source_name,
            algorithm=state.algorithm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x
        if self.smooth_scale is not None:
            scale = self.smooth_scale.to(device=x.device, dtype=torch.float32)
            work = (work.float() / scale).to(dtype=x.dtype)
        if self.activation_qdq:
            work = self.format_backend.quantize_activation(work).to(dtype=x.dtype)
        return F.linear(work, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"algorithm={self.algorithm!r}, activation_qdq={self.activation_qdq}, "
            f"smooth={self.smooth_scale is not None}, source_name={self.source_name!r}"
        )
