from __future__ import annotations

from dataclasses import dataclass

import torch

from ..state import ActivationStats, SmoothResult


@dataclass(frozen=True)
class SmoothConfig:
    """标准 SmoothQuant 配置。"""

    alpha: float = 0.5
    eps: float = 1e-5
    scale_min: float | None = None
    scale_max: float | None = None

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.scale_min is not None and self.scale_min <= 0:
            raise ValueError("scale_min must be positive")
        if self.scale_max is not None and self.scale_max <= 0:
            raise ValueError("scale_max must be positive")
        if self.scale_min is not None and self.scale_max is not None:
            if self.scale_min > self.scale_max:
                raise ValueError("scale_min cannot exceed scale_max")


def compute_smooth_scale(
    weight: torch.Tensor,
    activation_stats: ActivationStats,
    config: SmoothConfig,
) -> torch.Tensor:
    config.validate()
    if weight.ndim != 2:
        raise ValueError(f"Smooth expects 2-D Linear weight, got {tuple(weight.shape)}")
    activation_stats.validate(in_features=weight.shape[1])

    act_absmax = activation_stats.absmax.to(device=weight.device, dtype=torch.float32)
    weight_absmax = weight.detach().float().abs().amax(dim=0)
    scale = act_absmax.clamp_min(config.eps).pow(config.alpha)
    scale = scale / weight_absmax.clamp_min(config.eps).pow(1.0 - config.alpha)
    if config.scale_min is not None or config.scale_max is not None:
        minimum = config.scale_min if config.scale_min is not None else -torch.inf
        maximum = config.scale_max if config.scale_max is not None else torch.inf
        scale = scale.clamp(min=minimum, max=maximum)
    return scale.clamp_min(config.eps).contiguous()


@torch.no_grad()
def apply_smooth(
    weight: torch.Tensor,
    activation_stats: ActivationStats,
    config: SmoothConfig,
) -> SmoothResult:
    scale = compute_smooth_scale(weight, activation_stats, config)
    transformed = (weight.detach().float() * scale.unsqueeze(0)).to(
        dtype=weight.dtype, device=weight.device
    ).contiguous()
    return SmoothResult(
        weight=transformed,
        scale=scale.detach().cpu().float().contiguous(),
        alpha=config.alpha,
        metadata={
            "transform": "smoothquant",
            "alpha": config.alpha,
            "eps": config.eps,
            "input_compensation": "x / smooth_scale before activation QDQ",
        },
        report={
            "smooth_scale_min": float(scale.min().item()),
            "smooth_scale_max": float(scale.max().item()),
            "smooth_scale_mean": float(scale.mean().item()),
            "weight_absmax_before": float(weight.detach().float().abs().max().item()),
            "weight_absmax_after": float(transformed.detach().float().abs().max().item()),
        },
    )
