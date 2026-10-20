from __future__ import annotations

import torch

from ..state import CurvatureStats, ModuleCalibrationStats


def transform_curvature_for_smooth(
    curvature: CurvatureStats,
    smooth_scale: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> CurvatureStats:
    """执行 H_s = D^{-1} H D^{-1}，不修改原始 Hessian。"""

    curvature.validate()
    scale = smooth_scale.detach().to(device=curvature.matrix.device, dtype=torch.float32)
    if scale.ndim != 1 or scale.numel() != curvature.in_features:
        raise ValueError(
            f"smooth scale shape={tuple(scale.shape)}, expected ({curvature.in_features},)"
        )
    inverse = scale.clamp_min(eps).reciprocal()
    transformed = inverse[:, None] * curvature.matrix.float() * inverse[None, :]
    return CurvatureStats(
        matrix=transformed.detach().cpu().contiguous(),
        nsamples=curvature.nsamples,
        in_features=curvature.in_features,
        normalization=curvature.normalization,
        domain="smoothed",
        module_name=curvature.module_name,
    )


def transform_module_calibration_for_smooth(
    module: ModuleCalibrationStats,
    smooth_scale: torch.Tensor,
) -> ModuleCalibrationStats:
    return ModuleCalibrationStats(
        module_name=module.module_name,
        in_features=module.in_features,
        activation=module.activation,
        curvature=(
            None
            if module.curvature is None
            else transform_curvature_for_smooth(module.curvature, smooth_scale)
        ),
    )
