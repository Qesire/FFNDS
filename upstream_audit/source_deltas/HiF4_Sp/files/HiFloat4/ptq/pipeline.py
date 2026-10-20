from __future__ import annotations

import torch

from .algorithms.gptq_original import OriginalGPTQConfig, quantize_gptq_original
from .algorithms.rtn import RTNConfig, quantize_rtn
from .experiments.two_stage_hifx4 import TwoStageHiF4GPTQConfig, quantize_gptq_two_stage
from .formats.base import WeightFormatBackend
from .formats.hifx4 import HIFX4Format
from .state import ActivationStats, CurvatureStats, PreparedWeightState, QuantizedWeightState
from .transforms.hessian_domain import transform_curvature_for_smooth
from .transforms.magr import MagRConfig, apply_magr
from .transforms.smooth import SmoothConfig, apply_smooth


@torch.no_grad()
def prepare_weight_transforms(
    weight: torch.Tensor,
    *,
    activation_stats: ActivationStats | None = None,
    curvature: CurvatureStats | None = None,
    smooth_config: SmoothConfig | None = None,
    magr_config: MagRConfig | None = None,
) -> PreparedWeightState:
    current_weight = weight.detach().clone()
    current_curvature = curvature
    transforms: list[str] = []
    metadata: dict[str, object] = {}
    report: dict[str, object] = {}
    smooth_scale = None
    quantization_reference_weight = None

    if smooth_config is not None:
        if activation_stats is None:
            raise ValueError("Smooth requires activation_stats")
        smooth_result = apply_smooth(current_weight, activation_stats, smooth_config)
        current_weight = smooth_result.weight
        smooth_scale = smooth_result.scale
        transforms.append("smoothquant")
        metadata.update(smooth_result.metadata)
        report.update(smooth_result.report)
        if current_curvature is not None:
            current_curvature = transform_curvature_for_smooth(
                current_curvature, smooth_result.scale
            )

    if magr_config is not None:
        if current_curvature is None:
            raise ValueError("MagR requires curvature stats")
        quantization_reference_weight = current_weight.detach().clone()
        magr_result = apply_magr(current_weight, current_curvature, magr_config)
        current_weight = magr_result.weight
        transforms.append(f"magr_{magr_config.mode}")
        metadata.update(magr_result.metadata)
        report.update(magr_result.report)

    return PreparedWeightState(
        weight=current_weight,
        curvature=current_curvature,
        smooth_scale=smooth_scale,
        transforms=transforms,
        metadata=metadata,
        report=report,
        quantization_reference_weight=quantization_reference_weight,
    )


@torch.no_grad()
def _attach_prepared(
    state: QuantizedWeightState,
    prepared: PreparedWeightState,
    *,
    algorithm_name: str,
    format_name: str,
) -> QuantizedWeightState:
    state.smooth_scale = prepared.smooth_scale
    state.transforms = list(prepared.transforms)
    state.metadata.update(prepared.metadata)
    state.metadata["pipeline"] = [*prepared.transforms, algorithm_name, format_name]
    state.report.update(prepared.report)

    if prepared.quantization_reference_weight is not None:
        if prepared.curvature is None:
            raise ValueError("MagR final-error accounting requires curvature")
        q = state.dequantized_weight.detach().float()
        magr_weight = prepared.weight.detach().to(device=q.device, dtype=torch.float32)
        reference = prepared.quantization_reference_weight.detach().to(
            device=q.device, dtype=torch.float32
        )
        hessian = prepared.curvature.as_mean().to(device=q.device, dtype=torch.float32)
        hessian = 0.5 * (hessian + hessian.T)

        def weighted(error: torch.Tensor) -> float:
            return float(torch.sum((error @ hessian) * error).detach().cpu().item())

        pre_error = magr_weight - reference
        quant_error = q - magr_weight
        final_error = q - reference
        pre_loss = weighted(pre_error)
        quant_loss = weighted(quant_error)
        final_loss = weighted(final_error)
        state.report.update(
            {
                "magr_pre_hessian_error": pre_loss,
                "magr_quant_hessian_error": quant_loss,
                "magr_final_hessian_error": final_loss,
                "magr_cross_hessian_term": final_loss - pre_loss - quant_loss,
                "magr_final_relative_weight_error": float(
                    (
                        torch.linalg.vector_norm(final_error)
                        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
            }
        )
    return state


@torch.no_grad()
def quantize_prepared_rtn(
    prepared: PreparedWeightState,
    *,
    format_backend: WeightFormatBackend,
    config: RTNConfig | None = None,
) -> QuantizedWeightState:
    state = quantize_rtn(prepared.weight, format_backend, config or RTNConfig())
    return _attach_prepared(
        state, prepared, algorithm_name="rtn", format_name=format_backend.name
    )


@torch.no_grad()
def quantize_prepared_gptq(
    prepared: PreparedWeightState,
    *,
    format_backend: WeightFormatBackend,
    config: OriginalGPTQConfig | None = None,
) -> QuantizedWeightState:
    """Default GPTQ path: preserved repository implementation."""
    if prepared.curvature is None:
        raise ValueError("GPTQ requires prepared.curvature")
    state = quantize_gptq_original(
        prepared.weight,
        prepared.curvature,
        format_backend,
        config or OriginalGPTQConfig(),
    )
    return _attach_prepared(
        state, prepared, algorithm_name="gptq_original", format_name=format_backend.name
    )


@torch.no_grad()
def quantize_prepared_gptq_two_stage(
    prepared: PreparedWeightState,
    *,
    format_backend: HIFX4Format,
    config: TwoStageHiF4GPTQConfig | None = None,
) -> QuantizedWeightState:
    if prepared.curvature is None:
        raise ValueError("two-stage GPTQ requires prepared.curvature")
    state = quantize_gptq_two_stage(
        prepared.weight,
        prepared.curvature,
        format_backend,
        config or TwoStageHiF4GPTQConfig(),
    )
    return _attach_prepared(
        state, prepared, algorithm_name="gptq_two_stage", format_name=format_backend.name
    )


@torch.no_grad()
def build_smooth_hifx4_weight(
    weight: torch.Tensor,
    activation_stats: ActivationStats,
    *,
    smooth_config: SmoothConfig | None = None,
    magr_config: MagRConfig | None = None,
    rtn_config: RTNConfig | None = None,
    curvature: CurvatureStats | None = None,
    format_backend: HIFX4Format | None = None,
) -> tuple[QuantizedWeightState, CurvatureStats | None]:
    prepared = prepare_weight_transforms(
        weight,
        activation_stats=activation_stats,
        curvature=curvature,
        smooth_config=smooth_config or SmoothConfig(),
        magr_config=magr_config,
    )
    state = quantize_prepared_rtn(
        prepared,
        format_backend=format_backend or HIFX4Format(),
        config=rtn_config or RTNConfig(),
    )
    return state, prepared.curvature


@torch.no_grad()
def build_smooth_hifx4_gptq_weight(
    weight: torch.Tensor,
    activation_stats: ActivationStats,
    curvature: CurvatureStats,
    *,
    smooth_config: SmoothConfig | None = None,
    magr_config: MagRConfig | None = None,
    gptq_config: OriginalGPTQConfig | None = None,
    format_backend: HIFX4Format | None = None,
) -> QuantizedWeightState:
    """Smooth + preserved repository hifx4 GPTQ."""
    prepared = prepare_weight_transforms(
        weight,
        activation_stats=activation_stats,
        curvature=curvature,
        smooth_config=smooth_config or SmoothConfig(),
        magr_config=magr_config,
    )
    return quantize_prepared_gptq(
        prepared,
        format_backend=format_backend or HIFX4Format(),
        config=gptq_config or OriginalGPTQConfig(),
    )


@torch.no_grad()
def build_smooth_hifx4_two_stage_gptq_weight(
    weight: torch.Tensor,
    activation_stats: ActivationStats,
    curvature: CurvatureStats,
    *,
    smooth_config: SmoothConfig | None = None,
    magr_config: MagRConfig | None = None,
    two_stage_config: TwoStageHiF4GPTQConfig | None = None,
    format_backend: HIFX4Format | None = None,
) -> QuantizedWeightState:
    prepared = prepare_weight_transforms(
        weight,
        activation_stats=activation_stats,
        curvature=curvature,
        smooth_config=smooth_config or SmoothConfig(),
        magr_config=magr_config,
    )
    return quantize_prepared_gptq_two_stage(
        prepared,
        format_backend=format_backend or HIFX4Format(),
        config=two_stage_config or TwoStageHiF4GPTQConfig(),
    )
