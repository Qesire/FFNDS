from __future__ import annotations

from typing import Any

import torch

from ...state import CurvatureStats, MagRResult
from .config import MagRConfig
from .linf_prox import prox_group_linf


def _relative_change(current: torch.Tensor, previous: torch.Tensor, eps: float) -> float:
    numerator = torch.linalg.vector_norm(current - previous)
    denominator = torch.linalg.vector_norm(previous).clamp_min(eps)
    return float((numerator / denominator).detach().cpu().item())


def _weighted_quadratic(error: torch.Tensor, hessian: torch.Tensor) -> float:
    return float(
        (0.5 * torch.sum((error @ hessian) * error)).detach().cpu().item()
    )


def _sum_group_linf(
    weight: torch.Tensor,
    group_size: int,
    *,
    allow_partial_groups: bool,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("MagR regularizer expects 2-D Linear weight")
    in_features = int(weight.shape[1])
    full_width = (in_features // group_size) * group_size
    if full_width != in_features and not allow_partial_groups:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )
    value = weight.new_zeros(())
    if full_width:
        grouped = weight[:, :full_width].reshape(
            weight.shape[0], full_width // group_size, group_size
        )
        value = value + grouped.abs().amax(dim=-1).sum()
    if full_width < in_features:
        value = value + weight[:, full_width:].abs().amax(dim=-1).sum()
    return value


def _group_count(weight: torch.Tensor, group_size: int) -> int:
    """Number of row-wise groups contributing to the summed group-Linf norm."""
    if weight.ndim != 2:
        raise ValueError("MagR regularizer expects 2-D Linear weight")
    rows, cols = map(int, weight.shape)
    return rows * ((cols + group_size - 1) // group_size)


def _tree_level_value(weight: torch.Tensor, group_size: int, config: MagRConfig) -> torch.Tensor:
    value = _sum_group_linf(
        weight, group_size, allow_partial_groups=config.allow_partial_groups
    )
    if config.normalize_tree_group_count:
        value = value / float(max(1, _group_count(weight, group_size)))
    return value


def magr_regularizer(weight: torch.Tensor, config: MagRConfig) -> torch.Tensor:
    """返回实际被优化的凸 surrogate，而不是 bit-exact hifx4 误差。"""

    config.validate()
    if config.mode == "group64":
        return _sum_group_linf(
            weight, 64, allow_partial_groups=config.allow_partial_groups
        )
    if config.mode == "leaf4":
        return _sum_group_linf(
            weight, 4, allow_partial_groups=config.allow_partial_groups
        )

    lambda64, lambda8, lambda4 = config.resolved_level_weights()
    return (
        lambda64 * _tree_level_value(weight, 64, config)
        + lambda8 * _tree_level_value(weight, 8, config)
        + lambda4 * _tree_level_value(weight, 4, config)
    )


def prox_hif4_tree(
    value: torch.Tensor,
    *,
    tau: float,
    config: MagRConfig,
) -> tuple[torch.Tensor, int, float]:
    """用 Dykstra 计算三层嵌套 group 正则之和的 proximal。

    正则在输出行之间可分，因此按行分块运行 Dykstra，避免为整层权重同时
    保存三份 correction tensor。
    """

    lambda64, lambda8, lambda4 = config.resolved_level_weights()
    levels = []
    for group_size, coefficient in ((64, lambda64), (8, lambda8), (4, lambda4)):
        if config.normalize_tree_group_count:
            coefficient = coefficient / float(max(1, _group_count(value, group_size)))
        levels.append((group_size, coefficient))
    rows = int(value.shape[0])
    chunk_size = rows if config.row_chunk_size is None else int(config.row_chunk_size)
    output = torch.empty_like(value)
    total_iterations = 0
    max_change = 0.0

    for row_start in range(0, rows, chunk_size):
        row_end = min(row_start + chunk_size, rows)
        x = value[row_start:row_end].clone()
        corrections = [torch.zeros_like(x) for _ in range(3)]
        chunk_change = 0.0
        chunk_iterations = config.dykstra_iters
        for iteration in range(1, config.dykstra_iters + 1):
            previous_cycle = x.clone()
            for index, (group_size, coefficient) in enumerate(levels):
                shifted = x + corrections[index]
                updated = prox_group_linf(
                    shifted,
                    group_size=group_size,
                    radius=tau * coefficient,
                    allow_partial_groups=config.allow_partial_groups,
                    row_chunk_size=None,
                )
                corrections[index] = shifted - updated
                x = updated
            chunk_change = _relative_change(x, previous_cycle, config.eps)
            if config.dykstra_tol > 0 and chunk_change <= config.dykstra_tol:
                chunk_iterations = iteration
                break
        output[row_start:row_end] = x
        total_iterations += chunk_iterations
        max_change = max(max_change, chunk_change)
    return output, total_iterations, max_change


def _prox_regularizer(
    value: torch.Tensor,
    *,
    tau: float,
    config: MagRConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if tau == 0:
        return value.clone(), {"dykstra_iterations": 0, "dykstra_relative_change": 0.0}
    if config.mode == "group64":
        return (
            prox_group_linf(
                value,
                group_size=64,
                radius=tau,
                allow_partial_groups=config.allow_partial_groups,
                row_chunk_size=config.row_chunk_size,
            ),
            {"dykstra_iterations": 0, "dykstra_relative_change": 0.0},
        )
    if config.mode == "leaf4":
        return (
            prox_group_linf(
                value,
                group_size=4,
                radius=tau,
                allow_partial_groups=config.allow_partial_groups,
                row_chunk_size=config.row_chunk_size,
            ),
            {"dykstra_iterations": 0, "dykstra_relative_change": 0.0},
        )
    output, iterations, change = prox_hif4_tree(value, tau=tau, config=config)
    return output, {
        "dykstra_iterations": int(iterations),
        "dykstra_relative_change": float(change),
    }


def _spectral_value_power(
    hessian: torch.Tensor,
    *,
    iterations: int,
    tolerance: float,
    seed: int,
    eps: float,
) -> tuple[float, int]:
    size = int(hessian.shape[0])
    generator = torch.Generator(device=hessian.device)
    generator.manual_seed(int(seed))
    vector = torch.randn(size, device=hessian.device, dtype=hessian.dtype, generator=generator)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(eps)
    previous = None
    estimate = hessian.new_zeros(())

    for iteration in range(1, iterations + 1):
        product = hessian @ vector
        norm = torch.linalg.vector_norm(product)
        if float(norm.detach().cpu().item()) <= eps:
            return 0.0, iteration
        vector = product / norm
        estimate = torch.dot(vector, hessian @ vector).clamp_min(0)
        current = float(estimate.detach().cpu().item())
        if previous is not None and tolerance > 0:
            if abs(current - previous) <= tolerance * max(abs(current), eps):
                return current, iteration
        previous = current
    return float(estimate.detach().cpu().item()), iterations


def _prepare_hessian(
    curvature: CurvatureStats,
    config: MagRConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, int, float]:
    curvature.validate()
    raw = curvature.as_mean().to(device=device, dtype=torch.float32).clone()
    if config.symmetrize_curvature:
        raw = 0.5 * (raw + raw.T)

    if config.spectral_method == "exact":
        spectral_value = float(
            torch.linalg.eigvalsh(raw).max().clamp_min(0).detach().cpu().item()
        )
        spectral_iterations = 1
    else:
        spectral_value, spectral_iterations = _spectral_value_power(
            raw,
            iterations=config.power_iters,
            tolerance=config.power_tol,
            seed=config.random_seed,
            eps=config.eps,
        )

    if config.normalize_curvature:
        denominator = max(spectral_value, config.eps)
        objective_hessian = raw / denominator if spectral_value > config.eps else torch.zeros_like(raw)
        step_size = config.step_scale
    else:
        objective_hessian = raw
        step_size = config.step_scale / max(spectral_value, config.eps)
    return raw, objective_hessian, spectral_value, spectral_iterations, step_size


@torch.no_grad()
def apply_magr(
    weight: torch.Tensor,
    curvature: CurvatureStats,
    config: MagRConfig | None = None,
) -> MagRResult:
    """在 Smooth 后的权重域运行 MagR，并保持输入 CurvatureStats 不变。"""

    cfg = config or MagRConfig()
    cfg.validate()
    if weight.ndim != 2 or not torch.is_floating_point(weight):
        raise ValueError("MagR weight must be a floating-point 2-D tensor")
    if curvature.in_features != int(weight.shape[1]):
        raise ValueError("curvature/weight in_features mismatch")
    if not cfg.allow_partial_groups and int(weight.shape[1]) % 64 != 0:
        raise ValueError("strict hifx4 MagR requires in_features divisible by 64")

    original_device = weight.device
    original_dtype = weight.dtype
    work_device = original_device if cfg.compute_device is None else torch.device(cfg.compute_device)
    reference = weight.detach().to(work_device, dtype=torch.float32).clone()
    current = reference.clone()
    raw_hessian, objective_hessian, spectral_value, spectral_iterations, step_size = (
        _prepare_hessian(curvature, cfg, work_device)
    )

    lambda64, lambda8, lambda4 = cfg.resolved_level_weights()
    initial_regularizer = float(magr_regularizer(reference, cfg).detach().cpu().item())
    initial_objective = cfg.alpha * initial_regularizer
    checkpoints: list[dict[str, Any]] = []
    checkpoint_set = {int(x) for x in cfg.checkpoint_iterations}
    if 0 in checkpoint_set:
        checkpoints.append(
            {
                "iteration": 0,
                "data_loss_normalized": 0.0,
                "regularizer": initial_regularizer,
                "objective": initial_objective,
                "relative_weight_change": 0.0,
            }
        )

    executed = 0
    final_relative_change = 0.0
    dykstra_iterations_total = 0
    dykstra_last_change = 0.0
    tau = step_size * cfg.alpha

    for iteration in range(1, cfg.outer_iters + 1):
        previous = current
        gradient = (current - reference) @ objective_hessian
        proposal = current - step_size * gradient
        current, prox_report = _prox_regularizer(proposal, tau=tau, config=cfg)
        executed = iteration
        final_relative_change = _relative_change(current, previous, cfg.eps)
        dykstra_iterations_total += int(prox_report["dykstra_iterations"])
        dykstra_last_change = float(prox_report["dykstra_relative_change"])

        if iteration in checkpoint_set:
            error = current - reference
            regularizer = float(magr_regularizer(current, cfg).detach().cpu().item())
            data_loss = _weighted_quadratic(error, objective_hessian)
            checkpoints.append(
                {
                    "iteration": iteration,
                    "data_loss_normalized": data_loss,
                    "regularizer": regularizer,
                    "objective": data_loss + cfg.alpha * regularizer,
                    "relative_weight_change": final_relative_change,
                }
            )
        if cfg.outer_tol > 0 and final_relative_change <= cfg.outer_tol:
            break

    error = current - reference
    final_regularizer = float(magr_regularizer(current, cfg).detach().cpu().item())
    data_loss_normalized = _weighted_quadratic(error, objective_hessian)
    data_loss_raw = _weighted_quadratic(error, raw_hessian)
    final_objective = data_loss_normalized + cfg.alpha * final_regularizer
    relative_weight_change = float(
        (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(reference).clamp_min(cfg.eps)
        )
        .detach()
        .cpu()
        .item()
    )

    result = MagRResult(
        weight=current.to(original_device, dtype=original_dtype).contiguous(),
        metadata={
            "magr_transform": "magr",
            "magr_mode": cfg.mode,
            "magr_objective": "spectral_normalized_hessian_plus_hierarchical_linf_surrogate",
            "magr_alpha": float(cfg.alpha),
            "magr_outer_iters_requested": int(cfg.outer_iters),
            "magr_outer_iters_executed": int(executed),
            "magr_step_scale": float(cfg.step_scale),
            "magr_step_size": float(step_size),
            "magr_normalize_curvature": bool(cfg.normalize_curvature),
            "magr_spectral_method": cfg.spectral_method,
            "magr_spectral_value": float(spectral_value),
            "magr_spectral_iterations": int(spectral_iterations),
            "magr_curvature_domain": curvature.domain,
            "magr_prox_solver": "dykstra" if cfg.mode == "hif4_tree" else "group_linf_exact",
            "magr_allow_partial_groups": bool(cfg.allow_partial_groups),
            "magr_row_chunk_size": cfg.row_chunk_size,
            "magr_lambda64": lambda64,
            "magr_lambda8": lambda8,
            "magr_lambda4": lambda4,
            "magr_normalize_tree_group_count": bool(cfg.normalize_tree_group_count),
            "magr_normalize_tree_lambdas": bool(cfg.normalize_tree_lambdas),
        },
        report={
            "magr_regularizer_before": initial_regularizer,
            "magr_regularizer_after": final_regularizer,
            "magr_objective_before": initial_objective,
            "magr_objective_after": final_objective,
            "magr_data_loss_normalized": data_loss_normalized,
            "magr_pre_hessian_error": 2.0 * data_loss_raw,
            "magr_relative_weight_change": relative_weight_change,
            "magr_outer_relative_change": final_relative_change,
            "magr_dykstra_iterations_total": dykstra_iterations_total,
            "magr_dykstra_last_relative_change": dykstra_last_change,
            "magr_checkpoints": checkpoints,
        },
    )
    if not torch.isfinite(result.weight).all():
        raise RuntimeError("MagR produced NaN or Inf")
    return result


__all__ = [
    "apply_magr",
    "magr_regularizer",
    "prox_hif4_tree",
]
