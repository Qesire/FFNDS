from __future__ import annotations

import torch


def project_onto_l1_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """按最后一维批量投影到 L1 球；实现 Duchi et al. 的排序算法。"""

    if radius < 0:
        raise ValueError("L1-ball radius cannot be negative")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if x.shape[-1] <= 0:
        raise ValueError("last dimension cannot be empty")
    if radius == 0:
        return torch.zeros_like(x)

    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    abs_flat = flat.abs()
    l1 = abs_flat.sum(dim=1, keepdim=True)
    inside = l1 <= float(radius)

    sorted_abs, _ = torch.sort(abs_flat, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_abs, dim=1)
    indices = torch.arange(
        1, flat.shape[1] + 1, device=flat.device, dtype=flat.dtype
    ).unsqueeze(0)
    active = sorted_abs * indices > (cumulative - float(radius))
    rho = active.sum(dim=1).clamp_min(1)
    row = torch.arange(flat.shape[0], device=flat.device)
    theta = (cumulative[row, rho - 1] - float(radius)) / rho.to(flat.dtype)
    projected = flat.sign() * (abs_flat - theta.unsqueeze(1)).clamp_min(0)
    projected = torch.where(inside, flat, projected)
    return projected.reshape(original_shape)


def prox_linf(x: torch.Tensor, radius: float) -> torch.Tensor:
    """prox_{radius ||.||_inf}(x) = x - Pi_{||.||_1 <= radius}(x)."""

    if radius < 0:
        raise ValueError("prox radius cannot be negative")
    if radius == 0:
        return x.clone()
    return x - project_onto_l1_ball(x, radius)


def prox_group_linf(
    x: torch.Tensor,
    *,
    group_size: int,
    radius: float,
    allow_partial_groups: bool = True,
    row_chunk_size: int | None = None,
) -> torch.Tensor:
    """沿二维 Linear 权重最后一维，对连续 group 独立施加 L-infinity prox。"""

    if x.ndim != 2:
        raise ValueError(f"groupwise prox expects 2-D weight, got {tuple(x.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if radius < 0:
        raise ValueError("prox radius cannot be negative")
    if radius == 0:
        return x.clone()

    out_features, in_features = map(int, x.shape)
    full_width = (in_features // group_size) * group_size
    if full_width != in_features and not allow_partial_groups:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )

    if row_chunk_size is not None and row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive or None")
    chunk = out_features if row_chunk_size is None else int(row_chunk_size)
    output = torch.empty_like(x)
    for row_start in range(0, out_features, chunk):
        row_end = min(row_start + chunk, out_features)
        source = x[row_start:row_end]
        if full_width:
            grouped = source[:, :full_width].reshape(
                row_end - row_start, full_width // group_size, group_size
            )
            output[row_start:row_end, :full_width] = prox_linf(
                grouped, radius
            ).reshape(row_end - row_start, full_width)
        if full_width < in_features:
            output[row_start:row_end, full_width:] = prox_linf(
                source[:, full_width:], radius
            )
    return output
