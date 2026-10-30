"""Explicit A5-inspired sign/scale adapter and delta arithmetic.

The original runtime reconstruction (beta * gamma64 * ternary) survives, but
its parameter producer does not. The new LS/mean-absolute recipe is named an
adapter, never an original-A5 reproduction. No native QDQ is implemented here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ADAPTER = "A5_INSPIRED_MEANABS64_ROW_LS_V1"


def proxy_weight(weight, group_size=64):
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError("full original input groups are required")
    w = weight.float()
    ternary = w.sign()
    gamma = w.abs().reshape(w.shape[0], -1, group_size).mean(-1)
    base = gamma.repeat_interleave(group_size, 1) * ternary
    denominator = base.square().sum(1)
    beta = torch.where(denominator > 0, (w*base).sum(1)/denominator.clamp_min(1e-30), 0)
    return (beta[:, None] * base).to(weight.dtype), gamma, beta


def proxy_hidden(v, weight, bias):
    # Bias is included in both historical/current preactivations. It cancels
    # in their linear difference, not inside either nonlinear evaluation.
    return F.gelu(F.linear(v, weight, bias), approximate="tanh")


def delta_hidden(old_pre, delta_pre):
    return F.gelu(old_pre + delta_pre, approximate="tanh") - F.gelu(old_pre, approximate="tanh")


def hwg(weight2, current_gate):
    """Sensitivity of actual *post-Q2* coordinates; original output layout."""
    if weight2.ndim != 2 or current_gate.numel() != weight2.shape[0]:
        raise ValueError("this implementation requires a verified row-constant gate")
    return (weight2.float().square() * current_gate.float().reshape(-1, 1).square()).sum(0).sqrt()


def stable_keep(scores, k):
    if not 0 <= k <= scores.shape[-1]:
        raise ValueError("invalid K")
    # Group/channel index is the deterministic tie-break, including all-zero
    # scores. No dependence on CUDA topk's unspecified tied-index order.
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(-1, indices, True)


def group_scores(scores, group_size=64):
    if scores.shape[-1] % group_size:
        raise ValueError("partial hidden groups are illegal")
    return scores.square().reshape(*scores.shape[:-1], -1, group_size).sum(-1)


def expand_groups(mask, group_size=64):
    return mask.repeat_interleave(group_size, -1)


def retained_delta(old_y, old_v2, current_v2, weight2, mask):
    # V2 is Q2(h/smooth2), not h and not Q2(h1-h0). b2 already occurs in oldY.
    delta = (current_v2.float() - old_v2.float()) * mask
    return old_y.float() + F.linear(delta, weight2.float(), None)


def signed_support_temporal(old_contributions, new_contributions, old_mask, new_mask):
    """Exact support/value split of a changing sparse approximation."""
    value_change = (new_contributions-old_contributions)*new_mask
    support_change = old_contributions*(new_mask.to(old_contributions.dtype)-old_mask.to(old_contributions.dtype))
    return value_change, support_change
