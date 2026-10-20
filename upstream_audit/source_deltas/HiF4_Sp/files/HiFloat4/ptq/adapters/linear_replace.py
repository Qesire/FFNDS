from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from ..algorithms.rtn import RTNConfig
from ..formats.hifx4 import HIFX4Format
from ..pipeline import build_smooth_hifx4_weight
from ..runtime.hifx4_linear import HIFX4Linear
from ..state import ActivationStats, CurvatureStats, QuantizedWeightState
from ..transforms.smooth import SmoothConfig
from .module_utils import get_module, set_module


def replace_linear_from_states(
    model: nn.Module,
    states: Mapping[str, QuantizedWeightState],
    *,
    activation_qdq: bool = True,
    force_fp32: bool = True,
) -> dict[str, Any]:
    replaced: list[dict[str, Any]] = []
    for name, state in states.items():
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        replacement = HIFX4Linear.from_state(
            module,
            state,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=name,
        )
        set_module(model, name, replacement)
        replaced.append(
            {
                "name": name,
                "algorithm": state.algorithm,
                "transforms": list(state.transforms),
                "smooth": state.smooth_scale is not None,
            }
        )
    return {"replaced_count": len(replaced), "replaced": replaced}


def apply_smooth_hifx4_to_model(
    model: nn.Module,
    module_names: Sequence[str],
    activation_stats: Mapping[str, ActivationStats],
    *,
    curvature_stats: Mapping[str, CurvatureStats] | None = None,
    smooth_config: SmoothConfig | None = None,
    rtn_config: RTNConfig | None = None,
    activation_qdq: bool = True,
    force_fp32: bool = True,
) -> dict[str, Any]:
    format_backend = HIFX4Format(force_fp32=force_fp32)
    states: dict[str, QuantizedWeightState] = {}
    reports: list[dict[str, Any]] = []
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        if name not in activation_stats:
            raise KeyError(f"missing activation stats for {name!r}")
        curvature = None if curvature_stats is None else curvature_stats.get(name)
        state, smoothed_curvature = build_smooth_hifx4_weight(
            module.weight.detach(),
            activation_stats[name],
            smooth_config=smooth_config,
            rtn_config=rtn_config,
            curvature=curvature,
            format_backend=format_backend,
        )
        states[name] = state
        reports.append(
            {
                "name": name,
                "curvature_domain": (
                    None if smoothed_curvature is None else smoothed_curvature.domain
                ),
                "metadata": state.metadata,
                "report": state.report,
            }
        )
    replacement = replace_linear_from_states(
        model,
        states,
        activation_qdq=activation_qdq,
        force_fp32=force_fp32,
    )
    return {
        "format": "hifx4",
        "algorithm": "smooth+rtn",
        "activation_qdq": activation_qdq,
        **replacement,
        "reports": reports,
    }


def replace_linear_from_module_states(
    model: nn.Module,
    module_states: Mapping[str, "ModuleQuantizedState"],
    *,
    activation_qdq: bool = True,
    force_fp32: bool = True,
) -> dict[str, Any]:
    """使用完整 artifact 模块状态替换 Linear，并采用 artifact 中保存的 bias。"""

    from ..state import ModuleQuantizedState

    replaced: list[dict[str, Any]] = []
    for name, module_state in module_states.items():
        if not isinstance(module_state, ModuleQuantizedState):
            raise TypeError(f"{name}: expected ModuleQuantizedState")
        module_state.validate()
        original = get_module(model, name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        state = module_state.weight
        replacement = HIFX4Linear(
            original.in_features,
            original.out_features,
            bias=module_state.bias is not None or original.bias is not None,
            weight_qdq=state.dequantized_weight.to(
                device=original.weight.device, dtype=original.weight.dtype
            ),
            bias_tensor=(
                original.bias.detach()
                if module_state.bias is None and original.bias is not None
                else None
                if module_state.bias is None
                else module_state.bias.to(
                    device=original.weight.device, dtype=original.weight.dtype
                )
            ),
            smooth_scale=state.smooth_scale,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=name,
            algorithm=state.algorithm,
        )
        set_module(model, name, replacement)
        replaced.append(
            {
                "name": name,
                "algorithm": state.algorithm,
                "transforms": list(state.transforms),
                "smooth": state.smooth_scale is not None,
            }
        )
    return {"replaced_count": len(replaced), "replaced": replaced}
