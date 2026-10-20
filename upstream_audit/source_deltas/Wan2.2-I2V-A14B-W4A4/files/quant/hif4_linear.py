from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hif4_backend import hif4_qdq_tensor, hif4_rtn_qdq_weight


class HiF4FakeQuantLinear(nn.Module):
    """
    HiF4 W4A4 fake-quant Linear.

    Semantics:
      - Weight: one-time HiF4 RTN QDQ at replacement/build time.
      - Activation: online HiF4 QDQ in every forward.
      - Compute: floating F.linear, usually BF16 GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        weight_qdq: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation_qdq = bool(activation_qdq)
        self.force_fp32 = bool(force_fp32)
        self.source_name = source_name

        if weight_qdq is None:
            weight_qdq = torch.empty((out_features, in_features), dtype=torch.bfloat16)

        self.weight = nn.Parameter(weight_qdq, requires_grad=False)

        if bias:
            if bias_tensor is None:
                bias_tensor = torch.zeros(out_features, dtype=weight_qdq.dtype, device=weight_qdq.device)
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        mod: nn.Linear,
        *,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
        quant_device: str = "cuda",
    ) -> "HiF4FakeQuantLinear":
        weight = mod.weight.detach()
        weight_qdq = hif4_rtn_qdq_weight(
            weight,
            dim=-1,
            force_fp32=force_fp32,
            quant_device=quant_device,
            return_device=weight.device,
        ).to(weight.dtype)

        bias_tensor = None if mod.bias is None else mod.bias.detach().clone()

        return cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            weight_qdq=weight_qdq,
            bias_tensor=bias_tensor,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=source_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_qdq:
            x = hif4_qdq_tensor(x, dim=-1, force_fp32=self.force_fp32, kind="activation").to(x.dtype)

        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def _get_parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def should_quantize_linear(
    name: str,
    mod: nn.Linear,
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> bool:
    lname = name.lower()

    if exclude_regex and re.search(exclude_regex, name):
        return False
    if include_regex and not re.search(include_regex, name):
        return False

    default_exclude = r"(time|timestep|embed|embedding|modulation|adaln|head|final|norm|patch|vae|t5|text)"
    default_include = r"(self_attn|cross_attn|attn|ffn|mlp|feed_forward|to_q|to_k|to_v|to_out|proj|q_proj|k_proj|v_proj|o_proj|gate|up|down)"

    if re.search(default_exclude, lname):
        return False
    if re.search(default_include, lname):
        return True

    return False


@torch.no_grad()
def replace_linear_with_hif4(
    model: nn.Module,
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    activation_qdq: bool = True,
    force_fp32: bool = True,
    dry_run: bool = False,
    quant_device: str = "cuda",
) -> dict[str, Any]:
    selected = []
    skipped = []

    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            ok = should_quantize_linear(name, mod, include_regex=include_regex, exclude_regex=exclude_regex)
            row = {
                "name": name,
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": mod.bias is not None,
                "dtype": str(mod.weight.dtype),
                "device": str(mod.weight.device),
                "selected": ok,
            }
            if ok:
                selected.append(row)
            else:
                skipped.append(row)

    if not dry_run:
        for row in selected:
            name = row["name"]
            parent, child = _get_parent(model, name)
            old = getattr(parent, child)

            new = HiF4FakeQuantLinear.from_linear(
                old,
                activation_qdq=activation_qdq,
                force_fp32=force_fp32,
                source_name=name,
                quant_device=quant_device,
            )

            setattr(parent, child, new)

    return {
        "dry_run": dry_run,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": selected,
        "skipped": skipped,
    }
