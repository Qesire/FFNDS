import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mxfp4_rtn_backend import (
    get_mxfp4_rtn_stats,
    mxfp4_rtn_qdq_activation,
    mxfp4_rtn_qdq_weight,
)

DEFAULT_EXCLUDE_REGEX = (
    r"(time|timestep|embed|embedding|adaln|modulation|norm|ln|head|final|"
    r"patch|vae|t5|text|condition|pos|rope)"
)

class MXFP4RTNFakeQuantLinear(nn.Module):
    def __init__(
        self,
        q_weight,
        bias,
        in_features,
        out_features,
        activation_qdq=True,
        block_size=32,
        force_fp32=True,
        weight_scale_method="ocp_floor",
        act_scale_method="safe_ceil",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation_qdq = bool(activation_qdq)
        self.block_size = int(block_size)
        self.force_fp32 = bool(force_fp32)
        self.weight_scale_method = str(weight_scale_method)
        self.act_scale_method = str(act_scale_method)

        self.register_buffer("weight", q_weight.detach().clone())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        lin: nn.Linear,
        activation_qdq=True,
        block_size=32,
        force_fp32=True,
        weight_scale_method="ocp_floor",
        act_scale_method="safe_ceil",
    ):
        q_weight = mxfp4_rtn_qdq_weight(
            lin.weight.detach(),
            dim=-1,
            block_size=block_size,
            force_fp32=force_fp32,
            scale_method=weight_scale_method,
        )
        bias = lin.bias.detach() if lin.bias is not None else None
        return cls(
            q_weight=q_weight,
            bias=bias,
            in_features=lin.in_features,
            out_features=lin.out_features,
            activation_qdq=activation_qdq,
            block_size=block_size,
            force_fp32=force_fp32,
            weight_scale_method=weight_scale_method,
            act_scale_method=act_scale_method,
        )

    def forward(self, x):
        if self.activation_qdq:
            x = mxfp4_rtn_qdq_activation(
                x,
                dim=-1,
                block_size=self.block_size,
                force_fp32=self.force_fp32,
                scale_method=self.act_scale_method,
            )
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype, device=x.device)
        return F.linear(x, weight, bias)

def _match(pattern, name):
    if not pattern:
        return True
    return re.search(pattern, name) is not None

def _excluded(pattern, name):
    if not pattern:
        return False
    return re.search(pattern, name) is not None

def replace_linear_with_mxfp4_rtn(
    module: nn.Module,
    prefix="",
    include_regex=None,
    exclude_regex=DEFAULT_EXCLUDE_REGEX,
    activation_qdq=True,
    block_size=32,
    force_fp32=True,
    weight_scale_method="ocp_floor",
    act_scale_method="safe_ceil",
    dry_run=False,
):
    report = {
        "selected": 0,
        "skipped": 0,
        "replaced": 0,
        "dry_run": bool(dry_run),
        "activation_qdq": bool(activation_qdq),
        "block_size": int(block_size),
        "force_fp32": bool(force_fp32),
        "weight_scale_method": str(weight_scale_method),
        "act_scale_method": str(act_scale_method),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_names": [],
        "skipped_names": [],
    }

    def rec(parent, pfx):
        for name, child in list(parent.named_children()):
            fq = f"{pfx}.{name}" if pfx else name

            if isinstance(child, nn.Linear):
                if _excluded(exclude_regex, fq) or not _match(include_regex, fq):
                    report["skipped"] += 1
                    report["skipped_names"].append(fq)
                    continue

                report["selected"] += 1
                report["selected_names"].append(fq)

                if not dry_run:
                    qlin = MXFP4RTNFakeQuantLinear.from_linear(
                        child,
                        activation_qdq=activation_qdq,
                        block_size=block_size,
                        force_fp32=force_fp32,
                        weight_scale_method=weight_scale_method,
                        act_scale_method=act_scale_method,
                    )
                    setattr(parent, name, qlin)
                    report["replaced"] += 1
            else:
                rec(child, fq)

    rec(module, prefix)
    report["stats"] = get_mxfp4_rtn_stats()
    return report
