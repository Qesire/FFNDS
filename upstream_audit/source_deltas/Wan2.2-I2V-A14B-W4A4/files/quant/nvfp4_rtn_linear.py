from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nvfp4_rtn_backend import DEFAULT_GROUP_SIZE, get_nvfp4_rtn_stats, nvfp4_rtn_qdq_activation, nvfp4_rtn_qdq_weight

DEFAULT_EXCLUDE_REGEX = r"(time|timestep|embed|embedding|modulation|adaln|head|final|norm|patch|vae|t5|text)"
DEFAULT_INCLUDE_REGEX = r"(self_attn|cross_attn|attn|ffn|mlp|feed_forward|to_q|to_k|to_v|to_out|proj|q_proj|k_proj|v_proj|o_proj|gate|up|down)"


def should_quantize_linear(name: str, mod: nn.Linear, *, include_regex=None, exclude_regex=None) -> bool:
    del mod
    lname=name.lower()
    if exclude_regex and re.search(exclude_regex,name): return False
    if include_regex and not re.search(include_regex,name): return False
    if re.search(DEFAULT_EXCLUDE_REGEX,lname): return False
    return re.search(DEFAULT_INCLUDE_REGEX,lname) is not None


def iter_selected_linears(model: nn.Module, *, prefix="", include_regex=None, exclude_regex=None) -> Iterable:
    for local_name,mod in list(model.named_modules()):
        if not isinstance(mod,nn.Linear): continue
        full_name=f"{prefix}.{local_name}" if prefix else local_name
        if should_quantize_linear(full_name,mod,include_regex=include_regex,exclude_regex=exclude_regex):
            yield local_name,full_name,mod


def _get_parent(root,name):
    parts=name.split('.'); parent=root
    for p in parts[:-1]: parent=getattr(parent,p)
    return parent,parts[-1]


def load_activation_scale_map(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"NVFP4 activation scale sidecar missing: {p}")
    from safetensors.torch import load_file
    raw=load_file(str(p),device="cpu"); out={}
    for key,value in raw.items():
        name=key[:-len('.input_global_scale')] if key.endswith('.input_global_scale') else key
        if value.numel()!=1: raise ValueError(f"NVFP4 scale {key} must be scalar")
        scalar=value.reshape(1).to(torch.float32).contiguous(); v=float(scalar.item())
        if not math.isfinite(v) or v<=0: raise ValueError(f"bad NVFP4 scale {key}: {v}")
        out[name]=scalar
    return out


class NVFP4RTNFakeQuantLinear(nn.Module):
    def __init__(self, *, q_weight,bias,input_global_scale,in_features,out_features,source_name,
                 activation_qdq=True,group_size=16,force_fp32=True,weight_meta=None):
        super().__init__(); self.in_features=int(in_features); self.out_features=int(out_features)
        self.source_name=str(source_name); self.activation_qdq=bool(activation_qdq)
        self.group_size=int(group_size); self.force_fp32=bool(force_fp32); self.weight_meta=dict(weight_meta or {})
        self.weight=nn.Parameter(q_weight.detach(),requires_grad=False)
        if bias is None: self.register_parameter('bias',None)
        else: self.bias=nn.Parameter(bias.detach().clone(),requires_grad=False)
        self.register_buffer('input_global_scale',input_global_scale.detach().reshape(1).to(torch.float32).clone(),persistent=True)

    @classmethod
    @torch.no_grad()
    def from_linear(cls,lin,*,source_name,input_global_scale,activation_qdq=True,group_size=16,
                    force_fp32=True,quant_device='cuda:0',chunk_rows=128,collect_error_stats=False):
        q_weight,meta=nvfp4_rtn_qdq_weight(lin.weight.detach(),group_size=group_size,force_fp32=force_fp32,
            quant_device=quant_device,chunk_rows=chunk_rows,return_device=lin.weight.device,
            collect_error_stats=collect_error_stats)
        return cls(q_weight=q_weight,bias=None if lin.bias is None else lin.bias.detach(),input_global_scale=input_global_scale,
            in_features=lin.in_features,out_features=lin.out_features,source_name=source_name,
            activation_qdq=activation_qdq,group_size=group_size,force_fp32=force_fp32,weight_meta=meta)

    def forward(self,x):
        if self.activation_qdq:
            x=nvfp4_rtn_qdq_activation(x,self.input_global_scale,group_size=self.group_size,force_fp32=self.force_fp32)
        weight=self.weight.to(device=x.device,dtype=x.dtype)
        bias=None if self.bias is None else self.bias.to(device=x.device,dtype=x.dtype)
        return F.linear(x,weight,bias)


@torch.no_grad()
def replace_linear_with_nvfp4_rtn(model: nn.Module, *, prefix: str, activation_scales: dict,
    include_regex=None,exclude_regex=None,activation_qdq=True,group_size=16,force_fp32=True,
    quant_device='cuda:0',chunk_rows=128,dry_run=False,allow_missing_scales=False,collect_error_stats=False):
    selected=list(iter_selected_linears(model,prefix=prefix,include_regex=include_regex,exclude_regex=exclude_regex))
    selected_names=[f for _,f,_ in selected]; missing=[n for n in selected_names if n not in activation_scales]
    if activation_qdq and missing and not allow_missing_scales:
        raise KeyError(f"Missing {len(missing)} NVFP4 input_global_scale entries; first={missing[:10]}")
    report={"prefix":prefix,"selected":len(selected),"replaced":0,"missing_activation_scales":missing,
        "activation_scale_count_available":len(activation_scales),"dry_run":bool(dry_run),"activation_qdq":bool(activation_qdq),
        "group_size":int(group_size),"force_fp32":bool(force_fp32),"quant_device":str(quant_device),
        "chunk_rows":int(chunk_rows),"include_regex":include_regex,"exclude_regex":exclude_regex,
        "selection_policy":"exact_match_to_successful_HiF4_selector","selected_names":selected_names,"layers":[]}
    if dry_run: return report
    for local_name,full_name,old in selected:
        scale=activation_scales.get(full_name,torch.tensor([1.0],dtype=torch.float32))
        parent,leaf=_get_parent(model,local_name)
        new=NVFP4RTNFakeQuantLinear.from_linear(old,source_name=full_name,input_global_scale=scale,
            activation_qdq=activation_qdq,group_size=group_size,force_fp32=force_fp32,
            quant_device=quant_device,chunk_rows=chunk_rows,collect_error_stats=collect_error_stats)
        setattr(parent,leaf,new); report['replaced']+=1
        report['layers'].append({"name":full_name,"in_features":int(old.in_features),"out_features":int(old.out_features),
            "bias":old.bias is not None,"input_global_scale":float(scale.reshape(()).item()),"weight_meta":new.weight_meta})
    report['stats_after_replace']=get_nvfp4_rtn_stats(); return report
