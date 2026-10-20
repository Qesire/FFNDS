#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq
from HiFloat4.ptq.state import CurvatureStats
from HiFloat4.ptq.transforms.magr import MagRConfig, apply_magr


def load(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(path)
    return obj


def curvature(name: str, rec: dict[str, Any], domain: str) -> CurvatureStats:
    h = rec.get("hessian", rec.get("matrix"))
    out = CurvatureStats(
        matrix=h.detach().float().cpu(), nsamples=int(rec["nsamples"]),
        in_features=int(rec.get("in_features", h.shape[0])),
        normalization=str(rec.get("normalization", "mean_xtx")),
        domain=str(rec.get("domain", "")), module_name=name,
    )
    out.validate()
    if out.domain != domain:
        raise ValueError(f"{name}: {out.domain} != {domain}")
    return out


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(a.float()-b.float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item())


def weighted(a: torch.Tensor, b: torch.Tensor, h: torch.Tensor) -> float:
    e=(a.float()-b.float()).cpu(); h=h.float().cpu()
    return float(torch.sum((e@h)*e).item())


def rtn_error(w: torch.Tensor, device: str) -> float:
    q=native_hif4_weight_qdq(w.to(device), force_fp32=True).cpu().float()
    return rel(q,w.cpu())


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else math.inf


def main() -> None:
    ap=argparse.ArgumentParser(description="Match normalized MagR-tree one-step displacement to group64.")
    ap.add_argument('--smooth-shard', required=True)
    ap.add_argument('--hessian-shard', required=True)
    ap.add_argument('--expected-curvature-domain', choices=['smooth_a4_qdq','smooth_bf16_noqdq','smooth_a4','smooth_a16'], required=True)
    ap.add_argument('--alphas', nargs='+', type=float, default=[1e-6,3e-6,1e-5,3e-5,1e-4])
    ap.add_argument('--group64-alpha', type=float, default=1e-4)
    ap.add_argument('--compute-device', default='cuda:0')
    ap.add_argument('--dykstra-iters', type=int, default=32)
    ap.add_argument('--output', required=True)
    a=ap.parse_args()

    smooth=load(Path(a.smooth_shard)); hess=load(Path(a.hessian_shard))
    meta=smooth.get('__meta__',{}); expert=str(meta.get('expert',''))
    records=smooth.get(expert,{}); hrecords=hess.get(expert,{})
    if set(records)!=set(hrecords): raise RuntimeError('module mismatch')
    base_cfg=MagRConfig(mode='group64',alpha=a.group64_alpha,outer_iters=1,spectral_method='exact',compute_device=a.compute_device)
    tree_cfgs={alpha:MagRConfig(mode='hif4_tree',alpha=alpha,outer_iters=1,spectral_method='exact',compute_device=a.compute_device,dykstra_iters=a.dykstra_iters,normalize_tree_group_count=True,normalize_tree_lambdas=True) for alpha in a.alphas}
    rows=[]
    for name in sorted(records):
        w=records[name]['weight'].detach().float().cpu(); c=curvature(name,hrecords[name],a.expected_curvature_domain)
        b=apply_magr(w,c,base_cfg).weight.float().cpu()
        base={'disp':rel(b,w),'data':weighted(b,w,c.as_mean()),'rtn':rtn_error(b,a.compute_device)}
        candidates={}
        for alpha,cfg in tree_cfgs.items():
            t=apply_magr(w,c,cfg).weight.float().cpu()
            candidates[str(alpha)]={'disp':rel(t,w),'data':weighted(t,w,c.as_mean()),'rtn':rtn_error(t,a.compute_device)}
        rows.append({'name':name,'group64':base,'tree':candidates})
        print(name)
    aggregate=[]
    base_disp=median([r['group64']['disp'] for r in rows]); base_data=median([r['group64']['data'] for r in rows]); base_rtn=median([r['group64']['rtn'] for r in rows])
    for alpha in a.alphas:
        key=str(alpha)
        d=median([r['tree'][key]['disp'] for r in rows]); data=median([r['tree'][key]['data'] for r in rows]); rtn=median([r['tree'][key]['rtn'] for r in rows])
        score=abs(math.log(max(d,1e-30)/max(base_disp,1e-30))) + 0.25*math.log1p(data/max(base_data,1e-30)) + 0.25*math.log1p(rtn/max(base_rtn,1e-30))
        aggregate.append({'alpha':alpha,'median_disp':d,'disp_ratio_to_group64':d/max(base_disp,1e-30),'median_weighted_preservation_error':data,'median_rtn_reconstruction_rel_error':rtn,'score':score})
    recommended=min(aggregate,key=lambda x:x['score'])['alpha']
    report={'artifact':'magr_tree_alpha_search_v48','expert':expert,'block':meta.get('block'),'curvature_domain':a.expected_curvature_domain,'group64_alpha':a.group64_alpha,'group64_medians':{'disp':base_disp,'weighted_preservation_error':base_data,'rtn_reconstruction_rel_error':base_rtn},'candidates':aggregate,'recommended_alpha':recommended,'modules':rows}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='modules'},ensure_ascii=False,indent=2))
    print('MAGR_TREE_ALPHA_SEARCH_V48_OK')
if __name__=='__main__': main()
