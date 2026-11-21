"""Compile CAL3 secant-kernel and rank-16 joint readouts for the frozen HHLL FFNs."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from fit_orig import CAL, fit, make_queries, thash


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def verified(ref):
    path=Path(ref["path"])
    assert digest(path)==ref["sha256"],path
    return path


def save_json(path, obj):
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj,indent=1,allow_nan=False))
    os.replace(tmp,path)


def load_inputs(r,expected):
    a,b=[r[k] for k in ("anchor","hit")]
    assert a["seed"]==b["seed"]==42 and a["source"]==b["source"]==r["source"]
    assert a["input_identity"]["checkpoint_sha256"]==b["input_identity"]["checkpoint_sha256"]==expected
    assert a["token_id_sha256_le_int64"]==b["token_id_sha256_le_int64"]
    assert (a["step"],b["step"])==((0,1) if r["expert"]=="high" else (2,3))
    aa,bb=[torch.load(verified(x["panel"]),weights_only=True,map_location="cpu") for x in (a,b)]
    assert all(torch.equal(aa[k],bb[k]) for k in ("token_thw","token_indices"))
    ids=bb["token_indices"].numpy()
    coords=bb["token_thw"].numpy()
    assert len(ids)==512 and len(set(ids))==512 and hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()==a["token_id_sha256_le_int64"]
    _,_,q,edge=make_queries(ids,coords)
    i,j=np.unique(np.sort(edge,axis=1),axis=0).T.copy()
    return aa["U"].cuda(),bb["U"].cuda(),bb["gate"].cuda().float().reshape(-1),q,i,j


def compile_site(bind, pairs, expert, layer, root):
    path=root/"params"/expert/f"{layer:02}.pt"
    receipt=path.with_suffix(".json")
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        assert previous["sha256"]==digest(path) and previous["code_sha256"]==digest(__file__)
        return previous
    start=time.monotonic()
    ck=bind["checkpoints"][expert]
    keys=[f"blocks.{layer}.ffn.{name}" for name in ("0.weight","0.bias","2.weight","2.bias")]
    with safe_open(ck["path"],framework="pt",device="cpu") as f:
        native=[f.get_tensor(k) for k in keys]
        assert tuple(native[0].shape)==(13824,5120) and tuple(native[2].shape)==(5120,13824)
        assert all(v.dtype==torch.bfloat16 for v in native)
        weight_hash={k:thash(t) for k,t in zip(keys,native)}
        w1,b1,w2,b2=(t.cuda().float() for t in native)
    if expert+f"{layer:02}" in bind["sites"]:
        assert weight_hash==bind["sites"][expert+f"{layer:02}"]["ffn_tensor_sha256"]
    raw=[pairs[s,expert,layer] for s in CAL]
    inputs=[load_inputs(r,ck["sha256"]) for r in raw]
    ops=[]
    for u0,u1,gate,q,i,j in inputs:
        z0,z1=[F.linear(u.float(),w1,b1) for u in (u0,u1)]
        ops.append((z1-z0,F.gelu(z1,approximate="tanh")-F.gelu(z0,approximate="tanh")))
    numerator=sum((dz.double()*dh).mean(0) for dz,dh in ops)
    energy=sum(dz.double().square().mean(0) for dz,dh in ops)
    beta=(numerator/energy.clamp_min(1e-30)).float()
    scores=torch.zeros_like(beta,dtype=torch.float64)
    for (dz,dh),(_,_,gate,*_) in zip(ops,inputs):
        scores+=(dh-dz*beta).square().mean(0)*((w2.square().T@gate.square()).clamp_min(0))
    sel=scores.topk(2560).indices
    check={}
    if layer==19:
        ref=json.loads((root/"src"/f"reference_{expert}.json").read_text())
        ref_idx=torch.tensor(ref["selected"],device="cuda")
        intersection=len(set(sel.tolist())&set(ref_idx.tolist()))
        assert intersection>=2558,intersection
        check["calculated_topk_intersection"]=intersection
        # Reuse the frozen b19 support, including its channel ordering.
        sel=ref_idx
    packs=[]
    for (dz,dh),(_,_,_,q,i,j) in zip(ops,inputs):
        e=dh-dz*beta
        packs.append(dict(z=e[:,sel],target=F.linear(dh,w2),
                          base=F.linear(dz*beta,w2)+F.linear(e[:,sel],w2[:,sel]),q=q,i=i,j=j))
    scale=torch.stack([r["z"][r["q"]].square().mean(0) for r in packs]).mean(0).sqrt().clamp_min(1e-8)
    left,right,stats=fit(packs,scale,gamma=1)
    if layer==19:
        check["trace_ratio"]=stats["trace_svd_rank16"]/ref["trace"]
        assert .995<check["trace_ratio"]<1.005,check
    sigma=right.norm(dim=1)
    zmap=left*sigma
    v=right/sigma[:,None]
    assert (v@v.T-torch.eye(16,device="cuda")).abs().max().item()<2e-5
    omitted=beta.clone()
    omitted[sel]=0
    zops=[]
    w1bf,b1bf=w1.bfloat16(),b1.bfloat16()
    for u0,u1,*_ in inputs:
        z0=F.linear(u0,w1bf,b1bf)[:,sel]
        z1=F.linear(u1,w1bf[sel],b1bf[sel])
        dh=(F.gelu(z1,approximate="tanh").float()-F.gelu(z0,approximate="tanh").float()).bfloat16().float()
        zops.append((z1.float()-z0.float(),dh))
    kb=(sum((dz.double()*dh).mean(0) for dz,dh in zops)/sum(dz.double().square().mean(0) for dz,_ in zops).clamp_min(1e-30)).float()
    ks=torch.stack([(dh-dz*kb).square().mean(0) for dz,dh in zops]).mean(0).sqrt().clamp_min(1e-8)
    params=dict(sel=sel,A0=((w1.T*omitted)@w2.T).bfloat16(),zmap=zmap,jmap=(w1[sel].T*beta[sel])@zmap,v=v,kernel_beta=kb,kernel_scale=ks)
    assert all(torch.isfinite(t).all() for t in params.values())
    cpu={k:t.cpu().contiguous() for k,t in params.items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".pt.tmp")
    torch.save(cpu,tmp)
    os.replace(tmp,path)
    data=dict(expert=expert,layer=layer,path=str(path),sha256=digest(path),bytes=path.stat().st_size,fit=stats,
              cal_sources=list(CAL),input_panel_sha256=[[x[k]["panel"]["sha256"] for k in ("anchor","hit")] for x in raw],
              ffn_tensor_sha256=weight_hash,checkpoint_sha256=ck["sha256"],code_sha256=digest(__file__),
              shapes={k:list(t.shape) for k,t in cpu.items()},dtypes={k:str(t.dtype) for k,t in cpu.items()},reference=check,
              wall_seconds=time.monotonic()-start,selection_sha256=thash(sel))
    save_json(receipt,data)
    print(expert,layer,round(data["wall_seconds"],2),check,flush=True)
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    args=parser.parse_args()
    root=args.stage.resolve()
    bind_path=root/"src/BINDINGS.json"
    assert digest(bind_path)=="8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6"
    bind=json.loads(bind_path.read_text())
    cal_pairs=json.loads(verified(bind["input_manifests"]["cal_pairs"]).read_text())["input_pairs"]
    pairs={(r["source"],r["expert"],r["layer"]):r for r in cal_pairs}
    assert len(pairs)==880
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    all_sites={}
    with torch.inference_mode():
        for expert in ("high","low"):
            ck=bind["checkpoints"][expert]
            assert Path(ck["path"]).stat().st_size==ck["bytes"]
            for layer in [19,*[i for i in range(40) if i!=19]]:
                data=compile_site(bind,pairs,expert,layer,root)
                all_sites[expert+f"{layer:02}"]=data
    save_json(root/"PARAMS.json",dict(status="PASS",schema="FFNDS_JOINT_GLS_CAL3_PARAMETERS_V1",
              sites=all_sites,slurm_job=os.environ.get("SLURM_JOB_ID"),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
              source_bindings_sha256=digest(bind_path),cal_pairs_sha256=bind["input_manifests"]["cal_pairs"]["sha256"]))
