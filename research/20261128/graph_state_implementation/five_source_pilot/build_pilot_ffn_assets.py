"""Build s512 factors and an independent least-squares gauge from frozen CAL packs."""
import hashlib
import json
from pathlib import Path
import sys

import torch

from ffnds_optim.ffn import tensor_digest


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


@torch.inference_mode()
def run(base_path, out):
    base=json.loads(base_path.read_bytes())
    manifest=Path(base["cal_manifest"]["path"])
    assert sha(manifest)==base["cal_manifest"]["sha256"]
    data=json.loads(manifest.read_bytes())
    sites={f"{e}{i:02d}" for e in ("high","low") for i in range(40)}
    assert data["status"]=="PASS" and set(data["sites"])==sites
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261128)
    torch.cuda.manual_seed_all(20261128)
    rank=512
    dest={name:out/name for name in ("factor","gauge")}
    for path in dest.values():path.mkdir(parents=True,exist_ok=False)
    receipt=dict(status="running",rank=rank,oversampling=64,power_iterations=2,cal_manifest_sha256=sha(manifest),sites={})
    for name in sorted(sites):
        ref=data["sites"][name]
        path=Path(ref["path"])
        assert sha(path)==ref["sha256"] and path.stat().st_size==ref["bytes"]
        pack=torch.load(path,map_location="cpu",mmap=True,weights_only=True)
        a=pack["A0"].to(device="cuda",dtype=torch.float32)
        j,v=[pack[k].to(device="cuda",dtype=torch.float64) for k in ("jmap","v")]
        gram=(j.T@j)*(v@v.T)
        ridge=1e-6*float(gram.diag().mean())
        gram.diagonal().add_(ridge)
        b=((a.double()@v.T)*j).sum(0)
        c=torch.linalg.solve(gram,-b).to(pack["jmap"].dtype).cpu()
        assert torch.isfinite(c).all()
        gauge_path=dest["gauge"]/(name+".pt")
        torch.save(dict(gauge=c,source_pack_sha256=ref["sha256"],ridge=ridge),gauge_path)
        u,s,w=torch.svd_lowrank(a,q=rank+64,niter=2)
        root=s[:rank].sqrt()
        left=(u[:,:rank]*root).bfloat16().cpu().contiguous()
        right=(root[:,None]*w[:,:rank].T).bfloat16().cpu().contiguous()
        target=dest["factor"]/(name+".pt")
        torch.save(dict(left=left,right=right,gauge=None,rank=rank,format_version=1,target="A0",
            method="randomized_weight_frobenius_initializer",source_pack_sha256=ref["sha256"],
            **{f"source_{k.lower()}_sha256":tensor_digest(pack[k]) for k in ("A0","jmap","v")}),target)
        receipt["sites"][name]=dict(factor_sha256=sha(target),gauge_sha256=sha(gauge_path),
            randomized_energy_fraction=float(s[:rank].square().sum()/a.square().sum()),gauge_ridge=ridge,
            gauge_linf=float(c.abs().max()))
        print("FFN_ASSET_READY",name,flush=True)
    receipt["status"]="complete"
    (out/"FFN_ASSETS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    run(*(Path(s).resolve() for s in sys.argv[1:]))
