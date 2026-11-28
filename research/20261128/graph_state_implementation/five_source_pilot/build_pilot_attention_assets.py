"""Capture paired historical actions on the three actual frozen CAL trajectories."""
import argparse
import json
from pathlib import Path

import torch

from ffnds_optim.config import OptimizationConfig
from ffnds_optim.runtime import OptimizedRuntime
from ffnds_optim.compile_attention import compile_asset


CAL = ("vbi_0032","vbi_0304","vbi_0456")


def capture(args):
    root,args_path=args.out.resolve(),args.candidates.resolve()
    files={k:Path(getattr(args,k)).resolve() for k in ("runtime","fit","binding","config")}
    raw=json.loads(args_path.read_bytes())
    indexed={r["source_id"]:r for r in raw}
    root.mkdir(parents=True,exist_ok=False)
    selection={"sources":[dict(indexed[s],seed=42) for s in CAL]}
    # The earlier catalog and current schema bind the identical prompt/image bytes.
    manifest=root/"CAL_SOURCES.json"
    manifest.write_text(json.dumps(selection,ensure_ascii=False)+"\n",encoding="utf-8")
    model=OptimizedRuntime(files["runtime"],files["fit"],OptimizationConfig.load(files["config"]),binding_path=files["binding"])
    model.register_manifest(manifest)
    legacy=model.legacy_attention
    ids=legacy.panel_ids
    held=ids[~torch.isin(ids,legacy.probes)][:128]
    assert len(held)==128
    samples=[[] for _ in range(40)]
    orig=legacy.approximate
    import attention_mass_centered_low as module
    fit=module.schur_fit
    moments=legacy.get_moments
    low_count=0

    def observe(q,k,v,cache,arm):
        nonlocal low_count
        if arm!="OUT-D":
            return orig(q,k,v,cache,arm)
        coefficient=[]
        values=[]
        def read_fit(*a,**kw):
            result=fit(*a,**kw)
            coefficient.append(result[0])
            return result
        def read_moments(x):
            result=moments(x)
            values.append(result[1])
            return result
        module.schur_fit,legacy.get_moments=read_fit,read_moments
        try:
            output=orig(q,k,v,cache,arm)
        finally:
            module.schur_fit,legacy.get_moments=fit,moments
        assert len(coefficient)==len(values)==1 and low_count<40
        _,_,c,design,unused=cache
        assert unused is None and c.shape[-1]==128 and design.shape[-1]==31
        gamma=coefficient[0][:,1:]*legacy.fits["low"][2]
        z=torch.cat((c.index_select(1,held),design.index_select(1,held)),-1).cpu().contiguous()
        g=torch.cat((torch.diag_embed(values[0][:,0]),-gamma),dim=1).cpu().contiguous()
        samples[low_count].append(dict(Z=z,G=g))
        low_count+=1
        return output

    legacy.approximate=observe
    for source in CAL:
        low_count=0
        model.render(source,"calibration")
        assert low_count==40
        print("REAL_CAL_TRAJECTORY",source,flush=True)
    for layer, pairs in enumerate(samples):
        torch.save(dict(schema="ffnds_attention_joint_cal_v1",expert="low",layer=layer,
            calibration=dict(split="CAL",rows="nonprobe",source_ids=list(CAL),row_count=len(held)),samples=pairs),root/f"low{layer:02d}.pt")


def compile_sites(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for layer in range(40):
        name=f"low{layer:02d}.pt"
        compile_asset(args.input/name,out/name,rank=64,steps=8,tile_rows=128,device="cuda")
        print("LOW_JOINT_ASSET_READY",name,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="action",required=True)
    c=sub.add_parser("capture")
    for name in ("runtime","fit","binding","config","candidates","out"):
        c.add_argument("--"+name,required=True,type=Path)
    b=sub.add_parser("compile")
    for name in ("input","out"):
        b.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    (capture if args.action=="capture" else compile_sites)(args)
