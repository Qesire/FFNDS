"""Exercise the online low-noise adapter with real archived Q/K/V on an A100."""
from pathlib import Path
from types import SimpleNamespace
import sys

from safetensors.torch import load_file
import torch

from attention_online_structured import (
    Assets, BlockMask, F, HEADS, N, OnlineAttention, TILES, TOP, BLOCK, design,
    digest, load, native_flash, readout, relative, save,
)


@torch.inference_mode()
def main(fit,panels,config_path):
    config=load(config_path)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    online=OnlineAttention(SimpleNamespace(base=load(Path("base.json"))),fit)
    assets=Assets(panels,fit/"plan_assets","cuda")
    raw=Path(config["controls"]["vbi_0392"])
    wide=load(raw)["backgrounds"]["wide8192"]
    audit=[]
    for layer in (0,39):
        site=next(x for x in assets.groups["self","low"] if x.source=="vbi_0392"
                  and x.row["background"]=="wide8192" and x.row["layer"]==layer)
        path=site.root/site.index_row["support_file"]
        assert digest(path)==site.row["support_sha256"]
        blocks=load_file(str(path),device="cuda")["selected_blocks"]
        count=torch.full((1,HEADS,TILES),TOP,device="cuda",dtype=torch.int32)
        support=BlockMask.from_kv_blocks(count,F.pad(blocks,(0,TILES-TOP))[None],BLOCK_SIZE=BLOCK,seq_lengths=(N,N))
        d=assets.prepare(site)
        endpoints=[]
        for step in (2,3):
            key=f"endpoints/self/s{step:02d}_b{layer:02d}"
            endpoint=raw.parent/"wide8192"/(key+".safetensors")
            assert digest(endpoint)==wide["assets"][key]["sha256"]
            data=load_file(str(endpoint),device="cuda")
            qp=data["q_effective"]
            # Only the 512 registered positions are compared with the archive.
            # Other rows fill the physical full-length kernel shape.
            q=qp.index_select(0,torch.arange(N,device="cuda")%len(qp))
            q.index_copy_(0,online.panel_ids,qp)
            endpoints.append((q,data["k_effective"],data["v_effective"]))
        q0,k0,v0=endpoints[0]
        dense,lse=native_flash(q0,k0,v0)
        maps,lam,sr,sm=online.fits["low"]
        for arm in ("FIN-C0","OUT-D"):
            cache,recomputed=online.low_cache(q0,k0,v0,support,dense[0],lse[0],arm)
            candidate,teacher,selected=online.approximate(*endpoints[1],cache,arm)
            ref,_=readout(d,design(d,maps),arm,assets.p,lam[arm],sr,sm,assets.perm)
            rec=dict(layer=layer,arm=arm,unregistered_rows_filled=True,
                     common_logit_lse_recomputations=recomputed,
                     probe_relative_l2=relative(teacher,d["y"][:,assets.p]),
                     selected_relative_l2=relative(selected[:,online.panel_ids],d["selected"]),
                     readout_relative_l2=relative(candidate[online.panel_ids[assets.h]].transpose(0,1),ref[:,assets.h].bfloat16()))
            audit.append(rec)
            print("PANEL_AUDIT",rec,flush=True)
    save(Path("PANEL_PATCH.json"),dict(status="complete",scope="archived real Q/K/V; registered queries only; no new model execution",rows=audit))


if __name__=="__main__":
    main(*(Path(s).resolve() for s in sys.argv[1:4]))
