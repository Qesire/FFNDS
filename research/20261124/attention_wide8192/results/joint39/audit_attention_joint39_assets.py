"""Read-only independent audit of sealed joint-fork vector files on the school host."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    config=json.load(sys.stdin)
    base=Path(config["stage"])
    verified=[]
    for mode in config["modes"]:
        root=base/mode
        h,rel=(root/"RESULTS.sha256").read_text().strip().split(maxsplit=1)
        target=root/rel
        assert target.resolve()==(root/"output/JOINT39_RESULT.json").resolve()
        assert digest(target)==h
        r=json.loads(target.read_bytes())
        assert r["status"]=="complete" and r["job"]==config[mode] and r["plan_commit"]==config["plan_commit"]
        for background,parent in r["results"].items():
            for expert,record in parent["experts"].items():
                a=record["asset"]
                path=Path(a["path"]).resolve()
                assert path.is_relative_to(root/"output"/"forks")
                assert path.stat().st_size==a["bytes"] and digest(path)==a["sha256"]
                t=load_file(str(path),device="cpu")
                p,q=t["probe_rows"],t["held_rows"]
                assert p.shape==(64,) and q.shape==(448,) and set(p.tolist()).isdisjoint(q.tolist())
                assert t["X_P64_H448"].shape==(512,5120) and t["joint_phi"].shape==(512,48)
                assert t["block_B00"].dtype==torch.bfloat16 and t["flow_B00"].shape==(512,64)
                assert torch.equal(t["flow_joint"][p],t["flow_B00"][p].float())
                assert torch.equal(t["flow_independent"][p],t["flow_B00"][p])
                for metric,key in (("joint_flow","flow_joint"),("independent_flow","flow_independent"),("B11_refit","flow_B11_refit")):
                    e=(t[key].double()-t["flow_B00"].double())[q]
                    torch.testing.assert_close(e.square().sum(),torch.tensor(record["flow_scores"][metric]["point"],dtype=e.dtype))
                B10,B01,B11,B00=(t["flow_"+x].double()[q] for x in ("B10","B01","B11_fixed","B00"))
                terms=torch.stack((B10-B00,B01-B00,B11-B10-B01+B00)).flatten(1)
                expected=torch.tensor(record["branch_gram"]["flow"]["gram"],dtype=terms.dtype)
                torch.testing.assert_close(terms@terms.T,expected,rtol=2e-10,atol=2e-8)
                z=t["joint_phi"].double()
                scale=z[p].square().mean(0).sqrt()
                z=z*torch.where(scale>0,scale.reciprocal(),0)
                zh,zp=z[q],z[p]
                residual=(t["flow_B00"].double()-t["flow_cheap"].double())[q]
                floor=torch.linalg.lstsq(zh,residual,driver="gelsd").solution
                reg=len(q)*record["joint_config"]["ridge_per_query"]
                normal=zh.T@zh+reg*torch.eye(zh.shape[1],dtype=z.dtype)
                all_h=torch.linalg.solve(normal,zh.T@residual)
                diagnostic=dict(H448_oracle_floor_point=float((zh@floor-residual).square().sum()),
                    H448_same_ridge_point=float((zh@all_h-residual).square().sum()),
                    P64_rank=int(torch.linalg.matrix_rank(zp)),
                    P64_condition=float(torch.linalg.cond(zp)))
                assert diagnostic["H448_oracle_floor_point"]<=diagnostic["H448_same_ridge_point"]+1e-7
                verified.append(dict(job=r["job"],source=parent["source"],background=background,
                    expert=expert,asset_sha256=a["sha256"],bytes=a["bytes"],diagnostic=diagnostic))
    assert len(verified)==4*len(config["modes"])
    print(json.dumps(dict(status="verified",vectors=verified,count=len(verified))))


if __name__=="__main__":
    main()
