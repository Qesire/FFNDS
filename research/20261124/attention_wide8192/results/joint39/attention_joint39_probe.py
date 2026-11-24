"""Check actual P64 SDPA shapes against archived native full-query attention."""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from attention_panel_fit import digest, save


@torch.inference_mode()
def main(raw, panel, output):
    torch.backends.cuda.matmul.allow_tf32=False
    data=json.loads(raw.read_bytes())
    p=torch.tensor(json.loads(panel.read_bytes())["probe_rows_in_production_order"],device="cuda")
    rows=[]
    for source,index_path in data.items():
        index_path=Path(index_path)
        ind=json.loads(index_path.read_bytes())
        assert ind["status"]=="complete" and ind["source"]["record_id"]==source
        for background,arm in ind["backgrounds"].items():
            for step in (1,3):
                key=f"endpoints/self/s{step:02d}_b39"
                path=index_path.parent/background/(key+".safetensors")
                assert digest(path)==arm["assets"][key]["sha256"]
                ins=load_file(str(path),device="cuda:0")
                q=ins["q_effective"].index_select(0,p)
                k,v=(ins[x+"_effective"] for x in ("k","v"))
                assert q.shape==(64,40,128) and k.shape==v.shape==(31668,40,128)
                args=[x.transpose(0,1).unsqueeze(0) for x in (q,k,v)]
                z=F.scaled_dot_product_attention(*args,attn_mask=None,is_causal=False,dropout_p=0).squeeze(0).transpose(0,1)
                exact=ins["o_native"].index_select(0,p)
                diff=z.float()-exact.float()
                scale=float(exact.float().norm())
                assert scale>0 and float(diff.norm())/scale<0.006
                rows.append(dict(source=source,background=background,step=step,layer=39,
                    query_count=len(q),all_key_count=len(k),relative_l2=float(diff.norm())/scale,
                    max_abs=float(diff.abs().max()),bitwise_equal=torch.equal(z,exact),input_sha256=arm["assets"][key]["sha256"]))
                print("P64",source,background,step,rows[-1]["relative_l2"],flush=True)
    assert len(rows)==8
    save(output,dict(status="complete",rows=rows,max_relative_l2=max(r["relative_l2"] for r in rows)))


if __name__=="__main__":
    main(*(Path(s).resolve() for s in sys.argv[1:]))
