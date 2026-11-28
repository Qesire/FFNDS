"""Recreate only the missing native FFN kernel features from the sealed fork states."""
import json
from pathlib import Path
import sys

from safetensors import safe_open
import torch

from attention_joint39 import Assets, Experiment, Layer39, digest, equal_trace, ffn_operands, load, save, tensor_file


PLAN="a1c6c38443f6dcda34b3aaaf26c2d335a37f1a93"
SOURCES=("vbi_0392","vbi_1013")


class ExportFFN(Layer39):
    def __init__(self,*args,archived,**kwargs):
        super().__init__(*args,**kwargs)
        self.archived=archived

    def replay(self,info,model,m,c,f,pre):
        expert=info["expert"]
        old=self.archived["experts"][expert]
        filename=Path(old["asset"]["path"])
        assert digest(filename)==old["asset"]["sha256"]
        want=("X_P64_H448","A_native","native_live_Y","cheap_U","joint_phi","probe_rows","held_rows")
        with safe_open(str(filename),framework="pt",device="cpu") as previous:
            data={k:previous.get_tensor(k).to(self.ids.device) for k in want}
        for now,ref in (("X","X_P64_H448"),("true_A","A_native"),("live_Y","native_live_Y")):
            assert torch.equal(info[now],data[ref]),(self.source,self.background,expert,now)
        assert torch.equal(self.p,data["probe_rows"]) and torch.equal(self.h,data["held_rows"])
        cached,prepared=self.cached.pop(expert)
        _,q,x=ffn_operands(data["cheap_U"],cached,prepared)
        # This reproduces the exact previously stored cheap path, with its
        # original BF16 anchors and selected channel ordering.
        assert q.dtype==torch.float32 and q.shape==(512,16)
        assert torch.equal(q,data["joint_phi"][:,32:]),(self.source,self.background,expert,"FFN q16 witness")
        assert prepared["zmap"].shape[0]==2560 and x.shape==(512,2560)
        assert x.dtype==torch.float32 and bool(torch.isfinite(x).all())
        path=self.dest/(expert+".safetensors")
        record=tensor_file(path,dict(ffn_kernel_x=x,probe_rows=self.p,held_rows=self.h))
        self.rows[expert]=dict(expert=expert,step=info["step"],source=self.source,background=self.background,
            source_asset_sha256=old["asset"]["sha256"],asset=record,live_X=True,live_A=True,live_Y=True,
            cheap_q16_bitwise=True,kernel_feature_width=x.shape[-1],normalization_divisor=2560)
        print("KERNEL",self.source,self.background,expert,record["sha256"],flush=True)


@torch.inference_mode()
def main(reduced,frozen,fit,identity):
    info=load(identity)
    assert info["plan_commit"]==PLAN
    assert digest(reduced/"RESULTS.sha256")==info["panel_results_sha256"]
    assets=Assets(reduced,frozen,"cuda:0")
    panel_ids=torch.tensor(load(frozen/"attention_panel_index.json")["token_ids"],device="cuda:0")
    original={}
    refs={}
    for source,spec in info["previous"].items():
        path=Path(spec["fork"])
        assert digest(path)==spec["fork_sha256"]
        previous=load(path)
        assert previous["status"]=="complete" and previous["sources"]==[source]
        original[source]=previous["results"]
        raw=load(Path(spec["raw"]))
        assert raw["status"]=="complete" and raw["source"]["record_id"]==source
        refs[source]=raw["backgrounds"]
    assert set(original)==set(SOURCES)
    exp=Experiment()
    report=exp.output/"KERNEL_EXPORT.json"
    r=dict(status="running",job=exp.job,plan_commit=PLAN,gpu=exp.state["gpu"],records=[],trace_controls=[],
           panel_results_sha256=info["panel_results_sha256"],producer_sha256=digest(Path(__file__)))
    save(report,r)
    for background in ("Dense","wide8192"):
        method=exp.build("wide8192") if background=="wide8192" else None
        for source in SOURCES:
            writer=ExportFFN(exp,assets,panel_ids,fit,source,background,method,
                exp.output/"kernel_features"/source/background,archived=original[source][background])
            with writer.bind():
                run=exp.render(source,background,method=method)
            assert set(writer.rows)=={"high","low"} and writer.waiting is None and len(writer.cached)==0
            assert equal_trace(run,refs[source][background]["record"]),(source,background)
            if method is not None:
                k=run["counters"]
                assert (k["variant"],k["tile_rows"],k["anchors"],k["hits"],k["anchor_fallback"])==("wide8192",8192,80,80,0)
            r["trace_controls"].append(dict(source=source,background=background,exact_trace=True))
            r["records"].extend(writer.rows.values())
            save(report,r)
    assert len(r["records"])==8 and len(r["trace_controls"])==4
    r["status"]="complete"
    save(report,r)


if __name__=="__main__":
    main(*(Path(x).resolve() for x in sys.argv[1:]))
