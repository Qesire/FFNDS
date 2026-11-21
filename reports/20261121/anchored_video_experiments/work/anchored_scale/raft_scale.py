"""Native VBench RAFT on paired originals. Reduce each pair in memory; persist only statistics."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
OUT=WORK.parent/"outputs/FFNDS_anchor_scale24"
SCOPE=WORK.parents[2]/"2026-11-18/gen-j/work/state_coupled/quality_representation_v1/dynamic_degree/scope.json"
sys.path.insert(0,str(WORK/"video_joint_eval"))
import diagnostic0800 as archived

WINDOWS=[(0,80),*((i,i+16) for i in range(0,80,16))]
HIST=OUT.parent/"FFNDS_joint_DEV3"


def read(p):
    return json.loads(p.read_bytes())


def atomic(p,obj):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=1),encoding="utf8")
    temp.replace(p)


def angular(dot, product, keep):
    cc=dot[keep]/product[keep]
    return float(np.degrees(np.arccos(np.clip(cc,-1,1))).sum()),int(keep.sum())


def frame_stats(f,d):
    f=f.astype(np.float64)
    d=d.astype(np.float64)
    fn=np.linalg.norm(f,axis=0)
    dn=np.linalg.norm(d,axis=0)
    dot=np.sum(f*d,axis=0)
    diff=fn-dn
    angle,count=angular(dot,fn*dn,(fn>.001)&(dn>.001))
    return dict(n=dot.size,angle=angle,n_valid=count,dot=float(dot.sum()),
                dense2=float(np.square(dn).sum()),epe=float(np.linalg.norm(f-d,axis=0).mean()),
                c_amp=float(fn.mean()),d_amp=float(dn.mean()),mae=float(np.abs(diff).mean()),
                excess=float(np.maximum(diff,0).mean()),deficit=float(np.maximum(-diff,0).mean()))


def grid(flow):
    # DIS uses a half-size image, sampled at [2::4,2::4]. Use the same pixel centres and units.
    return cv2.resize(flow.transpose(1,2,0),(416,232),interpolation=cv2.INTER_AREA)[2::4,2::4].astype(np.float64)


def grid_stats(f,d):
    f,d=grid(f),grid(d)
    fn,dn=np.linalg.norm(f,axis=-1),np.linalg.norm(d,axis=-1)
    dot=(f*d).sum(axis=-1)
    angle,n=angular(dot,fn*dn,(fn>=.5)&(dn>=.5))
    return dict(grid_angle=angle,grid_valid=n,grid_n=dot.size,
                grid_dot=float(dot.sum()),grid_d2=float(np.square(dn).sum()),
                grid_epe=float(np.linalg.norm(f-d,axis=-1).mean()))


def reduce(rows):
    def total(k):
        return sum(r[k] for r in rows)
    def ratio(a,b):
        return total(a)/total(b) if total(b)>0 else None
    out={k:total(k)/len(rows) for k in ("c_amp","d_amp","epe","mae","excess","deficit","grid_epe","candidate_dynamic","dense_dynamic")}
    out.update(gain=ratio("dot","dense2"),angle_deg=ratio("angle","n_valid"),
               angle_support=ratio("n_valid","n"),grid_gain=ratio("grid_dot","grid_d2"),
               grid_active_angle_deg=ratio("grid_angle","grid_valid"),grid_support=ratio("grid_valid","grid_n"),
               grid_dense_rms=(total("grid_d2")/total("grid_n"))**.5)
    return out


@torch.no_grad()
def infer(src,model,padder,identity):
    start=time.perf_counter()
    paths={arm:OUT/arm/(src+".mp4") for arm in ("dense","candidate")}
    receipts={a:read(p.with_suffix(".json")) for a,p in paths.items()}
    sha={a:archived.sha(p) for a,p in paths.items()}
    for a,r in receipts.items():
        if (r["source"]!=src or r["video"]["mp4_sha256"]!=sha[a]
            or (r["video"]["height"],r["video"]["width"],r["video"]["fps"],r["video"]["frames"])!=(464,832,16,81)):
            raise ValueError(f"Invalid registered video: {src}/{a}")
    if receipts["dense"]["initial"]!=receipts["candidate"]["initial"]:
        raise ValueError(f"Initial noise mismatch: {src}")
    frames={a:model.get_frames(str(p)) for a,p in paths.items()}
    for seq in frames.values():
        if len(seq)!=41:
            raise ValueError("Expected 41 sampled frames from an 81-frame video")
    model.set_params(frame=frames["dense"][0],count=41)
    rows=[]
    scores={a:[] for a in frames}
    for i in range(40):
        flow={}
        for a,seq in frames.items():
            im,nxt=padder(seq[i].shape).pad(seq[i],seq[i+1])
            _,f=model.model(im,nxt,iters=20,test_mode=True)
            if tuple(f.shape)!=(1,2,464,832):
                raise ValueError("Unexpected RAFT output geometry")
            scores[a].append(float(model.get_score(im,f)))
            flow[a]=f[0].detach().cpu().numpy()
        f,d=flow["candidate"],flow["dense"]
        row=frame_stats(f,d)
        row.update(grid_stats(f,d),candidate_dynamic=scores["candidate"][-1],dense_dynamic=scores["dense"][-1])
        rows.append(row)
    windows={f"{a}-{b}":reduce(rows[a//2:b//2]) for a,b in WINDOWS}
    res=dict(source=src,sha=sha,runtime=identity,seconds=time.perf_counter()-start,windows=windows,
             dynamic_degree={a:bool(model.check_move(seq)) for a,seq in scores.items()})
    if src in {"vbi_0392","vbi_0800","vbi_1013"}:
        prior=read(HIST/(src+"_raft.json"))
        if prior["arms"]["Joint"]["video_sha256"]!=sha["candidate"] or prior["arms"]["Dense"]["video_sha256"]!=sha["dense"]:
            raise ValueError("Historical video identity changed")
        old=prior["arms"]["Joint"]
        w=windows["0-80"]
        test={"flow":w["candidate_dynamic"]-old["summary"]["raft_continuous_px_per_2_frames"],
              "epe":w["epe"]-old["summary"]["raft_endpoint_error_px"],
              "mae":w["mae"]-old["summary"]["raft_magnitude_mae_px"],
              "deg":w["angle_deg"]-old["summary"]["raft_direction_error_deg"],
              "support":w["angle_support"]-old["summary"]["raft_direction_valid_fraction"]}
        for v in old["windows"]:
            name=f"{v['first_frame']}-{v['last_frame']}"
            now=windows[name]
            for k,l in zip(("c_amp","d_amp","epe","mae"),archived.WINDOW_METRICS):
                test[name+"."+k]=now[k]-v[l]
        res["legacy_deltas"]=test
    return res


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--source",action="append")
    args=cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    sample=read(HIST/"vbi_0392_raft.json")["runtime"]
    if archived.sha(SCOPE)!=sample["scope_sha256"]: raise ValueError("Scope binding mismatch")
    model,idn=archived.load_metric(read(SCOPE),WORK/"raft_vbench_local")
    idn.update(scope=archived.sha(SCOPE),device=torch.cuda.get_device_name(),torch=str(torch.__version__),
               opencv=cv2.__version__,numpy=np.__version__,iters=20,precision="float32",script=archived.sha(Path(__file__)))
    out=OUT/"raft"
    out.mkdir(exist_ok=True)
    cohort=read(OUT/"COHORT.json")
    chosen=args.source or list(dict.fromkeys(["vbi_0392","vbi_0800","vbi_1013","vbi_0024","vbi_0864"]+
                                              [v["record_id"] for v in cohort]))
    if not set(chosen).issubset({v["record_id"] for v in cohort}): raise ValueError(chosen)
    for s in chosen:
        path=out/(s+".json")
        hashes={a:archived.sha(OUT/a/(s+".mp4")) for a in ("dense","candidate")}
        if path.is_file():
            cache=read(path)
            if cache["runtime"]==idn and cache["sha"]==hashes: continue
        result=infer(s,model,model.infer.__globals__["InputPadder"],idn)
        atomic(path,result)
        print(s,round(result["seconds"],2),max([0.]+[abs(v) for v in result.get("legacy_deltas",{}).values()]),flush=True)


if __name__=="__main__":
    main()
