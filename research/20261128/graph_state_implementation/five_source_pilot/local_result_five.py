"""Read signed local receipts and build a source-by-source playable screening index."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, geometric_mean

import cv2
import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def read_latent(path, tag):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==tag["sha256"]
    size=int.from_bytes(raw[:8],"little")
    meta=json.loads(raw[8:8+size])["latent"]
    a,b=meta["data_offsets"]
    data=raw[8+size+a:8+size+b]
    if meta["dtype"]=="BF16":
        x=(np.frombuffer(data,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
    else:
        x=np.frombuffer(data,dtype={"F32":"<f4","F16":"<f2"}[meta["dtype"]])
    assert math.prod(meta["shape"])==x.size
    return x.astype(np.float64)


def pixel_compare(video,ref):
    clips=[cv2.VideoCapture(str(p)) for p in (video,ref)]
    n=0; squared=0.; temporal=0.; prev=None
    try:
        assert all(c.isOpened() for c in clips)
        while True:
            (oa,a),(ob,b)=(s.read() for s in clips)
            if not (oa or ob):break
            assert oa and ob and a.shape==b.shape==(464,832,3)
            diff=np.subtract(a,b,dtype=np.int16)
            squared+=float(np.square(diff,dtype=np.int32).sum(dtype=np.float64))
            if prev is not None:temporal+=float(np.abs(diff-prev).sum(dtype=np.float64))
            prev=diff; n+=1
    finally:
        for c in clips:c.release()
    assert n==81
    pixels=464*832*3
    return math.sqrt(squared/(n*pixels))/255, temporal/((n-1)*pixels*255)


def speed(proposed, ref, names):
    assert proposed["job"]==ref["job"] and proposed["gpu"]==ref["gpu"] and ref["profile"]["name"]=="frozen"
    slots=[(proposed["records"][s]["record"],ref["records"][s]["record"]) for s in names]
    assert all(a["initial"]==b["initial"] for a,b in slots)
    return dict(wall_speedup=geometric_mean(b["wall"]/a["wall"] for a,b in slots),
                gpu_speedup=geometric_mean(sum(b["step_ms"])/sum(a["step_ms"]) for a,b in slots),
                wall_candidate_seconds=fmean(a["wall"] for a,_ in slots),
                wall_reference_seconds=fmean(b["wall"] for _,b in slots),
                peak_candidate_gib=max(a["peak_allocated"] for a,_ in slots)/2**30,
                peak_reference_gib=max(b["peak_allocated"] for _,b in slots)/2**30)


def run(root):
    base_dir=root/"00_frozen"
    frozen=load(base_dir/"candidate.json")
    names=frozen["sources"]
    assert len(names)==5
    all_results={}
    page=["# 五源优化实测原片与逐源指标", "", "所有原片均校验 MP4 SHA-256。像素与时间误差使用同一个 OpenCV 解码器，对照为冻结 centered + wide8192。", "",
          "| 配置 | 来源 | 原片 | 冻结原片 | 端点 latent 相对 L2 | RGB RMSE | 帧间差分 RGB MAE |", "|---|---|---|---|---:|---:|---:|"]
    for path in sorted(root.glob("*/candidate.json")):
        profile=path.parent.name; data=load(path); rows=[]
        assert data["sources"]==names and data["status"]=="complete" and set(data["records"])==set(names)
        for s in names:
            a,b=data["records"][s],frozen["records"][s]
            ca,cb=a["record"],b["record"]
            assert ca["initial"]==cb["initial"]
            video=path.parent/(s+".mp4"); ref=base_dir/(s+".mp4")
            for p,r in ((video,ca),(ref,cb)):
                assert hashlib.sha256(p.read_bytes()).hexdigest()==r["video"]["mp4_sha256"]
            rgb,temporal=(0.,0.) if ca["video"]["decoded_rgb24_sha256"]==cb["video"]["decoded_rgb24_sha256"] else pixel_compare(video,ref)
            latent=None
            if a["final_latent"] and b["final_latent"]:
                x,y=(read_latent(p.parent/r["final_latent"]["file"],r["final_latent"]) for p,r in ((video,a),(ref,b)))
                latent=float(np.linalg.norm(x-y)/np.linalg.norm(y))
            rows.append(dict(source_id=s,exact_frozen_trace=a["exact_206224_centered_trace"],rgb_rmse_0to1=rgb,
                             temporal_delta_rgb_mae_0to1=temporal,final_latent_relative_l2=latent))
            t="—" if latent is None else f"{latent:.5%}"
            page.append(f"| {profile} | {s} | [候选](<{video.as_posix()}>) | [冻结](<{ref.as_posix()}>) | {t} | {rgb:.6f} | {temporal:.6f} |")
        path_ref=path.parent/"same_gpu_reference.json"
        timing=speed(data,load(path_ref),names) if path_ref.is_file() else None
        all_results[profile]=dict(profile=data["profile"],semantics=data["semantics"],quality=rows,timing=timing)
        print(profile,"exact",sum(r["exact_frozen_trace"] for r in rows),"RGB",round(fmean(r["rgb_rmse_0to1"] for r in rows),6),timing,flush=True)
    (root/"VIDEO_INDEX.md").write_text("\n".join(page)+"\n",encoding="utf-8")
    (root/"LOCAL_CHECK.json").write_text(json.dumps(all_results,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path); a=p.parse_args()
    run(a.root.resolve())
