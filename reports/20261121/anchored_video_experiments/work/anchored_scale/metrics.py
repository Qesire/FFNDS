"""Assess paired anchor/Dense videos with the DEV3 raster, geometry, and flow recipe."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"video_joint_eval"))
import evaluate as dev3

video = dev3.load_module("tracked_patches", HERE.parent/"track1013/run.py")
cv2.setNumThreads(1)
WINDOWS = [(0, 80), *((i, i+16) for i in range(0, 80, 16))]
VERSION = "anchor-paired-1"
MOVING = 0.5  # original pixels per two frames
OUT = HERE.parents[1]/"outputs/FFNDS_anchor_scale24"


def read(path):
    return json.loads(path.read_bytes())


def write_json(path, record):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf8")
    temp.replace(path)


def appearance(frames):
    detail = []
    for frame in frames:
        y = frame.astype(np.float32)
        mid = cv2.GaussianBlur(y, (0, 0), 1)-cv2.GaussianBlur(y, (0, 0), 4)
        contrast = y.std()
        detail.append(float(mid.std()/contrast) if contrast > 0 else None)
    small = np.stack([cv2.resize(f, (416, 232), interpolation=cv2.INTER_AREA)[2::4, 2::4] for f in frames]).astype(np.float32)
    # At each odd frame's centre, the component unexplained by interpolation of its neighbours.
    highpass = small[1::2] - 0.5*(small[::2][:-1]+small[2::2])
    return np.array(detail, dtype=np.float64), highpass


def mean_valid(values):
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def temporal_record(high, quiet):
    vals = np.abs(high[quiet])
    return dict(quiet_fraction=float(quiet.mean()), n=int(len(vals)),
                mean_abs=float(vals.mean()) if len(vals) else None,
                p95_abs=float(np.percentile(vals, 95)) if len(vals) else None)


def directed(ref, arm):
    result = dev3.flow_stats(ref, arm)
    a = np.linalg.norm(ref, axis=-1)
    b = np.linalg.norm(arm, axis=-1)
    ref_active = a >= MOVING
    active = ref_active & (b >= MOVING)
    dot = (ref*arm).sum(axis=-1)
    cosine = dot[active]/(a[active]*b[active])
    count = int(np.count_nonzero(active))
    result.update(reference_active_fraction=float(ref_active.mean()),
                  candidate_active_fraction=float(np.mean(b>=MOVING)), joint_active_count=count,
                  joint_active_fraction=float(active.mean()), dense_rms=float(np.sqrt((a*a).mean())),
                  energy_direction_deg=(math.degrees(math.acos(float(np.clip(result["energy_weighted_direction_cosine"],-1,1))))
                                        if result["energy_weighted_direction_cosine"] is not None else None),
                  mean_active_angle_deg=float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))).mean()) if count else None)
    result["reference_quiet"] = result["dense_rms"] < MOVING
    # The legacy common/difference field above is horizontal. Add its vertical counterpart.
    vertical = dev3.flow_stats(ref.swapaxes(1,2), arm.swapaxes(1,2))
    result["vertical_normalized_common_difference"] = vertical["normalized_common_difference"]
    result["vertical_common_difference_error_gram_px2"] = vertical["common_difference_error_gram_px2"]
    return result


def measure(src, dense, candidate):
    files = {"dense":dense, "candidate":candidate}
    frames = {k:video.decode(p) for k,p in files.items()}
    flows = {k:dev3.motion(f) for k,f in frames.items()}
    image = {k:appearance(f) for k,f in frames.items()}
    result = dict(source=src, version=VERSION, opencv=cv2.__version__, frame_count=81, fps=16,
                  hashes={k:dev3.digest(p) for k,p in files.items()}, windows={})
    for start,end in WINDOWS:
        span = slice(start//2,end//2)
        stats = directed(flows["dense"][span], flows["candidate"][span])
        quiet = {k:np.linalg.norm(v[span],axis=-1)<MOVING for k,v in flows.items()}
        both = quiet["dense"] & quiet["candidate"]
        stats["appearance"] = {}
        for k,(texture,highpass) in image.items():
            stats["appearance"][k] = dict(midband_norm=mean_valid(texture[start:end+1]),
                    own_quiet_temporal=temporal_record(highpass[span],quiet[k]),
                    matched_quiet_temporal=temporal_record(highpass[span],both))
        result["windows"][f"{start}-{end}"] = stats
    if src == "vbi_1013":
        result["tracked_regions"] = dev3.fingers(video,{"Dense":dense,"Candidate":candidate})
    return result


def axes(m, key):
    raw = m["common_difference_error_gram_px2"]
    other = m["vertical_common_difference_error_gram_px2"]
    return math.sqrt((raw[key][key]+other[key][key])/2)


def flat(m):
    image = m["appearance"]
    d,c = (image[k] for k in ("dense","candidate"))
    texture, old = c["midband_norm"], d["midband_norm"]
    return dict(reference_quiet=m["reference_quiet"], dense_amp=m["dense_amplitude"],
                candidate_amp=m["arm_amplitude"], epe=m["endpoint_error"],
                gain=m["signed_scale"] if not m["reference_quiet"] else None,
                angle_deg=m["mean_active_angle_deg"], both_moving=m["joint_active_fraction"],
                excess=m["mean_amplitude_excess"], deficit=m["mean_amplitude_deficit"],
                common_err=axes(m,0), differential_err=axes(m,1), texture=texture,
                dense_texture=old, texture_ratio=texture/old if old and texture is not None else None,
                quiet_fraction=c["matched_quiet_temporal"]["quiet_fraction"],
                quiet_candidate=c["matched_quiet_temporal"]["mean_abs"], quiet_dense=d["matched_quiet_temporal"]["mean_abs"])


def cohort(root):
    obj=read(root/"COHORT.json")
    entries=obj if isinstance(obj,list) else obj["sources"]
    if isinstance(entries,dict):
        return {s:dict(v,source=s) for s,v in entries.items()}
    return {(v.get("source") or v["record_id"]):v for v in entries}


def paired_receipts(source, paths):
    records = {k:read(p.with_suffix(".json")) for k,p in paths.items()}
    hashes = {k:dev3.digest(p) for k,p in paths.items()}
    for k,r in records.items():
        if (r["source"] != source or r["video"]["mp4_sha256"] != hashes[k] or
                r["experts"] != [0,0,1,1]):
            raise ValueError(f"Source, schedule or video hash disagrees for {source}/{k}")
    d,c = records["dense"],records["candidate"]
    if c["initial"] != d["initial"]:
        raise ValueError(f"Unpaired initial noise at {source}")
    return hashes, dict(initial=c["initial"],same_step0_latent=c["latents"][0]==d["latents"][0],
                        same_step0_flow=c["flows"][0]==d["flows"][0])


def summarize(root, metadata, force_complete):
    records={}
    for src in metadata:
        path=root/"metrics"/(src+".json")
        if path.is_file():
            records[src]=read(path)
    if force_complete and len(records)!=len(metadata):
        raise ValueError(f"Missing results for {set(metadata)-set(records)}")
    if not records:
        return
    rows=[]
    for src,r in records.items():
        row=metadata[src]
        for window,metrics in r["windows"].items():
            rows.append(dict(source=src, group=row.get("cohort_group"), cluster=row.get("source_cluster_id"),
                             category=row.get("category"),window=window, **flat(metrics)))
    with (root/"指标逐源逐窗.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    subsets={"ALL":rows}
    for key in ("group","category"):
        for label in sorted({str(r[key]) for r in rows}):
            name=label if key=="group" else f"category={label}"
            subsets[name]=[r for r in rows if str(r[key])==label]
    summary={}
    for group,group_rows in subsets.items():
        out={}
        for window in records[next(iter(records))]["windows"]:
            items=[r for r in group_rows if r["window"]==window]
            stats={}
            for k in list(items[0])[6:]:
                a=[r[k] for r in items if r[k] is not None]
                q=np.percentile(a,[25,50,75]).tolist() if a else [None]*3
                stats[k]=dict(valid_sources=len(a),q25=q[0],median=q[1],q75=q[2])
            out[window]=dict(sources=len(items),reference_quiet_sources=sum(r["reference_quiet"] for r in items),statistics=stats)
        summary[group]=out
    write_json(root/"指标汇总.json",summary)
    from metrics_review import write_review
    write_review(root, metadata, rows, summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=OUT)
    parser.add_argument("--source",action="append")
    parser.add_argument("--partial",action="store_true",help="Process transferred pairs only")
    args=parser.parse_args()
    root=args.root
    meta=cohort(root)
    dst=root/"metrics"
    dst.mkdir(exist_ok=True,parents=True)
    for src in (args.source or meta):
        if src not in meta:
            raise ValueError(src)
        dense,cand=[root/arm/(src+".mp4") for arm in ("dense","candidate")]
        if args.partial and not (dense.is_file() and cand.is_file()):
            continue
        hashes, history=paired_receipts(src, {"dense":dense,"candidate":cand})
        output=dst/(src+".json")
        if output.is_file():
            old=read(output)
            if old["hashes"]==hashes and old["version"]==VERSION and old.get("pair")==history:
                continue
        record=measure(src,dense,cand)
        if record["hashes"]!=hashes:
            raise ValueError(f"Pair changed during measurement: {src}")
        record["pair"]=history
        write_json(output,record)
        print(src,flush=True)
    summarize(root,meta,not(args.partial or args.source))


if __name__=="__main__":
    main()
