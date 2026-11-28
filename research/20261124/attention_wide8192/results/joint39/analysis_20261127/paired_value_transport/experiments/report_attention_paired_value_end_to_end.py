"""Verify and score four-step videos from the paired-value rollout."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from review_attention_c_schedule import measure


COMPARISONS = {
    "diagonal_to_Dense": ("diagonal", "Dense"),
    "paired_to_Dense": ("paired", "Dense"),
    "diagonal_to_wide8192": ("diagonal", "wide8192"),
    "paired_to_wide8192": ("paired", "wide8192"),
    "paired_to_diagonal": ("paired", "diagonal"),
}


def pixel_errors(paths):
    streams = {arm:cv2.VideoCapture(str(path)) for arm,path in paths.items()}
    pixel_sq = dict.fromkeys(COMPARISONS,0.0)
    first_sq = dict.fromkeys(COMPARISONS,0.0)
    last_sq = dict.fromkeys(COMPARISONS,0.0)
    delta_sq = dict.fromkeys(COMPARISONS,0.0)
    previous = None
    frames = 0
    count = 0
    try:
        assert all(s.isOpened() for s in streams.values())
        while True:
            decoded = {k:s.read() for k,s in streams.items()}
            if not any(valid for valid,_ in decoded.values()):
                break
            assert all(valid and bgr.shape==(464,832,3) for valid,bgr in decoded.values())
            pixels = {k:value.astype(np.float32) for k,(_,value) in decoded.items()}
            count = next(iter(pixels.values())).size
            for key,(a,b) in COMPARISONS.items():
                difference = pixels[a]-pixels[b]
                err = float(np.square(difference).sum(dtype=np.float64))
                if frames == 0:
                    first_sq[key] = err
                pixel_sq[key] += err
                last_sq[key] = err
                if previous is not None:
                    delta_sq[key] += float(np.square(difference-(previous[a]-previous[b])).sum(dtype=np.float64))
            previous = pixels
            frames += 1
    finally:
        for stream in streams.values():
            stream.release()
    assert frames == 81
    out = {}
    for key,error in pixel_sq.items():
        rmse = math.sqrt(error/(frames*count))/255
        out[key] = dict(rmse_0_1=rmse,psnr_db=-20*math.log10(rmse) if rmse else None,
                         first_frame_rmse_0_1=math.sqrt(first_sq[key]/count)/255,
                         last_frame_rmse_0_1=math.sqrt(last_sq[key]/count)/255,
                         temporal_difference_rmse_0_1=math.sqrt(delta_sq[key]/((frames-1)*count))/255)
    return out


def summarize(receipt, media):
    raw = json.loads(receipt.read_bytes())
    assert raw["status"] == "complete" and raw["probe_queries_per_candidate"] == 5120
    result = dict(job=raw["job"],plan_commit=raw["plan_commit"], decoder="OpenCV "+cv2.__version__, sources={})
    for sid,entry in raw["records"].items():
        records = {name:item["record"] for name,item in entry["candidates"].items()} | entry["controls"]
        paths = {name:media/f"{sid}_{name}.mp4" for name in records}
        motion = {}
        for name,path in paths.items():
            identity = records[name]["video"]["mp4_sha256"]
            summary = measure(path,identity)
            motion[name] = {k:summary[k] for k in ("median_flow_spatial_p90_px","mean_adjacent_luma_mae","late_median_flow_spatial_p90_px")}
        assert entry["identical_prefix_steps"]==3
        assert all(value[field]["relative_l2"]==0 for value in entry["paired_vs_diagonal"][:3] for field in ("latent","flow"))
        result["sources"][sid] = dict(fold=entry["fold"], tensor_errors=entry["errors"], paired_vs_diagonal=entry["paired_vs_diagonal"],
                                      decoded_pixel_errors=pixel_errors(paths), within_video_motion_proxies=motion)
        print("VIDEO_SCORE_VERIFIED",sid,flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt",type=Path)
    p.add_argument("media",type=Path)
    p.add_argument("out",type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.receipt,a.media),indent=2,allow_nan=False)+"\n",encoding="utf-8")
