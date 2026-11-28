"""Quantify temporal change in the six fingerprint-matched A/C/C0 DEV clips."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "outputs"
ZOOM = (416, 232)
FLOW = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def read(out, name):
    return json.loads((out / name).read_bytes())


def measure(path, expected):
    assert sha256(path.read_bytes()).hexdigest() == expected
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = [int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)]
    previous = first = None
    changes, flow_p90 = [], []
    frames = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                changes.append(float(cv2.absdiff(previous, luma).mean()))
                small = cv2.resize(luma, ZOOM, interpolation=cv2.INTER_AREA)
                displacement = cv2.calcOpticalFlowFarneback(previous_small, small, None, **FLOW)
                flow_p90.append(float(np.quantile(np.linalg.norm(displacement, axis=2), .9)))
            else:
                first = luma
                small = cv2.resize(luma, ZOOM, interpolation=cv2.INTER_AREA)
            previous, previous_small = luma, small
            frames += 1
    finally:
        cap.release()
    assert frames == 81 and fps > 0 and len(changes) == len(flow_p90) == 80
    return dict(frames=frames, fps=fps, geometry=size, mean_adjacent_luma_mae=mean(changes),
                late_adjacent_luma_mae=mean(changes[60:]), first_last_luma_mae=float(cv2.absdiff(first, previous).mean()),
                median_flow_spatial_p90_px=median(flow_p90), late_median_flow_spatial_p90_px=median(flow_p90[60:]),
                mean_adjacent_luma_mae_by_quarter=[mean(changes[i:i+20]) for i in range(0,80,20)],
                flow_p90_px_by_transition=flow_p90, adjacent_luma_mae_by_transition=changes)


def main(directory):
    baseline = read(directory, "attention_online_quality_194188.json")
    scheduled = read(directory, "attention_c_schedule_quality_196295.json")
    receipt = read(directory, "attention_online_schedule_196295.json")
    assert receipt["status"] == scheduled["status"] == "complete" and receipt["quality"] == scheduled["quality"]
    archived = {r["source"]:r["record"]["video"]["mp4_sha256"] for r in baseline["renders"] if r["arm"] == "C"}
    out = dict(quality_job="196295", reference_job="194188", video_decoder=f"OpenCV {cv2.__version__}",
               luma_mae_units="8-bit grayscale levels on decoded full-size frames", flow_geometry=list(ZOOM),
               flow_parameters=FLOW, flow_summary="spatial p90 magnitude in pixels per frame, median over 80 transitions", sources={})
    for source in ("vbi_0392", "vbi_1013"):
        target = scheduled["quality"][source]["record"]["video"]["mp4_sha256"]
        ref = baseline["control"][source]["wide8192"]["video"]["mp4_sha256"]
        assert scheduled["references"][source]["mp4_sha256"]==archived[source]
        folders = {a: directory / "attention_online_videos_194188" for a in ("A","C")}
        folders["C0"] = directory / "attention_c_schedule_videos_196295"
        checks = dict(A=ref,C=archived[source],C0=target)
        metrics = {a:measure(folders[a]/f"{source}_{a}.mp4",checksum) for a,checksum in checks.items()}
        assert len({(r["frames"],r["fps"],tuple(r["geometry"])) for r in metrics.values()}) == 1
        out["sources"][source]=dict(mp4_sha256=checks,metrics=metrics)
    (directory / "attention_c_schedule_motion_196295.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf8")
    for source, item in out["sources"].items():
        for arm,m in item["metrics"].items():
            print(source,arm,*(round(m[k],4) for k in ("mean_adjacent_luma_mae","late_adjacent_luma_mae",
                         "median_flow_spatial_p90_px","late_median_flow_spatial_p90_px","first_last_luma_mae")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=OUT)
    main(parser.parse_args().directory)
