"""FFNDS video evidence: all-frame objective analysis and fail-closed audit.

Runs only on already generated videos; does not load a generative model.
Requires numpy and opencv-python for 'evaluate'; 'audit' uses stdlib only.
These inexpensive diagnostics are not VBench, perceptual scores, or human acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = "ffnds.video.metrics.v1"
REQUIRED_IDENTITY = ("source_id", "seed", "model_digest", "protocol_sha256", "run_sha256")
REQUIRED_COST = ("generation_wall_seconds", "ffn_boundary_seconds",
                 "setup_seconds", "peak_gpu_memory_mib")
FAMILIES = ("artifact", "temporal_motion", "spatial_detail", "luma_color", "runtime")
PAIRED_FAMILIES = ("paired_spatial", "paired_temporal", "paired_motion", "static_region")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values):
    if not values:
        raise ValueError("Empty measurement series")
    values = sorted(float(v) for v in values)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Non-finite metric")
    def percentile(p):
        i = p * (len(values) - 1)
        lo = math.floor(i)
        return values[lo] + (values[math.ceil(i)] - values[lo]) * (i - lo)
    return {"mean": statistics.fmean(values), "p50": percentile(.5),
            "p95": percentile(.95), "max": values[-1], "n": len(values)}


def validate(report, require_acceptance=False):
    failures = []
    if report.get("schema") != SCHEMA:
        failures.append("incorrect schema")
    identity, cost = report.get("identity", {}), report.get("cost", {})
    for key in REQUIRED_IDENTITY:
        if key not in identity or identity[key] in ("", None):
            failures.append("missing identity." + key)
    for key in REQUIRED_COST:
        if key not in cost or not isinstance(cost[key], (int, float)) or not math.isfinite(cost[key]) or cost[key] < 0:
            failures.append("invalid cost." + key)
    for key in FAMILIES + (PAIRED_FAMILIES if report.get("reference") else ()):
        if not isinstance(report.get("metrics", {}).get(key), dict) or not report["metrics"][key]:
            failures.append("missing metric family " + key)
    artifact = report.get("metrics", {}).get("artifact", {})
    if artifact.get("decoded_frames", 0) < 2 or artifact.get("fps", 0) <= 0:
        failures.append("invalid frame coverage")
    if artifact.get("decoded_frames") != artifact.get("verified_full_decode_frames"):
        failures.append("not all frames verified")
    if report.get("reference") and report.get("metrics", {}).get("paired_spatial", {}).get("frames") != artifact.get("decoded_frames"):
        failures.append("paired frames incomplete")
    ext = report.get("external_quality", {})
    for key in ("vbench_i2v", "vbench2", "human_review"):
        record = ext.get(key)
        if not isinstance(record, dict) or record.get("status") not in ("PASS", "FAIL", "NOT_RUN", "NOT_APPLICABLE"):
            failures.append("missing external score status " + key)
        elif record["status"] in ("NOT_RUN", "NOT_APPLICABLE") and not record.get("reason"):
            failures.append("missing non-computation reason " + key)
    if report.get("quality_accepted") is True:
        if not all(ext.get(k, {}).get("status") == "PASS" for k in ("vbench_i2v", "human_review")):
            failures.append("unsupported accepted quality")
    if require_acceptance and report.get("quality_accepted") is not True:
        failures.append("quality acceptance not documented")
    return failures


def evaluate(candidate, reference, identity_record):
    import cv2
    import numpy as np
    caps = [cv2.VideoCapture(str(p)) for p in ([candidate] + ([reference] if reference else []))]
    try:
        if not all(c.isOpened() for c in caps):
            raise ValueError("Video decode failed")
        fps_values = [float(c.get(cv2.CAP_PROP_FPS)) for c in caps]
        if min(fps_values) <= 0 or max(fps_values) - min(fps_values) > 1e-3:
            raise ValueError("Bad/mismatched frame rates")
        metrics = {family: {} for family in FAMILIES + (PAIRED_FAMILIES if reference else ())}
        data = {name: [] for name in ("temporal_mae", "temporal_flow_mean",
                   "edge_energy", "mean_encoded_luma", "luma_temporal_abs")}
        if reference:
            data.update({name: [] for name in ("pixel_rmse", "pixel_mae",
                        "encoded_luma_rmse", "mean_signed_luma_drift",
                        "temporal_delta_rmse", "motion_flow_l2", "flow_magnitude_ratio",
                        "static_luma_delta_error", "edge_energy_ratio")})
        old = None
        count = 0
        frame_shape = None
        while True:
            reads = [cap.read() for cap in caps]
            if not any(ok for ok, _ in reads):
                break
            if not all(ok for ok, _ in reads):
                raise ValueError("Reference and candidate have different frame counts")
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for _, frame in reads]
            if frame_shape is None:
                frame_shape = list(frames[0].shape)
            if any(list(f.shape) != frame_shape for f in frames):
                raise ValueError("Frame shape changed/mismatched")
            # BT.709 weights are applied to gamma-encoded pixels, not to linear luminance.
            gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
            for v in gray:
                if not np.all(np.isfinite(v)):
                    raise ValueError("Nonfinite decoded pixels")
            data["edge_energy"].append(float(np.mean(np.abs(cv2.Laplacian(gray[0], cv2.CV_32F)))))
            data["mean_encoded_luma"].append(float(np.mean(frames[0] @ np.array([.2126,.7152,.0722],dtype=np.float32))))
            if reference:
                diff = frames[0] - frames[1]
                data["pixel_rmse"].append(float(np.sqrt(np.mean(diff**2))))
                data["pixel_mae"].append(float(np.mean(np.abs(diff))))
                d_luma = (frames[0]-frames[1]) @ np.array([.2126,.7152,.0722],dtype=np.float32)
                data["encoded_luma_rmse"].append(float(np.sqrt(np.mean(d_luma**2))))
                data["mean_signed_luma_drift"].append(float(np.mean(d_luma)))
                edge_r = float(np.mean(np.abs(cv2.Laplacian(gray[1],cv2.CV_32F))))
                data["edge_energy_ratio"].append(float(data["edge_energy"][-1]/max(edge_r,1e-9)))
            if old is not None:
                data["temporal_mae"].append(float(np.mean(np.abs(frames[0]-old[0]))))
                data["luma_temporal_abs"].append(float(np.mean(np.abs(gray[0]-old[1][0]))))
                flows = []
                for g, oldg in zip(gray, old[1]):
                    h, w = g.shape
                    scale = min(1.0, 320.0/max(h,w))
                    dims = (max(8,round(w*scale)), max(8,round(h*scale)))
                    prev_s = cv2.resize(oldg,dims)
                    curr_s = cv2.resize(g,dims)
                    flow = cv2.calcOpticalFlowFarneback(prev_s,curr_s,None,.5,3,15,3,5,1.2,0)
                    flows.append(flow)
                data["temporal_flow_mean"].append(float(np.mean(np.linalg.norm(flows[0],axis=2))))
                if reference:
                    delta = (frames[0]-old[0])-(frames[1]-old[2][1])
                    data["temporal_delta_rmse"].append(float(np.sqrt(np.mean(delta**2))))
                    data["motion_flow_l2"].append(float(np.mean(np.linalg.norm(flows[0]-flows[1],axis=2))))
                    m0 = float(np.mean(np.linalg.norm(flows[1],axis=2)))
                    data["flow_magnitude_ratio"].append(float(data["temporal_flow_mean"][-1]/max(m0,1e-9)))
                    # Fixed low-motion mask is obtained ONLY from dense flow.
                    mask = np.linalg.norm(flows[1],axis=2) <= .25
                    if np.any(mask):
                        d0 = cv2.resize(gray[0]-old[1][0],flows[1].shape[1::-1])
                        d1 = cv2.resize(gray[1]-old[1][1],flows[1].shape[1::-1])
                        data["static_luma_delta_error"].append(float(np.mean(np.abs(d0[mask]-d1[mask]))))
            old = (frames[0],gray,frames)
            count += 1
        if count < 2:
            raise ValueError("Need at least two fully decoded frames")
        metrics["artifact"] = {"decoded_frames":count,"verified_full_decode_frames":count,
                               "fps":fps_values[0],"frame_shape":frame_shape,
                               "candidate_sha256":digest(candidate),
                               "reference_sha256":digest(reference) if reference else None}
        metrics["temporal_motion"] = {"adjacent_RGB_MAE_0to1":summary(data["temporal_mae"]),
                                      "downsampled_Farneback_mean_flow_px":summary(data["temporal_flow_mean"])}
        metrics["spatial_detail"] = {"laplacian_abs_energy_candidate":summary(data["edge_energy"]),
                                      "not_a_perceptual_sharpness_score":True}
        metrics["luma_color"] = {"mean_encoded_BT709_luma":summary(data["mean_encoded_luma"]),
                                 "adjacent_encoded_luma_abs":summary(data["luma_temporal_abs"])}
        metrics["runtime"] = dict(identity_record["cost"])
        if reference:
            metrics["paired_spatial"] = {"frames":count,"RGB_RMSE_0to1":summary(data["pixel_rmse"]),
                                          "RGB_MAE_0to1":summary(data["pixel_mae"]),
                                          "encoded_luma_RMSE_0to1":summary(data["encoded_luma_rmse"]),
                                          "signed_encoded_luma_drift_0to1":summary(data["mean_signed_luma_drift"])}
            metrics["paired_temporal"] = {"adjacent_delta_RGB_RMSE_0to1":summary(data["temporal_delta_rmse"])}
            metrics["paired_motion"] = {"downsampled_Farneback_flow_L2_px":summary(data["motion_flow_l2"]),
                                        "candidate_to_reference_flow_magnitude_ratio":summary(data["flow_magnitude_ratio"])}
            metrics["static_region"] = {"reference_flow_threshold_downsampled_px":.25,
                                         "uncompensated_low_reference_motion_luma_delta_abs":summary(data["static_luma_delta_error"]) if data["static_luma_delta_error"] else {"status":"NOT_ENOUGH_STATIC_PIXELS"},
                                         "warning":"Reference-flow low-motion pixels are not camera-compensated background masks or official temporal flicker scores."}
        report = {"schema":SCHEMA,"identity":identity_record["identity"],"cost":identity_record["cost"],
                  "candidate":{"path":str(candidate),"sha256":digest(candidate)},
                  "reference":{"path":str(reference),"sha256":digest(reference)} if reference else None,
                  "metrics":metrics,
                  "external_quality":{k:{"status":"NOT_RUN","reason":"Independent official benchmark or manual review not yet attached"}
                                      for k in ("vbench_i2v","vbench2","human_review")},
                  "quality_accepted":False,"scope":"ALL_FRAMES_EVALUATED_NO_OFFICIAL_OR_HUMAN_ACCEPTANCE"}
        return report
    finally:
        for c in caps:
            c.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    e=sub.add_parser("evaluate")
    e.add_argument("--candidate",type=Path,required=True)
    e.add_argument("--reference",type=Path)
    e.add_argument("--identity-json",type=Path,required=True)
    e.add_argument("--out",type=Path,required=True)
    a=sub.add_parser("audit")
    a.add_argument("--report",type=Path,required=True)
    a.add_argument("--require-acceptance",action="store_true")
    args=p.parse_args()
    if args.action=="evaluate":
        inputs=json.loads(args.identity_json.read_text())
        if args.out.exists():
            raise FileExistsError("Report is immutable; choose new output path")
        report=evaluate(args.candidate,args.reference,inputs)
        errors=validate(report)
        if errors:
            raise ValueError("Incomplete video metric report: "+"; ".join(errors))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        print("CORE_METRICS_COMPLETE__QUALITY_NOT_ACCEPTED",args.out)
    else:
        report=json.loads(args.report.read_text())
        errors=validate(report,args.require_acceptance)
        if errors:
            raise SystemExit("INVALID_METRIC_REPORT: "+"; ".join(errors))
        print("CORE_METRICS_COMPLETE", "QUALITY_ACCEPTED" if report.get("quality_accepted") else "QUALITY_PENDING")


if __name__=="__main__":
    main()
