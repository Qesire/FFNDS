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
REQUIRED_IDENTITY = ("source_id", "seed", "expected_frames", "input_prompt_digest", "model_digest", "protocol_sha256", "run_sha256")
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
    if report.get("reference"):
        ref_identity=report.get("reference_identity",{})
        for key in ("source_id","seed","input_prompt_digest"):
            if identity.get(key)!=ref_identity.get(key) or ref_identity.get(key) in (None,""):
                failures.append("reference does not match source/seed/prompt: "+key)
        if not ref_identity.get("run_sha256"):
            failures.append("reference run receipt identity absent")
    for key in FAMILIES + (PAIRED_FAMILIES if report.get("reference") else ()):
        if not isinstance(report.get("metrics", {}).get(key), dict) or not report["metrics"][key]:
            failures.append("missing metric family " + key)
    required_metric_fields={
        "temporal_motion":("adjacent_RGB_MAE_0to1","downsampled_Farneback_mean_flow_px"),
        "spatial_detail":("laplacian_abs_energy_candidate",),
        "luma_color":("mean_encoded_BT709_luma","adjacent_encoded_luma_abs"),
        "runtime":REQUIRED_COST,
    }
    if report.get("reference"):
        required_metric_fields.update({
            "paired_spatial":("RGB_RMSE_0to1","RGB_MAE_0to1","encoded_luma_RMSE_0to1",
                              "signed_encoded_luma_drift_0to1","candidate_to_reference_laplacian_energy_ratio"),
            "paired_temporal":("adjacent_delta_RGB_RMSE_0to1",),
            "paired_motion":("downsampled_Farneback_flow_L2_px","candidate_to_reference_flow_magnitude_ratio"),
            "static_region":("reference_flow_threshold_downsampled_px",
                             "uncompensated_low_reference_motion_luma_delta_abs","warning"),
        })
    for family,fields in required_metric_fields.items():
        values=report.get("metrics",{}).get(family,{})
        if not isinstance(values,dict): values={}

        for key in fields:
            if key not in values:
                failures.append("missing metric "+family+"."+key)
    if not isinstance(identity.get("expected_frames"),int) or identity["expected_frames"] < 2:
        failures.append("missing valid expected_frames")
    artifact = report.get("metrics", {}).get("artifact", {})
    if artifact.get("decoded_frames", 0) < 2 or artifact.get("fps", 0) <= 0:
        failures.append("invalid frame coverage")
    if artifact.get("decoded_frames") != artifact.get("verified_full_decode_frames"):
        failures.append("not all frames verified")
    if artifact.get("decoded_frames") != identity.get("expected_frames"):
        failures.append("not all expected frames decoded")
    if report.get("reference") and report.get("metrics", {}).get("paired_spatial", {}).get("frames") != artifact.get("decoded_frames"):
        failures.append("paired frames incomplete")
    frames = report.get("per_frame", [])
    if not isinstance(frames, list) or len(frames) != artifact.get("decoded_frames", -1):
        failures.append("missing or incomplete per-frame metric series")
    elif [f.get("frame") for f in frames] != list(range(len(frames))):
        failures.append("per-frame indices not contiguous")
    elif any(("laplacian_abs_edge_energy" not in f or "mean_encoded_luma" not in f
              or (f["frame"]>0 and ("candidate_adjacent_RGB_MAE_0to1" not in f
                                    or "candidate_downsampled_Farneback_flow_mean_px" not in f)))
             for f in frames):
        failures.append("intrinsic per-frame quality observations incomplete")
    elif report.get("reference") and any("paired_RGB_RMSE_0to1" not in f or
                                      (f["frame"] > 0 and "paired_temporal_delta_RGB_RMSE_0to1" not in f)
                                      for f in frames):
        failures.append("paired framewise quality metrics incomplete")
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
        framewise = []
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
            luma = [f @ np.array([.2126,.7152,.0722],dtype=np.float32) for f in frames]
            for v in gray:
                if not np.all(np.isfinite(v)):
                    raise ValueError("Nonfinite decoded pixels")
            data["edge_energy"].append(float(np.mean(np.abs(cv2.Laplacian(gray[0], cv2.CV_32F)))))
            item={"frame":count,"laplacian_abs_edge_energy":data["edge_energy"][-1]}
            data["mean_encoded_luma"].append(float(np.mean(luma[0])))
            item["mean_encoded_luma"]=data["mean_encoded_luma"][-1]
            if reference:
                diff = frames[0] - frames[1]
                data["pixel_rmse"].append(float(np.sqrt(np.mean(diff**2))))
                data["pixel_mae"].append(float(np.mean(np.abs(diff))))
                d_luma = luma[0]-luma[1]
                data["encoded_luma_rmse"].append(float(np.sqrt(np.mean(d_luma**2))))
                data["mean_signed_luma_drift"].append(float(np.mean(d_luma)))
                item["paired_RGB_RMSE_0to1"]=data["pixel_rmse"][-1]
                item["paired_encoded_luma_RMSE_0to1"]=data["encoded_luma_rmse"][-1]
                item["paired_signed_encoded_luma_drift_0to1"]=data["mean_signed_luma_drift"][-1]
                edge_r = float(np.mean(np.abs(cv2.Laplacian(gray[1],cv2.CV_32F))))
                data["edge_energy_ratio"].append(float(data["edge_energy"][-1]/max(edge_r,1e-9)))
                item["candidate_to_reference_laplacian_energy_ratio"]=data["edge_energy_ratio"][-1]
            if old is not None:
                data["temporal_mae"].append(float(np.mean(np.abs(frames[0]-old[0]))))
                data["luma_temporal_abs"].append(float(np.mean(np.abs(luma[0]-old[3][0]))))
                item["candidate_adjacent_RGB_MAE_0to1"]=data["temporal_mae"][-1]
                item["candidate_adjacent_encoded_luma_abs"]=data["luma_temporal_abs"][-1]
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
                item["candidate_downsampled_Farneback_flow_mean_px"]=data["temporal_flow_mean"][-1]
                if reference:
                    delta = (frames[0]-old[0])-(frames[1]-old[2][1])
                    data["temporal_delta_rmse"].append(float(np.sqrt(np.mean(delta**2))))
                    item["paired_temporal_delta_RGB_RMSE_0to1"]=data["temporal_delta_rmse"][-1]
                    data["motion_flow_l2"].append(float(np.mean(np.linalg.norm(flows[0]-flows[1],axis=2))))
                    item["paired_Farneback_flow_L2_px"]=data["motion_flow_l2"][-1]
                    m0 = float(np.mean(np.linalg.norm(flows[1],axis=2)))
                    data["flow_magnitude_ratio"].append(float(data["temporal_flow_mean"][-1]/max(m0,1e-9)))
                    # Fixed low-motion mask is obtained ONLY from dense flow.
                    mask = np.linalg.norm(flows[1],axis=2) <= .25
                    if np.any(mask):
                        d0 = cv2.resize(luma[0]-old[3][0],flows[1].shape[1::-1])
                        d1 = cv2.resize(luma[1]-old[3][1],flows[1].shape[1::-1])
                        data["static_luma_delta_error"].append(float(np.mean(np.abs(d0[mask]-d1[mask]))))
                        item["paired_low_motion_luma_delta_abs"]=data["static_luma_delta_error"][-1]
            framewise.append(item)
            old = (frames[0],gray,frames,luma)
            count += 1
        if count < 2:
            raise ValueError("Need at least two fully decoded frames")
        if count != identity_record["identity"]["expected_frames"]:
            raise ValueError("Decoded frame count differs from preregistered expected_frames")
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
                                          "candidate_to_reference_laplacian_energy_ratio":summary(data["edge_energy_ratio"]),
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
                  "reference_identity":identity_record.get("reference_identity") if reference else None,
                  "metrics":metrics,"per_frame":framewise,
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
