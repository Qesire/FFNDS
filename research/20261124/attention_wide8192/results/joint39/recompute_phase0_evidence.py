#!/usr/bin/env python3
"""Recompute phase0 receipts and compare actual calibration matrices.

Requires NumPy and SymPy. This runs no model and decodes no video.
The old transform is available in results/replay_evidence.zip.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import zipfile

import numpy as np
import sympy as sp


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_arrays(path):
    data = Path(path).read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8+n])
    dtype = {"F32": "<f4", "F64": "<f8"}
    arrays = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        arrays[name] = np.frombuffer(
            data[8+n+start:8+n+end], dtype=dtype[entry["dtype"]]
        ).reshape(entry["shape"]).astype(np.float64)
    return arrays, data


def old_transform(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("calibration/self_high_transform.npz")]
            if len(names) != 1:
                raise ValueError(names)
            data = z.read(names[0])
    else:
        data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, data


def first_existing(paths):
    return next((p for p in paths if p.exists()), paths[0])


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=first_existing([here/"phase0_source", here]))
    parser.add_argument("--old-transform", type=Path, default=here.parent/"replay_evidence.zip")
    parser.add_argument("--old-motion", type=Path, default=here/"C_SCHEDULE_MOTION_196295.json")
    parser.add_argument("--old-schedule", type=Path, default=here/"C_SCHEDULE_196295.json.gz")
    parser.add_argument("--output", type=Path, default=here/"phase0_evidence_recomputed.json")
    args = parser.parse_args()
    root = args.source_dir
    result = read_json(root/"PHASE0_CAL_RESULT_196471.json")
    motion = read_json(root/"PHASE0_CAL_MOTION_196471.json")
    human = read_json(root/"PHASE0_CAL_HUMAN_REVIEW_20261125.json")
    arrays, new_bytes = safe_arrays(root/"PHASE0_BASIS_196471.safetensors")
    old, old_bytes = old_transform(args.old_transform)
    assert sha(new_bytes) == result["fit_basis_sha256"]
    assert sha(old_bytes) == result["calibration"]["high"]["transform_sha256"]
    assert result["fixed_outabs_lambda"] == .1
    assert len(result["fit_sites"]) == 24 and len(result["held_diagnostics"]) == 16
    assert human["candidate_job"] == result["job"] == "196471"

    held = {}
    for row in result["held_diagnostics"]:
        ratio = row["point_error"]["phase0_basis"]/row["point_error"]["step1_basis"]
        assert math.isclose(ratio, row["phase0_to_step1_ratio"], rel_tol=1e-14)
        held.setdefault(row["source"], {}).setdefault(row["background"], []).append({
            "layer": row["layer"], "ratio": ratio, "point_error": row["point_error"]})
    for src, backgrounds in held.items():
        assert backgrounds["Dense"] == backgrounds["wide8192"]
    held_summary = {
        source: {
            background: {
                "layers": rows,
                "equal_layer_geomean": statistics.geometric_mean(x["ratio"] for x in rows),
                "pooled_energy_ratio": sum(x["point_error"]["phase0_basis"] for x in rows)/
                                      sum(x["point_error"]["step1_basis"] for x in rows),
            } for background, rows in backgrounds.items()
        } for source, backgrounds in held.items()
    }

    t0 = arrays["basis"]
    t1 = old["out_current_basis"].astype(np.float64)
    assert t0.shape == t1.shape == (64, 31)
    q0, q1 = np.linalg.qr(t0)[0], np.linalg.qr(t1)[0]
    cosines = np.linalg.svd(q0.T@q1, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    k0, k1 = t0@t0.T, t1@t1.T
    rms_ratio = arrays["rms"]/old["out_current_rms"]
    geometry = {
        "metric": "Euclidean geometry on the same fixed 64-dimensional raw selected-output coordinates; no task-sensitivity metric assumed",
        "principal_angles_degrees": angles.tolist(),
        "principal_angle_median_degrees": float(np.median(angles)),
        "principal_angle_max_degrees": float(angles.max()),
        "mean_squared_cosine_overlap": float(np.mean(cosines**2)),
        "kernel_relative_frobenius": float(np.linalg.norm(k0-k1)/np.linalg.norm(k1)),
        "kernel_definition": "K=T T^T acts on raw projected selected-output coordinates, before BF16 rounding and constant feature",
        "rms_new_over_old_min_median_max": [float(rms_ratio.min()),float(np.median(rms_ratio)),float(rms_ratio.max())],
        "retained_standardized_feature_energy_new": float(np.sum(arrays["spectrum"][:31]**2)/np.sum(arrays["spectrum"]**2)),
        "retained_standardized_feature_energy_old": float(np.sum(old["out_current_spectrum"][:31]**2)/np.sum(old["out_current_spectrum"]**2)),
        "energy_scope": "Each energy fraction describes its own CAL Gram, not retained native residual energy or motion information",
    }

    previous = read_json(args.old_motion)["sources"]["vbi_0392"]["metrics"]
    luma, flow = motion["adjacent_luma_mae_by_transition"], motion["flow_p90_px_by_transition"]
    assert len(luma) == len(flow) == 80
    recomputed = {
        "mean_adjacent_luma_mae": statistics.mean(luma),
        "late_adjacent_luma_mae": statistics.mean(luma[-20:]),
        "median_flow_spatial_p90_px": statistics.median(flow),
        "late_median_flow_spatial_p90_px": statistics.median(flow[-20:]),
        "mean_adjacent_luma_mae_by_quarter": [statistics.mean(luma[i:i+20]) for i in range(0,80,20)],
    }
    for name, value in recomputed.items():
        if isinstance(value, list):
            assert all(math.isclose(a,b,rel_tol=1e-13) for a,b in zip(value,motion[name]))
        else:
            assert math.isclose(value,motion[name],rel_tol=1e-13)
    ratios = {
        arm: {name: recomputed[name]/previous[arm][name]
              for name in ["mean_adjacent_luma_mae", "late_adjacent_luma_mae",
                           "median_flow_spatial_p90_px", "late_median_flow_spatial_p90_px"]}
        for arm in ["C","C0"]
    }

    import gzip
    data = args.old_schedule.read_bytes()
    if args.old_schedule.suffix == ".gz":
        data = gzip.decompress(data)
    schedule = json.loads(data)
    old_record = schedule["quality"]["vbi_0392"]["record"]
    current = result["record"]
    assert old_record["initial"] == current["initial"]
    trace_identity = {
        "initial_matches_C0": True,
        "flow_hash_matches_C0": [a==b for a,b in zip(old_record["flows"],current["flows"])],
        "latent_hash_matches_C0": [a==b for a,b in zip(old_record["latents"],current["latents"])],
        "mp4_hash_matches_C0": old_record["video"]["mp4_sha256"] == current["video"]["mp4_sha256"],
        "scope": "Only identity checks; hashes do not estimate tensor distance. Raw trace tensors were not loaded.",
    }

    # Exact real-arithmetic interpretation of the actual headwise computation.
    n,d,m,r = 31668,128,64,31
    b = sp.MatrixSymbol("B",n,d)
    pv = sp.MatrixSymbol("Pv",d,m)
    transform = sp.MatrixSymbol("T",m,r)
    weight = sp.MatrixSymbol("W",r,d)
    bias = sp.MatrixSymbol("b",1,d)
    ones = sp.MatrixSymbol("ones",n,1)
    before = b+ones*bias+(b*pv*transform)*weight
    after = b*(sp.Identity(d)+pv*transform*weight)+ones*bias
    difference = sp.expand(before-after)
    assert difference == sp.ZeroMatrix(n,d)
    affine = {
        "symbolic_verified": True, "symbolic_difference": str(difference),
        "formula": "B + [1, B Pv T] [b;W] = B (I + Pv T W) + 1 b",
        "coefficient_sharing": "W,b are fitted per layer/head/current call from P64, then shared across non-P queries; T shared",
        "channel_update_rank_upper_bound": 31,
        "rounding": "Actual features use Q_BF16(B Pv T); extra term [Q_BF16(B Pv T)-B Pv T] W remains; final casting and P overwrite are separate",
        "exact_query_relation": "Equal selected-output rows have equal predicted rows outside P even with deterministic BF16 feature rounding",
        "inference_scope": "This identifies the implemented function family; it does not prove that actual motion errors lie outside it",
    }
    out = {
        "evidence_commit": "c5e4ee8168708a8e57cdb981103c4dbf715e3bfa",
        "scope": "Actual receipt aggregates and calibration-matrix algebra; no model run, no video decode, no tensor-error recomputation",
        "hashes": {"new_basis":sha(new_bytes),"old_high_transform":sha(old_bytes)},
        "fit_sources":result["cal_sources"],"fit_panel_count":24,"held_panel_count":16,
        "held_background_scores_identical": True,"held_summary":held_summary,
        "basis_geometry":geometry,"motion_recomputed":recomputed,
        "motion_candidate_over_reference":ratios,"trajectory_identity":trace_identity,
        "outabs_affine_family":affine,"human_review":human,
        "timing_scope": "One captured cold quality render; 114.269 s is not confirmatory timing and must not be compared to warm C/C0 timing",
    }
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),
                      "held_geomeans":{s:b["wide8192"]["equal_layer_geomean"] for s,b in held_summary.items()},
                      "angle_median":geometry["principal_angle_median_degrees"],
                      "kernel_relative_frobenius":geometry["kernel_relative_frobenius"],
                      "motion_ratios":ratios,"affine_identity_verified":True},indent=2))


if __name__ == "__main__":
    main()
