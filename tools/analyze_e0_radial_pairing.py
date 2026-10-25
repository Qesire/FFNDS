"""Distinguish shared geometry from token pairing in saved native FFN operands.

No model, CUDA, native norm or native FFN is called. Directions are the same
decoded FP64 group panels as the historical ideal-LN analysis.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/e0_radial_pairing_saved_20261025.json"
STATES = ["pre_FFN_state", "actual_post_FFN_state", "decoded_leave_group_out_state"]


def center_channels(x):
    return x - x.mean(axis=1, keepdims=True)


def expected_trace(matrix, labels=None, derange=True):
    """Exact permutation expectation, retaining singleton strata explicitly.

    Each non-singleton off-diagonal entry has marginal probability 1/(m-1)
    under a uniform stratum derangement. No sampling/independence assumption
    about the scientific data is needed for this combinatorial identity.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(matrix)
    if matrix.shape != (n, n) or n < 2:
        raise ValueError("Need a square matrix with at least two rows")
    labels = np.zeros(n, dtype=int) if labels is None else np.asarray(labels)
    if labels.shape != (n,):
        raise ValueError("Strata must label every row")
    total, moved = 0.0, 0
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        panel = matrix[np.ix_(ids, ids)]
        m = len(ids)
        if m == 1:
            total += float(panel[0, 0])
        elif derange:
            total += float((panel.sum() - np.trace(panel)) / (m - 1))
            moved += m
        else:
            total += float(panel.sum() / m)
            moved += m
    return total, moved


def pairing_statistics(state, delta, temporal, spatial, eps=1e-6):
    """State row i vs direction row j; channel centering is the LN operation."""
    state, delta = [np.asarray(x, dtype=np.float64) for x in (state, delta)]
    if state.shape != delta.shape or state.ndim != 2 or eps < 0:
        raise ValueError("Need paired matrices and a nonnegative epsilon")
    if not all(np.isfinite(x).all() for x in (state, delta)):
        raise ValueError("Nonfinite operand")
    x, d = center_channels(state), center_channels(delta)
    s, v = np.sum(x*x, axis=1), np.sum(d*d, axis=1)
    energy = float(np.sum(delta*delta))
    if np.any(s <= 0) or np.any(v <= 0) or energy <= 0:
        raise ValueError("Degenerate direction or residual")
    dot = x @ d.T
    de = state.shape[1] * eps
    # ||Pd||^2 - ||Pd - x (x.Pd)/(||x||^2 + D eps)||^2.
    radial = ((s + 2*de) / (s + de)**2)[:, None] * dot**2
    cosine = dot / np.sqrt(s[:, None] * v[None, :])
    matrices = {"radial_fraction": (radial, energy),
                "cosine_squared": (cosine**2, len(state)),
                "signed_cosine": (cosine, len(state))}
    out = {}
    for name, (matrix, norm) in matrices.items():
        out["paired_" + name] = float(np.trace(matrix) / norm)
        for control, labels, derange in [
            ("all_permutations", None, False), ("deranged", None, True),
            ("temporal_deranged", temporal, True),
            ("spatial_deranged", spatial, True)
        ]:
            value, moved = expected_trace(matrix, labels, derange)
            out[control + "_" + name] = value / norm
            if name == "radial_fraction":
                out[control + "_moved_rows"] = moved
        out["paired_minus_deranged_" + name] = (
            out["paired_" + name] - out["deranged_" + name])
    return out


def common_fraction(x):
    """Position-common vector energy AFTER per-row channel centering."""
    x = center_channels(np.asarray(x, dtype=np.float64))
    return float(len(x) * np.sum(x.mean(axis=0)**2) / np.sum(x*x))


def describe(state, delta, temporal, spatial, eps):
    raw = pairing_statistics(state, delta, temporal, spatial, eps)
    common = np.broadcast_to(state.mean(axis=0, keepdims=True), state.shape)
    raw["common_state_direction_radial_fraction"] = pairing_statistics(
        common, delta, temporal, spatial, eps)["paired_radial_fraction"]
    return {
        "state_position_common_fraction": common_fraction(state),
        "direction_position_common_fraction": common_fraction(delta),
        "raw": raw,
        "position_centered": pairing_statistics(
            state-state.mean(axis=0, keepdims=True),
            delta-delta.mean(axis=0, keepdims=True), temporal, spatial, eps)
    }


def aggregate(cases):
    out = {"group_cases": len(cases), "states": {}}
    for state in STATES:
        records = [c["states"][state] for c in cases]
        out["states"][state] = {
            key: float(np.mean([r[key] for r in records])) for key in
            ["state_position_common_fraction", "direction_position_common_fraction"]}
        for representation in ["raw", "position_centered"]:
            out["states"][state][representation] = {
                key: float(np.mean([r[representation][key] for r in records]))
                for key in records[0][representation]}
    return out


def analyze(raw_root):
    plan = json.loads(PLAN.read_text())
    mp = ROOT / plan["accepted_manifest"]
    if sha256(mp) != plan["accepted_manifest_sha256"]:
        raise ValueError("Accepted collection manifest changed")
    manifest = json.loads(mp.read_text())
    rp = ROOT / plan["reference_geometry"]
    reference = json.loads(rp.read_text())
    old = {(c["source_index"], c["step"], c["block"], c["group"]): c
           for c in reference["cases"]}
    out = {"id": plan["id"], "status": "COMPLETE_SAVED_PAIRING_DISCRIMINATION",
           "plan_sha256": sha256(PLAN), "analysis_sha256": sha256(Path(__file__)),
           "accepted_manifest_sha256": sha256(mp), "reference_geometry_sha256": sha256(rp),
           "source_revision": "40ea8efaca384c278c1ab9124f977d246bbc97bc",
           "source_clusters": 6, "uniform_rows_per_scope": 32,
           "new_model_generations": 0, "new_GPU_calls": 0,
           "new_native_FFN_calls": 0, "new_native_norm_calls": 0,
           "model_training_updates": 0, "epsilon": plan["epsilon"],
           "scope": plan["scope"], "sources": [], "cases": []}
    max_error = 0.0
    for source in manifest["sources"]:
        index = source["index"]
        run_path = raw_root / f"source{index:02d}_attempt1/RUN.json"
        if sha256(run_path) != source["run_sha256"]:
            raise ValueError("Run binding changed")
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Require accepted passive collection")
        out["sources"].append({"index": index, "run_sha256": source["run_sha256"]})
        for scope in run["scopes"]:
            if scope["block"] not in plan["blocks"]:
                continue
            path = run_path.parent / "observed" / scope["path"]
            wp = run_path.parent / "observed" / scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Operand/weight binding changed")
            a, w = load_arrays(path), load_arrays(wp)
            rows = scope["row_plan"]
            ids = [rows["rows"].index(row) for row in rows["uniform_rows"]]
            coordinates = np.asarray(rows["coordinates"])[ids]
            temporal = coordinates[:, 0] // 7
            spatial = 2*(coordinates[:, 1] >= 15) + (coordinates[:, 2] >= 26)
            if len(ids) != 32 or scope["weight_binding"]["groups"] != plan["groups"]:
                raise ValueError("Unexpected fixed rows/groups")
            q = a["q2_post"][ids].astype(np.float64)
            pre, post = [a[key][ids].astype(np.float64) for key in
                         ["residual_pre_ffn", "residual_post_ffn"]]
            for k, group in enumerate(plan["groups"]):
                delta = (q[:, 64*group:64*(group+1)] @
                         w["W2_panel"][:, 64*k:64*(k+1)].astype(np.float64).T
                         ) * a["gate"].astype(np.float64)
                record = {"source_index": index, "step": scope["step"],
                          "block": scope["block"], "group": group,
                          "payload_sha256": scope["sha256"],
                          "weight_panel_sha256": scope["weight_binding"]["sha256"],
                          "states": {}}
                for name, state in zip(STATES, [pre, post, post-delta]):
                    result = describe(state, delta, temporal, spatial, plan["epsilon"])
                    expected = old[(index, scope["step"], scope["block"], group)][name]["radial_removed_fraction"]
                    error = abs(result["raw"]["paired_radial_fraction"] - expected)
                    max_error = max(max_error, error)
                    if error > 1e-11:
                        raise ValueError("Historical radial identity did not reproduce")
                    record["states"][name] = result
                out["cases"].append(record)
            print(f"source{index:02d} step{scope['step']} block{scope['block']:02d} verified", flush=True)
    if len(out["cases"]) != 480:
        raise ValueError("Incomplete saved scope")
    out["historical_radial_checks"] = {"comparisons": 1440, "max_absolute_error": max_error}
    out["offline_decoded_panel_matmuls"] = len(out["cases"])
    out["by_block"] = {str(b): aggregate([c for c in out["cases"] if c["block"] == b])
                       for b in plan["blocks"]}
    out["block0_by_source"] = {str(i): aggregate([c for c in out["cases"] if
                              c["block"] == 0 and c["source_index"] == i])
                              for i in plan["source_indices"]}
    out["block0_by_step"] = {str(s): aggregate([c for c in out["cases"] if
                            c["block"] == 0 and c["step"] == s]) for s in plan["steps"]}
    out["block0_by_group"] = {str(g): aggregate([c for c in out["cases"] if
                             c["block"] == 0 and c["group"] == g]) for g in plan["groups"]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detail-out", type=Path, required=True,
                        help="Full scalar case records in the HF cache")
    args = parser.parse_args()
    if args.out.exists() or args.detail_out.exists():
        raise FileExistsError("Preserve the first result")
    hf = Path.home() / ".cache/huggingface"
    detail_relative = args.detail_out.resolve().relative_to(hf.resolve())
    result = analyze(args.raw_root)
    write_json(args.detail_out, result)
    public = {k: v for k, v in result.items() if k != "cases"}
    public["detail_artifact"] = {"HF_relative": str(detail_relative),
                                "sha256": sha256(args.detail_out),
                                "bytes": args.detail_out.stat().st_size,
                                "group_case_records": len(result["cases"])}
    write_json(args.out, public)


if __name__ == "__main__":
    main()
