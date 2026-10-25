"""Saved native operand descriptions: joint directions and same-coordinate reuse.

All finite LN calculations are ideal FP64 mathematics, not native interventions.
Temporal oracles use target-step values and cannot serve as online policies.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_e0_geometry_rankings import compare, scores
from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def quantiles(x):
    return np.quantile(x, [0, .25, .5, .75, 1]).tolist()


def joint_description(state, delta):
    singles = scores(state, delta)
    result = {"subsets": {}}
    for size in [2, 3]:
        subsets = list(itertools.combinations(range(delta.shape[1]), size))
        directions = np.stack([delta[:, list(s)].sum(axis=1) for s in subsets], axis=1)
        measured = scores(state, directions)
        unary_energy = np.stack([singles["gated_energy"][:, list(s)].sum(axis=1) for s in subsets], axis=1)
        unary_LN = np.stack([singles["finite_ideal_LN"][:, list(s)].sum(axis=1) for s in subsets], axis=1)
        target = measured["finite_ideal_LN"]
        gram = np.einsum("rgd,rhd->rgh", delta, delta)
        reconstructed = np.stack([gram[:, s, :][:, :, s].sum(axis=(1, 2)) for s in subsets], axis=1)
        # Strong cancellation makes relative-to-joint energy ill conditioned.
        # Check floating-point reconstruction against the input energy scale.
        identity_error = float(np.max(np.abs(reconstructed-measured["gated_energy"])/unary_energy))
        if identity_error > 1e-10:
            raise ValueError("Joint signed-Gram identity failed")
        result["subsets"][str(size)] = {
            "subsets_by_panel_index": [list(s) for s in subsets],
            "row_count": len(state), "gram_identity_max_relative_error": identity_error,
            "joint_raw_over_individual_energy_quantiles": quantiles(measured["gated_energy"]/unary_energy),
            "joint_finite_over_individual_LN_quantiles": quantiles(target/unary_LN),
            "choices": {k: compare(v, target) for k, v in {
                "individual_gated_energy_sum": unary_energy,
                "individual_finite_LN_sum": unary_LN,
                "joint_gated_energy": measured["gated_energy"],
                "joint_LN_tangent": measured["LN_tangent_energy"],
                "joint_uncentered_angle": measured["uncentered_angle_times_2d"],
            }.items()}}
    pairs = list(itertools.combinations(range(delta.shape[1]), 2))
    x = state[:, None, :]
    z = x-x.mean(axis=-1, keepdims=True)
    n2 = np.sum(z*z, axis=-1, keepdims=True)
    pd = delta-delta.mean(axis=-1, keepdims=True)
    tangent = pd-z*np.sum(z*pd, axis=-1, keepdims=True)/(n2+state.shape[-1]*1e-6)
    before = np.stack([np.sum(delta[:, a]*delta[:, b], axis=-1) for a, b in pairs], axis=1)
    after = np.stack([np.sum(tangent[:, a]*tangent[:, b], axis=-1) for a, b in pairs], axis=1)
    result["cross_term_sign_changes_after_LN_projection"] = {
        "pairs": before.size, "positive_to_negative": int(np.sum((before>0)&(after<0))),
        "negative_to_positive": int(np.sum((before<0)&(after>0))),
        "raw_positive": int(np.sum(before>0)), "projected_positive": int(np.sum(after>0))}
    return result


def temporal_description(previous, current, gate):
    a, b = previous*gate, current*gate
    target_energy = float(np.sum(b*b))
    error = float(np.sum((a-b)**2))
    dot, previous_energy = float(np.sum(a*b)), float(np.sum(a*a))
    scale = dot/previous_energy
    mean_corrected = a-a.mean(axis=0, keepdims=True)+b.mean(axis=0, keepdims=True)
    return {"relative_SSE_reuse": error/target_energy,
            "current_gate_weighted_cosine": dot/np.sqrt(previous_energy*target_energy),
            "previous_over_current_gated_energy": previous_energy/target_energy,
            "oracle_one_scalar": scale,
            "relative_SSE_oracle_scalar": float(np.sum((scale*a-b)**2))/target_energy,
            "relative_SSE_oracle_current_mean": float(np.sum((mean_corrected-b)**2))/target_energy,
            "relative_SSE_previous_mean_only": float(np.sum((a.mean(axis=0, keepdims=True)-b)**2))/target_energy,
            "target_gated_energy": target_energy,
            "reuse_gated_error_energy": error}


def aggregate_joint(records):
    result = {"scopes": len(records), "subsets": {}}
    for size in ["2", "3"]:
        parts = [r["joint"]["subsets"][size] for r in records]
        out = {"choices": {}, "gram_identity_max_relative_error": max(p["gram_identity_max_relative_error"] for p in parts)}
        for key in parts[0]["choices"]:
            ss = [p["choices"][key] for p in parts]
            chosen, best = sum(v["chosen_target_loss"] for v in ss), sum(v["best_target_loss"] for v in ss)
            out["choices"][key] = {
                "rows": sum(v["rows"] for v in ss),
                "changed_rows": sum(v["chosen_group_changed_rows"] for v in ss),
                "relative_excess_ideal_LN_loss": chosen/best-1,
                "chosen_ideal_LN_loss": chosen, "oracle_ideal_LN_loss": best}
        for key in ["joint_raw_over_individual_energy_quantiles", "joint_finite_over_individual_LN_quantiles"]:
            out[key+"_scope_median_range"] = [min(p[key][2] for p in parts), max(p[key][2] for p in parts)]
        result["subsets"][size] = out
    result["cross_terms"] = {k: sum(r["joint"]["cross_term_sign_changes_after_LN_projection"][k] for r in records)
                             for k in records[0]["joint"]["cross_term_sign_changes_after_LN_projection"]}
    return result


def analyze(root):
    manifest_path = ROOT/"research/20261025/E0_NATIVE_OBSERVATIONS.json"
    manifest = json.loads(manifest_path.read_text())
    result = {"status": "COMPLETE_SAVED_OPERAND_JOINT_AND_TEMPORAL_DESCRIPTION",
              "accepted_manifest_sha256": sha256(manifest_path),
              "tool_sha256": {p.name: sha256(p) for p in [Path(__file__), ROOT/"tools/analyze_e0_geometry_rankings.py"]},
              "source_clusters": 6, "new_GPU_calls": 0, "new_native_FFN_calls": 0,
              "new_generations": 0, "scope": "Fixed five decoded groups, saved 32 uniform coordinates, exposed CAL24 sources. Ideal LN subset directions and target-informed temporal representation controls only; no native omission, executed reuse, downstream or video gain.",
              "joint_scopes": [], "temporal_pairs": []}
    for source in manifest["sources"]:
        run_path = root/f"source{source['index']:02d}_attempt1/RUN.json"
        if sha256(run_path) != source["run_sha256"]:
            raise ValueError("Accepted run binding changed")
        run = json.loads(run_path.read_text())
        by_position = {}
        for scope in run["scopes"]:
            path = run_path.parent/"observed"/scope["path"]
            wp = run_path.parent/"observed"/scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Saved operand binding changed")
            a, w = load_arrays(path), load_arrays(wp)
            plan = scope["row_plan"]
            ids = [plan["rows"].index(row) for row in plan["uniform_rows"]]
            y, gate = a["ffn_output"][ids].astype(np.float64), a["gate"].astype(np.float64)
            by_position[scope["step"], scope["block"]] = (scope, y, gate)
            if scope["block"] == 39:
                continue
            q, weights = a["q2_post"][ids].astype(np.float64), w["W2_panel"].astype(np.float64)
            groups = scope["weight_binding"]["groups"]
            delta = np.stack([(q[:, g*64:(g+1)*64] @ weights[:, k*64:(k+1)*64].T)*gate
                              for k, g in enumerate(groups)], axis=1)
            result["joint_scopes"].append({"source_index": source["index"], "step": scope["step"],
                "block": scope["block"], "groups": groups, "payload_sha256": scope["sha256"],
                "weight_panel_sha256": scope["weight_binding"]["sha256"],
                "joint": joint_description(a["residual_post_ffn"][ids].astype(np.float64), delta)})
        for step in [1, 2, 3]:
            for block in [0, 10, 20, 30, 39]:
                prev, y0, _ = by_position[step-1, block]
                current, y1, gate = by_position[step, block]
                if prev["row_plan"] != current["row_plan"]:
                    raise ValueError("Temporal coordinates changed")
                result["temporal_pairs"].append({"source_index": source["index"], "previous_step": step-1,
                    "target_step": step, "block": block, "same_expert": prev["expert"]==current["expert"],
                    "previous_payload_sha256": prev["sha256"], "current_payload_sha256": current["sha256"],
                    "metrics": temporal_description(y0, y1, gate)})
    result["joint_by_block"] = {str(b): aggregate_joint([r for r in result["joint_scopes"] if r["block"]==b])
                                for b in [0, 10, 20, 30]}
    result["temporal_by_block"] = {}
    for same in [True, False]:
        subset = [r for r in result["temporal_pairs"] if r["same_expert"]==same]
        result["temporal_by_block"]["same_expert" if same else "expert_crossing"] = {
            str(b): {"pairs": len([r for r in subset if r["block"]==b]),
                     "metrics": {k: {"mean": float(np.mean([r["metrics"][k] for r in subset if r["block"]==b])),
                                     "quantiles": quantiles([r["metrics"][k] for r in subset if r["block"]==b])}
                                 for k in subset[0]["metrics"]}}
            for b in [0, 10, 20, 30, 39]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the first result")
    write_json(args.out, analyze(args.raw_root))


if __name__ == "__main__":
    main()
