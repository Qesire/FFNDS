"""Describe rankings of five saved decoded group directions in ideal LN geometry.

No native action is executed. All scores use post-FFN residuals and complete
post-Q2 group operands, so none is a feasible pre-FFN selector or video label.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def scores(state, directions, eps=1e-6):
    """Return [row,group] scores; normalized chord distances avoid 1-cos loss."""
    x, delta = np.asarray(state, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    if x.ndim != 2 or delta.ndim != 3 or delta.shape[::2] != x.shape or eps < 0:
        raise ValueError("Expected state [rows,width], directions [rows,groups,width]")
    if not np.isfinite(x).all() or not np.isfinite(delta).all():
        raise ValueError("Require finite operands")
    x = x[:, None, :]
    z = x - x.mean(axis=-1, keepdims=True)
    y = x - delta
    yc = y - y.mean(axis=-1, keepdims=True)
    n2 = np.sum(z*z, axis=-1, keepdims=True)
    yn2 = np.sum(yc*yc, axis=-1, keepdims=True)
    xn2, yraw2 = np.sum(x*x, axis=-1, keepdims=True), np.sum(y*y, axis=-1, keepdims=True)
    if np.any(n2 <= 0) or np.any(yn2 <= 0) or np.any(xn2 <= 0) or np.any(yraw2 <= 0):
        raise ValueError("Degenerate angular comparison")
    width = x.shape[-1]
    var = n2 / width + eps
    pd = delta - delta.mean(axis=-1, keepdims=True)
    tangent = pd - z * np.sum(z*pd, axis=-1, keepdims=True) / (n2 + width*eps)
    energy = np.sum(delta*delta, axis=-1)
    return {
        "gated_energy": energy,
        "rstd_scaled_energy": energy / var[..., 0],
        "mean_projected_energy": np.sum(pd*pd, axis=-1) / var[..., 0],
        "LN_tangent_energy": np.sum(tangent*tangent, axis=-1) / var[..., 0],
        "finite_ideal_LN": np.sum((yc / np.sqrt(yn2/width+eps) - z / np.sqrt(var))**2, axis=-1),
        "centered_angle_times_2d": width * np.sum((yc/np.sqrt(yn2) - z/np.sqrt(n2))**2, axis=-1),
        "uncentered_angle_times_2d": width * np.sum((y/np.sqrt(yraw2) - x/np.sqrt(xn2))**2, axis=-1),
    }


def compare(score, target):
    """Posthoc choose one of the five groups per row, with tolerance-only ties."""
    score, target = np.asarray(score), np.asarray(target)
    chosen, best = score.argmin(axis=1), target.argmin(axis=1)
    losses = target[np.arange(len(target)), chosen]
    optimum = target.min(axis=1)
    flips = valid = matched = matched_flips = 0
    for a, b in itertools.combinations(range(score.shape[1]), 2):
        ds, dt = score[:, a]-score[:, b], target[:, a]-target[:, b]
        non_tie = (np.abs(ds) > 1e-12*np.maximum(score[:, a], score[:, b])) & (
            np.abs(dt) > 1e-12*np.maximum(target[:, a], target[:, b]))
        reverse = (ds*dt < 0) & non_tie
        # A descriptive stratum, not a tuned selection rule or a deployed action.
        close = np.abs(ds) <= .1*np.maximum(score[:, a], score[:, b])
        valid += int(non_tie.sum())
        flips += int(reverse.sum())
        matched += int((non_tie & close).sum())
        matched_flips += int((reverse & close).sum())
    return {"rows": len(score), "chosen_group_changed_rows": int(np.sum(chosen != best)),
            "valid_pairs": valid, "pair_order_reversals": flips,
            "score_within_ten_percent_pairs": matched, "reversals_among_close_pairs": matched_flips,
            "chosen_target_loss": float(losses.sum()), "best_target_loss": float(optimum.sum()),
            "relative_excess_target_loss": float(losses.sum()/optimum.sum()-1)}


def aggregate(scopes):
    out = {"scopes": len(scopes), "rows": sum(s["uniform_rows"] for s in scopes), "scores": {}}
    for key in scopes[0]["row_comparisons_to_finite"]:
        records = [s["row_comparisons_to_finite"][key] for s in scopes]
        totals = {k: sum(r[k] for r in records) for k in records[0] if k != "relative_excess_target_loss"}
        totals["relative_excess_target_loss"] = totals["chosen_target_loss"]/totals["best_target_loss"]-1
        totals["chosen_group_changed_fraction"] = totals["chosen_group_changed_rows"]/totals["rows"]
        totals["pair_order_reversal_fraction"] = totals["pair_order_reversals"]/totals["valid_pairs"]
        totals["scopes_aggregate_argmin_changed"] = int(sum(
            np.argmin(s["aggregate_scores"][key]) != np.argmin(s["aggregate_scores"]["finite_ideal_LN"])
            for s in scopes))
        out["scores"][key] = totals
    return out


def analyze(raw_root):
    accepted = ROOT/"research/20261025/E0_NATIVE_OBSERVATIONS.json"
    manifest = json.loads(accepted.read_text())
    result = {"status": "COMPLETE_SAVED_OPERAND_RANKING_DESCRIPTION", "epsilon": 1e-6,
              "accepted_observations_sha256": sha256(accepted),
              "analysis_sha256": sha256(Path(__file__)), "new_native_FFN_calls": 0,
              "new_model_generations": 0, "GPU_calls": 0,
              "scope": "Six exposed source clusters, four steps, blocks0/10/20/30, 32 uniform rows, five fixed64 groups. Ideal FP64 finite LN labels from decoded panel directions, not BF16 action effects, downstream or video labels. All quantities are posthoc.",
              "scopes": []}
    for source in manifest["sources"]:
        index = source["index"]
        run_path = raw_root/f"source{index:02d}_attempt1/RUN.json"
        if sha256(run_path) != source["run_sha256"]:
            raise ValueError("Accepted run binding changed")
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Unaccepted run")
        for scope in run["scopes"]:
            if scope["block"] == 39:
                continue
            path = run_path.parent/"observed"/scope["path"]
            wp = run_path.parent/"observed"/scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Saved operand binding changed")
            a, w = load_arrays(path), load_arrays(wp)
            plan = scope["row_plan"]
            ids = [plan["rows"].index(row) for row in plan["uniform_rows"]]
            groups = scope["weight_binding"]["groups"]
            q, weights = a["q2_post"][ids].astype(np.float64), w["W2_panel"].astype(np.float64)
            delta = np.stack([(q[:, g*64:(g+1)*64] @ weights[:, k*64:(k+1)*64].T)*a["gate"]
                              for k, g in enumerate(groups)], axis=1)
            s = scores(a["residual_post_ffn"][ids], delta)
            record = {"source_index": index, "step": scope["step"], "block": scope["block"],
                      "groups": groups, "uniform_rows": len(ids), "payload_sha256": scope["sha256"],
                      "weight_panel_sha256": scope["weight_binding"]["sha256"],
                      "aggregate_scores": {k: v.sum(axis=0).tolist() for k, v in s.items()},
                      "row_comparisons_to_finite": {k: compare(v, s["finite_ideal_LN"]) for k, v in s.items()},
                      "radial_effect_beyond_rstd": compare(s["rstd_scaled_energy"], s["LN_tangent_energy"]),
                      "max_relative_centered_angle_vs_finite_error": float(np.max(
                          np.abs(s["centered_angle_times_2d"]-s["finite_ideal_LN"])/s["finite_ideal_LN"]))}
            result["scopes"].append(record)
    result["by_block"] = {str(b): aggregate([s for s in result["scopes"] if s["block"] == b])
                          for b in [0, 10, 20, 30]}
    result["by_source"] = {str(i): aggregate([s for s in result["scopes"] if s["source_index"] == i])
                           for i in manifest["source_indices"]}
    result["pooled"] = aggregate(result["scopes"])
    fixed = []
    for s in result["scopes"]:
        other = [t for t in result["scopes"] if (t["block"], t["step"]) == (s["block"], s["step"])
                 and t["source_index"] != s["source_index"]]
        chosen = int(np.argmin(np.sum([t["aggregate_scores"]["finite_ideal_LN"] for t in other], axis=0)))
        target = s["aggregate_scores"]["finite_ideal_LN"]
        fixed.append({"source_index": s["source_index"], "step": s["step"], "block": s["block"],
                      "group_from_other_five_sources": s["groups"][chosen],
                      "finite_loss": target[chosen], "per_scope_oracle_loss": min(target),
                      "per_row_oracle_loss": s["row_comparisons_to_finite"]["finite_ideal_LN"]["best_target_loss"]})
    result["leave_one_source_out_fixed_group"] = {
        "scope": "Five fixed groups only, separately by step/block. Sum ideal finite labels from other five discovery-exposed sources; choose one fixed group. Descriptive reference, no fresh video holdout, learned policy or native execution.",
        "scopes": fixed,
        "by_block": {str(b): {k: sum(t[k] for t in fixed if t["block"] == b)
                              for k in ["finite_loss", "per_scope_oracle_loss", "per_row_oracle_loss"]}
                     for b in [0, 10, 20, 30]}}
    result["offline_decoded_panel_matmuls"] = len(result["scopes"])*5
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the first analysis result")
    write_json(args.out, analyze(args.raw_root))


if __name__ == "__main__":
    main()
