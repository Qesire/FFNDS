"""Interpret accepted passive captures at paired native QDQ boundaries on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import analyze_run, load_arrays, paired, quantiles, ratio
from analyze_legacy_ffn_structure import profile
from ffnds_observation_common import sha256, write_json


def panel_decomposition(q2, weights, gate, groups):
    """Separate constant means and row variation in decoded signed geometry."""
    parts = np.stack([(q2[:, 64*g:64*(g+1)] @ weights[:, 64*k:64*(k+1)].T) * gate
                      for k, g in enumerate(groups)])
    means = parts.mean(axis=1, keepdims=True)
    centered = parts - means
    total, ct = parts.sum(axis=0), centered.sum(axis=0)
    numerator, denominator = np.sum(total ** 2), np.sum(parts ** 2)
    mean_energy = len(q2) * np.sum(means.sum(axis=0) ** 2)
    variation_energy = np.sum(ct ** 2)
    return {"groups": groups,
            "raw_sum_energy_over_sum_individual_energies": ratio(numerator, denominator),
            "centered_sum_energy_over_sum_individual_energies": ratio(variation_energy, np.sum(centered ** 2)),
            "mean_sum_energy_over_sum_individual_mean_energies": ratio(np.sum(means.sum(axis=0) ** 2), np.sum(means ** 2)),
            "constant_mean_fraction_of_summed_panel_output_energy": ratio(mean_energy, numerator),
            "mean_variation_identity_relative_error": ratio(abs(numerator - mean_energy - variation_energy), numerator),
            "scope": "Five fixed panels and sampled rows; decoded FP64, gate included, b2 excluded; no native action or whole hidden-space claim"}


def interpret(run_path):
    # First enforce acceptance, every payload/weight hash and coordinate pairing.
    atlas = analyze_run(run_path)
    record = json.loads(run_path.read_text())
    out = {"status": "COMPLETE_ACCEPTED_NATIVE_PASSIVE_MECHANISM_DESCRIPTION",
           "run_sha256": sha256(run_path), "source_id": record["source_id"],
           "independent_source_clusters": 1, "atlas": atlas, "scopes": [],
           "model_calls": 0, "GPU_calls": 0,
           "claim_scope": "Paired descriptive native D-Q values; no causal skip, video-quality or speed claim",
           "analysis_source_sha256": {p.name: sha256(p) for p in [Path(__file__),
               Path(__file__).with_name("analyze_ffnds_observations.py"),
               Path(__file__).with_name("analyze_legacy_ffn_structure.py")]}}
    for scope in record["scopes"]:
        a = load_arrays(run_path.parent / "observed" / scope["path"])
        lookup = {r: i for i, r in enumerate(scope["row_plan"]["rows"])}
        ids = [lookup[r] for r in scope["row_plan"]["uniform_rows"]]
        x = {k: v[ids].astype(np.float64) for k, v in a.items() if v.ndim == 2 and v.shape[0] == len(lookup)}
        z, h, pre, post = [x[k] for k in ["z", "gelu", "q2_pre", "q2_post"]]
        induced = (pre != 0) & (post == 0)
        gate = a["gate"].astype(np.float64)
        ffn = x["ffn_output"] * gate
        residual = x["residual_pre_ffn"]
        item = {"step": scope["step"], "block": scope["block"], "expert": scope["expert"],
                "rows": len(ids), "z_negative_fraction": float(np.mean(z < 0)),
                "GELU_zero_fraction": float(np.mean(h == 0)),
                "pre_Q2_zero_fraction": float(np.mean(pre == 0)),
                "post_Q2_zero_fraction": float(np.mean(post == 0)),
                "Q2_new_zero_fraction": float(np.mean(induced)),
                "Q2_new_zero_pre_energy_fraction": ratio(np.sum(pre[induced] ** 2), np.sum(pre ** 2)),
                "Q2_new_zero_negative_z_fraction": ratio(np.count_nonzero(induced & (z < 0)), np.count_nonzero(induced)),
                "profiles": {k: profile(x[k]) for k in ["q1_pre", "q1_post", "q2_pre", "q2_post"]},
                "gated_FFN_energy_over_pre_residual_energy": ratio(np.sum(ffn ** 2), np.sum(residual ** 2)),
                "gated_FFN_residual_cosine_quantiles": quantiles(np.sum(ffn * residual, axis=1) /
                    np.maximum(np.linalg.norm(ffn, axis=1) * np.linalg.norm(residual, axis=1), np.finfo(float).tiny)),
                "BF16_residual_writeback_pair": paired(residual + ffn, x["residual_post_ffn"])}
        weights = load_arrays(run_path.parent / "observed" / scope["weight_binding"]["path"])
        item["signed_panel_mean_variation"] = panel_decomposition(
            post, weights["W2_panel"].astype(np.float64), gate,
            scope["weight_binding"]["groups"])
        out["scopes"].append(item)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve first native interpretation")
    write_json(args.out, interpret(args.run))


if __name__ == "__main__":
    main()
