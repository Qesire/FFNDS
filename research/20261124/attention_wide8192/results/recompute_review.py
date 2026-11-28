#!/usr/bin/env python3
"""Recompute the 2026-11-24 attention review from the published replay ZIP.

Requires NumPy and pandas. Reads the ZIP directly; no extraction, model access,
GPU, or network. Additional comparisons are exploratory and leave the original
FIT_RESULT and its frozen decision unchanged. Ratios use equal-site log means
within each source, followed by an equal-source log mean.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


DEV = ("vbi_0392", "vbi_0800", "vbi_1013")
KEYS = ["source_id", "source_cluster_id", "module", "expert", "background", "layer"]
METRICS = ("point", "spatial", "temporal")
COMMIT = "b1e15c2df4d1dad9a285798e552192ee4fca0bd8"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_rows(frame, arm, level="P64_estimate"):
    if arm in ("SP", "ANCH"):
        level = "zero_fit_reference"
    return frame[(frame.arm == arm) & (frame.fit_level == level)]


def ratio(frame, arm_a, arm_b, metric="point", *, module="self", background=None,
          expert=None, level_a="P64_estimate", level_b=None, noise_filter=True):
    """Source-macro geometric ratio; retains paired site identity."""
    d = frame[frame.source_id.isin(DEV) & (frame.module == module)]
    if background is not None:
        d = d[d.background == background]
    if expert is not None:
        d = d[d.expert == expert]
    a = selected_rows(d, arm_a, level_a)
    b = selected_rows(d, arm_b, level_b or level_a)
    paired = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if noise_filter and metric in METRICS:
        paired = paired[(paired[metric + "_a"] > paired[metric + "_native_noise_a"])
                        & (paired[metric + "_b"] > paired[metric + "_native_noise_b"])]
    paired = paired[(paired[metric + "_a"] > 0) & (paired[metric + "_b"] > 0)].copy()
    paired["log_ratio"] = np.log(paired[metric + "_a"] / paired[metric + "_b"])
    means = paired.groupby("source_id").log_ratio.mean()
    assert set(means.index) == set(DEV), (arm_a, arm_b, metric, means)
    return {
        "ratio": float(np.exp(means.mean())),
        "source_ratios": {source: float(np.exp(means[source])) for source in DEV},
        "paired_sites": len(paired),
        "site_improvements": int((paired.log_ratio < 0).sum()),
    }


def percent(r):
    return 100.0 * (r - 1.0)


def metric_percentages(frame, a, b, **kwargs):
    return {metric: percent(ratio(frame, a, b, metric, **kwargs)["ratio"])
            for metric in METRICS}


def build_review(replay_path: Path, fit_path: Path, evidence_path: Path):
    zip_bytes = replay_path.read_bytes()
    evidence_bytes = evidence_path.read_bytes()
    fit_bytes = fit_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    fit = json.loads(fit_bytes)
    assert sha256(zip_bytes) == evidence["replay_zip_sha256"], "Replay ZIP SHA mismatch"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        expected = evidence["sourced_artifacts"]
        assert len(expected) == 16, "Unexpected frozen artifact count"
        for name, digest in expected.items():
            assert sha256(archive.read(name)) == digest, f"SHA mismatch: {name}"
        frame = pd.read_csv(io.BytesIO(archive.read("site_metrics.csv")))
        witness = json.loads(archive.read("WITNESS_RESULT.json"))["records"]
        recipes = {name: json.loads(archive.read(f"calibration/{name}_recipe.json"))
                   for name in ("self_high", "self_low", "text_cross_high", "text_cross_low")}

    assert len(frame) == 4608
    assert not frame.duplicated(KEYS + ["arm", "fit_level"]).any()
    clusters = frame[frame.source_id.isin(DEV)][["source_id", "source_cluster_id"]].drop_duplicates()
    assert len(clusters) == 3 and clusters.source_cluster_id.nunique() == 3

    # Verify all published comparisons, preserving the frozen noise-floor rule.
    differences = []
    for area, comparisons in fit["contrasts"].items():
        module, background = area.split("/")
        background = None if background == "both" else background
        for comparison, metrics in comparisons.items():
            a, b = comparison.split("/")
            for metric, published in metrics.items():
                if published["ratio"] is None:
                    continue
                is_floor = metric == "head_floor"
                measured = ratio(frame, a, b, "head" if is_floor else metric,
                                 module=module, background=background,
                                 level_a="family_floor_head" if is_floor else "P64_estimate")
                differences.append(abs(measured["ratio"] - published["ratio"]))
                for source in DEV:
                    assert np.isclose(measured["source_ratios"][source],
                                      published["source_ratios"][source], rtol=1e-12, atol=1e-12)
    assert len(differences) == 168 and max(differences) < 1e-12

    out = {
        "status": "EXPLORATORY_REANALYSIS_OF_FROZEN_REAL_PANEL_RESULTS",
        "commit": COMMIT,
        "new_model_executions": 0,
        "independent_unit": "source_cluster",
        "DEV_independent_units": 3,
        "DEV_source_clusters": dict(clusters.values.tolist()),
        "aggregation": "Equal-site mean log ratio within each source, then equal-source mean; exponentiate. Negative percent changes favor numerator.",
        "scope": "Self-attention DEV unless explicitly named text_cross; nested sites and heads are not independent samples. New comparisons are exploratory; frozen FIT_RESULT is unchanged.",
        "integrity": {"replay_zip_sha256": sha256(zip_bytes), "replay_zip_matches": True,
                      "verified_artifact_fingerprints": 16, "site_metric_rows": len(frame),
                      "fit_result_sha256": sha256(fit_bytes), "evidence_sha256": sha256(evidence_bytes)},
        "published_contrast_verification": {"count": len(differences),
                                            "max_absolute_ratio_difference": max(differences)},
    }

    out["C0_over_FIN_D"] = {}
    for bg in ("Dense", "wide8192", None):
        r = ratio(frame, "FIN-C0", "FIN-D", background=bg)
        out["C0_over_FIN_D"][bg or "both"] = {
            "downstream_percent_changes": metric_percentages(frame, "FIN-C0", "FIN-D", background=bg),
            "head_floor_percent_change": percent(ratio(frame, "FIN-C0", "FIN-D", "head",
                                                        background=bg, level_a="family_floor_head")["ratio"]),
            "point_site_improvements": r["site_improvements"], "paired_sites": r["paired_sites"],
        }
    out["C1_over_C0"] = {
        bg: {metric + "_percent_change": percent(ratio(frame, "FIN-C1", "FIN-C0", metric,
                                                       background=bg)["ratio"])
             for metric in (*METRICS, "head", "rho_mse", "w_mse")}
        for bg in ("Dense", "wide8192")
    }
    out["C0_over_OUT_ABS_by_expert"] = {
        f"{bg}/{expert}": metric_percentages(frame, "FIN-C0", "OUT-ABS", background=bg, expert=expert)
        for bg in ("Dense", "wide8192") for expert in ("high", "low")
    }

    # High-expert predictions are identical between these two routes. With four
    # layers per expert, an equal-site log mean is half of the low-only contrast.
    routes = {"candidate": "high OUT-ABS; low FIN-C0", "reference": "high OUT-ABS; low OUT-D",
              "interpretation": "Additional matched-freedom reference, not the original CAL-selected reference. Actual pooled CAL preferred OUT-ABS slightly over OUT-D in low expert.",
              "backgrounds": {}}
    for bg in ("Dense", "wide8192"):
        rows = {}
        for metric in METRICS:
            r = ratio(frame, "FIN-C0", "OUT-D", metric, background=bg, expert="low")
            assert r["paired_sites"] == 12
            rows[metric] = {
                "percent_change": percent(float(np.sqrt(r["ratio"]))),
                "source_percent_changes": {s: percent(float(np.sqrt(v))) for s, v in r["source_ratios"].items()},
            }
        routes["backgrounds"][bg] = rows
    out["matched_expert_routes"] = routes

    d = frame[frame.source_id.isin(DEV) & (frame.module == "self") &
              (frame.background == "wide8192") & (frame.fit_level == "P64_estimate")]
    energies = {"scope": "Arithmetic mean across 24 sites, head space before W_O/gate; these are not the primary macro ratios.",
                "identity": "E_head = E_moment + E_directed_mass + signed_cross", "arms": {}}
    for arm in ("FIN-C0", "FIN-C1"):
        q = d[d.arm == arm]
        assert len(q) == 24
        moment = q.w_mse * (40 * 448 * 128)
        mass = q["head"] - moment - q.signed_mass_moment_cross
        assert (mass >= 0).all()
        energies["arms"][arm] = {
            "moment": float(moment.mean()), "directed_mass": float(mass.mean()),
            "signed_cross": float(q.signed_mass_moment_cross.mean()),
            "composite_head": float(q["head"].mean()),
        }
    out["wide8192_head_error_decomposition"] = energies

    out["CAL_lambda_boundary"] = {}
    for name, recipe in recipes.items():
        compact = {}
        for arm, candidates in recipe["lambda_scores"].items():
            lambdas = [x["lambda"] for x in candidates]
            scores = [x["mean_SP_normalized_point"] for x in candidates]
            compact[arm] = scores
            assert recipe["lambda_values"][arm] == max(lambdas)
            assert all(a > b for a, b in zip(scores, scores[1:]))
        out["CAL_lambda_boundary"][name] = {
            "grid": lambdas, "chosen_lambda_all_arms": 0.1,
            "all_tuned_scores_decrease_to_upper_boundary": True,
            "LOSO_mean_SP_normalized_point_by_arm": compact,
        }
    out["CAL_expert_selection_scores"] = {}
    for expert in ("high", "low"):
        recipe = recipes[f"self_{expert}"]
        scores = {arm: next(x["mean_SP_normalized_point"] for x in rows
                            if x["lambda"] == recipe["lambda_values"][arm])
                  for arm, rows in recipe["lambda_scores"].items()}
        out["CAL_expert_selection_scores"][expert] = {
            "OUT-ABS": scores["OUT-ABS"], "OUT-D": scores["OUT-D"], "FIN-C0": scores["FIN-C0"],
            "output_only_winner": "OUT-ABS", "reported_exploratory_route_choice": "OUT-ABS" if expert == "high" else "FIN-C0",
        }

    out["P64_head_error_ratios"] = {
        "meaning": "Same-site head-space energy ratios; H448 estimator and family floor are in-sample H fits, not held-out predictors.",
        "backgrounds": {},
    }
    for bg in ("Dense", "wide8192"):
        out["P64_head_error_ratios"]["backgrounds"][bg] = {
            arm: {"over_H448_estimator": ratio(frame, arm, arm, "head", background=bg,
                                               level_b="all_H_estimator_reference")["ratio"],
                  "over_family_floor": ratio(frame, arm, arm, "head", background=bg,
                                              level_b="family_floor_head")["ratio"]}
            for arm in ("OUT-D", "OUT-ABS", "FIN-D", "FIN-C0", "FIN-C1")
        }

    cross = [r for r in witness if r["kind"] == "text_cross"]
    self_witness = [r for r in witness if r["kind"] == "self"]
    assert len(cross) == 96 and len(self_witness) == 8
    out["text_cross_step_medians"] = {
        "scope": "96 CAL+DEV panels; two signed endpoint directions; site/direction descriptive medians, not independent samples or model runs.",
        "steps": {},
    }
    for step in ("0.25", "0.5", "1.0"):
        rows = {}
        for direction in ("both", "anchor_toward_hit", "hit_toward_anchor"):
            directions = ("anchor_toward_hit", "hit_toward_anchor") if direction == "both" else (direction,)
            vals = [r["directions"][d][step] for r in cross for d in directions]
            rows[direction] = {
                "observations": len(vals),
                "first_order_relative_L2": float(np.median([v["tangent_relative"] for v in vals])),
                "second_order_relative_L2": float(np.median([v["second_order_relative"] for v in vals])),
            }
        out["text_cross_step_medians"]["steps"][step] = rows
    out["self_full_key_witness"] = {
        "panels": 8, "heads_per_panel": 4,
        "max_identity_absolute_error": max(r["identity_max_abs"] for r in self_witness),
        "pooled_first_order_relative_L2": float(np.sqrt(sum(r["remainder_energy"] for r in self_witness) /
                                                        sum(r["finite_energy"] for r in self_witness))),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-zip", "fit-result", "evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = build_review(args.replay_zip, args.fit_result, args.evidence)
    rendered = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    assert len(rendered.encode("utf-8")) < 15360, "Keep this review summary below 15 KiB"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": len(rendered.encode("utf-8")),
                      "verified_fingerprints": 16, **out["published_contrast_verification"],
                      "new_model_executions": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
