"""Source-held-out calibration of saved FFN temporal amplitudes; no model calls.

All predictions are evaluated on 32 captured rows of dense reference prefixes.
Current outputs are labels, never prediction features. Oracle bounds are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def energy(x):
    return float(np.sum(x * x))


def target_energy(case):
    return energy(case["current"] * case["gate"])


def feature(case, name):
    """Only current/previous pre-FFN inputs and the previous FFN output."""
    previous, bias = case["previous"], case["bias"]
    if name == "fixed_bias_preserving_scale":
        return previous - bias, bias
    if name.startswith("input_"):
        x0, x1 = case["input_previous"], case["input_current"]
        old = np.sum(x0 * x0, axis=1, keepdims=True)
        if np.any(old <= 0):
            raise ValueError("Degenerate saved input")
        if "projection" in name:
            ratio = np.sum(x0 * x1, axis=1, keepdims=True) / old
        elif "panel" in name:
            ratio = np.sqrt(energy(x1) / energy(x0))
        else:
            ratio = np.sqrt(np.sum(x1 * x1, axis=1, keepdims=True) / old)
        if "bias_preserving" in name:
            return ratio * (previous - bias), bias
        return ratio * previous, np.zeros_like(bias)
    return previous, np.zeros_like(bias)


def fit_scale(cases, name):
    """Minimize the mean of per-source relative SSE, not pooled raw SSE."""
    numerator = denominator = 0.0
    for case in cases:
        base, offset = feature(case, name)
        a = base * case["gate"]
        b = (case["current"] - offset) * case["gate"]
        norm = target_energy(case)
        numerator += float(np.sum(a * b)) / norm
        denominator += energy(a) / norm
    return numerator / denominator if denominator > 0 else 0.0


def fit_affine(cases):
    """One scalar plus a fixed channel vector; analytic weighted least squares."""
    width = cases[0]["previous"].shape[1]
    weight = np.zeros((1, width))
    old_sum, new_sum = np.zeros_like(weight), np.zeros_like(weight)
    for case in cases:
        w = case["gate"] ** 2 / target_energy(case)
        weight += len(case["previous"]) * w
        old_sum += w * case["previous"].sum(axis=0, keepdims=True)
        new_sum += w * case["current"].sum(axis=0, keepdims=True)
    old_mean = np.divide(old_sum, weight, out=np.zeros_like(weight), where=weight > 0)
    new_mean = np.divide(new_sum, weight, out=np.zeros_like(weight), where=weight > 0)
    numerator = denominator = 0.0
    for case in cases:
        w = case["gate"] ** 2 / target_energy(case)
        a, b = case["previous"] - old_mean, case["current"] - new_mean
        numerator += float(np.sum(w * a * b))
        denominator += float(np.sum(w * a * a))
    scale = numerator / denominator if denominator > 0 else 0.0
    return scale, new_mean - scale * old_mean


def score(case, prediction):
    error = (prediction - case["current"]) * case["gate"]
    mean = error.mean(axis=0, keepdims=True)
    centered = error - mean
    total = energy(error)
    mean_energy = len(error) * energy(mean)
    center_energy = energy(centered)
    if abs(total - mean_energy - center_energy) > 1e-10 * max(1.0, total):
        raise ValueError("Mean/centered orthogonal error identity failed")
    norm = target_energy(case)
    return {"SSE": total, "target_energy": norm, "relative_SSE": total / norm,
            "mean_error_over_target_energy": mean_energy / norm,
            "centered_error_over_target_energy": center_energy / norm}


def cosine(a, b):
    denominator = np.sqrt(energy(a) * energy(b))
    return float(np.sum(a * b)) / denominator if denominator > 0 else None


def describe(case):
    a, b = [case[k] * case["gate"] for k in ["previous", "current"]]
    ac, bc = [v - v.mean(axis=0, keepdims=True) for v in [a, b]]
    bias = np.broadcast_to(case["bias"] * case["gate"], a.shape)
    return {"raw_cosine": cosine(a, b), "spatially_centered_cosine": cosine(ac, bc),
            "bias_subtracted_cosine": cosine(a - bias, b - bias),
            "current_spatial_mean_energy_fraction": 1.0 - energy(bc) / energy(b),
            "current_bias_energy_over_output_energy": energy(bias) / energy(b),
            "input_panel_RMS_ratio": float(np.sqrt(energy(case["input_current"]) /
                                                       energy(case["input_previous"])))}


def oracle_predictions(case):
    previous, current, gate = [case[k] for k in ["previous", "current", "gate"]]
    scalar = fit_scale([case], "fixed_scale")
    scale, shift = fit_affine([case])
    a, b = previous * gate, current * gate
    row_scale = np.sum(a * b, axis=1, keepdims=True) / np.sum(a * a, axis=1, keepdims=True)
    return {"oracle_scalar": scalar * previous,
            "oracle_current_mean_shift": previous - previous.mean(axis=0, keepdims=True) +
                                         current.mean(axis=0, keepdims=True),
            "oracle_affine": scale * previous + shift,
            "oracle_row_scalar": row_scale * previous}, scalar


def evaluate(cases, names):
    records = []
    for case in cases:
        train = [c for c in cases if c["source_index"] != case["source_index"] and
                 (c["block"], c["target_step"]) == (case["block"], case["target_step"])]
        if len(train) != 5 or len({c["source_index"] for c in train}) != 5:
            raise ValueError("Require five different training sources, held out as a whole")
        row = {k: case[k] for k in ["source_index", "block", "target_step", "expert",
                                   "previous_payload_sha256", "current_payload_sha256"]}
        row.update(uniform_rows=len(case["previous"]), description=describe(case),
                   calibration_source_indices=sorted(c["source_index"] for c in train),
                   predictors={}, target_informed_bounds={})
        for name in names:
            if name == "reuse":
                prediction, params = case["previous"], {"scale": 1.0}
            elif name == "fixed_affine":
                scale, shift = fit_affine(train)
                prediction = scale * case["previous"] + shift
                params = {"scale": scale, "channel_shift_RMS": float(np.sqrt(np.mean(shift ** 2))),
                          "channel_shift_F64_sha256": hashlib.sha256(
                              shift.astype("<f8").tobytes()).hexdigest(), "fixed_parameters": shift.size + 1}
            else:
                scale = fit_scale(train, name)
                base, offset = feature(case, name)
                prediction, params = scale * base + offset, {"scale": scale, "fixed_parameters": 1}
            row["predictors"][name] = {**score(case, prediction), "calibration": params}
        bounds, oracle_scale = oracle_predictions(case)
        row["oracle_scalar_coefficient"] = oracle_scale
        row["target_informed_bounds"] = {name: score(case, value) for name, value in bounds.items()}
        # The oracle uses the same scalar family as fixed_scale and must dominate it.
        if (row["target_informed_bounds"]["oracle_scalar"]["relative_SSE"] >
                row["predictors"]["fixed_scale"]["relative_SSE"] + 1e-12):
            raise ValueError("Scalar oracle does not bound the held-out fixed scalar")
        records.append(row)
    return records


def aggregate(records):
    result = {"source_clusters": len({r["source_index"] for r in records}),
              "pairs": len(records), "predictors": {}, "target_informed_bounds": {}}
    for field in ["predictors", "target_informed_bounds"]:
        for name in records[0][field]:
            values = [r[field][name] for r in records]
            relative = [v["relative_SSE"] for v in values]
            result[field][name] = {"mean_relative_SSE": float(np.mean(relative)),
                "relative_SSE_quantiles": np.quantile(relative, [0, .25, .5, .75, 1]).tolist(),
                "pooled_SSE_over_pooled_target_energy": sum(v["SSE"] for v in values) /
                                                        sum(v["target_energy"] for v in values),
                "mean_error_over_target_energy": float(np.mean([v["mean_error_over_target_energy"] for v in values])),
                "centered_error_over_target_energy": float(np.mean([v["centered_error_over_target_energy"] for v in values]))}
    for name, metrics in result["predictors"].items():
        diffs = []
        for index in sorted({r["source_index"] for r in records}):
            source = [r for r in records if r["source_index"] == index]
            diffs.append(float(np.mean([r["predictors"][name]["relative_SSE"] -
                                       r["predictors"]["fixed_scale"]["relative_SSE"] for r in source])))
        metrics["source_mean_loss_differences_to_fixed_scale"] = diffs
        metrics["sources_better_than_fixed_scale"] = sum(d < -1e-12 for d in diffs)
    result["description_means"] = {key: float(np.mean([r["description"][key] for r in records]))
                                    for key in records[0]["description"]}
    return result


def load_cases(raw_root, plan):
    manifest_path = ROOT / plan["accepted_manifest"]
    if sha256(manifest_path) != plan["accepted_manifest_sha256"]:
        raise ValueError("Accepted observation manifest changed")
    manifest = json.loads(manifest_path.read_text())
    cases = []
    for source in manifest["sources"]:
        folder = raw_root / f"source{source['index']:02d}_attempt1"
        if sha256(folder / "RUN.json") != source["run_sha256"]:
            raise ValueError("Accepted raw RUN changed")
        run = json.loads((folder / "RUN.json").read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Require accepted passive capture")
        scopes = {(s["step"], s["block"]): s for s in run["scopes"]}
        for block in plan["blocks"]:
            for previous_step, target_step in plan["same_expert_pairs"]:
                pair = [scopes[step, block] for step in [previous_step, target_step]]
                if pair[0]["expert"] != pair[1]["expert"] or pair[0]["row_plan"] != pair[1]["row_plan"]:
                    raise ValueError("Changed expert or latent coordinates")
                arrays = []
                for scope in pair:
                    path = folder / "observed" / scope["path"]
                    if sha256(path) != scope["sha256"]:
                        raise ValueError("Accepted operand payload changed")
                    raw = load_arrays(path)
                    ids = [scope["row_plan"]["rows"].index(r) for r in scope["row_plan"]["uniform_rows"]]
                    if len(ids) != plan["uniform_rows_per_scope"]:
                        raise ValueError("Changed fixed uniform sample size")
                    arrays.append({**{k: raw[k][ids].astype(np.float64) for k in
                                       ["ffn_output", "modulated_input"]},
                                   **{k: raw[k].reshape(1, -1).astype(np.float64) for k in ["gate", "b2"]}})
                old, new = arrays
                if not np.array_equal(old["b2"], new["b2"]):
                    raise ValueError("Same-expert bias changed")
                cases.append({"source_index": source["index"], "block": block, "target_step": target_step,
                    "expert": pair[1]["expert"], "previous": old["ffn_output"], "current": new["ffn_output"],
                    "input_previous": old["modulated_input"], "input_current": new["modulated_input"],
                    "gate": new["gate"], "bias": new["b2"],
                    "previous_payload_sha256": pair[0]["sha256"], "current_payload_sha256": pair[1]["sha256"]})
    if sorted({c["source_index"] for c in cases}) != plan["source_indices"]:
        raise ValueError("Wrong source cohort")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "config/e0_temporal_saved_calibration_20261025.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve prior analysis output")
    plan = json.loads(args.plan.read_text())
    cases = load_cases(args.raw_root, plan)
    records = evaluate(cases, plan["comparators"])
    old = json.loads((ROOT / "research/20261025/E0_JOINT_AND_TEMPORAL.json").read_text())
    for row in records:
        previous = next(r for r in old["temporal_pairs"] if (r["source_index"], r["block"], r["target_step"]) ==
                        (row["source_index"], row["block"], row["target_step"]))
        for field, name, prior_key in [("predictors", "reuse", "relative_SSE_reuse"),
                                       ("target_informed_bounds", "oracle_scalar", "relative_SSE_oracle_scalar"),
                                       ("target_informed_bounds", "oracle_current_mean_shift", "relative_SSE_oracle_current_mean")]:
            if abs(row[field][name]["relative_SSE"] - previous["metrics"][prior_key]) > 1e-12:
                raise ValueError("Previous temporal observation does not reproduce")
    result = {"status": "COMPLETE_SAVED_SOURCE_HELD_OUT_TEMPORAL_CALIBRATION",
              "scope": plan["scope_limits"], "plan_sha256": sha256(args.plan),
              "tool_sha256": sha256(Path(__file__)), "accepted_manifest_sha256": plan["accepted_manifest_sha256"],
              "source_clusters": 6, "new_generations": 0, "new_GPU_calls": 0, "new_native_FFN_calls": 0,
              "predictions_arithmetic": "FP64 over saved BF16 operands, no native writeback or executed cache",
              "old_temporal_baseline_and_bounds_reproduce": True, "records": records,
              "by_block": {str(b): aggregate([r for r in records if r["block"] == b]) for b in plan["blocks"]},
              "by_block_target_step": {f"{b}/{s}": aggregate([r for r in records if
                  (r["block"], r["target_step"]) == (b, s)]) for b in plan["blocks"] for s in [1, 3]}}
    write_json(args.out, result)
    for block, row in result["by_block"].items():
        print(block, {k: round(v["mean_relative_SSE"], 6) for k, v in row["predictors"].items()})


if __name__ == "__main__":
    main()
