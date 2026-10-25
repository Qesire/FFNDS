"""Check a shared-reference moment null using accepted full native trajectories.

No model runs. The null fit is algebra on six squared distances, not a test
assuming millions of independent coordinates or an estimator of semantic truth.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["dense", "A", "B", "AB"]
PAIRS = list(itertools.combinations(range(4), 2))
DESIGN = np.array([[float(k in pair) for k in range(4)] for pair in PAIRS])


def inner(a, b):
    return float(np.sum(a * b))


def star_fit(distances):
    """Four nonnegative arm variances; exact enumeration of all active sets."""
    d = np.asarray(distances, dtype=np.float64)
    best = np.zeros(4)
    best_error = inner(d, d)
    for bits in range(1, 16):
        active = [k for k in range(4) if bits & (1 << k)]
        values = np.linalg.lstsq(DESIGN[:, active], d, rcond=None)[0]
        if np.any(values < 0):
            continue
        fit = np.zeros(4)
        fit[active] = values
        residual = DESIGN @ fit - d
        error = inner(residual, residual)
        if error < best_error:
            best, best_error = fit, error
    unconstrained = np.linalg.lstsq(DESIGN, d, rcond=None)[0]
    norm = inner(d, d)
    return {"nonnegative_arm_energies": dict(zip(ARMS, best.tolist())),
            "unconstrained_arm_energies": dict(zip(ARMS, unconstrained.tolist())),
            "relative_L2_discrepancy_of_six_squared_distances": float(np.sqrt(best_error / norm)) if norm else 0.0}


def subspaces(value):
    # Wan scheduler saves [C,T,H,W]; the transformer token grid is spatially
    # patchified and is a different object. Keep the native latent axes here.
    if value.ndim == 4:
        value = value[None, ...]
    if value.ndim != 5 or value.shape[0] != 1:
        raise ValueError("Require native [C,T,H,W] or single-batch equivalent")
    global_mean = value.mean(axis=(2, 3, 4), keepdims=True)
    per_time = value.mean(axis=(3, 4), keepdims=True)
    return {"full": value,
            "global_channel_mean": np.broadcast_to(global_mean, value.shape),
            "time_varying_spatial_mean": np.broadcast_to(per_time - global_mean, value.shape),
            "spatially_centered_per_time": value - per_time}


def moments(values):
    responses = [v - values[0] for v in values[1:]]
    gram = np.array([[inner(a, b) for b in responses] for a in responses])
    distances = [inner(values[a] - values[b], values[a] - values[b]) for a, b in PAIRS]
    c = responses[2] - responses[0] - responses[1]
    additive = responses[0] + responses[1]
    ea, eb, ez = np.diag(gram)
    c_energy = inner(c, c)
    identity = (distances[0] + distances[1] + distances[4] + distances[5] -
                distances[2] - distances[3])
    if abs(c_energy - identity) > 1e-10 * max(1.0, c_energy, sum(distances)):
        raise ValueError("Reference-independent distance identity failed")
    return {"squared_pair_distances": {f"{ARMS[a]}/{ARMS[b]}": d for (a, b), d in zip(PAIRS, distances)},
            "response_Gram_A_B_AB": gram.tolist(), "factorial_interaction_energy": c_energy,
            "additive_response_energy": inner(additive, additive),
            "AB_over_A_plus_B": float(ez / (ea + eb)) if ea + eb else None,
            "curvature_over_additive_energy": c_energy / inner(additive, additive) if inner(additive, additive) else None,
            "all_three_reference_cross_moments": [float(gram[0, 1]), float(gram[0, 2]), float(gram[1, 2])],
            "star_null_fit": star_fit(distances)}


def analyze(raw_root, plan):
    result_path = ROOT / plan["original_result"]
    accepted = json.loads(result_path.read_text())
    rows = []
    for source in accepted["sources"]:
        folder = raw_root / f"source{source['index']:02d}_attempt1"
        if sha256(folder / "RUN.json") != source["run_sha256"]:
            raise ValueError("Original accepted native run changed")
        run = json.loads((folder / "RUN.json").read_text())
        for step in plan["steps"]:
            trajectories = []
            for arm in ARMS:
                entry = run["phases"][arm]["trajectory"][step]
                path = folder / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("Original trajectory payload changed")
                trajectories.append(load_arrays(path))
            for field in plan["fields"]:
                values = [v[field].astype(np.float64) for v in trajectories]
                if any(v.shape != (16, 21, 58, 104) for v in values):
                    raise ValueError("Frozen native latent geometry changed")
                parts = [subspaces(v) for v in values]
                out = {name: moments([v[name] for v in parts]) for name in plan["subspaces"]}
                for key in ["factorial_interaction_energy", "additive_response_energy"]:
                    full = out["full"][key]
                    partitioned = sum(out[name][key] for name in plan["subspaces"] if name != "full")
                    if abs(full - partitioned) > 1e-10 * max(1.0, full):
                        raise ValueError("Orthogonal trajectory-subspace energy identity failed")
                original = source["steps"][step]["interaction"][field]
                for current, old in [(out["full"]["factorial_interaction_energy"], original["curvature_norm2"]),
                                     (out["full"]["AB_over_A_plus_B"], original["AB_over_A_plus_B_SSE"])]:
                    if abs(current - old) > 1e-10 * max(1.0, abs(old)):
                        raise ValueError("Original native scalar result does not reproduce")
                rows.append({"source_index": source["index"], "step": step, "field": field,
                             "tensor_shape": list(values[0].shape), "subspaces": out})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "config/e0_shared_reference_saved_20261025.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the first moment analysis")
    plan = json.loads(args.plan.read_text())
    rows = analyze(args.raw_root, plan)
    out = {"status": "COMPLETE_SAVED_FOUR_ARM_SHARED_REFERENCE_MOMENT_CHECK",
           "scope": plan["claim_scope"], "tool_sha256": sha256(Path(__file__)),
           "plan_sha256": sha256(args.plan), "accepted_result_sha256": sha256(ROOT / plan["original_result"]),
           "competing_explanation_record_sha256": sha256(ROOT / plan["competing_explanation_record"]),
           "source_clusters": 6, "new_generations": 0, "new_GPU_calls": 0,
           "original_scalar_and_subspace_identities_pass": True, "records": rows}
    write_json(args.out, out)
    for row in rows:
        if row["field"] == "flow" and row["step"] in [0, 3]:
            print(row["source_index"], row["step"], {name: round(v["star_null_fit"][
                "relative_L2_discrepancy_of_six_squared_distances"], 5) for name, v in row["subspaces"].items()})


if __name__ == "__main__":
    main()
