"""Describe same-coordinate temporal similarity with mean/bias controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json


def cosine(a, b):
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def analyze(root, indices):
    result = {"status": "COMPLETE_ACCEPTED_SAME_COORDINATE_TEMPORAL_DESCRIPTION",
              "source_clusters": len(indices), "sources": [], "pairs": [],
              "scope": "Native pre-gate FFN outputs at fixed latent coordinates. No object tracking, causal reuse action, quality or speed claim.",
              "source_sha256": {p.name: sha256(p) for p in [Path(__file__),
                  Path(__file__).with_name("analyze_ffnds_observations.py"),
                  Path(__file__).with_name("ffnds_observation_common.py")]}}
    for index in indices:
        run_path = root / f"source{index:02d}_attempt1/RUN.json"
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP" or run["source_index"] != index:
            raise ValueError("Require accepted correct source")
        result["sources"].append({"index": index, "source_id": run["source_id"], "run_sha256": sha256(run_path)})
        previous = {}
        for scope in sorted(run["scopes"], key=lambda s: (s["step"], s["block"])):
            path = run_path.parent / "observed" / scope["path"]
            if sha256(path) != scope["sha256"]:
                raise ValueError("Native observation changed")
            arrays = load_arrays(path)
            plan = scope["row_plan"]
            lookup = {r: i for i, r in enumerate(plan["rows"])}
            ids = [lookup[r] for r in plan["uniform_rows"]]
            y = arrays["ffn_output"][ids].astype(np.float64)
            bias = arrays["b2"].astype(np.float64)
            centered = y - y.mean(axis=0, keepdims=True)
            current = {"scope": scope, "y": y, "bias": bias, "centered": centered}
            if scope["block"] in previous:
                p = previous[scope["block"]]
                old = p["scope"]
                if old["row_plan"] != plan:
                    raise ValueError("Temporal rows are not identical coordinates")
                cross = np.sum(p["y"] * y)
                mean_cross = len(y) * np.sum(p["y"].mean(axis=0) * y.mean(axis=0))
                centered_cross = np.sum(p["centered"] * centered)
                result["pairs"].append({"source_index": index, "block": scope["block"],
                    "steps": [old["step"], scope["step"]], "experts": [old["expert"], scope["expert"]],
                    "same_expert": old["expert"] == scope["expert"], "uniform_rows": len(y),
                    "raw_Frobenius_cosine": cosine(p["y"], y),
                    "minus_b2_Frobenius_cosine": cosine(p["y"] - p["bias"], y - bias),
                    "row_mean_centered_Frobenius_cosine": cosine(p["centered"], centered),
                    "common_mean_fraction_of_cross_inner_product": float(mean_cross / cross) if cross != 0 else None,
                    "mean_centered_inner_product_identity_relative_error": float(abs(cross - mean_cross - centered_cross) / abs(cross)) if cross != 0 else None,
                    "mean_energy_fractions": [float(len(y) * np.sum(a.mean(axis=0) ** 2) / np.sum(a ** 2)) for a in [p["y"], y]]})
            previous[scope["block"]] = current
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve original temporal description")
    write_json(args.out, analyze(args.raw_root, args.indices))


if __name__ == "__main__":
    main()
