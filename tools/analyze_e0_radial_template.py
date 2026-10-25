"""Cross-source transfer of saved residual common directions, no model calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from analyze_e0_radial_pairing import center_channels
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/e0_radial_template_saved_20261025.json"


def fit_template(means, held_source):
    """Each source contributes one position mean; held source is never read."""
    training = [value for source, value in means.items() if source != held_source]
    if len(training) != 5:
        raise ValueError("Require exactly five training sources")
    return np.mean(training, axis=0)


def radial_fraction(state, delta, eps=1e-6):
    x, d = center_channels(state), center_channels(delta)
    norm = np.sum(x*x, axis=1)
    de = x.shape[1]*eps
    return float(np.sum((norm+2*de)/(norm+de)**2*np.sum(x*d, axis=1)**2)/np.sum(delta*delta))


def aggregate(records):
    keys = [k for k in records[0] if k not in ["source_index", "step", "block", "group", "state"]]
    return {"group_cases": len(records), **{k: float(np.mean([r[k] for r in records])) for k in keys}}


def analyze(raw_root):
    plan = json.loads(PLAN.read_text())
    parent_path = ROOT / plan["parent"]
    parent = json.loads(parent_path.read_text())
    manifest_path = ROOT / parent["accepted_manifest"]
    if sha256(manifest_path) != parent["accepted_manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = json.loads(manifest_path.read_text())
    means, scopes, bindings = {}, [], []
    for source in manifest["sources"]:
        index = source["index"]
        run_path = raw_root / f"source{index:02d}_attempt1/RUN.json"
        if sha256(run_path) != source["run_sha256"]:
            raise ValueError("Run changed")
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Unaccepted collection")
        for scope in run["scopes"]:
            if scope["block"] not in parent["blocks"]:
                continue
            path = run_path.parent / "observed" / scope["path"]
            wp = run_path.parent / "observed" / scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Operand changed")
            a = load_arrays(path)
            ids = [scope["row_plan"]["rows"].index(row) for row in scope["row_plan"]["uniform_rows"]]
            for state, field in zip(plan["states"], ["residual_pre_ffn", "residual_post_ffn"]):
                means.setdefault((scope["step"], scope["block"], state), {})[index] = (
                    a[field][ids].astype(np.float64).mean(axis=0, keepdims=True))
            scopes.append((index, scope, path, wp, ids))
            bindings.append({"source_index": index, "step": scope["step"], "block": scope["block"],
                             "payload_sha256": scope["sha256"],
                             "weight_panel_sha256": scope["weight_binding"]["sha256"]})
    records = []
    for index, scope, path, wp, ids in scopes:
        a, w = load_arrays(path), load_arrays(wp)
        for k, group in enumerate(parent["groups"]):
            delta = (a["q2_post"][ids, 64*group:64*(group+1)].astype(np.float64) @
                     w["W2_panel"][:, 64*k:64*(k+1)].astype(np.float64).T) * a["gate"].astype(np.float64)
            for state, field in zip(plan["states"], ["residual_pre_ffn", "residual_post_ffn"]):
                pool = means[(scope["step"], scope["block"], state)]
                template, own = fit_template(pool, index), pool[index]
                x = a[field][ids].astype(np.float64)
                t, o = center_channels(template), center_channels(own)
                cosine = float(np.sum(t*o)/np.sqrt(np.sum(t*t)*np.sum(o*o)))
                actual = radial_fraction(x, delta)
                own_score = radial_fraction(np.broadcast_to(own, x.shape), delta)
                other_score = radial_fraction(np.broadcast_to(template, x.shape), delta)
                records.append({"source_index": index, "step": scope["step"], "block": scope["block"],
                                "group": group, "state": state, "own_vs_other_mean_cosine": cosine,
                                "actual_paired_radial_fraction": actual,
                                "own_mean_radial_fraction": own_score,
                                "other_five_mean_radial_fraction": other_score,
                                "absolute_other_minus_own_fraction": abs(other_score-own_score),
                                "absolute_other_minus_actual_fraction": abs(other_score-actual)})
    if len(records) != 960:
        raise ValueError("Incomplete scope")
    return {"id": plan["id"], "status": "COMPLETE_SAVED_SOURCE_TRANSFER_DESCRIPTION",
            "analysis_sha256": sha256(Path(__file__)), "plan_sha256": sha256(PLAN),
            "parent_plan_sha256": sha256(parent_path), "accepted_manifest_sha256": sha256(manifest_path),
            "radial_pairing_helper_sha256": sha256(ROOT / "tools/analyze_e0_radial_pairing.py"),
            "source_clusters": 6, "new_model_generations": 0, "new_GPU_calls": 0,
            "new_native_FFN_calls": 0, "new_native_norm_calls": 0, "model_training_updates": 0,
            "offline_decoded_panel_matmuls": 480, "scope": plan["scope"],
            "source_bindings": bindings, "cases": records,
            "by_block_state": {str(b): {s: aggregate([r for r in records if r["block"] == b and r["state"] == s])
                              for s in plan["states"]} for b in parent["blocks"]},
            "block0_by_source": {str(i): {s: aggregate([r for r in records if r["block"] == 0 and
                                r["source_index"] == i and r["state"] == s]) for s in plan["states"]}
                                for i in plan["source_indices"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the first result")
    write_json(args.out, analyze(args.raw_root))


if __name__ == "__main__":
    main()
