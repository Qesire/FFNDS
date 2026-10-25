"""Ideal epsilon-LayerNorm geometry on accepted native saved signed operands.

This is decoded FP64 mathematics, not a native finite FFN intervention or video
evaluation. Final block39 is excluded because it has no next-block norm1.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays, ratio
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def geometry(state, delta, eps=1e-6):
    state, delta = np.asarray(state, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if state.shape != delta.shape or state.ndim != 2 or not np.isfinite(state).all() or not np.isfinite(delta).all():
        raise ValueError("Need paired finite residual/direction matrices")
    width = state.shape[1]
    if eps < 0:
        raise ValueError("Normalization epsilon must be nonnegative")
    centered = state - state.mean(axis=1, keepdims=True)
    norm2 = np.sum(centered ** 2, axis=1, keepdims=True)
    variance = norm2 / width + eps
    if np.any(variance <= 0):
        raise ValueError("Degenerate normalization")
    pd = delta - delta.mean(axis=1, keepdims=True)
    tangent = pd - centered * np.sum(centered * pd, axis=1, keepdims=True) / (norm2 + width * eps)
    energy = np.sum(delta ** 2)
    rstd2 = 1 / variance
    weighted = np.sum(rstd2 * delta ** 2)
    changed = state - delta
    changed = changed - changed.mean(axis=1, keepdims=True)
    finite = changed / np.sqrt(np.mean(changed ** 2, axis=1, keepdims=True) + eps) - centered / np.sqrt(variance)
    row_energy = np.sum(delta ** 2, axis=1, keepdims=True)
    isotropic = (2 - (width * eps / (norm2 + width * eps)) ** 2) / width
    return {"channel_mean_removed_fraction": ratio(energy - np.sum(pd ** 2), energy),
            "radial_removed_fraction": ratio(np.sum(pd ** 2) - np.sum(tangent ** 2), energy),
            "total_directional_energy_removed_fraction": ratio(energy - np.sum(tangent ** 2), energy),
            "rstd_weighted_energy_removed_fraction": ratio(weighted - np.sum(rstd2 * tangent ** 2), weighted),
            "finite_ideal_LN_response_over_rstd_scaled_direction_energy": ratio(np.sum(finite ** 2), weighted),
            "finite_vs_tangent_relative_SSE": ratio(np.sum((finite + tangent / np.sqrt(variance)) ** 2), np.sum(finite ** 2)),
            "isotropic_expected_removed_fraction": ratio(np.sum(row_energy * isotropic), energy),
            "rstd_weighted_isotropic_expected_removed_fraction": ratio(np.sum(rstd2 * row_energy * isotropic), weighted)}


def analyze(root, indices):
    source_paths = [ROOT / ".runtime/code/LightX2V/lightx2v" / p for p in [
        "models/networks/wan/weights/transformer_weights.py",
        "models/networks/wan/infer/transformer_infer.py",
        "common/ops/norm/layer_norm_weight.py"]]
    result = {"status": "COMPLETE_SAVED_NATIVE_OPERAND_IDEAL_LN_GEOMETRY",
              "source_clusters": len(indices), "indices": indices, "epsilon": 1e-6,
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), *source_paths]},
              "scope": "Five fixed decoded W2 group contributions; sampled uniform rows; actual residual operands; ideal FP64 epsilon-LN and algebraic leave-group-out controls. No native action, no actual nextnorm tensor, no downstream residual-bypass or video claim.",
              "excluded_block": 39, "new_FFN_calls": 0, "GPU_calls": 0, "sources": [], "cases": []}
    for index in indices:
        run_path = root / f"source{index:02d}_attempt1/RUN.json"
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Require accepted native collection")
        result["sources"].append({"index": index, "run_sha256": sha256(run_path)})
        for scope in run["scopes"]:
            if scope["block"] == 39:
                continue
            path = run_path.parent / "observed" / scope["path"]
            wp = run_path.parent / "observed" / scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Signed operand or weight panel changed")
            a, w = load_arrays(path), load_arrays(wp)
            lookup = {r: i for i, r in enumerate(scope["row_plan"]["rows"])}
            ids = [lookup[r] for r in scope["row_plan"]["uniform_rows"]]
            q2 = a["q2_post"][ids].astype(np.float64)
            pre, post = [a[k][ids].astype(np.float64) for k in ["residual_pre_ffn", "residual_post_ffn"]]
            for k, group in enumerate(scope["weight_binding"]["groups"]):
                delta = (q2[:, 64*group:64*(group+1)] @ w["W2_panel"][:, 64*k:64*(k+1)].astype(np.float64).T) * a["gate"].astype(np.float64)
                result["cases"].append({"source_index": index, "step": scope["step"],
                    "block": scope["block"], "group": group, "uniform_rows": len(ids),
                    "pre_FFN_state": geometry(pre, delta),
                    "actual_post_FFN_state": geometry(post, delta),
                    "decoded_leave_group_out_state": geometry(post - delta, delta)})
    result["offline_decoded_panel_matmuls"] = len(result["cases"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve first geometry result")
    write_json(args.out, analyze(args.raw_root, args.indices))


if __name__ == "__main__":
    main()
