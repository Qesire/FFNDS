#!/usr/bin/env python3
"""CPU-only residual replay of existing joint39 vector assets.

Prepared without access to the archived model vectors in this local review;
not executed on real joint39 vectors at preparation time. No model or GPU
execution is performed by this script.

The independent baseline MUST be flow_B11_fixed, whose P rows were not
overwritten. flow_independent is intentionally never loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

RIDGE = 0.1
KEYS = (
    "joint_phi", "probe_rows", "held_rows",
    "flow_B00", "flow_cheap", "flow_B11_fixed", "flow_joint",
)


def energies(error, held, edges):
    result = {"point": float(error[held].square().sum())}
    for name, edge in edges.items():
        i, j = edge.T
        result[name] = float((error[i] - error[j]).square().sum())
    return result


def ridge_weights(design, target, rows):
    x, y = design[rows], target[rows]
    normal = x.T @ x + len(rows) * RIDGE * torch.eye(
        x.shape[1], dtype=x.dtype, device="cpu"
    )
    return torch.linalg.solve(normal, x.T @ y)


def compare_baseline(base, truth, design, probe, held, edges):
    residual = truth - base
    weights = ridge_weights(design, residual, probe)
    prediction = base + design @ weights
    prediction[probe] = truth[probe]
    # Match ridge_flow's FP64 solve followed by its FP32 output.
    prediction = prediction.float()
    scores = energies(prediction.double() - truth, held, edges)

    zh = design[held]
    floor_weights = torch.linalg.lstsq(
        zh, residual[held], driver="gelsd"
    ).solution
    all_h_weights = ridge_weights(design, residual, held)
    # Match the published H448 diagnostics: score the FP64 linear residual,
    # without adding an output cast to the representation floor.
    floor_error = design @ floor_weights - residual
    all_h_error = design @ all_h_weights - residual
    return {
        "baseline": energies(base - truth, held, edges),
        "P64_residual_fit": scores,
        "H448_OLS_floor": energies(floor_error, held, edges),
        "H448_same_ridge": energies(all_h_error, held, edges),
        "P64_weight_norm": float(weights.norm()),
    }, prediction


def replay_asset(path, panel, edges):
    # safe_open reads only these small tensors, not the whole vector file.
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        values = {key: stream.get_tensor(key) for key in KEYS}
    probe = values["probe_rows"].long()
    held = values["held_rows"].long()
    phi = values["joint_phi"].double()
    truth = values["flow_B00"].double()
    if phi.shape != (512, 48) or truth.shape != (512, 64):
        raise ValueError(f"Unexpected joint39 shapes in {path}")
    if not torch.equal(probe, torch.tensor(panel["probe_rows_in_production_order"])):
        raise ValueError(f"P64 order does not match panel index: {path}")
    if not torch.equal(held, torch.tensor(panel["held_rows"])):
        raise ValueError(f"H448 rows do not match panel index: {path}")
    if probe.numel() != 64 or held.numel() != 448 or torch.isin(probe, held).any():
        raise ValueError(f"Invalid P64/H448 partition: {path}")
    for name, edge in edges.items():
        if edge.shape != (224, 2) or not torch.isin(edge, held).all():
            raise ValueError(f"{name} edges must lie entirely in H448")

    scale = phi[probe].square().mean(0).sqrt()
    inverse_scale = torch.where(scale > 0, scale.reciprocal(), 0)
    design = phi * inverse_scale
    cheap, cheap_prediction = compare_baseline(
        values["flow_cheap"].double(), truth, design, probe, held, edges
    )
    independent, _ = compare_baseline(
        values["flow_B11_fixed"].double(), truth, design, probe, held, edges
    )
    with path.open("rb") as asset_stream:
        asset_sha256 = hashlib.file_digest(asset_stream, "sha256").hexdigest()
    saved_joint = values["flow_joint"].double()
    difference = cheap_prediction.double() - saved_joint
    return {
        "asset": str(path),
        "asset_bytes": path.stat().st_size,
        "asset_sha256": asset_sha256,
        "rows": 512,
        "output_width": 64,
        "feature_columns": 48,
        "P64_rank": int(torch.linalg.matrix_rank(design[probe])),
        "P64_condition": float(torch.linalg.cond(design[probe])),
        "cheap_base": cheap,
        "B11_fixed_base": independent,
        "saved_joint_scores": energies(saved_joint - truth, held, edges),
        "cheap_replay_vs_saved_joint": {
            "H448_difference_energy": float(difference[held].square().sum()),
            "H448_max_abs": float(difference[held].abs().max()),
            "P64_max_abs": float(difference[probe].abs().max()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", required=True, type=Path,
                        help="Sealed joint39 .safetensors file; repeat for multiple sites.")
    parser.add_argument("--panel-index", required=True, type=Path,
                        help="Matching attention_panel_index.json with P/H and edge rows.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    panel_bytes = args.panel_index.read_bytes()
    panel = json.loads(panel_bytes)
    edges = {
        "spatial": torch.tensor(panel["spatial_edges_rows"], dtype=torch.long),
        "temporal": torch.tensor(panel["temporal_edges_rows"], dtype=torch.long),
    }
    with torch.inference_mode():
        rows = [replay_asset(path.resolve(), panel, edges) for path in args.asset]
    output = {
        "status": "COMPLETED_EXISTING_VECTOR_CPU_REPLAY",
        "new_model_executions": 0,
        "gpu_used": False,
        "torch_version": torch.__version__,
        "panel_index": str(args.panel_index.resolve()),
        "panel_index_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "ridge_per_query": RIDGE,
        "feature_source": "Same stored joint_phi, normalized only by P64 column RMS.",
        "teacher_budget": (
            "Unchanged P64 teacher flow labels at the same probe rows; "
            "H448 labels enter fitting only in the separately reported representation diagnostics."
        ),
        "independent_baseline": (
            "flow_B11_fixed before P overwrite; flow_independent is never read."
        ),
        "execution_cost_scope": (
            "The B11_fixed-base residual fit retains the independent computations "
            "and adds a solve/readout. This is a diagnostic additive construction; "
            "additional online solve/readout cost has not been timed."
        ),
        "precision": (
            "CPU FP64 solve; P64-fitted prediction cast to FP32; "
            "all scores accumulated in FP64. H448 reference fits stay FP64."
        ),
        "H448_reference_scope": "Fit and score on the same H448; diagnostic only.",
        "sites": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), "sites": len(rows),
                      "new_model_executions": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()

