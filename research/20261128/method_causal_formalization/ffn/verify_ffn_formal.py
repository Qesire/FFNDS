"""CPU algebra certificates for the frozen FFN mathematical representation.

Uses real frozen observer IDs for the graph/rank calculation. Numerical FFN and
GLS operands are explicitly synthetic; this script runs no model or benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--observer-json", type=Path, default=HERE / "FFN_OBSERVER_METADATA.json")
parser.add_argument("--source-manifest", type=Path, default=HERE / "FFN_SOURCE_BINDING.json")
parser.add_argument("--source-root", type=Path,
                    help="Optional root containing original relative source paths; rechecks snapshot hashes")
parser.add_argument("--output", type=Path, default=HERE / "FFN_FORMAL_VERIFIED.json")
args = parser.parse_args()
rng = np.random.default_rng(20261128)
checks = {}


def check(name, lhs, rhs, tol=2e-10):
    error = float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
    checks[name] = {"max_abs_error": error, "tolerance": tol, "pass": error < tol}
    assert error < tol, (name, error)


identity = json.loads(args.observer_json.read_text())
ids = np.array(identity["observer_ids"])
grid = tuple(identity.get("grid", (21, 29, 52)))
assert grid == (21, 29, 52), "This certificate is bound to the frozen latent geometry"
assert ids.shape == (64,) and len(np.unique(ids)) == 64
coords = np.column_stack(np.unravel_index(ids, grid))
p = len(ids)
pairs = set()
for i, c in enumerate(coords):
    same = np.flatnonzero((coords[:, 0] == c[0]) & (np.arange(p) != i))
    if len(same):
        dist = ((coords[same, 1:] - c[1:]) ** 2).sum(1)
        j = int(same[np.lexsort((ids[same], dist))[0]])
        pairs.add(tuple(sorted((i, j))))
D = np.stack([np.eye(p)[i] - np.eye(p)[j] for i, j in sorted(pairs)])
metric = np.eye(p) + p / len(pairs) * D.T @ D
inv_metric = np.linalg.inv(metric)
point_residual = rng.normal(size=(p, 7))
check("point_edge_metric", np.sum(point_residual * (metric @ point_residual)),
      np.sum(point_residual ** 2) + p / len(pairs) * np.sum((D @ point_residual) ** 2))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# Exact selected/omitted finite-increment decomposition, independent of smallness.
n, width, hidden, selected = 97, 24, 47, 17
u0, ut = rng.normal(size=(2, n, width))
w1 = rng.normal(size=(width, hidden)) / np.sqrt(width)
w2 = rng.normal(size=(hidden, width)) / np.sqrt(hidden)
b1, b2 = rng.normal(size=hidden), rng.normal(size=width)
alpha = rng.normal(size=hidden)
sel, omit = np.arange(selected), np.arange(selected, hidden)
z0, zt = u0 @ w1 + b1, ut @ w1 + b1
du, dz, dh = ut - u0, zt - z0, gelu(zt) - gelu(z0)
y0, yt = gelu(z0) @ w2 + b2, gelu(zt) @ w2 + b2
a0 = (w1[:, omit] * alpha[omit]) @ w2[omit]
base = y0 + du @ a0 + dh[:, sel] @ w2[sel]
omitted_residual = (dh[:, omit] - dz[:, omit] * alpha[omit]) @ w2[omit]
check("exact_omitted_remainder", yt - base, omitted_residual)

r = 16
zmap = rng.normal(size=(selected, r)) / np.sqrt(selected)
jmap = (w1[:, sel] * alpha[sel]) @ zmap
q = dh[:, sel] @ zmap - du @ jmap
check("mode_is_selected_nonlinear_residual", q, (dh[:, sel] - dz[:, sel] * alpha[sel]) @ zmap)
kernel_beta = rng.normal(size=selected)
kernel_scale = rng.uniform(0.5, 1.5, size=selected)
x = (dh[:, sel] - dz[:, sel] * kernel_beta) / kernel_scale
check("cal_and_kernel_slope_bridge", q,
      (x * kernel_scale) @ zmap + (dz[:, sel] * (kernel_beta - alpha[sel])) @ zmap)
v = np.linalg.qr(rng.normal(size=(width, r)))[0].T
pp = np.arange(p)
qp, xp = q[pp], x[pp]
k = selected
Kpp, Knp = xp @ xp.T / k, x @ xp.T / k
S = Kpp + inv_metric
SinvQ = np.linalg.solve(S, qp)
G = (qp.T @ SinvQ) * (v @ v.T)


def fit(target, q_all=q, q_p=qp, kernel_cross=Knp, covariance=S):
    iq = np.linalg.solve(covariance, q_p)
    gram = (q_p.T @ iq) * (v @ v.T)
    gains = np.linalg.solve(gram, np.sum(iq * (target @ v.T), axis=0))
    coefficient = np.linalg.solve(covariance, target - (q_p * gains) @ v)
    recovery = (q_all * gains) @ v + 0.5 * kernel_cross @ coefficient
    return recovery, gains, coefficient


target = rng.normal(size=(p, width))
recovered, gains, coefficient = fit(target)
normal_residual = target - (qp * gains) @ v
check("gls_normal_equations", np.sum(SinvQ * (normal_residual @ v.T), axis=0), np.zeros(r))
primal_weights = np.linalg.solve(xp.T @ metric @ xp / k + np.eye(k),
                                 xp.T @ metric @ normal_residual / np.sqrt(k))
check("kernel_ridge_primal_dual", x @ primal_weights / np.sqrt(k), Knp @ coefficient)
another = rng.normal(size=target.shape)
check("fixed_feature_recovery_linearity", fit(2 * target - 3 * another)[0],
      2 * fit(target)[0] - 3 * fit(another)[0])

c = rng.normal(size=r)
mode = (q * c) @ v
mode_recovery, mode_g, mode_C = fit(mode[pp])
check("mode_space_exact_recovery", mode_recovery, mode)
check("mode_coefficients_identified", mode_g, c)
check("mode_kernel_residual_zero", mode_C, np.zeros_like(mode_C))

# The required rank is tensor-atom rank, not rank(Q_P).
rank1q = rng.normal(size=(n, 1)) @ rng.uniform(0.5, 1.5, size=(1, r))
rank1mode = (rank1q * c) @ v
rank1rec, _, _ = fit(rank1mode[pp], q_all=rank1q, q_p=rank1q[pp])
check("rank1_q_can_identify_16_output_modes", rank1rec, rank1mode)
atom_design = np.stack([(rank1q[pp, j, None] * v[j]).reshape(-1) for j in range(r)], axis=1)

# Exact hard overwrite and perturbation identity at fixed features/current input.
teacher = rng.normal(size=(p, width))
def predict(base_value):
    result = base_value + fit(teacher - base_value[pp])[0]
    result[pp] = teacher
    return result

perturb = rng.normal(size=base.shape)
filtered = perturb - fit(perturb[pp])[0]
filtered[pp] = 0
check("base_error_after_refit_and_overwrite", predict(base + perturb) - predict(base), filtered)
unseen = rng.normal(size=base.shape)
unseen[pp] = 0
check("probe_nullspace_recovery_is_zero", fit(unseen[pp])[0], np.zeros_like(unseen))
check("probe_nullspace_residual_survives", predict(base + unseen) - predict(base), unseen)

# Paired gauge (static c) is a real identity when teacher and Q/X remain fixed.
Ac = a0 + (jmap * c) @ v
Bc = w2[sel] - (zmap * c) @ v
base_c = y0 + du @ Ac + dh[:, sel] @ Bc
check("paired_gauge_base_change", base_c - base, -mode)
check("paired_gauge_output_identity", predict(base_c), predict(base))

# Same-value limiting input gives Q=X=0, so current unregularized gain solve
# requires a separate zero-increment policy; no claim of observed model failure.
zero_gram = np.zeros((r, r))

# Finite-precision counterexample: arithmetic reordering is not a BF16 identity.
def bf16(values):
    """Finite FP32 values, BF16 round-to-nearest-even, returned as FP32."""
    bits = np.asarray(values, dtype=np.float32).copy().view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)


ub0 = bf16(rng.normal(size=(31, 24)))
ubt = bf16(rng.normal(size=(31, 24)))
ab = bf16(rng.normal(size=(24, 24)))
original = bf16(bf16(ubt - ub0) @ ab)
reordered = bf16(ubt @ ab) - bf16(ub0 @ ab)
rounding = original - reordered
assert np.count_nonzero(rounding) > 0

binding = json.loads(args.source_manifest.read_text())
source_hashes = binding["files"]
source_status = "immutable_manifest_only; source bytes not requested"
if args.source_root is not None:
    observed_hashes = {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest()
                       for name in source_hashes}
    assert observed_hashes == source_hashes, "Source snapshot bytes differ from the immutable binding"
    source_status = "all_source_hashes_reverified"
result = {
    "status": "PASS", "scope": "CPU algebra only; no model inference, ablation, GPU work or latency measurement",
    "research_head_supplied_by_parent": binding["research_head"],
    "inputs": {"observer_metadata_sha256": hashlib.sha256(args.observer_json.read_bytes()).hexdigest(),
               "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()},
    "actual_observer_geometry": {
        "grid": [21, 29, 52], "N": 31668, "P": p, "E": len(pairs),
        "incidence_rank": int(np.linalg.matrix_rank(D)),
        "graph_connected_components": int(p - np.linalg.matrix_rank(D)),
        "point_edge_stacked_rows": p + len(pairs),
        "point_edge_stacked_rank": int(np.linalg.matrix_rank(np.vstack([np.eye(p), D]))),
        "mode_plus_kernel_design_columns": 16 + p,
        "sampled_video_latent_frames": int(len(np.unique(coords[:, 0]))),
        "frame_probe_counts": {str(f): int(np.sum(coords[:, 0] == f)) for f in range(21)},
        "all_edges_same_video_frame": bool(all(coords[i, 0] == coords[j, 0] for i, j in pairs)),
        "observer_ids": ids.tolist(), "edge_probe_row_pairs": [list(pair) for pair in sorted(pairs)],
    },
    "synthetic_algebra_dimensions": {"N": n, "d": width, "m": hidden, "K": selected, "P": p, "modes": r},
    "checks": checks,
    "rank_certificate": {
        "rank_of_q_probe_example": int(np.linalg.matrix_rank(rank1q[pp])),
        "rank_of_tensor_atom_design_example": int(np.linalg.matrix_rank(atom_design)),
        "zero_increment_gram_rank": int(np.linalg.matrix_rank(zero_gram)),
    },
    "bf16_reorder_counterexample": {
        "mismatched_entries": int(np.count_nonzero(rounding)), "entries": int(rounding.size),
        "max_abs_difference": float(np.max(np.abs(rounding))),
        "scope": "Synthetic CPU round-to-nearest-even BF16 emulation with FP32 matmul, not a model error estimate or GPU execution equivalence test",
    },
    "source_sha256": source_hashes,
    "source_validation": source_status,
    "model_data_status": {
        "real_jmap_spectrum": "MISSING; no live full site jmap arrays were loaded",
        "real_v_orthogonality": "MISSING; compiler orthogonalization was source-verified, not remeasured",
        "real_online_gram_condition_numbers": "MISSING; no live Q_P or X_P arrays were loaded",
        "real_omitted_residual_recovery_error": "MISSING; no full current native FFN panel was computed",
        "interpretation": "Algebra certificates prove conditional identities, not actual model conditioning or recovery accuracy"
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": result["status"], "algebra_checks": len(checks),
                  "max_algebra_error": max(item["max_abs_error"] for item in checks.values()),
                  "actual_graph": {k: v for k, v in result["actual_observer_geometry"].items()
                                   if k not in ("observer_ids", "edge_probe_row_pairs")},
                  "rank_certificate": result["rank_certificate"],
                  "bf16_counterexample": result["bf16_reorder_counterexample"]}, indent=2))
