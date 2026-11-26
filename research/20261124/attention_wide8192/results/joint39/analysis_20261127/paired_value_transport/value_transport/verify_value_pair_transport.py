"""Conditional algebra at small synthetic shape; static costs at actual Wan shape."""
from pathlib import Path
import json
import numpy as np
from value_pair_transport import (
    value_moments, capture_value_pairs, fit_value_map, encode_centered,
    decode_centered, ridge_repair, sparsepr_v2_repair,
)

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261127)
n, d, selected_count, pair_count, probe_count = 257, 16, 32, 128, 64


def softmax(x):
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z/z.sum(axis=1, keepdims=True)


q, k, va = (rng.normal(size=(n, d)) for _ in range(3))
logits_a = q @ k.T / np.sqrt(d)
pi_a = softmax(logits_a)
mask = np.zeros((n, n), dtype=bool)
selected = np.argsort(logits_a, axis=1)[:, -selected_count:]
np.put_along_axis(mask, selected, True, axis=1)
pi_sa = softmax(np.where(mask, logits_a, -np.inf))
rho = 1-(pi_a*mask).sum(axis=1)
aa, sa = pi_a @ va, pi_sa @ va
history = encode_centered(aa, sa, rho, va)
pair_rows = rng.choice(n, pair_count, replace=False)
state = capture_value_pairs(va, pair_rows)
probe_rows = np.sort(rng.choice(n, probe_count, replace=False))
hold = np.ones(n, dtype=bool)
hold[probe_rows] = False

# A full channel mixing that diagonal scaling cannot generally represent.
rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
transform = rotation @ np.diag(rng.uniform(.7, 1.5, d))
shift = rng.normal(scale=.2, size=d)
vt = va @ transform + shift
at, st = pi_a @ vt, pi_sa @ vt
matrix, mean_t, scale_t = fit_value_map(state, vt, ridge=0)
_, scale_a, _ = value_moments(va)
matrix_truth = np.diag(scale_a) @ transform
base_pair = decode_centered(st, rho, history, mean_t, matrix)
base_diag = decode_centered(st, rho, history, mean_t, np.diag(scale_t))
projection = rng.normal(size=(d, 8))/np.sqrt(d)
features = (st-sa) @ projection
paired = ridge_repair(base_pair, features, at[probe_rows], probe_rows)
diagonal = ridge_repair(base_diag, features, at[probe_rows], probe_rows)
current = sparsepr_v2_repair(st, at[probe_rows], probe_rows)

# The same map trained only on a subset is evaluated on untouched key rows.
key_hold = np.ones(n, dtype=bool)
key_hold[pair_rows] = False
mean_a, _, inverse_a = value_moments(va)
xa = (va-mean_a)*inverse_a
value_unseen_error = np.max(np.abs((vt-mean_t-xa@matrix)[key_hold]))

# Diagonal positive value changes recover the existing diagonal special case.
gamma = rng.uniform(.5, 1.8, d)
vd = va*gamma + shift
md, mu_d, sd = fit_value_map(state, vd, ridge=0)
diag_special_error = np.max(np.abs(md-np.diag(sd)))

# General changes: exact separation of routing/mass and value-map error.
qt = q + .15*rng.normal(size=q.shape)
kt = k + .13*rng.normal(size=k.shape)
lg_t = qt @ kt.T / np.sqrt(d)
pi_t = softmax(lg_t)
pi_st = softmax(np.where(mask, lg_t, -np.inf))
v_noise = vt + .05*rng.normal(size=vt.shape)
mp, mu_p, sp = fit_value_map(state, v_noise, ridge=1e-3)
current_a, current_s = pi_t@v_noise, pi_st@v_noise
base = decode_centered(current_s, rho, history, mu_p, mp)
omitted_a = pi_a*(~mask)
omitted_t = pi_t*(~mask)
rho_t = omitted_t.sum(axis=1)
yt = v_noise-mu_p
value_defect = yt-xa@mp
term_prob = (omitted_t-omitted_a)@yt
term_mass = (rho-rho_t)[:, None]*(current_s-mu_p)
term_value = omitted_a@value_defect
decomposition_error = np.max(np.abs(current_a-base-term_prob-term_mass-term_value))
bound_rhs = rho*np.sum(omitted_a*np.sum(value_defect**2, axis=1)[None, :], axis=1)
bound_lhs = np.sum(term_value**2, axis=1)

N, H, D, layers, J = 31668, 40, 128, 40, 256
apply_flops = 2*N*H*D*D
stats_flops = 4*J*H*D*D
dense_attention = 4*N*N*H*D
whole_cycle = 4*layers*39.85196220416e12
out = {
    "scope": "Synthetic FP64 operator checks and actual-Wan-shape arithmetic; no new model/GPU performance results; recovery baseline is paper-defined v2 on supplied equal-weight probes, not full SparsePR",
    "synthetic_shape": {"tokens":n,"head_dim":d,"selected_keys":selected_count,
                        "value_pair_rows":pair_count,"native_probe_rows":probe_count},
    "general_affine_value_fixed_weights": {
        "sample_design_rank": int(np.linalg.matrix_rank(state.normalized_anchor_rows)),
        "identified_matrix_max_abs_error": float(np.max(np.abs(matrix-matrix_truth))),
        "unseen_value_max_abs_error": float(value_unseen_error),
        "base_max_abs_error": float(np.max(np.abs(at-base_pair))),
        "repaired_max_abs_error": float(np.max(np.abs(at-paired))),
        "diagonal_repaired_held_l2_error": float(np.linalg.norm((at-diagonal)[hold])),
        "current_affine_repaired_held_l2_error": float(np.linalg.norm((at-current)[hold])),
        "scope": "Existence witness for a new exact class; errors of other estimators are not empirical Wan improvements or forecast win rates",
    },
    "diagonal_special_case_matrix_max_abs_error": float(diag_special_error),
    "changed_QKV_error_decomposition_max_abs": float(decomposition_error),
    "omitted_probability_value_bound_max_violation": float(np.max(bound_lhs-bound_rhs)),
    "cost": {
        "N":N,"heads":H,"head_dim":D,"layers":layers,"value_pair_rows":J,
        "extra_BF16_anchor_pair_bytes": 2*J*H*D*layers,
        "extra_BF16_anchor_pair_GiB": 2*J*H*D*layers/2**30,
        "full_matrix_apply_FLOPs_per_layer_low_hit":apply_flops,
        "full_matrix_apply_FLOPs_whole_low_hit":apply_flops*layers,
        "value_pair_two_Gram_FLOPs_per_layer":stats_flops,
        "apply_over_one_dense_attention":apply_flops/dense_attention,
        "apply_over_whole_fourstep_principal_ledger":apply_flops*layers/whole_cycle,
        "rank8_optional_apply_FLOPs_per_layer_low_hit":4*N*H*D*8,
        "matrix_FP32_MiB_one_active_layer":4*H*D*D/2**20,
        "scope":"Logical bytes and designated GEMMs only; excludes solves, moments, gather, casts, padding, I/O and launch; no latency claim",
    },
}
assert out["general_affine_value_fixed_weights"]["sample_design_rank"] == d
assert out["general_affine_value_fixed_weights"]["identified_matrix_max_abs_error"] < 1e-12
assert out["general_affine_value_fixed_weights"]["repaired_max_abs_error"] < 1e-12
assert out["general_affine_value_fixed_weights"]["diagonal_repaired_held_l2_error"] > 1e-3
assert out["general_affine_value_fixed_weights"]["current_affine_repaired_held_l2_error"] > 1e-3
assert diag_special_error < 1e-12
assert decomposition_error < 1e-12
assert np.max(bound_lhs-bound_rhs) < 1e-12
(ROOT/"VALUE_PAIR_TRANSPORT_VERIFIED.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False,indent=2))
