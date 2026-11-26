"""Conditional operator equivalence and attention witnesses, CPU float64 only.

These are algebraic constructions, not Wan quality/speed experiments and not
an execution of the unpublished SparsePR implementation.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261127)


def reader(phi, index, weights, penalty):
    a = phi[index]
    return phi @ np.linalg.solve(a.T @ (weights[:, None] * a) + penalty,
                                 a.T * weights[None, :])


def recover(base, yp, index, k):
    out = base + k @ (yp - base[index])
    out[index] = yp
    return out


def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


n, p, q, d = 37, 8, 4, 7
index = np.array([1, 4, 8, 13, 18, 24, 29, 35])
phi = np.c_[np.ones(n), rng.normal(size=(n, q - 1))]
weight = rng.uniform(.5, 2., size=p)
s = rng.normal(size=(n, d))
y = rng.normal(size=(n, d))
r = rng.normal(size=(n, d))
k = reader(phi, index, weight, .4 * np.eye(q))
p_select = np.eye(n)[index]
mask = np.eye(n)
mask[index, index] = 0
ell = mask @ (np.eye(n) - k @ p_select)
without = recover(s, y[index], index, k)
with_history = recover(s + r, y[index], index, k)
identity_error = np.max(np.abs(with_history - without - ell @ r))

# No ridge, and the probes uniquely identify every dictionary coefficient.
k0 = reader(phi, index, weight, np.zeros((q, q)))
coeff = rng.normal(size=(q, d))
r_inside = phi @ coeff
equivalent_error = np.max(np.abs(recover(s + r_inside, y[index], index, k0)
                                 - recover(s, y[index], index, k0)))

# The same function class with nonzero regularization can change only the
# regularization center, even when the historical field lies in the dictionary.
penalty = .4 * np.eye(q)
a = phi[index]
h = a.T @ (weight[:, None] * a) + penalty
ridge_expected = mask @ phi @ np.linalg.solve(h, penalty @ coeff)
ridge_difference = (recover(s + r_inside, y[index], index, k)
                    - recover(s, y[index], index, k))
ridge_identity_error = np.max(np.abs(ridge_difference - ridge_expected))

# A prior can supply non-probe directions not observed in current labels.
r_hidden = rng.normal(size=(n, d))
r_hidden[index] = 0
y_hidden = s + r_hidden + phi @ coeff
hidden_without = recover(s, y_hidden[index], index, k0)
hidden_with = recover(s + r_hidden, y_hidden[index], index, k0)

# Fitting dense outputs vs residuals differs by the same baseline-preservation
# operator. It becomes equivalent only under suitable recovery/regularization.
dense_fit = k @ y[index]
dense_fit[index] = y[index]
dense_residual_difference_error = np.max(np.abs(without - dense_fit - ell @ s))

# With different dictionaries/readers, isolate the history and reader effects.
phi2 = np.c_[np.ones(n), rng.normal(size=(n, q - 1))]
k2 = reader(phi2, index, weight, .7 * np.eye(q))
mixed_difference = recover(s + r, y[index], index, k2) - without
mixed_expected = mask @ ((np.eye(n) - k2 @ p_select) @ r
                         + (k2 - k) @ (y[index] - s[index]))
mixed_identity_error = np.max(np.abs(mixed_difference - mixed_expected))

# The temporal dictionary is affine in the whole current tensor at fixed
# anchor, but its anchor-dependent offsets differ across query rows. It need
# not belong to the shared row-wise affine(S_t) family even with zero prior.
same_current = np.zeros((4, 1))
different_anchor = np.array([[-1.], [0.], [1.], [2.]])
temporal_phi = np.c_[np.ones(4), same_current - different_anchor]
current_phi = np.ones((4, 1))
temporal_probe = np.array([0, 3])
temporal_target = -different_anchor
temporal_reader = reader(temporal_phi, temporal_probe, np.ones(2), np.zeros((2, 2)))
current_reader = reader(current_phi, temporal_probe, np.ones(2), np.zeros((1, 1)))
temporal_output = recover(same_current, temporal_target[temporal_probe], temporal_probe, temporal_reader)
current_output = recover(same_current, temporal_target[temporal_probe], temporal_probe, current_reader)


def sparsepr_style_fixed_route(sparse, dense, probe, coverage, ridge=1.):
    """Per-head affine residual ridge with standardized sparse features.

    Probe-weighted mean/std and unpenalized intercept make the conditional
    positive-diagonal value-affine argument explicit. This is a mathematical
    realization of that regression family, not a claim to run author code.
    """
    w = coverage / coverage.sum()
    mean = (w[:, None] * sparse[probe]).sum(axis=0)
    variance = (w[:, None] * (sparse[probe] - mean)**2).sum(axis=0)
    # A scale of 1 for a constant feature represents its exact zero centered
    # feature; for positive variance no epsilon is introduced.
    std = np.where(variance > 1e-28, np.sqrt(variance), 1.)
    f = np.c_[np.ones(len(sparse)), (sparse - mean) / std]
    regularizer = ridge * np.eye(f.shape[1])
    regularizer[0, 0] = 0.
    read = reader(f, probe, coverage, regularizer)
    return recover(sparse, dense[probe], probe, read), f


# General current affine residual fitting is itself value-affine equivariant
# when the routing/coverage remain fixed and normalization is nondegenerate.
nq, nk, dv = 31, 13, 5
logits = rng.normal(size=(nq, nk))
prob = softmax(logits)
selected = np.array([0, 3, 5, 9])
ps = softmax(logits[:, selected])
value = rng.normal(size=(nk, dv))
s_a = ps @ value[selected]
y_a = prob @ value
probes = np.array([1, 5, 9, 15, 20, 26, 29])
coverage = prob[probes][:, selected].sum(axis=1)
gamma = rng.uniform(.4, 2.3, size=dv)
shift = rng.normal(size=dv)
value_t = value * gamma + shift
s_t = ps @ value_t[selected]
y_t = prob @ value_t
fit_a, f_a = sparsepr_style_fixed_route(s_a, y_a, probes, coverage)
fit_t, f_t = sparsepr_style_fixed_route(s_t, y_t, probes, coverage)
affine_error = np.max(np.abs(fit_t - (fit_a * gamma + shift)))
feature_invariance_error = np.max(np.abs(f_t - f_a))


def exact_attention_witness(prob, value, probe, gamma=1.7, shift=-.3):
    # The supplied probabilities are realized by ordinary scaled-dot-product
    # attention: keys are the standard basis, Q=sqrt(d_k)*log(prob).
    dk = prob.shape[1]
    keys = np.eye(dk)
    query = np.sqrt(dk) * np.log(prob)
    realized = softmax(query @ keys.T / np.sqrt(dk))
    value = value[:, None]
    y_a = realized @ value
    s_a = np.repeat(value[[0]], len(prob), axis=0)
    rho = 1. - realized[:, [0]]
    mu_a = value.mean(axis=0)
    sigma_a = value.std(axis=0)
    z = (y_a - (1 - rho) * s_a - rho * mu_a) / sigma_a
    vt = value * gamma + shift
    y_t = realized @ vt
    s_t = np.repeat(vt[[0]], len(prob), axis=0)
    transported = ((1 - rho) * s_t + rho * vt.mean(axis=0)
                   + z * vt.std(axis=0))
    covariance_weights = realized[probe, 0]
    f_a, _ = sparsepr_style_fixed_route(s_a, y_a, probe, covariance_weights)
    f_t, _ = sparsepr_style_fixed_route(s_t, y_t, probe, covariance_weights)
    return {
        "probability_realization_max_abs": float(np.max(np.abs(realized - prob))),
        "dense_anchor": y_a.ravel().tolist(),
        "sparse_anchor": s_a.ravel().tolist(),
        "rho_anchor": rho.ravel().tolist(),
        "z_anchor": z.ravel().tolist(),
        "probe_rows": probe.tolist(),
        "value_gamma": gamma, "value_shift": shift,
        "dense_hit": y_t.ravel().tolist(),
        "current_affine_sparse_feature_fit_hit": f_t.ravel().tolist(),
        "transported_history_hit": transported.ravel().tolist(),
        "history_transport_max_abs_error": float(np.max(np.abs(transported - y_t))),
        "current_affine_fit_l2_error": float(np.linalg.norm(f_t - y_t)),
        "current_affine_fit_equivariance_max_abs": float(np.max(np.abs(f_t - (f_a * gamma + shift)))),
        "scope": "Constructed exact attention, fixed top-1 support (key 0); all Q/K/V are known. Distinguishability concerns the restricted affine(S) readout, not information-theoretic impossibility from full Q/K/V. No real-model performance inference."
    }


p2 = np.array([.1, .25, .4])
two_key = exact_attention_witness(np.c_[1 - p2, p2], np.array([0., 1.]), np.array([1]))
three_key = exact_attention_witness(np.array([[.7, .2, .1], [.7, .15, .15], [.7, .1, .2]]),
                                    np.array([0., 0., 1.]), np.array([1]))

out = {
    "scope": "CPU float64 conditional operator identities and exact synthetic attention; no new model/GPU runs; not author SparsePR execution",
    "general_output_difference_identity_max_abs": float(identity_error),
    "unregularized_identifiable_history_in_dictionary_output_difference_max_abs": float(equivalent_error),
    "ridge_prior_center_identity_max_abs": float(ridge_identity_error),
    "ridge_same_dictionary_output_difference_l2": float(np.linalg.norm(ridge_difference)),
    "hidden_history": {
        "probe_history_norm": float(np.linalg.norm(r_hidden[index])),
        "no_history_error_l2": float(np.linalg.norm(hidden_without - y_hidden)),
        "with_history_error_l2": float(np.linalg.norm(hidden_with - y_hidden)),
        "unobserved_history_norm": float(np.linalg.norm(r_hidden)),
    },
    "fit_dense_vs_fit_residual_identity_max_abs": float(dense_residual_difference_error),
    "different_reader_difference_identity_max_abs": float(mixed_identity_error),
    "temporal_dictionary_without_history_base": {
        "current_sparse_rows": same_current.ravel().tolist(),
        "anchor_sparse_rows": different_anchor.ravel().tolist(),
        "probe_rows": temporal_probe.tolist(),
        "current_shared_affine_output": current_output.ravel().tolist(),
        "temporal_dictionary_output": temporal_output.ravel().tolist(),
        "temporal_target_max_abs_error": float(np.max(np.abs(temporal_output - temporal_target))),
        "output_difference_l2": float(np.linalg.norm(temporal_output - current_output)),
        "scope": "Synthetic readout construction, zero history base; the temporal feature dictionary itself preserves query-wise anchor information."
    },
    "fixed_route_standardized_affine_regression": {
        "feature_invariance_max_abs": float(feature_invariance_error),
        "value_affine_equivariance_max_abs": float(affine_error),
        "anchor_l2_error": float(np.linalg.norm(fit_a - y_a)),
        "hit_l2_error": float(np.linalg.norm(fit_t - y_t)),
    },
    "two_key_attention_witness": two_key,
    "three_key_constant_coverage_attention_witness": three_key,
}
for error in [identity_error, equivalent_error, ridge_identity_error,
              dense_residual_difference_error, mixed_identity_error,
              affine_error, feature_invariance_error,
              two_key["history_transport_max_abs_error"],
              three_key["history_transport_max_abs_error"]]:
    assert error < 1e-12, error
assert out["hidden_history"]["with_history_error_l2"] < 1e-12
assert out["hidden_history"]["no_history_error_l2"] > 1
assert out["temporal_dictionary_without_history_base"]["temporal_target_max_abs_error"] < 1e-12
assert out["temporal_dictionary_without_history_base"]["output_difference_l2"] > .5
assert three_key["current_affine_fit_l2_error"] > .1
(ROOT / "EQUIVALENCE_VERIFIED.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
