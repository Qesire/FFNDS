"""Recompute finite update geometry and check task-sign identities.

Real values come only from the archived job 201276 Grams. Synthetic witnesses
are separately labeled; no model inference or output-interpolation experiment
is represented by this script.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def update_geometry(g, background=False):
    g = np.asarray(g, dtype=np.float64)
    rr, tt = g[0, 0], g[1, 1]
    d2 = rr + tt - 2 * g[0, 1]
    raw_dot_d = g[0, 1] - rr
    if background:
        rr += g[2, 2] + 2 * g[0, 2]
        tt += g[2, 2] + 2 * g[1, 2]
        raw_dot_d += g[1, 2] - g[0, 2]
    delta = 2 * raw_dot_d + d2
    assert np.isclose(delta, tt - rr, rtol=1e-10, atol=1e-9)
    denominator = np.sqrt(rr * d2)
    cosine = raw_dot_d / denominator
    threshold = -0.5 * np.sqrt(d2 / rr)
    margin = -delta / (2 * denominator)
    assert np.isclose(margin, threshold - cosine, rtol=1e-10, atol=1e-12)
    return dict(raw_energy=float(rr), transport_energy=float(tt), ratio=float(tt / rr),
                update_energy=float(d2), reference_dot_update=float(raw_dot_d),
                update_cosine=float(cosine), benefit_cosine_threshold=float(threshold),
                energy_change=float(delta), normalized_sign_margin=float(margin),
                margin_scope="Euclidean error-ball sensitivity with the observed update fixed; not a sampling uncertainty or generalization bound")


def algebraic_witnesses():
    rng = np.random.default_rng(20261126)
    n, p, f, c = 17, 5, 3, 4
    ids = np.array([0, 3, 7, 9, 15])
    P = np.eye(n)[ids]
    mask = np.eye(n)
    mask[ids, ids] = 0
    psi = rng.normal(size=(n, f))
    xp = P @ psi
    K = psi @ np.linalg.solve(xp.T @ xp + 0.1 * p * np.eye(f), xp.T)
    L = mask @ (np.eye(n) - K @ P)
    truth, base, cache = (rng.normal(size=(n, c)) for _ in range(3))
    da, ds = (np.exp(rng.normal(size=c)) for _ in range(2))

    def predict(past):
        z = base + past + K @ (P @ (truth - base - past))
        z[ids] = truth[ids]
        return z

    raw = predict(cache * da)
    trans = predict(cache * ds)
    known_update = L @ (cache * (ds - da))
    assert np.max(np.abs(trans - raw - known_update)) < 1e-12
    assert np.max(np.abs(P @ known_update)) < 1e-12
    commutation = np.max(np.abs((L @ cache) * ds - L @ (cache * ds)))
    # Existing teacher and fixed features suffice for any prescribed alpha;
    # this does not identify which alpha benefits an unobserved target.
    alpha = 0.37
    interpolated = predict(cache * da + alpha * cache * (ds - da))
    assert np.max(np.abs(interpolated - (raw + alpha * known_update))) < 1e-12

    # A compressed-observation fiber witness. It is not a claim that a real
    # network can change its output while its entire Q/K/V input is fixed.
    observer = np.array([[1.0, 0.0]])
    update = np.array([0.0, 1.0])
    errors = (np.array([0.0, -1.0]), np.array([0.0, 1.0]))
    changes = [float(2 * e @ update + update @ update) for e in errors]
    assert np.array_equal(observer @ errors[0], observer @ errors[1])
    assert changes[0] < 0 < changes[1]

    # Scalar benefit observability is weaker than full output observability.
    U = np.eye(2)
    J = np.eye(2)
    observed_update = np.array([1.0, 0.0])
    kernel_mode = np.array([0.0, 1.0])
    functional = observed_update @ J.T @ J @ U
    assert np.allclose(functional, observer[0])
    assert np.linalg.norm(J @ U @ kernel_mode) == 1
    assert functional @ kernel_mode == 0

    return dict(scope="Synthetic algebraic witnesses, not real model measurements",
                cache_update_identity_max_abs=float(np.max(np.abs(trans - raw - known_update))),
                probe_null_update_max_abs=float(np.max(np.abs(P @ known_update))),
                query_channel_commutation_max_abs=float(commutation),
                prescribed_alpha_identity_max_abs=float(np.max(np.abs(interpolated - (raw + alpha * known_update)))),
                compressed_observation_counterexample_energy_changes=changes,
                scalar_task_sign_observed_while_full_output_unobserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("TASK_SIGN_VERIFIED.json"))
    args = parser.parse_args()
    raw, digest = load(args.raw)
    result = dict(job=raw["job"], receipt_sha256_uncompressed=digest,
                  source_commit="6baa2a44ffaac5cd90d9a4f4df81809a651a784d",
                  independent_sources=2, nested_sites=16, sources={})
    for sid, source in raw["sources"].items():
        result["sources"][sid] = {}
        for key, site in source["sites"].items():
            tail = site["wan_wide8192_tail"]
            before = np.asarray(site["innovation_decomposition"]["route_shape_scale_gram_physical"])
            after = np.asarray(site["innovation_decomposition"]["route_shape_scale_post_fit_gram_physical"])
            stage = {name: update_geometry(g) for name, g in tail["stage_gram_raw_transport"].items()}
            dense = {name: update_geometry(g, background=True) for name, g in tail["stage_gram_raw_transport_background"].items()}
            result["sources"][sid][key] = dict(stages=stage, dense=dense,
                component_remaining_energy_ratio=dict(zip(("route", "shape", "scale"), map(float, after.diagonal() / before.diagonal()))),
                postfit_scale_penalty_fraction_of_raw=float((after[2, 2] + 2 * after[:2, 2].sum()) / after.sum()))
    result["synthetic_math"] = algebraic_witnesses()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(job=raw["job"], sources=2, sites=16, out=str(args.out), synthetic=result["synthetic_math"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
