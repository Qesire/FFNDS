"""CPU witnesses for the recovery interface and speedup composition.

The arrays below are explicitly synthetic. No model or performance run occurs.
Source hashes bind the implementation semantics inspected in the report.
"""
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np


def gm(x):
    return float(np.exp(np.log(np.asarray(x, dtype=float)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('attention-code', 'centered-code', 'ffn-hit', 'panel-fit'):
        parser.add_argument('--'+label, type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('CORE_COMPOSITION_VERIFIED.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(20261126)
    n, p, d, r, m, k = 257, 64, 128, 32, 16, 2560
    ids = np.sort(rng.choice(n, p, replace=False))
    other = np.setdiff1d(np.arange(n), ids)
    phi = np.column_stack((np.ones(n), rng.normal(size=(n, r-1))))
    phip = phi[ids]
    rho = rng.uniform(-1e-5, 1, (n, 1))
    selected = rng.normal(size=(n, d))
    history = rng.normal(size=(n, d))
    mu = rng.normal(size=(1, d))
    sigma = np.exp(rng.normal(scale=.2, size=(1, d)))
    teacher = rng.normal(size=(p, d))
    alpha = 6.4
    solve = np.linalg.solve(phip.T @ phip + alpha*np.eye(r), phip.T)
    K = phi @ solve

    base = (1-rho)*selected + rho*mu + history*sigma
    theta = solve @ (teacher-base[ids])
    dense = base + phi@theta
    dense[ids] = teacher

    # Decode only P to fit, then evaluate the existing expression by row tiles.
    basep = (1-rho[ids])*selected[ids] + rho[ids]*mu + history[ids]*sigma
    tile_theta = solve @ (teacher-basep)
    tiled = np.empty_like(dense)
    for start in range(0, n, 73):
        ix = slice(start, start+73)
        b = (1-rho[ix])*selected[ix]+rho[ix]*mu+history[ix]*sigma
        tiled[ix] = b+phi[ix]@tile_theta
    tiled[ids] = teacher

    # Equivalent observation lift + residual action, including exact P overwrite.
    H = np.zeros((n, p)); H[ids, np.arange(p)] = 1
    H[other] = K[other]
    L = np.eye(n); L[:, ids] -= K; L[ids] = 0
    composed = H@teacher+L@base
    P = np.zeros((p, n)); P[np.arange(p), ids] = 1

    # A feature-span shift is suppressed according to the existing ridge.
    U = rng.normal(size=(r, d))
    feature_update = L@(phi@U)
    ridge_prediction = alpha*phi@np.linalg.solve(phip.T@phip+alpha*np.eye(r), U)
    ridge_prediction[ids] = 0

    # FFN lift has 16 gain modes + a probe kernel, not the attention ridge.
    Q = rng.normal(size=(n, m)); V = rng.normal(size=(m, d))
    X = rng.normal(size=(n, k)); XP = X[ids]; QP = Q[ids]
    system = XP@XP.T/k + .75*np.eye(p)
    iq = np.linalg.solve(system, QP)
    gain_matrix = (QP.T@iq)*(V@V.T)
    kernel = X@XP.T/k
    def ffn_lift(target):
        gain = np.linalg.solve(gain_matrix, (iq*(target@V.T)).sum(0))
        coeff = np.linalg.solve(system, target-(QP*gain)@V)
        return (Q*gain)@V + .5*kernel@coeff
    t1, t2 = rng.normal(size=(2, p, d))
    ffn_linearity = np.max(np.abs(ffn_lift(t1+2*t2)-ffn_lift(t1)-2*ffn_lift(t2)))

    errors = {
        "probe_first_tiled_vs_fullfield_max_abs": float(np.max(np.abs(tiled-dense))),
        "observation_and_residual_composition_max_abs": float(np.max(np.abs(composed-dense))),
        "partition_of_identity_max_abs": float(np.max(np.abs(H@P+L-np.eye(n)))),
        "ridge_feature_shift_identity_max_abs": float(np.max(np.abs(feature_update-ridge_prediction))),
        "FFN_gain_kernel_lift_linearity_max_abs": float(ffn_linearity),
    }
    assert max(errors.values()) < 1e-11

    # Illustrative timings only: matched geometric means telescope, arithmetic
    # means of per-input speedups do not generally multiply.
    t0 = np.array([80., 160.]); td = np.array([40., 20.]); tdm = np.array([10., 20.])
    sd, sm, st = t0/td, td/tdm, t0/tdm
    composition = {
        "scope": "Synthetic timing example, not a model benchmark",
        "original": t0.tolist(), "distilled": td.tolist(), "distilled_plus_method": tdm.tolist(),
        "per_source_distillation_speedup": sd.tolist(), "per_source_method_speedup": sm.tolist(),
        "per_source_total_speedup": st.tolist(),
        "geomean_distillation": gm(sd), "geomean_method": gm(sm), "geomean_total": gm(st),
        "geomean_product": gm(sd)*gm(sm),
        "arithmetic_mean_speedup_product": float(sd.mean()*sm.mean()),
        "arithmetic_mean_total_speedup": float(st.mean()),
        "ratio_of_time_means_product": float((t0.mean()/td.mean())*(td.mean()/tdm.mean())),
        "ratio_of_time_means_total": float(t0.mean()/tdm.mean()),
    }
    assert np.allclose(sd*sm, st)
    assert np.isclose(composition["geomean_product"], composition["geomean_total"])
    assert not np.isclose(composition["arithmetic_mean_speedup_product"],composition["arithmetic_mean_total_speedup"])

    identity = [{"role": label, "supplied_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for label in ('attention_code', 'centered_code', 'ffn_hit', 'panel_fit')
                if (path := getattr(args,label)) is not None]
    result = {
        "status": "VERIFIED_CPU_ALGEBRA_ONLY",
        "scope": "Synthetic real-arithmetic witnesses at P64, attention rank32, FFN16 modes and selected width2560; no GPU/model/video/timing execution",
        "dimensions": dict(rows=n, probes=p, output_channels=d, attention_features=r, FFN_modes=m, selected_FFN_channels=k),
        "identities": errors, "speedup_composition": composition,
        "execution_shape_budget": {
            "actual_N": 31668, "actual_heads": 40, "actual_head_dim": 128,
            "one_full_FP32_response_bytes": 31668*40*128*4,
            "P64_FP32_response_bytes": 64*40*128*4,
            "tile8192_FP32_response_bytes": 8192*40*128*4,
            "scope": "Static temporary sizes only; not a prediction of peak reduction or latency. Current compiler lifetime/aliasing may already remove some buffers."
        },
        "source_identity": identity,
    }
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ['status','identities','speedup_composition','execution_shape_budget']},indent=2))


if __name__ == '__main__':
    main()
