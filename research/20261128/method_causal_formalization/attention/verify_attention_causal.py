"""CPU algebra certificates using frozen CAL transforms, not model experiments.

Actual model data: P_v, high/low feature transforms, regularization recipes.
Synthetic operands: Q/K/V, current feature rows, coefficients, state perturbations.
No generation, GPU run, calibration refit or quality prediction is performed.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).copy()
    bits += np.uint32(0x7FFF) + ((bits >> 16) & 1)
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32).astype(np.float64)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)


def maxabs(x):
    return float(np.max(np.abs(x)))


def rank(x):
    return int(np.linalg.matrix_rank(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal-assets", type=Path,
        help="Portable CAL_ASSETS.json bundle; repository default is ../../graph_state_theory/CAL_ASSETS.json")
    parser.add_argument("--cal-root", type=Path,
        help="Alternative materialized CAL root with plan_assets/ and calibration/")
    parser.add_argument("--input-manifest", type=Path, default=HERE/"ATTENTION_INPUT_MANIFEST.json")
    parser.add_argument("--source-root", type=Path,
        help="Optional repository checkout for explicit source fingerprint verification")
    parser.add_argument("--out", type=Path, default=HERE/"ATTENTION_CAUSAL_CERTIFICATES.json")
    args = parser.parse_args()
    manifest = json.loads(args.input_manifest.read_text())
    if args.cal_root is not None and args.cal_assets is not None:
        parser.error("Choose --cal-root or --cal-assets")
    if args.cal_root is None and args.cal_assets is None:
        candidates = [HERE.parents[1]/"graph_state_theory/CAL_ASSETS.json",
                      HERE.parents[1]/"round20_theory/CAL_ASSETS.json"]
        args.cal_assets = next((p for p in candidates if p.is_file()), None)
        if args.cal_assets is None:
            parser.error("Supply --cal-assets or --cal-root")
    assets = {}
    if args.cal_assets is not None:
        packed = json.loads(args.cal_assets.read_text())
        for row in packed["files"]:
            content = base64.b64decode(row["base64"], validate=True)
            assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]
            assets[row["path"]] = content

    def cal_bytes(relative):
        return ((args.cal_root/relative).read_bytes() if args.cal_root is not None else assets[relative])

    rng = np.random.default_rng(manifest["seed"])
    report = dict(
        scope="Exact frozen CAL bytes plus explicitly synthetic algebra operands; no model experiment",
        source_evidence_head=manifest["source_evidence_head"],
        model_forwards=0,
        gpu_runs=0,
        sources=manifest["source_fingerprints"],
        source_bytes_rechecked=args.source_root is not None,
        input_manifest_sha256=sha(args.input_manifest),
        calibration={},
    )
    if args.source_root is not None:
        for row in report["sources"].values():
            path = args.source_root/row["repo_path"]
            assert sha(path) == row["sha256"], path
    pv_bytes = cal_bytes("plan_assets/bases/P_v.npy")
    pv_sha = hashlib.sha256(pv_bytes).hexdigest()
    assert pv_sha == manifest["pv_sha256"]
    # Production explicitly casts the stored NumPy P_v to float32.
    pv = np.load(io.BytesIO(pv_bytes))[:, :64].astype(np.float32).astype(np.float64)
    readers = {}
    for expert, key in (("high", "out_current"), ("low", "out_delta")):
        transform_bytes = cal_bytes(f"calibration/self_{expert}_transform.npz")
        transform_sha = hashlib.sha256(transform_bytes).hexdigest()
        recipe = manifest["frozen_regularization"][expert]
        assert transform_sha == recipe["transform_sha256"]
        with np.load(io.BytesIO(transform_bytes)) as data:
            basis = data[f"{key}_basis"].astype(np.float32).astype(np.float64)
            spectrum = data[f"{key}_spectrum"].astype(np.float64)
            sr = float(data["output_rms"]) * np.sqrt(128)
        effective = pv @ basis
        values = np.linalg.svd(effective, compute_uv=False)
        readers[expert] = (pv, basis, effective)
        report["calibration"][expert] = dict(
            transform_sha256=transform_sha, pv_sha256=pv_sha,
            shape=list(effective.shape), effective_rank=rank(effective),
            sigma_max=float(values[0]), sigma_min=float(values[-1]),
            condition_number=float(values[0] / values[-1]),
            scale_r=sr, lambda_value=recipe["lambda_value"], alpha=recipe["alpha"],
            recipe_sha256=recipe["recipe_sha256"],
            regularization_source="Validated recipe values copied to input manifest; portable bundle contains transform bytes",
            standardized_feature_variance_retained_31_of_64=float((spectrum[:31] ** 2).sum() / (spectrum ** 2).sum()),
            variance_scope="CAL normalized feature Gram energy; not output residual, low history or video quality energy",
        )

    # Ridge certificate with the actual feature matrices, synthetic current rows.
    n, p, d, r, alpha = 193, 64, 128, 31, 6.4
    selected = rng.normal(size=(n, d))
    phi = bf16((selected @ readers["high"][0]) @ readers["high"][1])
    x = np.concatenate([np.ones((n, 1)), phi], axis=1)
    xp = x[:p]
    ridge = np.linalg.solve(xp.T @ xp + alpha * np.eye(r + 1), xp.T)
    theta = rng.normal(size=(r + 1, d))
    residual = x @ theta
    reconstructed = x @ (ridge @ residual[:p])
    bias = x @ (alpha * np.linalg.solve(xp.T @ xp + alpha * np.eye(r + 1), theta))
    constructed_error = residual - reconstructed
    post_error = constructed_error.copy()
    post_error[:p] = 0
    observation = np.zeros((p, n))
    observation[np.arange(p), np.arange(p)] = 1
    l = np.eye(n) - x @ ridge @ observation
    post_l = l.copy()
    post_l[:p] = 0
    invisible = np.zeros((n, d))
    invisible[p + 7, 3] = 1
    report["synthetic_ridge_certificate"] = dict(
        actual_CAL_reader="high", design_shape=list(x.shape), probe_design_rank=rank(xp),
        alpha=alpha, intercept_regularized=True,
        in_feature_space_bias_identity_max_abs=maxabs(constructed_error - bias),
        in_feature_space_post_nonprobe_relative_error=float(np.linalg.norm(post_error) / np.linalg.norm(residual[p:])),
        pre_overwrite_error_operator_rank=rank(l),
        post_overwrite_error_operator_rank=rank(post_l), expected_post_rank=n-p,
        invisible_residual_probe_norm=float(np.linalg.norm(invisible[:p])),
        invisible_residual_unrecovered_norm=float(np.linalg.norm(post_l @ invisible)),
        interpretation="Witnesses ridge bias and unobserved innovation; these errors are not Wan measurements",
    )

    # Exact low decomposition, including arbitrary stored mass and BF16 history.
    # Route is chosen at anchor and then frozen for the current step.
    q0, k0, v0 = (rng.normal(size=(n, d)) for _ in range(3))
    q1, k1, v1 = (a + .2 * rng.normal(size=a.shape) for a in (q0, k0, v0))
    block, top = 16, 3
    nblocks = (n + block - 1) // block
    qp = np.stack([q0[i*block:(i+1)*block].mean(0) for i in range(nblocks)])
    kp = np.stack([k0[i*block:(i+1)*block].mean(0) for i in range(nblocks)])
    routes = np.argsort(-(qp @ kp.T), axis=-1, kind="stable")[:, :top]
    keep = np.zeros((n, n), dtype=bool)
    for qi in range(n):
        keep[qi] = np.isin(np.arange(n) // block, routes[qi // block])

    def fields(q, k, v):
        logits = q @ k.T / np.sqrt(d)
        probabilities = softmax(logits)
        s = softmax(np.where(keep, logits, -np.inf)) @ v
        o = probabilities @ v
        rho = np.where(keep, 0, probabilities).sum(1, keepdims=True)
        mu, sigma = v.mean(0, keepdims=True), v.std(0, keepdims=True)
        z = (v - mu) / sigma
        c = np.where(keep, 0, probabilities) @ z
        return o, s, rho, mu, sigma, c, probabilities, z

    o0, s0, rho0, mu0, sigma0, c0, probs0, z0 = fields(q0, k0, v0)
    o1, s1, rho1, mu1, sigma1, c1, probs1, z1 = fields(q1, k1, v1)
    stored_rho = rho0 + 1e-6 * rng.normal(size=rho0.shape)
    stored_c = bf16((o0 - (1 - stored_rho) * s0 - stored_rho * mu0) / sigma0)
    base = (1 - stored_rho) * s1 + stored_rho * mu1 + stored_c * sigma1
    defect = (rho1 - stored_rho) * (mu1 - s1) + (c1 - stored_c) * sigma1
    probability_drift = np.where(keep, 0, probs1 - probs0) @ z0
    standardized_value_drift = np.where(keep, 0, probs1) @ (z1 - z0)
    # Sufficient transport condition: same attention probabilities and global
    # positive diagonal affine transform of values, so standardized values match.
    diagonal = rng.uniform(.4, 2., size=(1, d))
    shift = rng.normal(size=(1, d))
    v_affine = v0 * diagonal + shift
    oa, sa, rhoa, mua, sigmaa, ca, _, _ = fields(q0, k0, v_affine)
    transport = (1-rho0) * sa + rho0 * mua + c0 * sigmaa
    report["synthetic_low_transport_certificate"] = dict(
        support_chosen_at_anchor=True, current_route_recomputed=False,
        stored_mass_and_quantized_history_defect_identity_max_abs=maxabs((o1-base)-defect),
        centered_drift_probability_plus_value_identity_max_abs=maxabs((c1-c0)-probability_drift-standardized_value_drift),
        positive_diagonal_affine_value_transport_max_abs=maxabs(oa-transport),
        note="Synthetic softmax operands verify identities only; stored rho perturbation is not an observed runtime deviation",
    )

    # Nonanticipative fixed encoder: G_t has a changing graph subspace.
    gamma1, gamma2 = (rng.normal(size=(r, d)) for _ in range(2))
    sigma1_, sigma2_ = (rng.uniform(.4, 2., size=(d,)) for _ in range(2))
    g1 = np.concatenate([np.diag(sigma1_), -gamma1], axis=0)
    g2 = np.concatenate([np.diag(sigma2_), -gamma2], axis=0)
    m1, m2 = gamma1 / sigma1_[None], gamma2 / sigma2_[None]
    union = np.concatenate([g1, g2], axis=1)
    encoder, _ = np.linalg.qr(rng.normal(size=(d+r, 64)))
    residual_reader = g1 - encoder @ (encoder.T @ g1)
    u, _, _ = np.linalg.svd(residual_reader, full_matrices=False)
    missed = u[:, 0]
    report["synthetic_causal_state_certificate"] = dict(
        state_width=d+r, output_width=d, fixed_encoder_rank=64,
        single_future_reader_rank=rank(g1), two_future_union_rank=rank(union),
        graph_subspace_formula_rhs=d+rank(m2-m1),
        invisible_state_encoding_max_abs=maxabs(missed @ encoder),
        invisible_state_output_norm=float(np.linalg.norm(missed @ g1)),
        same_normalized_reader_union_rank=rank(np.concatenate([g1, np.concatenate([np.diag(sigma2_), -m1*sigma2_[None]], axis=0)], axis=1)),
        interpretation="Existence/rank certificate, not the measured rank of current Wan CAL trajectories",
    )

    N, H, D, P, layers = 31668, 40, 128, 64, 40
    partial = N % 128
    full_keys = 31*128
    tail_keys = full_keys-(128-partial)
    raw = H*N*(D+31)*2 + H*N*4
    code_rank = 64
    code = H*(N-P)*code_rank*2
    sidecar = H*P*(D+31)*2
    basis_bytes = H*(D+31)*code_rank*2
    encoded = code+sidecar+basis_bytes+H*N*4+P*8
    report["production_static_counts"] = dict(
        N=N,H=H,D=D,model_width=H*D,probe_rows=P,feature_width=31,design_width=32,
        query_block_size=128,key_blocks=248,selected_key_blocks=31,last_key_block_rows=partial,
        selected_valid_keys_range=[tail_keys,full_keys],
        selected_key_fraction_range=[tail_keys/N,full_keys/N],
        P_fraction=P/N, denoising_steps=4, layers_per_step=40,
        anchor_sites=80,hit_sites=80,probe_query_rows=80*P,
        raw_low_history_payload_bytes_per_layer=raw,
        raw_low_history_payload_GiB_40_layers=raw*layers/2**30,
        joint64_payload_bytes_per_layer_including_probe_ids=encoded,
        joint64_payload_GiB_40_layers=encoded*layers/2**30,
        payload_scope="rho + C/A or code + basis + exact probe C/A; excludes routes, QKV, model weights, temporaries and allocator",
        baseline_attention_core_work_fraction_range=[.5+.75*tail_keys/N+.5*P/N,.5+.75*full_keys/N+.5*P/N],
        ideal_partition_attention_core_work_fraction_range=[.5+.5*tail_keys/N+.5*P/N,.5+.5*full_keys/N+.5*P/N],
        work_scope="QK/AV multiply-add proxy only. Frozen graph: 80 full anchors + 120 selected calls + 80 P64 full-key teachers vs 160 full attention; shared partition removes 40 duplicated selected-anchor scans. Not end-to-end or latency prediction.",
        wide8192_semantics="FFN token tile size, not attention width or support size",
    )
    checks = {
        "ridge_identity": report["synthetic_ridge_certificate"]["in_feature_space_bias_identity_max_abs"] < 1e-10,
        "ridge_post_rank": report["synthetic_ridge_certificate"]["post_overwrite_error_operator_rank"] == n-p,
        "low_defect_identity": report["synthetic_low_transport_certificate"]["stored_mass_and_quantized_history_defect_identity_max_abs"] < 1e-10,
        "low_drift_identity": report["synthetic_low_transport_certificate"]["centered_drift_probability_plus_value_identity_max_abs"] < 1e-10,
        "affine_transport_identity": report["synthetic_low_transport_certificate"]["positive_diagonal_affine_value_transport_max_abs"] < 1e-10,
        "joint_rank_identity": report["synthetic_causal_state_certificate"]["two_future_union_rank"] == report["synthetic_causal_state_certificate"]["graph_subspace_formula_rhs"],
        "fixed_encoder_invisible_direction": report["synthetic_causal_state_certificate"]["invisible_state_encoding_max_abs"] < 1e-10,
    }
    assert all(checks.values()), checks
    report["algebra_checks"] = checks
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
