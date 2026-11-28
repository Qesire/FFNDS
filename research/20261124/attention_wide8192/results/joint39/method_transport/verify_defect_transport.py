"""Finite attention-defect transport: exact algebra and explicit synthetic witnesses.

No Wan activation, model inference, GPU or performance measurement is used.
The optional archived P_k supplies only a real 128-dimensional grouping direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x - x.max(axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def groups_from_keys(k: np.ndarray, block: int, direction: np.ndarray):
    groups = []
    parents = []
    for b, start in enumerate(range(0, len(k), block)):
        ids = np.arange(start, min(start + block, len(k)))
        z = (k[ids] - k[ids].mean(axis=0)) @ direction
        for mask in (z >= 0, z < 0):
            if np.any(mask):
                groups.append(ids[mask])
                parents.append(b)
    return groups, np.asarray(parents)


def operators(q, k, groups, parents, selected):
    """Dense probability matrices are materialized only in this small CPU witness."""
    exact_logits = q @ k.T / np.sqrt(k.shape[-1])
    approx_logits = np.zeros_like(exact_logits)
    for ids, parent in zip(groups, parents):
        mean_logit = q @ k[ids].mean(axis=0) / np.sqrt(k.shape[-1])
        approx_logits[:, ids] = mean_logit[:, None]
        keep = np.asarray([parent in row for row in selected])
        approx_logits[np.ix_(keep, ids)] = exact_logits[np.ix_(keep, ids)]
    return softmax(exact_logits), softmax(approx_logits)


def value_coordinates(v):
    mu = v.mean(axis=0)
    centered = v - mu
    # Exact constant columns have zero defect. Avoid introducing an arbitrary floor.
    zero = np.ptp(v, axis=0) == 0
    sigma = np.sqrt(np.mean(centered * centered, axis=0))
    sigma[zero] = 0.0
    standardized = np.divide(centered, sigma, out=np.zeros_like(v), where=sigma > 0)
    return mu, sigma, standardized


def mean_observation(v, groups):
    counts = np.asarray([len(g) for g in groups], dtype=np.float64)
    w = counts / counts.sum()
    means = np.stack([v[g].mean(axis=0) for g in groups])
    mu = w @ means
    return np.sqrt(w)[:, None] * (means - mu), counts


def maxabs(a):
    return float(np.max(np.abs(a)))


def run(basis_path: Path | None):
    rng = np.random.default_rng(251125)
    nq, nk, d, block = 96, 768, 128, 8
    direction = rng.normal(size=d)
    basis_evidence = None
    if basis_path is not None:
        raw = basis_path.read_bytes()
        p = np.load(basis_path)
        assert p.shape[0] == d
        direction = p[:, 0].astype(np.float64)
        basis_evidence = {
            'path': str(basis_path),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'shape': list(p.shape),
            'use': 'Grouping direction only; all Q/K/V below are synthetic.'
        }
    direction /= np.linalg.norm(direction)
    feature_rng = np.random.default_rng(251126)
    pk_features, _ = np.linalg.qr(feature_rng.normal(size=(d, 15)))
    pv_features, _ = np.linalg.qr(feature_rng.normal(size=(d, 15)))
    q = rng.normal(size=(nq, d))
    k = rng.normal(size=(nk, d))
    v = rng.normal(size=(nk, d))
    groups, parents = groups_from_keys(k, block, direction)
    selected = [{i % (nk // block), (i + 17) % (nk // block)} for i in range(nq)]
    p, p0 = operators(q, k, groups, parents, selected)
    r = p - p0
    a, g = p @ v, p0 @ v
    e = a - g
    _, sigma, vs = value_coordinates(v)
    c = np.divide(e, sigma, out=np.zeros_like(e), where=sigma > 0)

    # General finite routing/value split with genuinely changed effective Q/K.
    q1 = q + .12 * rng.normal(size=q.shape)
    k1 = k + .08 * rng.normal(size=k.shape)
    v1 = v + .15 * rng.normal(size=v.shape)
    p1, p01 = operators(q1, k1, groups, parents, selected)
    r1 = p1 - p01
    e1 = (p1 - p01) @ v1
    finite_error = maxabs(e1 - e - ((r1-r) @ v1 + r @ (v1-v)))
    _, sigma1, vs1 = value_coordinates(v1)
    c1 = r1 @ vs1
    normalized_split = maxabs(c1-c-((r1-r)@vs1+r@(vs1-vs)))

    # A large finite Q/K dual change plus key translation leaves both routes unchanged.
    rot, _ = np.linalg.qr(rng.normal(size=(d, d)))
    l = rot @ np.diag(np.linspace(.7, 1.4, d))
    q2 = np.linalg.solve(l, q.T).T
    k2 = k @ l + rng.normal(size=(1, d)) * .2
    p2, p02 = operators(q2, k2, groups, parents, selected)
    route_exact = max(maxabs(p2-p), maxabs(p02-p0))

    # Per-channel positive affine value evolution; normalized defect is unchanged.
    gains = np.geomspace(.2, 5, d)
    shift = rng.normal(size=d) * .7
    v2 = v * gains + shift
    a2, g2 = p2 @ v2, p02 @ v2
    _, sigma2, vs2 = value_coordinates(v2)
    raw_prediction = g2 + e
    norm_prediction = g2 + c * sigma2
    normalized_exact = maxabs(norm_prediction-a2)
    raw_error = float(np.linalg.norm(raw_prediction-a2))

    # Entire current-feature/ridge method: independent positive affine changes
    # at anchor and hit preserve both normalized states separately.
    selected_mask = np.asarray([
        [j//block in row for j in range(nk)] for row in selected
    ], dtype=bool)
    probes = np.arange(64)
    def candidate(v_anchor, v_hit, pp, pp0):
        _, sa, _ = value_coordinates(v_anchor)
        ca = np.divide(r@v_anchor, sa, out=np.zeros((nq,d)), where=sa>0)
        _, sh, _ = value_coordinates(v_hit)
        ah, gh = pp@v_hit, pp0@v_hit
        selected_weight = pp0*selected_mask
        mass = selected_weight.sum(axis=1, keepdims=True)
        bh = (selected_weight/mass)@v_hit
        contrast = np.divide(gh-bh, sh, out=np.zeros_like(gh), where=sh>0)
        psi = np.column_stack((np.ones(nq),q1@pk_features/np.sqrt(d),contrast@pv_features,mass))
        cs = np.divide(ah-gh, sh, out=np.zeros_like(ah), where=sh>0)
        target = cs[probes]-ca[probes]
        xp = psi[probes]
        coef = np.linalg.solve(xp.T@xp+6.4*np.eye(32),xp.T@target)
        result = gh+(ca+psi@coef)*sh
        result[probes] = ah[probes]
        return result,ca,psi,coef
    base_pred, base_cache, base_psi, base_coef = candidate(v,v1,p1,p01)
    gain_a=np.geomspace(.3,2.,d)
    gain_h=np.geomspace(.8,3.,d)[::-1]
    gauge_rng=np.random.default_rng(251127)
    shift_a=gauge_rng.normal(size=d)*.5
    shift_h=gauge_rng.normal(size=d)*.9
    gauge_pred,gauge_cache,gauge_psi,gauge_coef=candidate(
        v*gain_a+shift_a,v1*gain_h+shift_h,p1,p01
    )

    # Arbitrary affine channel transport inferred from group means.
    t = np.eye(d) + rng.normal(size=(d, d)) * (.2 / np.sqrt(d))
    b = rng.normal(size=d)
    vt = v @ t + b
    x, counts = mean_observation(v, groups)
    y, _ = mean_observation(vt, groups)
    xp = np.linalg.pinv(x)
    null_completion = np.eye(d) - xp @ x
    that = null_completion + xp @ y
    at, gt = p2 @ vt, p02 @ vt
    transport_exact = maxabs(gt + e @ that-at)
    singular = np.linalg.svd(x, compute_uv=False)

    # Finite residual formula for an imperfect affine model and changed routing.
    rem = .05 * rng.normal(size=v.shape)
    vm = v @ t + b + rem
    y_m, _ = mean_observation(vm, groups)
    that_m = null_completion + xp @ y_m
    em = r1 @ vm
    model_identity = maxabs(em-e@t-((r1-r)@vm+r@rem))
    obs_noise = y_m-x@t
    action_error_identity = maxabs(
        e @ (that_m-t) - (e @ null_completion @ (np.eye(d)-t) + e @ xp @ obs_noise)
    )

    # Target action can be identified although the transport matrix itself is not.
    xr = np.array([[1., 0.], [-1., 0.]])
    er = np.array([[2., 0.], [-3., 0.]])
    tr = np.array([[2., 3.], [7., 11.]])
    xrp = np.linalg.pinv(xr)
    tr_hat = np.eye(2)-xrp@xr+xrp@(xr@tr)
    rank_def_target = maxabs(er@tr_hat-er@tr)
    transport_parameter_difference = float(np.linalg.norm(tr_hat-tr))

    # An observable-space miss gives two identical summaries but different actions.
    er_miss = np.array([[0., 1.]])
    tr2 = tr.copy()
    tr2[1] += np.array([5., -4.])
    ambiguous_summaries = maxabs(xr@tr-xr@tr2)
    ambiguous_target = float(np.linalg.norm(er_miss@tr-er_miss@tr2))

    # Zero-scale and large-common-mean branches are distinguished explicitly.
    vzero = v.copy()
    vzero[:, 0] = 3.
    vzero[:, 1] = 0.
    _, szero, vz = value_coordinates(vzero)
    ezero = r @ (vzero-vzero.mean(axis=0))
    czero = r @ vz
    zero_exact = maxabs(ezero[:, :2])
    zero_reconstruction = maxabs(ezero-czero*szero)

    # Symbolic finite transport identity; zero-row-sum is imposed algebraically.
    ra, rb, rc, rd = sp.symbols('ra rb rc rd')
    rr = sp.Matrix([[ra, rb, -ra-rb], [rc, rd, -rc-rd]])
    s1, s2, s3, s4 = sp.symbols('s1 s2 s3 s4')
    rr1 = sp.Matrix([[s1, s2, -s1-s2], [s3, s4, -s3-s4]])
    vv = sp.Matrix(3, 2, sp.symbols('v:6'))
    tt = sp.Matrix(2, 2, sp.symbols('t:4'))
    bb = sp.Matrix(1, 2, sp.symbols('b:2'))
    ee = sp.Matrix(3, 2, sp.symbols('u:6'))
    vnext = vv*tt + sp.ones(3, 1)*bb + ee
    symbolic = all(sp.simplify(z) == 0 for z in rr1*vnext-rr*vv*tt-(rr1-rr)*vnext-rr*ee)

    # Actual model-shape arithmetic; dense operators above are not production paths.
    n, h, dh, layers, hits = 31668, 40, 128, 40, 80
    groups_actual = 2 * ((n + 127)//128)
    full_pair_mac = 2*n*n*dh*h
    fit_mac = h*dh*groups_actual*dh
    apply_mac = n*h*dh*dh
    bytes_inverse = layers*h*dh*groups_actual*4
    bytes_null = layers*h*dh*dh*4
    baseline_new_cache = layers*n*h*dh*2 + layers*n*h
    budgets = {
        'dimensions': {'tokens':n, 'heads':h, 'head_channels':dh, 'layers':layers, 'anchor_sites_per_fourstep':80, 'hit_sites_per_fourstep':hits, 'group_slots_per_head':groups_actual},
        'normalized_defect_cache_extra_bytes':0,
        'normalized_defect_optional_anchor_scales_fp32_bytes':layers*h*dh*4,
        'normalized_hit_decode_scalar_multiplies_per_layer':n*h*dh,
        'T_fit_MAC_per_layer':fit_mac,
        'Ea_T_MAC_per_layer':apply_mac,
        'T_fit_and_apply_over_full_attention_pair_MAC':(fit_mac+apply_mac)/full_pair_mac,
        'T_fit_and_apply_TFLOP_per_fourstep':2*(fit_mac+apply_mac)*hits/1e12,
        'anchor_factorization_count_per_fourstep':80*h,
        'anchor_factorization_shape':[groups_actual,dh],
        'full_rank_QR_explicit_Q_and_triangular_solve_major_TFLOP_per_fourstep':
            (5*groups_actual*dh**2-4*dh**3/3)*h*80/1e12,
        'rank_deficient_factorization_note':
            'Rank-revealing SVD/QR and null-space completion have algorithm-dependent setup costs; count 3200 small decompositions, not free setup.',
        'L_anchor_fp32_GiB_40layers':bytes_inverse/2**30,
        'P_null_anchor_fp32_GiB_40layers':bytes_null/2**30,
        'E_and_group_labels_GiB_40layers':baseline_new_cache/2**30,
        'E_labels_L_GiB_40layers':(baseline_new_cache+bytes_inverse)/2**30,
        'E_labels_L_Pnull_GiB_40layers':(baseline_new_cache+bytes_inverse+bytes_null)/2**30,
        'T_fp32_scratch_MiB_per_layer':h*dh*dh*4/2**20,
        'T_transport_read_E_BF16_GiB_per_fourstep':hits*n*h*dh*2/2**30,
        'T_transport_write_if_materialized_FP32_GiB_per_fourstep':hits*n*h*dh*4/2**30,
        'scope':'Arithmetic and tensor payload only; includes no kernel/launch/QR timing, peak prediction, or GPU speed measurement.'
    }
    source_candidates = [Path(__file__).parent/'source'/'attention_online_structured.py',
                         Path(__file__).parent.parent/'attention_online_structured.py']
    source = next((p for p in source_candidates if p.exists()), source_candidates[0])
    source_audit = {
        'local_path':str(source),
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
        'rho_semantics':'rho_a=1-Z_selected/Z_full; per-query omitted probability mass.',
        'u0_semantics':'u0=A_a-(1-rho_a)*B_a; stored BF16.',
        'scale_r_semantics':'Fixed CAL fit scale; not current/anchor V standard deviation transport.',
        'hit_base':'(1-rho_a)*B_s+u0',
    }
    checks = {
        'signed_row_mass_max_abs': maxabs(r.sum(axis=1)),
        'finite_route_value_split_max_abs':finite_error,
        'standardized_route_shape_split_max_abs':normalized_split,
        'dual_QK_with_key_translation_probability_max_abs':route_exact,
        'positive_diagonal_affine_normalized_prediction_max_abs':normalized_exact,
        'same_case_raw_defect_prediction_frobenius_error':raw_error,
        'standardized_values_positive_diagonal_invariance_max_abs':maxabs(vs2-vs),
        'independent_anchor_hit_positive_affine_cache_max_abs':maxabs(gauge_cache-base_cache),
        'independent_anchor_hit_positive_affine_features_max_abs':maxabs(gauge_psi-base_psi),
        'independent_anchor_hit_positive_affine_coefficients_max_abs':maxabs(gauge_coef-base_coef),
        'independent_anchor_hit_positive_affine_final_output_max_abs':maxabs(gauge_pred-base_pred*gain_h-shift_h),
        'full_T_group_mean_observation_rank':int(np.linalg.matrix_rank(x)),
        'full_T_group_mean_observation_sigma_min':float(singular[-1]),
        'full_T_group_mean_observation_sigma_max':float(singular[0]),
        'full_T_group_mean_observation_condition':float(singular[0]/singular[-1]),
        'full_T_parameter_max_abs':maxabs(that-t),
        'full_T_prediction_max_abs':transport_exact,
        'imperfect_affine_model_finite_identity_max_abs':model_identity,
        'target_action_estimation_identity_max_abs':action_error_identity,
        'target_sensitivity_norm_E_L':float(np.linalg.norm(e@xp,2)),
        'generic_bound_norm_E_over_sigma_min_X':float(np.linalg.norm(e,2)/singular[-1]),
        'rank_deficient_observable_action_max_abs':rank_def_target,
        'rank_deficient_transport_parameter_difference':transport_parameter_difference,
        'unobserved_direction_two_summary_difference':ambiguous_summaries,
        'unobserved_direction_two_target_difference':ambiguous_target,
        'zero_scale_columns_defect_max_abs':zero_exact,
        'zero_scale_decoding_identity_max_abs':zero_reconstruction,
        'standardized_defect_max_abs':maxabs(c),
        'standardized_defect_bound_sqrt_2Nk':float(np.sqrt(2*nk)),
        'symbolic_finite_transport_identity':bool(symbolic),
    }
    exact_checks=[v for key,v in checks.items() if key.endswith('_max_abs') and 'standardized_defect_max_abs'!=key]
    assert max(exact_checks)<1e-10, checks
    assert symbolic
    assert raw_error>1e-3
    assert ambiguous_target>1 and ambiguous_summaries==0
    assert maxabs(c)<=np.sqrt(2*nk)+1e-12
    return {
        'date':'2026-11-25',
        'evidence_commit':'7c8196fa5380dbe67dd9772f942553085fde55b3',
        'scope':'Mathematical identities and explicitly synthetic Q/K/V. Archived P_k is only a direction. No Wan activations, GPU inference, video or latency result.',
        'synthetic_shape':{'query_rows':nq,'key_rows':nk,'head_channels':d,'block_size':block,'nonempty_groups':len(groups)},
        'archived_direction':basis_evidence,
        'source_audit':source_audit,
        'checks':checks,
        'actual_shape_budget':budgets,
    }


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--basis', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('defect_transport_verified.json'))
    args=parser.parse_args()
    result=run(args.basis)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
