"""CPU construction checks, exact algebra, and the actual FFNDS shape budget.

The K/V activations below are synthetic. P_k/P_v are real archived FFNDS
projection assets. No new model inference or CUDA execution is performed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import zipfile
import numpy as np
import sympy as sp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(k, block, direction, split):
    out = []
    for b, start in enumerate(range(0, len(k), block)):
        ids = np.arange(start, min(start + block, len(k)))
        if not split:
            out.append((b, ids))
            continue
        centered = k[ids] - k[ids].mean(0)
        side = centered @ direction >= 0
        for keep in (side, ~side):
            if keep.any():
                out.append((b, ids[keep]))
    return out


def hybrid(q, k, v, selected_blocks, block, direction, split, frozen_groups=None):
    """One-head NumPy reference. Retains every query; no N x N matrix."""
    scale = q.shape[-1] ** -0.5
    cells = groups(k, block, direction, split) if frozen_groups is None else frozen_groups
    outputs, lses = [], []
    for i, query in enumerate(q):
        chosen = set(selected_blocks[i])
        logits, values = [], []
        for b, ids in cells:
            if b in chosen:
                # Each key belongs to exactly one cell, so selected keys stay exact.
                logits.extend((k[ids] @ query * scale).tolist())
                values.extend(v[ids])
            else:
                logits.append(float(k[ids].mean(0) @ query * scale + np.log(len(ids))))
                values.append(v[ids].mean(0))
        logits = np.asarray(logits)
        values = np.asarray(values)
        shift = logits.max()
        weights = np.exp(logits - shift)
        outputs.append(weights @ values / weights.sum())
        lses.append(shift + np.log(weights.sum()))
    return np.asarray(outputs), np.asarray(lses)


def dense(q, k, v):
    logits = q @ k.T / np.sqrt(q.shape[-1])
    shift = logits.max(1, keepdims=True)
    weights = np.exp(logits - shift)
    return weights @ v / weights.sum(1, keepdims=True), (shift[:, 0] + np.log(weights.sum(1)))


def finite_error(q, k, v, selected_blocks, block, direction, split, baseline, lse0, frozen_groups=None):
    cells = groups(k, block, direction, split) if frozen_groups is None else frozen_groups
    result, bounds = [], []
    for i, query in enumerate(q):
        num = np.zeros(v.shape[1])
        mass = 0.0
        bound_num = 0.0
        for b, ids in cells:
            if b in set(selected_blocks[i]):
                continue
            kb, vb = k[ids].mean(0), v[ids].mean(0)
            dk, dv = k[ids] - kb, v[ids] - vb
            t = dk @ query / np.sqrt(k.shape[1])
            a = np.expm1(t).mean()
            c = (np.expm1(t)[:, None] * dv).mean(0)
            weight = len(ids) * np.exp(kb @ query / np.sqrt(k.shape[1]) - lse0[i])
            num += weight * (a * (vb - baseline[i]) + c)
            mass += weight * a
            eps = np.max(np.abs(t))
            a_bound = 0.5 * np.exp(eps) * np.mean(t ** 2)
            c_bound = np.expm1(eps) * np.mean(np.linalg.norm(dv, axis=1))
            bound_num += weight * (a_bound * np.linalg.norm(vb-baseline[i]) + c_bound)
        result.append(num / (1 + mass))
        bounds.append(bound_num / (1 + mass))
    return np.asarray(result), np.asarray(bounds)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis-dir', type=Path, help='Directory containing the original P_k.npy/P_v.npy')
    parser.add_argument('--replay-zip', type=Path, default=here.parent/'replay_evidence.zip',
                        help='Existing replay archive; default matches results/joint39 publication location')
    parser.add_argument('--output', type=Path, default=here/'two_group_attention_verified.json')
    args = parser.parse_args()
    if args.basis_dir is not None:
        blobs = {name:(args.basis_dir/name).read_bytes() for name in ('P_k.npy','P_v.npy')}
    else:
        with zipfile.ZipFile(args.replay_zip) as archive:
            blobs = {name:archive.read('plan_assets/bases/'+name) for name in ('P_k.npy','P_v.npy')}
    pk,pv = (np.load(io.BytesIO(blobs[name]),allow_pickle=False) for name in ('P_k.npy','P_v.npy'))
    direction = pk[:, 0].astype(np.float64)
    direction /= np.linalg.norm(direction)
    value_direction = pv[:, 0].astype(np.float64)
    value_direction /= np.linalg.norm(value_direction)
    rng = np.random.default_rng(20261125)

    # Exact affine/projective identities, componentwise for vector outputs.
    z0, z, n0, n, ah, ch = sp.symbols("z0 z n0 n ah ch", positive=True)
    a, c = z/z0-1, (n-n0)/z0
    A0, A = n0/z0, n/z
    lift_identity = sp.simplify((A0+c)/(1+a)-A) == 0
    lifted_error = sp.simplify((A0+ch)/(1+ah)-A-((ch-c)-A*(ah-a))/(1+ah)) == 0
    g, h, gs, hs, vstar, Gs, Ga, Aa = sp.symbols("g h gs hs vstar Gs Ga Aa", real=True)
    bounded_error = sp.expand((vstar+gs*(A0-vstar)+hs)-(vstar+g*(A0-vstar)+h)
                              -((gs-g)*(A0-vstar)+(hs-h))) == 0
    E0, Es = sp.symbols("E0 Es", real=True)
    anchored_identity = sp.expand(Aa+(Gs-Ga)+(Es-E0)-Gs-Es).subs(Aa, Ga+E0) == 0

    # Actual head width and archived direction, with synthetic opposing K/V.
    d = 128
    n_selected = 128
    k = np.vstack([np.zeros((n_selected,d)), 4*direction, -4*direction])
    v = np.vstack([np.zeros((n_selected,d)), value_direction, -value_direction])
    t = np.array([-3., -1., 0., 1., 3.])
    q = t[:, None] * np.sqrt(d) / 4 * direction
    supports = [[0] for _ in q]
    exact, full_lse = dense(q,k,v)
    one, one_lse = hybrid(q,k,v,supports,128,direction,False)
    two, two_lse = hybrid(q,k,v,supports,128,direction,True)
    theory = (2*np.sinh(t)/(n_selected+2*np.cosh(t)))[:,None]*value_direction
    collision = {
        "q_dot_a_over_sqrt_d":t.tolist(),
        "exact_coefficient_along_value_direction":(exact@value_direction).tolist(),
        "one_group_max_output_norm":float(np.linalg.norm(one,axis=1).max()),
        "two_group_max_abs_error":float(np.max(np.abs(two-exact))),
        "closed_form_max_abs_error":float(np.max(np.abs(theory-exact))),
    }

    # A nontrivial random finite problem, with empty groups and incomplete tail.
    q = rng.normal(size=(11,d)) * 1.3
    k = rng.normal(size=(87,d))
    v = .4*k + rng.normal(size=(87,d))
    k[:8] = 0  # One empty sign group; no padded pseudo-token.
    supports = [sorted(rng.choice(11,size=3,replace=False).tolist()) for _ in q]
    exact, full_lse = dense(q,k,v)
    one, one_lse = hybrid(q,k,v,supports,8,direction,False)
    two, two_lse = hybrid(q,k,v,supports,8,direction,True)
    remainder, bound = finite_error(q,k,v,supports,8,direction,True,two,two_lse)
    error_residual = float(np.max(np.abs((exact-two)-remainder)))
    bound_slack = float(np.min(bound-np.linalg.norm(exact-two,axis=1)))
    refinement_margin = float(np.min(two_lse-one_lse))
    jensen_margin = float(np.min(full_lse-two_lse))
    shift = rng.normal(size=d)
    shifted, shifted_lse = hybrid(q,k,v+shift,supports,8,direction,True)
    translation_error = float(np.max(np.abs(shifted-two-shift)))
    permutation = rng.permutation(d)
    transformed,_ = hybrid(q,k,v[:,permutation]*1.7,supports,8,direction,True)
    linear_value_error = float(np.max(np.abs(transformed-two[:,permutation]*1.7)))
    gtrue = np.exp(two_lse-full_lse)
    vstar0 = v.mean(0)
    htrue = exact-vstar0-gtrue[:,None]*(two-vstar0)
    exact_rebuild = vstar0+gtrue[:,None]*(two-vstar0)+htrue
    hshift = (exact+shift)-(vstar0+shift)-gtrue[:,None]*((two+shift)-(vstar0+shift))
    no_division_in_lift = float(np.max(np.abs(exact-exact_rebuild)))
    h_translation_error = float(np.max(np.abs(htrue-hshift)))

    # Between-group covariance is exactly the response retained by grouping.
    between_cov = np.zeros((d,d))
    within_cov = np.zeros((d,d))
    total_cov = (v-v.mean(0)).T @ (k-k.mean(0))/len(k)
    # Use one parent spanning all 87 keys to check the partition theorem.
    for _, ids in groups(k,len(k),direction,True):
        w = len(ids)/len(k)
        km,vm = k[ids].mean(0),v[ids].mean(0)
        between_cov += w*np.outer(vm-v.mean(0),km-k.mean(0))
        within_cov += w*((v[ids]-vm).T@(k[ids]-km)/len(ids))
    covariance_residual = float(np.max(np.abs(total_cov-between_cov-within_cov)))

    # Freeze anchor memberships while recomputing all means from current K/V.
    frozen_cells = groups(k,8,direction,True)
    k_next=k+0.3*rng.normal(size=k.shape)
    v_next=v+0.2*rng.normal(size=v.shape)
    next_exact,_=dense(q,k_next,v_next)
    next_base,next_lse=hybrid(q,k_next,v_next,supports,8,direction,True,frozen_cells)
    next_rem,_=finite_error(q,k_next,v_next,supports,8,direction,True,next_base,next_lse,frozen_cells)
    frozen_partition_residual=float(np.max(np.abs(next_exact-next_base-next_rem)))

    N,H,D,B,selected,L = 31668,40,128,128,31,40
    blocks=math.ceil(N/B); omitted=blocks-selected
    budget = {
        "actual_model_shape":{"N":N,"heads":H,"head_dim":D,"blocks":blocks,"selected_blocks":selected,"last_valid_keys":N%B,"layers":L},
        "two_group_omitted_pair_ratio_vs_full":2*omitted/N,
        "two_group_all_blocks_then_mask_pair_ratio_vs_full":2*blocks/N,
        "two_group_KV_means_FP32_MiB":2*2*blocks*H*D*4/2**20,
        "two_group_valid_counts_int32_MiB":2*blocks*H*4/2**20,
        "all_coarse_score_materialization_FP32_GiB":N*H*2*blocks*4/2**30,
        "P64_lse_FP32_KiB":64*H*4/2**10,
        "P64_output_BF16_KiB":64*H*D*2/2**10,
        "one_full_position_residual_BF16_40layers_GiB":N*H*D*2*L/2**30,
        "current_low_rho_FP32_40layers_GiB":N*H*4*L/2**30,
        "current_low_phi31_BF16_40layers_GiB":N*H*31*2*L/2**30,
        "residual_only_saving_before_partition_labels_GiB":N*H*(4+31*2)*L/2**30,
        "frozen_group_labels_uint8_40layers_MiB":N*H*L/2**20,
        "frozen_group_labels_bitpacked_40layers_MiB":math.ceil(N*H*L/8)/2**20,
        "residual_plus_uint8_labels_40layers_GiB":N*H*(D*2+1)*L/2**30,
        "net_state_saving_with_uint8_labels_GiB":N*H*(4+31*2-1)*L/2**30,
        "one_split_projection_MACs_per_layer":N*H*D,
        "note":"Static arithmetic only. Summary reductions, mask/normalization, extra anchor evaluation and kernel cost require measurement."
    }
    checks={
        "lift_identity":bool(lift_identity),"lifted_error_identity":bool(lifted_error),
        "bounded_coordinate_error_identity":bool(bounded_error),"anchor_identity":bool(anchored_identity),
        "finite_error_max_abs_residual":error_residual,"finite_bound_min_slack":bound_slack,
        "minimum_refinement_lse_increase":refinement_margin,"minimum_true_minus_coarse_lse":jensen_margin,
        "value_translation_max_abs_error":translation_error,"value_linear_map_max_abs_error":linear_value_error,
        "bounded_lift_reconstruction_max_abs_error":no_division_in_lift,
        "bounded_residual_translation_max_abs_error":h_translation_error,
        "total_covariance_decomposition_max_abs_error":covariance_residual,
        "frozen_partition_current_KV_finite_identity_max_abs_error":frozen_partition_residual,
        "retained_value_convex_hull_coordinate_check":bool(np.all(two>=v.min(0)-1e-12) and np.all(two<=v.max(0)+1e-12)),
    }
    assert all([lift_identity,lifted_error,bounded_error,anchored_identity])
    assert collision["two_group_max_abs_error"]<1e-12 and collision["one_group_max_output_norm"]<1e-12
    assert error_residual<1e-12 and bound_slack>=-1e-12
    assert refinement_margin>=-1e-12 and jensen_margin>=-1e-12
    assert max(translation_error,linear_value_error,no_division_in_lift,h_translation_error,covariance_residual)<1e-12
    assert frozen_partition_residual<1e-12
    result={
        "scope":"Exact algebra plus explicitly synthetic activations; archived FFNDS projection bytes and actual dimensions. No model/GPU run; no performance/quality prediction.",
        "assets":{name.replace('.npy','_sha256'):hashlib.sha256(blob).hexdigest() for name,blob in blobs.items()},
        "collision_example":collision,"checks":checks,"budget":budget,
        "implementation_binding":{"current_P_teacher":"F.scaled_dot_product_attention returns output only in actual code", "existing_full_anchor_helper":"native_flash returns ATen Flash attention output and LSE", "P64_output_plus_lse_backend_validation":"NOT_EXECUTED; no local torch package or GPU run"}
    }
    out=args.output
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
