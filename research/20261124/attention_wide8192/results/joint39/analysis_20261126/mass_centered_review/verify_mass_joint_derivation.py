"""Finite geometry of job 201500 and separate algebraic witnesses.

Real outputs are derived from the archived Wan Grams. Synthetic tests establish
identities only; no model inference, new timing or video assessment is made.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def geometry(g, dense=False):
    g = np.asarray(g, dtype=float)
    rr, tt, rt = g[0, 0], g[1, 1], g[0, 1]
    d2 = rr + tt - 2*rt
    dot = rt - rr
    background_cross = 0.
    if dense:
        background_cross = 2*(g[1, 2]-g[0, 2])
        rr += g[2, 2] + 2*g[0, 2]
        tt += g[2, 2] + 2*g[1, 2]
        dot += background_cross/2
    assert d2 > 0 and rr > 0
    change = 2*dot + d2
    assert np.isclose(change, tt-rr, rtol=1e-10, atol=1e-9)
    den = np.sqrt(rr*d2)
    return dict(raw_energy=float(rr), centered_energy=float(tt), ratio=float(tt/rr),
                update_energy=float(d2), twice_error_update_dot=float(2*dot),
                delta=float(change), update_cosine=float(dot/den),
                benefit_cosine_threshold=float(-0.5*np.sqrt(d2/rr)),
                normalized_sign_margin=float(-change/(2*den)),
                background_cross_delta=float(background_cross))


def synthetic():
    rng = np.random.default_rng(201500)
    nq, nk, dim, npb = 31, 97, 11, 8
    def normalized():
        x = np.exp(rng.normal(size=(nq, nk)))
        return x/x.sum(1, keepdims=True)
    sa, ss, ta, ts = [normalized() for _ in range(4)]
    va, vs = (rng.normal(size=(nk, dim))+rng.normal(size=(1, dim)) for _ in range(2))
    rho = rng.uniform(-0.2, 1.2, size=(nq, 1))
    w = 1-rho
    mu_a, mu_s = va.mean(0), vs.mean(0)
    da, ds = va.std(0), vs.std(0)
    aa, ass, ba, bs = sa@va, ss@vs, ta@va, ts@vs
    ca = (aa-w*ba-rho*mu_a)/da
    cs = (ass-w*bs-rho*mu_s)/ds
    base = w*bs+rho*mu_s+ca*ds
    exact = w*bs+rho*mu_s+cs*ds
    error = np.max(np.abs(exact-ass))
    state_error = np.max(np.abs(ass-base-(cs-ca)*ds))
    signed_mass = max(np.max(np.abs((sa-w*ta).sum(1, keepdims=True)-rho)),
                      np.max(np.abs((ss-w*ts).sum(1, keepdims=True)-rho)))

    # Fixed recovery acts on the query axis, while means/scales use channels.
    p = np.arange(npb)
    psi = rng.normal(size=(nq, 5)); xp = psi[p]
    K = psi@np.linalg.solve(xp.T@xp+0.1*npb*np.eye(5), xp.T)
    L = np.eye(nq); L[:, p] -= K; L[p] = 0
    dm = mu_s-mu_a; dd = ds-da
    delta = rho*dm + ca*dd
    separated = (L@rho)*dm+(L@ca)*dd
    separated_error = np.max(np.abs(L@delta-separated))

    # Choice of an affine-equivariant center changes route-response prediction.
    cb = (aa-ba)/da
    base_b = bs+cb*ds
    center_change = rho*((bs-mu_s)-(ba-mu_a)*(ds/da))
    center_identity = np.max(np.abs(base_b-base-center_change))
    assert max(error,state_error,signed_mass,separated_error,center_identity) < 1e-11
    return dict(scope='Synthetic real-arithmetic witnesses with every scale column active; '
                      'no BF16 cache quantization, threshold truncation, or candidate model run',
                current_signed_mass_exact_readout_max_abs=float(error),
                exact_normalized_innovation_max_abs=float(state_error),
                signed_row_mass_max_abs=float(signed_mass),
                separated_mean_scale_recovery_max_abs=float(separated_error),
                changed_center_identity_max_abs=float(center_identity),
                rho_range=[float(rho.min()),float(rho.max())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('MASS_JOINT_DERIVATION_VERIFIED.json'))
    args=parser.parse_args()
    data=args.raw.read_bytes()
    if args.raw.suffix=='.gz':data=gzip.decompress(data)
    doc=json.loads(data)
    assert doc['job']=='201500' and doc['status']=='complete'
    sites={}
    for src,track in doc['fixed'].items():
        sites[src]={}
        for layer,row in track['sites'].items():
            tail=row['tail']; stages={k:geometry(v) for k,v in tail['stage_gram_raw_transport'].items()}
            targets={k:geometry(v,True) for k,v in tail['stage_gram_raw_transport_background'].items()}
            base=geometry(row['held_base_error_gram'])
            att=stages['attention_head']
            sites[src][layer]=dict(base=base,stages=stages,dense_targets=targets,
                anchor_relative_l2_raw=row['anchor_raw_relative_l2'],
                anchor_relative_l2_centered=row['anchor_relative_l2'],
                anchor_encoding_error_ratio=row['anchor_relative_l2']/row['anchor_raw_relative_l2'],
                raw_fit_to_base_energy=att['raw_energy']/base['raw_energy'],
                centered_fit_to_base_energy=att['centered_energy']/base['centered_energy'],
                held_update_energy_after_to_before_fit=att['update_energy']/base['update_energy'],
                realized_relative_tail_squared_gain=stages['block_exit']['ratio']/att['ratio'],
                self_residual_minus_gate_delta=stages['self_residual']['delta']-stages['attention_gate_response']['delta'])
    flat=[x for s in sites.values() for x in s.values()]
    out=dict(evidence_commit='ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63',job='201500',
             raw_sha256=hashlib.sha256(data).hexdigest(),
             scope='CPU arithmetic of archived Wan measurements plus separately labeled synthetic identities. '
                   'Held update ratios measure the implemented recovery and final BF16 writeback on H, '
                   'relative to the base update on H; they are not norms of an H-only recovery operator.',
             measurement_scope=dict(
               held_update_energy_after_to_before_fit={
                   'numerator':'Squared centered-minus-raw attention output norm on H, after fitted recovery, '
                               'final BF16 conversion, and P overwrite.',
                   'denominator':'Squared centered-minus-raw base norm on H, from FP32 base fields '
                                 'that already contain the respective quantized anchor histories.',
                   'dependency':'The fitted correction depends on P64 labels; H is the measurement subset, '
                                'not a closed input space for the recovery operator.',
                   'interpretation':'Realized direction energy ratio of recovery plus final writeback; '
                                    'not an isolated real-arithmetic L norm, projection fraction, or spectral estimate.'},
               raw_fit_to_base_energy='Actual raw error after fitted recovery and final BF16 writeback '
                                      'over raw base error, both measured on H.',
               centered_fit_to_base_energy='Actual centered error after fitted recovery and final BF16 writeback '
                                           'over centered base error, both measured on H.',
               implementation_identity_check='Source compare_shadow verifies L times the actual base difference '
                                             'before final BF16 conversion; FP32 arithmetic rounding is still present.',
               ideal_mean_scale_decomposition='rho times mean change plus C_a times scale change uses one '
                                              'unquantized ideal anchor with no support truncation; actual base '
                                              'differences also contain separate raw and centered encoding errors.'),
             new_model_executions=0,source_clusters=2,nested_sites=8,
             aggregate=dict(anchor_error_improvement_count=sum(x['anchor_encoding_error_ratio']<1 for x in flat),
               attention_improvement_count=sum(x['stages']['attention_head']['ratio']<1 for x in flat),
               dense_block_improvement_count=sum(x['dense_targets']['block_exit']['ratio']<1 for x in flat),
               held_update_after_fit_ratio_range=[min(x['held_update_energy_after_to_before_fit'] for x in flat),max(x['held_update_energy_after_to_before_fit'] for x in flat)],
               raw_fit_energy_above_base_count=sum(x['raw_fit_to_base_energy']>1 for x in flat),
               centered_fit_energy_above_base_count=sum(x['centered_fit_to_base_energy']>1 for x in flat)),
             sites=sites,synthetic=synthetic())
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='sites'},indent=2))


if __name__=='__main__':main()
