"""CPU mathematical verification of the real FFNDS GLS graph and state costs.

No model weights or activations are invented: operand examples are explicitly
synthetic; shapes, formats and operations come from inspected frozen source.
No CUDA, inference, video generation, or performance timing is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def bf16(a):
    """Round finite Float32 values to bfloat16 RNE, return Float64 values."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    words = a.view(np.uint32)
    rounded = (words + np.uint32(0x7fff) + ((words >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)


def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))


def fit_and_lift(target, q, x, v, ids, inv_metric, *, return_parts=False):
    """The inspected native_joint/fourstep_hit GLS formula, in Float64."""
    k = x.shape[1]
    qp, xp = q[ids], x[ids]
    system = xp @ xp.T / k + inv_metric
    aiq = np.linalg.solve(system, qp)
    gram = (qp.T @ aiq) * (v @ v.T)
    gain = np.linalg.solve(gram, (aiq * (target @ v.T)).sum(0))
    coeff = np.linalg.solve(system, target - (qp*gain) @ v)
    lifted = (q*gain) @ v + .5*(x @ xp.T / k) @ coeff
    return (lifted, gain, coeff) if return_parts else lifted


def numerical():
    rng = np.random.default_rng(20261128)
    n,d,k,r,p,s = 160,96,48,16,64,24
    ids = np.arange(p)
    normal = lambda shape, scale=.15: rng.normal(size=shape)*scale
    u0,u = normal((n,d)),normal((n,d))
    y0 = normal((n,d))
    a,w,b,bias = normal((d,d)),normal((d,k)),normal((k,d)),normal((k,))
    zmap = normal((k,r))
    beta_cal = rng.uniform(.1,.9,k)
    beta_kernel = beta_cal + normal((k,), .07)
    scale = rng.uniform(.4,1.3,k)
    e = beta_cal[:,None]*zmap
    j = w @ e
    v = np.linalg.qr(normal((d,r)))[0].T
    z0,z = u0@w+bias,u@w+bias
    dh = gelu(z)-gelu(z0)
    du,dz = u-u0,z-z0
    q = dh@zmap-du@j
    x = (dh-dz*beta_kernel)/scale
    original_base = y0+du@a+dh@b
    old_fold = (y0-u0@a)+u@a+dh@b
    q_by_z = dh@zmap-dz@e
    q_wrong_beta = dh@zmap-dz@(beta_kernel[:,None]*zmap)
    dense_teacher = normal((p,d))
    inv_metric = np.eye(p) + .02*np.ones((p,p))
    residual = dense_teacher-original_base[ids]
    base_out = original_base+fit_and_lift(residual,q,x,v,ids,inv_metric)

    # Coupled operator/state construction: the previous input is read only
    # through low rank coordinates and the narrow static jmap.
    lmat,rmat = normal((d,s)),normal((s,d))
    ar = lmat@rmat
    anchor_coordinates = u0@lmat
    anchor_j = u0@j
    candidate_base = y0+(u@lmat-anchor_coordinates)@rmat+dh@b
    candidate_q = dh@zmap-u@j+anchor_j
    candidate_out = candidate_base+fit_and_lift(dense_teacher-candidate_base[ids],q,x,v,ids,inv_metric)
    perturbation = du@(ar-a)
    predicted = perturbation-fit_and_lift(perturbation[ids],q,x,v,ids,inv_metric)
    delta = candidate_out-base_out
    delta[ids]=0;predicted[ids]=0
    ta,tb=normal((p,d)),normal((p,d))
    linearity = fit_and_lift(1.3*ta-.7*tb,q,x,v,ids,inv_metric)-1.3*fit_and_lift(ta,q,x,v,ids,inv_metric)+.7*fit_and_lift(tb,q,x,v,ids,inv_metric)
    # Mode errors are exactly reproduced by the GLS mode component when its
    # Gram is invertible, including the deployed half-strength kernel branch.
    mode_error = (q*normal((r,),.3))@v
    mode_recovered = fit_and_lift(mode_error[ids],q,x,v,ids,inv_metric)

    # Exact mode gauge: paired changes to the approximate branch matrices,
    # with Q/X and the native teacher unchanged.
    gc=normal((r,),.4)
    cv=gc[:,None]*v
    ag=a+j@cv
    bg=b-zmap@cv
    gauge_base=y0+du@ag+dh@bg
    lifted0,gain0,coeff0=fit_and_lift(residual,q,x,v,ids,inv_metric,return_parts=True)
    liftedg,gaing,coeffg=fit_and_lift(dense_teacher-gauge_base[ids],q,x,v,ids,inv_metric,return_parts=True)
    gauge_output_error=gauge_base+liftedg-base_out
    # Combine a fixed gauge with a low-rank A, preserving the matching B.
    quotient_base=y0+du@ar+dh@bg
    quotient_out=quotient_base+fit_and_lift(dense_teacher-quotient_base[ids],q,x,v,ids,inv_metric)
    quotient_perturbation=du@(ar-ag)
    quotient_pred=quotient_perturbation-fit_and_lift(quotient_perturbation[ids],q,x,v,ids,inv_metric)
    quotient_actual=quotient_out-base_out
    quotient_actual[ids]=0;quotient_pred[ids]=0
    # BF16 storage introduces an explicit base defect that is processed by
    # the same GLS lift when its Q/X are held fixed in this CPU example.
    agq,bgq=bf16(ag),bf16(bg)
    gauge_rounded_base=y0+du@agq+dh@bgq
    gauge_rounded_out=gauge_rounded_base+fit_and_lift(dense_teacher-gauge_rounded_base[ids],q,x,v,ids,inv_metric)
    gauge_store_error=du@(agq-ag)+dh@(bgq-bg)
    gauge_round_pred=gauge_store_error-fit_and_lift(gauge_store_error[ids],q,x,v,ids,inv_metric)
    gauge_round_actual=gauge_rounded_out-base_out
    gauge_round_actual[ids]=0;gauge_round_pred[ids]=0

    # Distinguish separable real endpoint features from the deployed extra
    # rounding of GELU differences.
    zz0,zz=bf16(z0),bf16(z)
    hh0,hh=bf16(gelu(zz0)),bf16(gelu(zz))
    dh_plain=hh-hh0;dh_b=bf16(dh_plain)
    xi0=(hh0-beta_kernel*zz0)/scale
    xi=(hh-beta_kernel*zz)/scale
    xb=(dh_b-(zz-zz0)*beta_kernel)/scale
    e_dh=dh_b-dh_plain
    feature_defect=xb-(xi-xi0)-e_dh/scale
    # Current J can itself carry compilation rounding. Its exact effect is
    # separated from the BF16 selected preactivation rounding.
    j_stored=j.astype(np.float32).astype(np.float64)
    q_old=dh_b@zmap-du@j_stored
    q_new=dh_b@zmap-(zz-zz0)@e
    q_defect=du@(j_stored-w@e)-((zz-zz0)-du@w)@e

    # A future query family need not fit the p=64 rows of one realized query.
    endpoint=normal((n,k))
    ee=np.linalg.qr(normal((k,12)))[0]
    query=normal((k,p))
    compressed=endpoint@ee@ee.T@query
    expected_defect=-endpoint@(np.eye(k)-ee@ee.T)@query
    vnull=np.linalg.svd(ee.T,full_matrices=True)[2][12]
    witness_same_code=float(np.linalg.norm(vnull@ee))
    witness_new_query=float(vnull@vnull)
    return {
        'operand_scope':'synthetic CPU arrays; actual P64/rank16 GLS formula and actual model static dimensions separately used',
        'dimensions':{'n':n,'d':d,'k':k,'rank':r,'probes':p,'factor_rank':s},
        'max_abs':{
            'old_intercept_real_identity':float(np.max(np.abs(old_fold-original_base))),
            'compiled_J_selected_factor_identity':float(np.max(np.abs(q_by_z-q))),
            'factor_cache_base_identity':float(np.max(np.abs(candidate_base-(y0+du@ar+dh@b)))),
            'factor_cache_q_identity':float(np.max(np.abs(candidate_q-q))),
            'GLS_lift_linearity':float(np.max(np.abs(linearity))),
            'GLS_compensated_factorization_error_identity':float(np.max(np.abs(delta-predicted))),
            'GLS_mode_subspace_exact_reproduction':float(np.max(np.abs(mode_recovered-mode_error))),
            'paired_mode_gauge_base_identity':float(np.max(np.abs(gauge_base-original_base+q@cv))),
            'paired_mode_gauge_gain_shift':float(np.max(np.abs(gaing-gain0-gc))),
            'paired_mode_gauge_kernel_coeff_identity':float(np.max(np.abs(coeffg-coeff0))),
            'paired_mode_gauge_full_output_identity':float(np.max(np.abs(gauge_output_error))),
            'gauge_plus_factorization_quotient_error_identity':float(np.max(np.abs(quotient_actual-quotient_pred))),
            'gauge_BF16_matrix_storage_defect_identity':float(np.max(np.abs(gauge_round_actual-gauge_round_pred))),
            'BF16_secant_endpoint_defect_identity':float(np.max(np.abs(feature_defect))),
            'BF16_J_selected_factor_defect_identity':float(np.max(np.abs((q_new-q_old)-q_defect))),
            'query_subspace_error_identity':float(np.max(np.abs(compressed-endpoint@query-expected_defect))),
        },
        'nonzero_distinctions':{
            'gauge_BF16_matrix_storage_changes_output_max_abs':float(np.max(np.abs(gauge_round_actual))),
            'mixing_beta_cal_and_beta_kernel_changes_q_max_abs':float(np.max(np.abs(q_wrong_beta-q))),
            'extra_BF16_difference_rounding_changes_x_max_abs':float(np.max(np.abs(e_dh/scale))),
            'compressed_endpoint_counterexample_code_distance':witness_same_code,
            'compressed_endpoint_counterexample_future_answer_distance':witness_new_query,
        }
    }


def budgets():
    n,d,m,k,l,p,rj=31668,5120,13824,2560,40,64,16
    gib=2**30
    sites=2*l
    dense_ffn=4*n*d*m
    a0=2*n*d*d
    sel=4*n*d*k
    basestate=2*l*n*(2*d+k)
    rows=[]
    for rank in (256,512,1024,1536,2048,2560):
        anchor=2*n*d*rank
        hit=4*n*d*rank
        janchor=2*n*d*rj
        state_bf=2*l*n*(d+k+rank+rj)
        state_mixed=2*l*n*(d+k+rank)+4*l*n*rj
        state_fp32_coordinates=2*l*n*(d+k)+4*l*n*(rank+rj)
        saving=sites*(a0-anchor-hit-janchor)
        rows.append({
            'rank':rank,
            'both_experts_BF16_A0_static_GiB':2*sites*d*d/gib,
            'both_experts_BF16_factors_static_GiB':4*sites*d*rank/gib,
            'both_experts_BF16_static_saved_GiB':(2*sites*d*d-4*sites*d*rank)/gib,
            'all_BF16_state_GiB':state_bf/gib,
            'BF16_projection_FP32_J_state_GiB':state_mixed/gib,
            'FP32_projection_and_J_state_GiB':state_fp32_coordinates/gib,
            'extra_exact_Ua_P64_if_preserving_original_qP_GiB':2*l*p*d/gib,
            'BF16_projection_FP32_J_saved_GiB':(basestate-state_mixed)/gib,
            'anchor_projection_TFLOP_per_site':anchor/1e12,
            'hit_linear_path_TFLOP_per_site':hit/1e12,
            'whole_cycle_A0_ratio_without_J':(anchor+hit)/a0,
            'whole_video_net_TFLOP_saved_including_J_anchor':saving/1e12,
            'FFN_major_GEMM_cycle_fraction_vs_dense_including_extra_J':(dense_ffn+sel+anchor+hit+janchor)/(2*dense_ffn),
        })
    return {
        'actual_model_shapes':dict(N=n,d=d,m=m,K=k,L=l,P=p,J_rank=rj,anchors=sites,hits=sites),
        'old_BF16_cache_GiB':basestate/gib,
        'old_intercept_with_FP32_J_and_exact_P_GiB':(2*l*n*(d+k)+4*l*n*rj+4*l*p*d)/gib,
        'old_intercept_anchor_added_TFLOP':sites*(a0+2*n*d*rj)/1e12,
        'only_Ua_recompute_anchor_endpoint':{'GiB':2*l*n*d/gib,'extra_TFLOP_per_video':sites*dense_ffn/1e12},
        'delete_Za_recompute_selected_W1_endpoint':{'GiB':4*l*n*d/gib,'extra_TFLOP_per_video':sites*2*n*d*k/1e12},
        'low_rank_coupled_compute_break_even_r_ignoring_J':d/3,
        'low_rank_coupled_compute_break_even_r_including_J':(d-rj)/3,
        'coupled_factorization_candidates':rows,
        'deployed_low_rank_output':{
            'rank_bound':rj+p,
            'dual_kernel_GEMM_FLOP_per_site':2*n*k*p+2*n*p*d,
            'primal_kernel_GEMM_FLOP_per_site':2*k*p*d+2*n*k*d,
            'primal_to_dual_kernel_FLOP_ratio':(2*k*p*d+2*n*k*d)/(2*n*k*p+2*n*p*d),
            'note':'The current N×64 kernel factorization is already far cheaper than expanding it to K×d weights. This bound concerns readout, not all history states.'
        }
    }


def main():
    root=Path(__file__).parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=root/'FFN_GRAPH_REWRITES_VERIFIED.json')
    args=parser.parse_args()
    out={'status':'CPU_math_and_static_costs_only','new_GPU_jobs':0,'new_model_forwards':0,'numerical':numerical(),'budgets':budgets()}
    assert max(out['numerical']['max_abs'].values())<1e-10
    assert out['numerical']['nonzero_distinctions']['compressed_endpoint_counterexample_code_distance']<1e-12
    sources=[
      'round15_sparsepr/ours/source/tools/ffnds_fourstep/fourstep_hit.py',
      'round15_sparsepr/ours/source/tools/ffnds_fourstep/fourstep_backend.py',
      'joint_graph/exact_port.py','evidence/native_joint.py','evidence/compile_sites.py',
      'round10_review/dense_audit/source/resident_historical.py',
      'round8_derivation/publication/SUFFICIENT_STATE_DERIVATION.md',
      'round19_results/source/research/20261127/centered_medium_eval/FROZEN_METHOD_BINDING.json'
    ]
    out['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources if Path(p).exists()}
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'max_identity_error':max(out['numerical']['max_abs'].values()),'candidates':out['budgets']['coupled_factorization_candidates']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
