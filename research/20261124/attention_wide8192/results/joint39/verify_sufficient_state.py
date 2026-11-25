"""Known-readout cache closure: symbolic proof, real CAL geometry, static cost.

Numerical activations below are explicitly synthetic identity witnesses, not Wan
activations or quality/performance measurements. No model/GPU inference is run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
from collections import defaultdict

import numpy as np
import sympy as sy


def bf16(x):
    """Round finite float32 values to BF16, round-to-nearest ties-to-even."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--endpoint-base", type=Path)
    parser.add_argument("--endpoint-extra", type=Path)
    parser.add_argument("--endpoint-scores", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sufficient_state_results.json"))
    args = parser.parse_args()
    n, d, k, r = 31668, 5120, 2560, 16
    U, Ua, Ya = [sy.MatrixSymbol(name, n, d) for name in ("U", "Ua", "Ya")]
    A = sy.MatrixSymbol("A", d, d)
    J = sy.MatrixSymbol("J", d, r)
    DH = sy.MatrixSymbol("DH", n, k)
    W = sy.MatrixSymbol("W", k, d)
    Z = sy.MatrixSymbol("Z", k, r)
    Vg = sy.MatrixSymbol("Vg", r, d)
    original = Ya + (U-Ua)*A + DH*W + (DH*Z-(U-Ua)*J)*Vg
    folded = (Ya-Ua*A) + U*A + DH*W + (DH*Z-U*J+Ua*J)*Vg
    ffn_zero = sy.simplify(sy.expand(original-folded)) == sy.ZeroMatrix(n, d)
    assert ffn_zero

    # One channel of the exact cache-error residualization identity.
    # K denotes the all-row ridge extension operator and P the row extraction.
    e, base, y, Kp, eP, baseP, yP = sy.symbols("e base y Kp eP baseP yP")
    low_error = sy.expand((base+e+Kp*(yP-baseP-eP))
                          -(base+Kp*(yP-baseP))-(e-Kp*eP))
    assert low_error == 0

    # Observe stored state (u[128], phi[31], rho) under a spanning family of
    # future B,W queries. This is an unrestricted readout-family calculation.
    dhead, feat = 128, 31
    obs = np.zeros((dhead+feat+1, dhead+feat+1))
    obs[:dhead,:dhead] = np.eye(dhead)
    for j in range(feat):
        obs[dhead+j, 0] = 1
        obs[dhead+j, dhead+j] = -1
    obs[-1,0] = 1
    obs[-1,-1] = -1
    family_rank = int(np.linalg.matrix_rank(obs))
    assert family_rank == 160

    rng = np.random.default_rng(20261125)
    ns, ds, ks, rs, ps = 128, 96, 48, 16, 64
    us = bf16(rng.normal(size=(ns,ds)))
    ua = bf16(rng.normal(size=(ns,ds)))
    ya = bf16(rng.normal(size=(ns,ds)))
    amat = bf16(rng.normal(scale=0.04,size=(ds,ds)))
    jmat = rng.normal(scale=0.04,size=(ds,rs)).astype(np.float32)
    dh = bf16(rng.normal(scale=0.3,size=(ns,ks)))
    w2 = bf16(rng.normal(scale=0.04,size=(ks,ds)))
    zm = rng.normal(scale=0.1,size=(ks,rs)).astype(np.float32)
    v = rng.normal(scale=0.05,size=(rs,ds)).astype(np.float32)
    # The actual branch uses BF16 difference and BF16 output for the A0 GEMM.
    du = us-ua
    old_linear = bf16(bf16(du) @ amat)
    old_base = ya + old_linear + dh @ w2
    old_q = dh @ zm - du @ jmat
    # Proposed folded fields: C BF16, J0 FP32. Anchor A0 projection outputs FP32.
    la = ua @ amat
    ccache = bf16(ya-la)
    jcache = ua @ jmat
    new_linear = bf16(us @ amat)
    new_base = ccache + new_linear + dh @ w2
    new_q = dh @ zm - us @ jmat + jcache
    # Protect original U_a,Y_a at P, and retain all original Z_a. Here those
    # protected operands are represented by the already evaluated old P paths.
    probes = np.arange(ps)
    new_base[probes] = old_base[probes]
    new_q[probes] = old_q[probes]
    x = rng.normal(size=(ns,ks)).astype(np.float32)
    teacher = rng.normal(size=(ps,ds)).astype(np.float32)

    def fit(base, q):
        xp, qp = x[probes].astype(np.float64), q[probes].astype(np.float64)
        target = teacher.astype(np.float64)-base[probes].astype(np.float64)
        vd = v.astype(np.float64)
        system = xp@xp.T/ks+np.eye(ps)
        iq = np.linalg.solve(system, qp)
        gain = np.linalg.solve((qp.T@iq)*(vd@vd.T),
                               (iq*(target@vd.T)).sum(axis=0))
        coeff = np.linalg.solve(system, target-(qp*gain)@vd)
        return gain, coeff

    go, co = fit(old_base,old_q)
    gn, cn = fit(new_base,new_q)
    assert np.array_equal(go,gn) and np.array_equal(co,cn)
    out_old = old_base.astype(np.float64)+(old_q.astype(np.float64)*go)@v + 0.5*(x@x[probes].T/ks)@co
    out_new = new_base.astype(np.float64)+(new_q.astype(np.float64)*gn)@v + 0.5*(x@x[probes].T/ks)@cn
    error_pred = new_base.astype(np.float64)-old_base + ((new_q.astype(np.float64)-old_q)*go)@v
    error_identity_max = float(np.max(np.abs((out_new-out_old)-error_pred)))
    assert error_identity_max < 1e-12
    numerical = {
        "scope":"Synthetic finite-precision identity witness; activations and FFN weights are not model data",
        "shape":{"rows":ns,"width":ds,"selected":ks,"modes":rs,"probes":ps},
        "protected_probe_gain_bitwise_identical":bool(np.array_equal(go,gn)),
        "protected_probe_coeff_bitwise_identical":bool(np.array_equal(co,cn)),
        "fixed_input_prefinal_cast_error_identity_max_abs":error_identity_max,
        "base_reordering_effect_max_abs":float(np.max(np.abs(new_base-old_base))),
        "q_reordering_effect_max_abs":float(np.max(np.abs(new_q-old_q))),
        "note":"Finite-precision differences are recorded to demonstrate non-equivalence, not estimate model error"
    }

    N,H,D,K,R,L,P = 31668,40,5120,2560,16,40,64
    gib = lambda b: b/2**30
    old_ffn = L*N*(2*D+K)*2
    folded_ffn = L*N*(D+K)*2 + L*N*R*4 + L*P*D*2*2
    low_u = L*N*H*128*2
    low_phi = L*N*H*31*2
    low_rho = L*N*H*4
    low_u8 = L*N*H*(128+2)+L*P*H*128*2
    low_phi8 = L*N*H*(31+2)+L*P*H*31*2
    f_anchor = 4*N*D*13824
    f_hit = 2*N*D*D+4*N*D*K
    extra_A = 2*N*D*D
    extra_J = 2*N*D*R
    budget = {
        "scope":"Static GEMM arithmetic and cache payload only; excludes metadata/workspaces and is not a latency or peak-memory measurement",
        "real_geometry":{"tokens":N,"heads":H,"head_dim":128,"width":D,"hidden":13824,"selected":K,"shared_modes":R,"resident_layers":L,"anchors_per_video":80,"hits_per_video":80,"probes":P},
        "FFN_original_BF16_fields_GiB":gib(old_ffn),
        "FFN_folded_C_BF16_Z_BF16_J_FP32_exact_UP_YP_BF16_GiB":gib(folded_ffn),
        "FFN_saved_GiB":gib(old_ffn-folded_ffn),
        "FFN_saved_fraction":1-folded_ffn/old_ffn,
        "FFN_folded_components_GiB":{"C_BF16":gib(L*N*D*2),"Z_BF16":gib(L*N*K*2),"J_FP32":gib(L*N*R*4),"exact_U_P_and_Y_P":gib(L*P*D*2*2)},
        "extra_anchor_A0_TFLOP_per_site":extra_A/1e12,
        "extra_anchor_J_TFLOP_per_site":extra_J/1e12,
        "extra_anchor_A0_TFLOP_per_video":80*extra_A/1e12,
        "extra_anchor_J_TFLOP_per_video":80*extra_J/1e12,
        "hidden_omitted_linear_projection_extra_TFLOP_per_video":80*2*N*(13824-K)*D/1e12,
        "hidden_full_residual_W2_extra_TFLOP_per_video":80*2*N*13824*D/1e12,
        "hidden_omitted_projection_to_dense_A0_projection_ratio":(13824-K)/D,
        "hidden_full_residual_projection_to_dense_A0_projection_ratio":13824/D,
        "native_anchor_W1_W2_TFLOP_per_video":80*f_anchor/1e12,
        "wide8192_anchor_hit_main_GEMM_TFLOP_per_video":80*(f_anchor+f_hit)/1e12,
        "extra_A0_relative_native_anchor_W1_W2":extra_A/f_anchor,
        "extra_A0_J_relative_wide8192_main_GEMM":(extra_A+extra_J)/(f_anchor+f_hit),
        "current_main_GEMM_ratio_vs_all_native_FFN":(f_anchor+f_hit)/(2*f_anchor),
        "folded_main_GEMM_ratio_vs_all_native_FFN":(f_anchor+f_hit+extra_A+extra_J)/(2*f_anchor),
        "low_attention_original_fields_GiB":gib(low_u+low_phi+low_rho),
        "low_attention_U_only_INT8_exact_P_fields_GiB":gib(low_u8+low_phi+low_rho),
        "low_attention_U_PHI_INT8_exact_P_fields_GiB":gib(low_u8+low_phi8+low_rho),
        "low_attention_original_components_GiB":{"u":gib(low_u),"phi":gib(low_phi),"rho":gib(low_rho)}
    }
    result={
        "evidence_commit":"873e1a879b91d544fdae11894d68bbc6acc854d1",
        "algebra":{"FFN_known_readout_fold_exact_in_reals":ffn_zero,
                   "fixed_design_cache_error_residualization_identity":str(low_error),
                   "low_attention_unrestricted_future_readout_observation_rank":family_rank,
                   "rank_scope":"Stored-state domain with arbitrary future B and W; not an empirical data intrinsic-dimension estimate"},
        "synthetic_numerical_witness":numerical,"budget":budget}
    if args.calibration and args.projection:
        with np.load(args.calibration,allow_pickle=False) as z:
            t=z["out_delta_basis"].astype(np.float64)
        p=np.load(args.projection,allow_pickle=False).astype(np.float64)[:,:64]
        m=p@t
        sv=np.linalg.svd(m,compute_uv=False)
        result["actual_low_CAL_geometry"]={
            "scope":"Actual frozen low CAL matrix and P_v; no activation/candidate quality computation",
            "calibration_sha256":hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
            "projection_sha256":hashlib.sha256(args.projection.read_bytes()).hexdigest(),
            "combined_shape":list(m.shape),"rank":int(np.linalg.matrix_rank(m)),
            "singular_min":float(sv[-1]),"singular_max":float(sv[0]),
            "condition_2":float(sv[0]/sv[-1])}
    if args.endpoint_scores or (args.endpoint_base and args.endpoint_extra):
        if args.endpoint_scores:
            records=[json.loads(args.endpoint_scores.read_text())]
            endpoint_hashes=records[0]["file_sha256"]
        else:
            files=[args.endpoint_base,args.endpoint_extra]
            records=[json.loads(p.read_text()) for p in files]
            endpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        rows=[row for d in records for row in d["results"]]
        groups=defaultdict(list)
        for row in rows:
            for metric in ("J","point","edge"):
                groups[row["expert"],row["source"],metric].append(
                    row["scores"]["actual_endpoint"][metric]/row["scores"]["original"][metric])
        summary={}
        for expert in ("high","low"):
            summary[expert]={}
            for metric in ("J","point","edge"):
                per_source=[math.exp(sum(map(math.log,v))/len(v))
                            for (e,src,m),v in groups.items() if e==expert and m==metric]
                summary[expert][metric]={"source_count":len(per_source),
                    "geomean":math.exp(sum(map(math.log,per_source))/len(per_source)),
                    "min":min(per_source),"max":max(per_source)}
        result["actual_archived_endpoint_scores"]={
            "scope":"Recomputed historical layer19 panel scores for actual_endpoint/original, 11 sources, two experts, two complementary folds; neither candidate execution nor full-video evidence",
            "analyzed_variant_definition":"actual_endpoint retains each position's complete endpoint state; compare its J/point/edge scores to original. The CAL-mean variant in the source receipt is not the numerator analyzed here.",
            "source_receipt_cal_mean_definition":records[0].get("source_receipt_cal_mean_definition",records[0].get("definition")),
            "input_binding_sha256":records[0]["sha256_inputs"],
            "file_sha256":endpoint_hashes,
            "records":len(rows),"aggregation":"Geometric mean over folds within each source, then equal-source geometric mean",
            "actual_endpoint_over_original":summary}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
