"""Static arithmetic of the current HHLL wide8192 + attention execution graph.

Counts useful GEMM multiply-accumulates (2 FLOPs/MAC), not GPU instructions or
latency. All constants bind archived real model dimensions and runtime code.
The extra-GEMM ledger still excludes solves, nonlinearities, reductions,
pointwise work, layout/packing, transfers, and non-block pipeline operations.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path


def ratio(value, denominator):
    x = Fraction(value, denominator)
    return {"fraction": str(x), "value": float(x),
            "fraction_saved": float(1-x), "arithmetic_speedup": float(1/x)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('theory_budget_verified.json'))
    args = parser.parse_args()
    N, D, M, K, P, G = 31668, 5120, 13824, 2560, 64, 16
    H, DH, L, B, TOP, C = 40, 128, 40, 128, 31, 512
    QB = (N+B-1)//B
    R = 32  # constant + 31 calibrated output features, per attention head
    assert H*DH == D and QB == 248 and TOP*B == 3968
    terms = {
        'self_QK_AV': 2*N*N*D,
        'self_QKV_O_projections': 4*N*D*D,
        'FFN_two_projections': 2*N*D*M,
        'cross_Q_O_projections': 2*N*D*D,
        'cross_K_V_projections': 2*C*D*D,
        'cross_QK_AV': 2*N*C*D,
    }
    attention, ffn = terms['self_QK_AV'], terms['FFN_two_projections']
    dense = sum(terms.values())
    retained = dense-attention-ffn
    dense_video = 4*L*dense

    # Source: fourstep_hit.make_hit. P operands are computed before the full
    # token loop; full loop includes the P rows again before exact overwrite.
    ff_hit = {
        'full_N_deltaU_A0': N*D*D,
        'full_N_selected_W1_W2': 2*N*D*K,
        'full_N_J_and_Z_maps': N*(D+K)*G,
        'full_N_kernel_to_P': N*K*P,
        'full_N_joint_readout': N*(G+P)*D,
        'P_base_and_feature_operands': P*(D*D+2*D*K+(D+K)*G),
        'P_native_FFN_teacher': 2*P*D*M,
        'P_covariance_X_XT': P*P*K,
        'P_gain_Gram_QT_IQ': G*G*P,
        'P_target_VT_and_Q_gain_V': 2*P*D*G,
    }
    major_hit = ff_hit['full_N_deltaU_A0']+ff_hit['full_N_selected_W1_W2']
    accounted_hit = sum(ff_hit.values())

    # Four denoising calls at one layer. High anchor needs only routing;
    # low anchor additionally computes selected attention and cache features.
    attn_extra_4 = {
        'two_anchor_pooled_QK_routes': 2*H*QB*QB*DH,
        'three_selected_feature_projections': 3*N*H*(DH*64+64*31),
        'two_full_N_ridge_readouts': 2*N*H*R*DH,
        'two_P_normal_Grams': 2*H*P*R*R,
        'two_P_ridge_RHS_products': 2*H*R*P*DH,
    }
    cases = []
    for keys in (TOP*B-(QB*B-N), TOP*B):
        # Each query uses TOP key blocks. At most one is the short tail.
        selected = 2*N*keys*D
        teacher = 2*P*N*D
        attn_4 = 2*attention+3*selected+2*teacher
        ffn_4_major = 2*ffn+2*major_hit
        ffn_4_accounted = 2*ffn+2*accounted_hit
        major = 4*retained+attn_4+ffn_4_major
        accounted = 4*retained+attn_4+sum(attn_extra_4.values())+ffn_4_accounted
        assert Fraction(attn_4,4*attention) == Fraction(1,2)+Fraction(3*keys,4*N)+Fraction(P,2*N)
        cases.append({
            'selected_keys_per_query': keys,
            'selected_density': float(Fraction(keys,N)),
            'attention_core_four_step_ratio': ratio(attn_4,4*attention),
            'self_attention_including_QKV_O_ratio': ratio(attn_4+4*terms['self_QKV_O_projections'],4*(attention+terms['self_QKV_O_projections'])),
            'whole_block_major_GEMM_ratio': ratio(major,4*dense),
            'whole_block_listed_GEMM_ratio': ratio(accounted,4*dense),
            'whole_video_major_GFLOPs': 2*major*L/1e9,
            'whole_video_listed_GFLOPs': 2*accounted*L/1e9,
            'whole_video_listed_MACs_before_rho_fallback': accounted*L,
        })
    # Observed fallback counts are all low-anchor query/head rows for a video.
    fallback = {}
    for source, rows in [('vbi_0392',8776),('vbi_1013',16485)]:
        extra = rows*N*DH  # QK only; probability reductions not counted
        fallback[source] = {
            'observed_query_head_rows': rows,
            'extra_QK_MACs': extra,
            'fraction_of_dense_block_GEMMs': float(Fraction(extra,dense_video)),
            'listed_whole_block_ratios_including_fallback': [
                ratio(c['whole_video_listed_MACs_before_rho_fallback']+extra,dense_video)
                for c in cases],
        }

    # Optimistic structural ceiling for exactly two native anchors and two
    # hits, when only FFN and self QK/AV are replaceable. All hit replacement
    # and anchor metadata work is set to zero for the bound.
    lower_bound = 4*retained+2*(attention+ffn)
    hit_only_fullblock_floor = 2*dense  # wider scope, all hit blocks free
    distance = [dict(selected_keys=c['selected_keys_per_query'],
        remaining_dense_fraction_above_zero_cost_bound=c['whole_block_listed_GEMM_ratio']['value']-float(Fraction(lower_bound,4*dense)),
        fraction_of_current_listed_GEMMs_above_bound=1-float(Fraction(lower_bound,4*dense))/c['whole_block_listed_GEMM_ratio']['value'],
        fraction_of_structural_savable_GEMMs_already_removed=c['whole_block_listed_GEMM_ratio']['fraction_saved']/float(1-Fraction(lower_bound,4*dense)))
        for c in cases]
    amortization=[]
    for hits in (1,2,4,9,19):
        for keys in (3892,3968):
            alpha,beta=Fraction(keys,N),Fraction(P,N)
            att_ratio=(1+(Fraction(hits)+Fraction(1,2))*alpha+hits*beta)/(hits+1)
            ff_ratio=(1+hits*Fraction(major_hit,ffn))/(hits+1)
            full=(retained+att_ratio*attention+ff_ratio*ffn)/dense
            amortization.append(dict(hits_per_expert_anchor=hits,denoiser_calls=2*(hits+1),selected_keys=keys,
                attention_core_ratio=float(att_ratio),FFN_major_ratio=float(ff_ratio),
                whole_block_major_ratio=float(full),arithmetic_speedup=float(1/full)))
    constant_overhead_wall_examples = []
    for overhead_fraction in [0, 0.1, 0.2, 0.3]:
        remaining = overhead_fraction+(1-overhead_fraction)*float(Fraction(lower_bound,4*dense))
        constant_overhead_wall_examples.append({'unchanged_fraction':overhead_fraction,
            'remaining_fraction_if_uniform_arithmetic_time':remaining,
            'speedup_if_uniform_arithmetic_time':1/remaining})
    out = {
        'status':'STATIC_USEFUL_MAC_ARITHMETIC_ON_BOUND_RUNTIME_GRAPH',
        'code_reference_commit':'13ffb127e71a5d4d1519ecb0df9c7b4d03efe601',
        'attention_evidence_commit':'ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63',
        'new_model_executions':0,'new_GPU_jobs':0,'latency_measured':False,
        'model': {'N':N,'D':D,'FFN_hidden':M,'FFN_selected':K,'FFN_modes':G,
            'P':P,'attention_heads':H,'head_dim':DH,'layers_per_expert':L,
            'text_keys_for_accounting':C,'attention_query_block':B,'top_key_blocks':TOP,
            'attention_fit_columns':R,'FFN_hit_tile_rows':8192,
            'denoiser_calls':4,'expert_schedule':['high','high','low','low'],
            'CFG_enabled':False,'native_anchor_steps':[0,2],'hit_steps':[1,3]},
        'cost_convention':'2 FLOPs per GEMM MAC. Count useful non-padded products, not actual issued GPU instructions.',
        'dense_MACs_per_layer_step':terms,
        'dense_TFLOPs_per_layer_step':{k:2*v/1e12 for k,v in terms.items()},
        'dense_TFLOPs_per_layer_step_total':2*dense/1e12,
        'dense_whole_video_PFLOPs_block_GEMMs':2*dense_video/1e15,
        'dense_shares':{'self_QK_AV':attention/dense,'FFN':ffn/dense,'retained_other':retained/dense},
        'FFN_hit_MACs':ff_hit,
        'FFN_major_hit_TFLOPs':2*major_hit/1e12,
        'FFN_listed_hit_TFLOPs':2*accounted_hit/1e12,
        'FFN_major_hit_ratio':ratio(major_hit,ffn),
        'FFN_major_anchor_hit_ratio':ratio(ffn+major_hit,2*ffn),
        'FFN_listed_anchor_hit_ratio':ratio(ffn+accounted_hit,2*ffn),
        'attention_extra_listed_MACs_four_steps_per_layer':attn_extra_4,
        'selected_support_cases':cases,
        'observed_omitted_mass_fallback_QK':fallback,
        'zero_cost_hit_FFN_and_attention_core_bound':ratio(lower_bound,4*dense),
        'zero_cost_entire_hit_block_bound_wider_scope':ratio(hit_only_fullblock_floor,4*dense),
        'distance_from_listed_current_budget_to_structural_bound':distance,
        'conditional_amortization_not_executed':{
            'FFN_formula':'(F + K*H + cache_extra) / ((K+1)*F)',
            'two_expert_attention_formula':'[1 + (K+1/2)*alpha + K*beta] / (K+1), excluding feature/routing/pointwise overhead',
            'conditions':'One anchor per expert; K hits from that anchor without refresh; native/selected/probe costs stay unchanged and quality remains acceptable. K=1 is current four-step HHLL; other K values change the denoiser-call count and are untested.',
            'major_GEMM_examples':amortization},
        'unchanged_overhead_examples_not_latency_predictions':constant_overhead_wall_examples,
        'memory':{
            'FFN_persistent_anchor_bytes':2*L*N*(2*D+K),
            'FFN_persistent_anchor_GiB':2*L*N*(2*D+K)/2**30,
            'attention_low_full_BF16_history_bytes':2*L*N*D,
            'attention_low_full_BF16_history_GiB':2*L*N*D/2**30,
            'attention_low_feature_BF16_and_rho_FP32_bytes':L*N*H*(2*31+4),
            'one_full_ND_BF16_field_GiB':2*N*D/2**30},
        'scope':[
            'The leading savings include both FFN and self-attention replacement, relative to the same four-step distilled model with native FFN/attention.',
            'Current centered transport preserves all leading GEMM call counts; it adds O(ND) moments/encoding/decoding and changes execution fusion.',
            'Listed-GEMM account is not all FLOPs: excludes Cholesky/triangular solves, GELU/softmax, means/variance, norms/RoPE, pointwise work, gather, copies, padding, kernel launch, offload and VAE/text/image encoding/save.',
            'Attention routing fractions are valid-key bounds; exact per-query supports were not archived here.',
            'The structural 1.58774x bound applies only to this schedule and retained matrix operations with uniform per-MAC cost; it is not a universal wall-time bound.',
            'Amdahl latency claims require measured time fractions and an explicit unchanged portion, not FLOP fractions.',
            'The four-step distilled runtime has CFG disabled; comparing to a teacher must count actual guided/unguided denoiser evaluations, not just nominal step count.'
        ],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['dense_TFLOPs_per_layer_step_total','dense_shares','FFN_major_anchor_hit_ratio','FFN_listed_anchor_hit_ratio','selected_support_cases','zero_cost_hit_FFN_and_attention_core_bound','memory']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
