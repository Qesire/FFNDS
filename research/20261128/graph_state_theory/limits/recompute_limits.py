"""CPU arithmetic and finite linear certificates for current FFNDS graph limits.

No model execution, video scoring, GPU experiment, or latency benchmark.
Real dimensions and measured times are archived model/run inputs. Random arrays
only witness the explicitly stated linear-state rank propositions.
"""
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing', type=Path,
                        default=Path('round19_results/speed_memory/SPEED_MEMORY_RECOMPUTED.json'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('LIMITS_RECOMPUTED.json'))
    args = parser.parse_args()
    measured = json.loads(args.timing.read_text())
    N,d,m,k,H,dh,L,P,G,C,r = 31668,5120,13824,2560,40,128,40,64,16,512,32
    qb = (N+127)//128
    assert qb == 248 and d == H*dh
    # All arithmetic below is FLOPs, 2 per multiply-accumulate.
    terms = {
        'self_QK_AV': 4*N*N*d,
        'self_QKV_O': 8*N*d*d,
        'FFN': 4*N*d*m,
        'cross_Q_O': 4*N*d*d,
        'cross_K_V': 4*C*d*d,
        'cross_QK_AV': 4*N*C*d,
    }
    A,F,D = terms['self_QK_AV'],terms['FFN'],sum(terms.values())
    R = D-A-F
    den = 4*L*D
    ffn_hit_major = 2*N*d*d + 4*N*d*k
    ffn_hit_extra = 2*(N*(d+k)*G+N*k*P+N*(G+P)*d+
                      P*(d*d+2*d*k+(d+k)*G)+2*P*d*m+
                      P*P*k+G*G*P+2*P*d*G)
    attention_extra_cycle = 2*(2*H*qb*qb*dh+
                                3*N*H*(dh*64+64*31)+
                                2*N*H*r*dh+2*H*P*r*r+2*H*r*P*dh)
    structural_floor = Fraction(4*R+2*(A+F),4*D)
    cases=[]
    for keys in (3892,3968):
        selected=4*N*keys*d
        probe=4*P*N*d
        attn_cycle=2*A+3*selected+2*probe
        ffn_cycle=2*F+2*ffn_hit_major
        major=4*R+attn_cycle+ffn_cycle
        listed=major+attention_extra_cycle+2*ffn_hit_extra
        residual = {
            'three_selected_attention_calls': Fraction(3*selected,4*D),
            'two_FFN_A0_actions': Fraction(2*2*N*d*d,4*D),
            'two_FFN_selected_W1_W2': Fraction(2*4*N*d*k,4*D),
            'two_attention_P64_teachers': Fraction(2*probe,4*D),
            'listed_feature_fit_and_readout_GEMMs': Fraction(attention_extra_cycle+2*ffn_hit_extra,4*D),
        }
        assert sum(residual.values(), Fraction(0)) == Fraction(listed,4*D)-structural_floor
        fused=listed-selected
        cases.append({
            'valid_selected_keys':keys,
            'major_fraction':float(Fraction(major,4*D)),
            'listed_fraction':float(Fraction(listed,4*D)),
            'listed_saved_percent':100*float(1-Fraction(listed,4*D)),
            'listed_arithmetic_speedup':float(Fraction(4*D,listed)),
            'remaining_above_floor_dense_pp':100*float(Fraction(listed,4*D)-structural_floor),
            'remaining_above_floor_current_percent':100*float(1-structural_floor/Fraction(listed,4*D)),
            'fraction_of_scope_savable_work_removed':float((1-Fraction(listed,4*D))/(1-structural_floor)),
            'remaining_budget_by_operation_dense_pp':{key:100*float(value) for key,value in residual.items()},
            'shared_anchor_partition_reduction':{
                'removed_selected_FLOPs_per_layer':selected,
                'removed_selected_FLOPs_per_video':selected*L,
                'saved_dense_pp':100*float(Fraction(selected,4*D)),
                'new_listed_saved_percent_excluding_new_vector_reductions':100*float(1-Fraction(fused,4*D)),
                'new_listed_arithmetic_speedup':float(Fraction(4*D,fused)),
                'scope':'Real-arithmetic single QK/AV scan yields selected and complement states; additional scalar/vector reductions and changed floating-point rounding not included.'}
        })
    dtime,mtime = measured['means']['dense'],measured['means']['centered']
    arithmetic_wall=[]
    for name,ratio in [('listed_low',cases[0]['listed_fraction']),('listed_high',cases[1]['listed_fraction']),
                       ('structural_zero_cost',float(structural_floor)),('all_hit_blocks_free',0.5)]:
        wall=dtime['outside_step_events_s']+ratio*dtime['gpu_s']
        arithmetic_wall.append({'case':name,'arithmetic_ratio':ratio,'gpu_s':ratio*dtime['gpu_s'],
                                 'wall_s':wall,'speedup_ratio_of_means':dtime['wall_s']/wall})
    hit_sensitivity=[]
    for removed in (0.1,0.25,0.5,1.0):
        wall=mtime['outside_step_events_s']+mtime['anchors_s']+(1-removed)*mtime['hits_s']
        hit_sensitivity.append({'fraction_of_current_hit_time_removed':removed,
                                'wall_s':wall,'improvement_over_current':mtime['wall_s']/wall,
                                'speedup_vs_D_ratio_of_means':dtime['wall_s']/wall,
                                'current_request_time_saved_percent':100*(mtime['wall_s']-wall)/mtime['wall_s']})
    ffn_state=2*L*N*(2*d+k)
    attention_state=2*L*N*d+2*L*N*H*31+4*L*N*H
    principal_traffic=4*ffn_state+2*attention_state
    ffn_u_only=2*L*N*d
    ffn_folded=2*L*N*(d+k)+4*L*N*G  # store the small J projection in FP32
    # CPU construction certificates: one fixed future readout vs a family.
    rng=np.random.default_rng(20261128)
    n,p,rank=257,64,32
    S=np.eye(n)[:p]
    Q=np.eye(n)[p:]
    maps=[];map_ranks=[]
    for i in range(3):
        x=np.column_stack((np.ones(n),rng.normal(size=(n,rank-1))))
        xp=x[:p]
        K=x@np.linalg.solve(xp.T@xp+6.4*np.eye(rank),xp.T)
        # Non-probe history dependence, after exact probe overwrite.
        Rq=Q@(np.eye(n)-K@S)
        maps.append(Rq)
        map_ranks.append(int(np.linalg.matrix_rank(Rq)))
    joint_ranks=[int(np.linalg.matrix_rank(np.vstack(maps[:i]))) for i in (1,2,3)]
    assert map_ranks == [n-p]*3 and joint_ranks[-1] == n
    # Rank-optimal encoder for a fixed readout and exact factor reconstruction.
    U,s,Vt=np.linalg.svd(maps[0],full_matrices=False)
    encoder=Vt[:n-p]
    decoder=U[:,:n-p]*s[:n-p]
    factor_error=float(np.max(np.abs(maps[0]-decoder@encoder)))
    assert factor_error < 1e-11
    report={
        'status':'CPU_EXACT_ARITHMETIC_AND_SYNTHETIC_LINEAR_CERTIFICATES',
        'date':'2026-11-28',
        'new_GPU_jobs':0,'model_forward_passes':0,'latency_benchmark_runs':0,
        'real_shape':{'N':N,'d':d,'m':m,'K':k,'H':H,'dh':dh,'L':L,'P':P,'attention_features':r,
                      'text_keys_assumed_in_ledger':C,'steps':4,'native_anchors':2,'hits':2},
        'dense_FLOPs_per_layer_step':terms,'dense_total_FLOPs':den,
        'structural_fixed_scope':{'floor_fraction':float(structural_floor),
                                 'max_saved_percent':100*float(1-structural_floor),
                                 'arithmetic_speedup':float(1/structural_floor),
                                 'scope':'Retain both complete native anchors and all QKV/O/cross GEMMs; zero out all hit FFN/core and auxiliary cache construction.'},
        'arithmetic_cases':cases,
        'expanded_scope_opportunities_dense_pp':{
            'all_hit_self_QKV_O':100*float(Fraction(terms['self_QKV_O'],2*D)),
            'all_hit_cross_Q_O':100*float(Fraction(terms['cross_Q_O'],2*D)),
            'all_hit_cross_K_V':100*float(Fraction(terms['cross_K_V'],2*D)),
            'all_hit_cross_core':100*float(Fraction(terms['cross_QK_AV'],2*D))},
        'timing':{
            'source':str(args.timing),'source_sha256':hashlib.sha256(args.timing.read_bytes()).hexdigest(),
            'source_evidence_commit':measured['source_commit'],
            'D':dtime,'M':mtime,
            'M_request_fractions':{'outside_step':mtime['outside_step_events_s']/mtime['wall_s'],
                                   'anchors':mtime['anchors_s']/mtime['wall_s'],
                                   'hits':mtime['hits_s']/mtime['wall_s']},
            'conditional_uniform_effective_cost_per_FLOP':{
                'assumption':'All listed GEMMs attain the original whole-denoiser effective FLOP/s, with Dense outside-step time fixed; not a physical latency upper bound or prediction.',
                'examples':arithmetic_wall},
            'conditional_hit_sensitivity':{'assumption':'Current anchor and outside-step time fixed; a hypothetical specified fraction of all hit time is removed. Ratios of arithmetic mean seconds, not registered geometric mean speedups.',
                                           'examples':hit_sensitivity}},
        'capacity_and_recomputation':{
            'FFN_BF16_main_GiB':ffn_state/2**30,'attention_main_GiB':attention_state/2**30,
            'combined_main_GiB':(ffn_state+attention_state)/2**30,
            'literal_graph_compulsory_main_history_traffic_GiB':principal_traffic/2**30,
            'traffic_formula':'4*C_FFN + 2*C_attention: two experts, one write/read each; excludes metadata, P64 rereads, intermediate state and physical transaction overhead.',
            'FFN_z0_recompute':{'main_GiB':2*L*N*2*d/2**30,'saved_GiB':2*L*N*k/2**30,
                                 'extra_dense_FLOP_pp':100*float(Fraction(2*L*2*N*d*k,den))},
            'FFN_fold_c0_and_j0':{'main_GiB':ffn_folded/2**30,'saved_GiB':(ffn_state-ffn_folded)/2**30,
                                   'extra_anchor_A0_FLOPs':2*L*2*N*d*d,
                                   'extra_anchor_A0_dense_FLOP_pp':100*float(Fraction(2*L*2*N*d*d,den)),
                                   'tiny_j0_dtype':'FP32',
                                   'scope':'Real-arithmetic c0=y0-u0 A0 and j0=u0 J; exact bounded-precision defect handled in the FFN derivation.'},
            'FFN_keep_u0_only':{'main_GiB':ffn_u_only/2**30,'saved_GiB':(ffn_state-ffn_u_only)/2**30,
                                 'extra_dense_FLOP_pp':100*float(Fraction(2*L*F,den)),
                                 'construction':'Recompute native FFN at anchor u0 on each hit to recover y0 and selected z0. Same weights, original kernel and arithmetic needed for bitwise replay.'},
            'layer_interleaved_anchor_replay':{
                'principal_history_single_layer_GiB':(ffn_state+attention_state)/(L*2**30),
                'two_BF16_hidden_streams_GiB':4*N*d/2**30,
                'extra_dense_FLOP_percent':50,
                'scope':'First native anchor determines next step input; subsequently replay anchor layer and execute hit layer in lockstep, consuming each local cache immediately. Includes one extra complete native-anchor replay per expert; 4 scheduler steps remain, denoiser computations increase. No standalone peak or speed claim.'}},
        'linear_state_certificate':{
            'theorem':'For fixed linear state s->Es and future linear task maps G_j, exact recovery for all s iff ker(E) is contained in intersection_j ker(G_j); minimal linear state dimension is rank(stack G_j). This is a linear dimension result, not a bit-complexity lower bound.',
            'synthetic_dimensions':{'N':n,'P':p,'features':rank},
            'individual_history_map_ranks':map_ranks,'joint_ranks_1_2_3_queries':joint_ranks,
            'fixed_map_rank_optimal_factorization_max_abs':factor_error,
            'scope':'Future features held fixed for each map. These random witnesses are not measured Wan state ranks or evidence that real model history is uncompressible.'}
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('structural_fixed_scope','arithmetic_cases','capacity_and_recomputation','linear_state_certificate')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
