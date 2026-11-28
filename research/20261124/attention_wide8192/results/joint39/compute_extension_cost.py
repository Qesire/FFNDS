"""Static MAC and invocation arithmetic for the explicit attention recipes.

This script has no latency model or model execution. The denominator is four
native QK128 + PV128 attention cores for one layer; other work is listed and
must be measured in the full cycle.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import json


def ratio_rows(alpha, beta):
    ordinary = Fraction(1, 2) + alpha + beta / 2
    current_b = Fraction(1, 2) + Fraction(25, 32)*alpha + beta / 2
    reference_c = Fraction(1, 2) + Fraction(3, 4)*alpha + beta / 2
    separate_schedule = Fraction(1, 4) + Fraction(33, 32)*alpha + 3*beta / 4
    return {
        'alpha':float(alpha),
        'original_all_step_selected':float(ordinary),
        'B_dependency_pruned_C0':float(current_b),
        'C_dependency_pruned_OUT_D':float(reference_c),
        'new_module_specific_schedule':float(separate_schedule),
        'B_minus_C':float(current_b-reference_c),
        'B_if_separate_low_anchor_full_QK_LSE':float(current_b+Fraction(1,8)),
        'C_if_separate_low_anchor_full_QK_LSE':float(reference_c+Fraction(1,8)),
        'new_schedule_if_separate_low_anchor_full_QK_LSE':float(separate_schedule+Fraction(1,8)),
    }


def main(output):
    n, p, layers, heads, d = 31668, 64, 40, 40, 128
    beta = Fraction(p,n)
    result = {
        'evidence_type':'STATIC_EXACT_RATIONAL_MAC_ARITHMETIC',
        'model_executions':0,
        'latency_measured':False,
        'model':{'tokens':n,'probe_rows':p,'layers':layers,'heads':heads,
                 'head_width':d,'residual_width':heads*d},
        'denominator':{
            'definition':'Four full QK128 + PV128 attention cores, one layer.',
            'MACs':4*2*n*n*heads*d,
            'excludes':'QKV/WO, unweighted global moments, projections, fitting, routing, layout, cache I/O, all non-self-attention work.',
        },
        'formulas':{
            'original':'1/2 + alpha + beta/2',
            'B':'1/2 + 25*alpha/32 + beta/2',
            'C':'1/2 + 3*alpha/4 + beta/2',
            'new_schedule':'1/4 + 33*alpha/32 + 3*beta/4',
            'extra_low_anchor_full_QK_LSE':'1/8',
            'full_cycle_time':'T_new/T_wide = 1 - f_self_core*(1-r_core) + epsilon_added',
        },
        'beta':float(beta),
        'selected_keys_bounds':[3892,3968],
        'at_bounds':[ratio_rows(Fraction(s,n),beta) for s in (3892,3968)],
        'at_alpha_one_eighth':ratio_rows(Fraction(1,8),beta),
        'selected_FIN_value_width':144,
        'selected_FIN_MAC_multiplier':float(Fraction(128+144,128+128)),
        'layer_call_counts':{
            'original_or_B_or_C':{'native_attention':2*layers,'P_attention':2*layers,
                'head_readouts':2*layers*heads},
            'original_selected':4*layers,
            'B_or_C_selected_after_pruning':3*layers,
            'new_module_specific_schedule':{'native_attention':layers,'P_attention':3*layers,
                'head_readouts':3*layers*heads,'selected':4*layers},
        },
        'static_capacity_arithmetic_GiB':{
            'original_FIN_attention_state_plan':15.241,
            'earlier_standalone_wide_peak':47.57,
            'sum_not_measured_joint_peak':float(Fraction(15241,1000)+Fraction(4757,100)),
        },
        'scheduler':{
            'sigmas':[str(v) for v in (Fraction(1),Fraction(15,16),Fraction(5,6),Fraction(5,8),Fraction(0))],
            'signed_update_rule':'x_next = x + (sigma_next-sigma)*v',
            'high_hit_abs_step':float(Fraction(5,48)),
            'low_hit_abs_step':float(Fraction(5,8)),
            'low_to_high_direct_amplitude_ratio':6,
            'low_to_high_direct_squared_ratio':36,
        },
        'assumptions':[
            'The original-all-step-selected row is the historical V128 interaction estimate; it omits added FIN moment channels.',
            'B/C keep two native attention anchors and two approximated hits.',
            'High OUT-ABS does not read high-anchor selected statistics; routing still follows the original Q/K block means and tie rules.',
            'Low C0 uses QK128 + PV144 at anchor and hit to return weighted moments.',
            'Low OUT-D also uses ANCH and requires low-anchor u/rho/full LSE.',
            'The new schedule replaces high step0 attention only; FFN retains native anchors0/2 on its own current inputs.',
            'A common selected density is a counting approximation; bounds account for the archived route block lengths.',
            'Native low-anchor full LSE is exposed by the original softmax, or an extra full QK pass is charged separately.',
            'Actual backend tiling or padding, BF16 arithmetic, memory traffic and throughput are not inferred from MAC counts.',
        ],
        'new_schedule_evidence':'Derived candidate only; original CAL has hit-step inputs and the new step0 trajectory is unmeasured.',
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['at_alpha_one_eighth'],indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'extension_cost_arithmetic.json')
    main(ap.parse_args().output)
