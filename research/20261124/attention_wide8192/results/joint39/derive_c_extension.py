"""Static exact arithmetic for C's proposed schedule and explicit cache.

No new model execution or GPU timing. All three attention variants of interest
here use OUT-ABS / OUT-D and V128; no FIN-C0 augmented-value cost is assumed.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import json


def main(output):
    n, p, layers, heads, width = 31668, 64, 40, 40, 128
    alpha, beta = Fraction(1,8), Fraction(p,n)
    current = Fraction(1,2)+Fraction(3,4)*alpha+beta/2
    proposed = Fraction(1,4)+alpha+3*beta/4
    assert current-proposed == (1-alpha-beta)/4
    size = {'u0_BF16':n*heads*width*2*layers,
            'rho_FP32':n*heads*4*layers,
            'anchor_project31_BF16':n*heads*31*2*layers}
    out = {
        'evidence_type':'STATIC_EXACT_RATIONAL_MAC_AND_EXPLICIT_CACHE_ARITHMETIC',
        'model_executions':0,'latency_measured':False,
        'alpha':float(alpha),'beta':float(beta),
        'current_C_core_fraction':float(current),
        'proposed_C_core_fraction':float(proposed),
        'proposed_fraction_of_current_C_core':float(proposed/current),
        'saved_fraction_of_four_dense_cores':float(current-proposed),
        'formula_current':'1/2 + 3*alpha/4 + beta/2',
        'formula_proposed':'1/4 + alpha + 3*beta/4',
        'formula_saved':'(1-alpha-beta)/4',
        'calls_current':{'native_attention':2*layers,'recovery':2*layers,'P_full_key_queries':2*layers*p},
        'calls_proposed':{'native_attention':layers,'recovery':3*layers,'P_full_key_queries':3*layers*p},
        'explicit_low_cache_GiB':{k:float(Fraction(v,2**30)) for k,v in size.items()},
        'explicit_low_cache_total_GiB':float(Fraction(sum(size.values()),2**30)),
        'cache_scope':'Only u0, rho and projected anchor data. Excludes BlockMask and workspaces. Not a simultaneous peak measurement.',
        'assumptions':[
            'Proposed attention uses OUT-ABS at steps0/1, native at step2 and OUT-D at step3.',
            'FFN retains native anchors at steps0/2 on the proposed trajectory current inputs.',
            'Common selected density alpha is an interaction-count approximation.',
            'The actual native backend already returns full O/LSE; rare omitted-mass reconciliation remains additional work.',
            'Core MAC denominator excludes projections, fitting, routing, state, layouts, FFN and decoding.',
            'No full-cycle latency or step0 quality prediction is made from this arithmetic.',
            'Low-anchor data remain required after changing the high schedule.',
        ]
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'c_extension_arithmetic.json')
    main(ap.parse_args().output)
