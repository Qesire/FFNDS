"""Prospective matched-count stripe-value analysis, with training-only fixed mask.

The caller must supply actual native losses. This tool does not infer them
from output splices or ideal means and never runs a model.
"""
import argparse
import json
import math
from pathlib import Path


def analyze(cases, train_indices=(1,11), evaluation_indices=(0,10), count=32):
    if tuple(train_indices)!=(1,11) or tuple(evaluation_indices)!=(0,10) or count!=32:
        raise ValueError('Frozen training/evaluation roles and32-point budget required')
    lookup={c['index']:c for c in cases}
    if len(cases)!=4 or len(lookup)!=4 or set(lookup)!=set(train_indices)|set(evaluation_indices):
        raise ValueError('Exactly the frozen four exposed development sources required')
    keys=('one_point_SSE','two_point_SSE','UB1_SSE','pre_W1_contrast_norm2')
    for case in cases:
        if any(len(case[k])!=64 or any(not math.isfinite(v) or v<0 for v in case[k]) for k in keys):
            raise ValueError('64 finite nonnegative stripe losses/signals required')
    improvement=[math.fsum(lookup[i]['one_point_SSE'][g]-lookup[i]['two_point_SSE'][g] for i in train_indices) for g in range(64)]
    fixed=sorted(sorted(range(64),key=lambda g:(-improvement[g],g))[:count])
    rows=[]
    for index in (*train_indices,*evaluation_indices):
        case=lookup[index]
        dynamic=sorted(sorted(range(64),key=lambda g:(-case['pre_W1_contrast_norm2'][g],g))[:count])
        oracle=sorted(sorted(range(64),key=lambda g:(-(case['one_point_SSE'][g]-case['two_point_SSE'][g]),g))[:count])
        risk=lambda selected:math.fsum(case['two_point_SSE'][g] if g in selected else case['one_point_SSE'][g] for g in range(64))
        rows.append({'index':index,'role':'training_for_fixed_mask' if index in train_indices else 'nonfit_exposed_development',
                     'UB1_SSE':math.fsum(case['UB1_SSE']),'fixed_SSE':risk(fixed),'dynamic_SSE':risk(dynamic),'oracle_SSE':risk(oracle),
                     'dynamic_two_point_stripes':dynamic,'oracle_two_point_stripes':oracle})
    totals={k:math.fsum(r[k] for r in rows if r['index'] in evaluation_indices) for k in ('UB1_SSE','fixed_SSE','dynamic_SSE','oracle_SSE')}
    return {'fixed_two_point_stripes':fixed,'fixed_training_indices':list(train_indices),'evaluation_indices':list(evaluation_indices),
            'two_point_count_per64':count,'per_source':rows,'evaluation_totals':totals,
            'evaluation_dynamic_gain_over_fixed':1-totals['dynamic_SSE']/totals['fixed_SSE'] if totals['fixed_SSE'] else None,
            'evaluation_dynamic_gain_over_UB1':1-totals['dynamic_SSE']/totals['UB1_SSE'] if totals['UB1_SSE'] else None,
            'scope':'Only valid for supplied actual native, same-row, same-count local losses. Old sources are development-exposed. A sampled-domain value is not full-scene policy, downstream flow or free-video quality.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Refuse to overwrite prior analysis')
    raw=json.loads(args.input.read_text())
    if raw['evidence_level']!='ACTUAL_NATIVE_COMPLETE_STRIPE_VALUES':
        raise ValueError('Ideal geometry/synthetic labels cannot substitute for native action values')
    result=analyze(raw['cases']);result['evidence_level']=raw['evidence_level']
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
