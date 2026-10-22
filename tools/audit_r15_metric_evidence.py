"""Recompute R14N0 metric scope/aggregation using public frozen scalar data only."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/20261022'

def load(name):
    return json.loads((BASE / name).read_text(encoding='utf-8'))

def best32(values):
    return set(sorted(range(64), key=lambda i: (-values[i], i))[:32])

def quantile(values, q):
    x=sorted(values); a=(len(x)-1)*q; i=int(a)
    return x[i] if i==len(x)-1 else x[i]+(a-i)*(x[i+1]-x[i])

def audit():
    result=load('R14N0_native_results.json')
    cases={r['index']:r for r in result['cases']}
    train=[cases[i] for i in [1,11]]; dev=[cases[i] for i in [0,10]]
    benefit=lambda c:[a-b for a,b in zip(c['one_point_SSE'],c['two_point_SSE'])]
    fixed=best32([math.fsum(benefit(c)[i] for c in train) for i in range(64)])
    common=best32([math.fsum(benefit(c)[i] for c in dev) for i in range(64)])
    per=[]
    for c in dev:
        assert all(len(c[k])==64 for k in ['one_point_SSE','two_point_SSE','UB1_SSE','pre_W1_contrast_norm2'])
        masks={'fixed':fixed,'dynamic':best32(c['pre_W1_contrast_norm2']),'oracle':best32(benefit(c)), 'posthoc_common':common}
        values={a:[c['two_point_SSE'][i] if i in mask else c['one_point_SSE'][i] for i in range(64)] for a,mask in masks.items()}
        values['UB1']=c['UB1_SSE']
        stats={a:dict(SSE=math.fsum(v),MSE=math.fsum(v)/(64*4*5120),stripe_SSE_p50=quantile(v,.5),stripe_SSE_p95=quantile(v,.95),stripe_SSE_max=max(v)) for a,v in values.items()}
        per.append(dict(source_index=c['index'],sampled_stripes=64,sampled_tokens=256,channels=5120,statistics=stats,beneficial_extra_point_groups=sum(v>0 for v in benefit(c)),oracle_better_than_UB1_stripes=sum(a<b for a,b in zip(values['oracle'],values['UB1']))))
    pooled={a:math.fsum(p['statistics'][a]['SSE'] for p in per) for a in per[0]['statistics']}
    common_old=load('R14N0_common_mask_reference.json')
    for key,field in [('fixed','registered_training_fixed_SSE'),('oracle','executed_native_oracle_SSE'),('posthoc_common','reference_common_SSE')]:
        assert math.isclose(pooled[key],common_old[field],rel_tol=1e-12)
    deficit=pooled['oracle']-pooled['UB1'];regret=pooled['dynamic']-pooled['oracle'];gap=pooled['dynamic']-pooled['UB1']
    assert math.isclose(deficit+regret,gap,rel_tol=1e-12)
    backend=load('R14_three_direction_backend_results.json')['median_ms']
    fractions=[.2,.5,.8]
    sf=backend['dense']/backend['three_fused_dynamic']
    # Illustrative counterexamples, deliberately not model data.
    counterexamples={
      'same_SSE_different_spatial_coherence': {'shared_error':[1,1,1,1],'contrast_error':[1,-1,1,-1],'both_SSE':4,'shared_mean_squared':1,'contrast_mean_squared':0},
      'temporal_static_score_rewards_freeze': {'assumption':'Official static-video adjacent-pixel MAE score; unchanged frames have MAE0','score_for_frozen_frames':1,'not_claim':'Good motion or overall video quality'},
      'pooled_vs_equal_source_ratio': {'reference_SSE':[1,100],'candidate_SSE':[2,90],'ratio_of_sums':92/101,'mean_source_ratios':(2+.9)/2},
      'eta_zero_not_population_certainty': {'repeated_numeric_difference':0,'inference':'Determinism on repeated input only; no source-sampling standard error'},
      'conditional_Amdahl':{'operator_speedup':sf,'unmeasured_FFN_fraction':{str(p):1/(1-p+p/sf) for p in fractions},'scope':'Illustrative p only, not a prediction of actual Wan speed'}}
    return dict(review='R15_METRIC_AUDIT_20261023',source_commit='aebd37f749b484f0caaf6224c0a1e507e327cf3f',source_sha256={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['R14N0_native_results.json','R14N0_common_mask_reference.json','R14_three_direction_backend_results.json']},per_source=per,pooled=pooled,relative_local_changes={a:pooled[a]/pooled['UB1']-1 for a in ['fixed','dynamic','oracle']},decomposition=dict(action_family_gap_to_UB1=deficit,selector_regret_within_family=regret,total_dynamic_gap=gap,action_family_fraction_of_gap=deficit/gap,selector_fraction_of_gap=regret/gap,train_to_development_common_part=pooled['fixed']-pooled['posthoc_common'],common_to_per_input_part=pooled['posthoc_common']-pooled['oracle'],transfer_share_of_fixed_oracle_gap=(pooled['fixed']-pooled['posthoc_common'])/(pooled['fixed']-pooled['oracle'])),aggregation=dict(ratio_of_sums=pooled['dynamic']/pooled['UB1'],equal_source_mean_ratio=statistics.mean(p['statistics']['dynamic']['SSE']/p['statistics']['UB1']['SSE'] for p in per),source_count=2,formal_population_CI=None,post_NMSE=None,unavailable='Teacher tensor energies/whole scenes/free candidate videos not in public scalar records; no reconstructed NMSE or video result.'),backend=dict(synthetic_same_run_ms=backend,synthetic_dynamic_over_fixed_latency=backend['three_fused_dynamic']/backend['three_fused_fixed'],synthetic_UB1_over_dynamic_speedup=backend['UB1_half_R256']/backend['three_fused_dynamic'],native_model_speedup=None),counterexamples=counterexamples,scope='Saved scalar recomputation and explicit illustrative algebra only. Row additivity and registered native composition controls establish finite sampled action values, not video/Pareto/global bounds. Posthoc common mask unexecuted. No historical gate changed.',new_model_runs=0,new_native_FFN_calls=0)

if __name__=='__main__':
    target=ROOT/'research/20261023/metric_evidence_recalculation_R15.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    value=audit();target.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':'PASS_SAVED_SCALAR_RECOMPUTATION','decomposition':value['decomposition'],'aggregation':value['aggregation'],'backend':value['backend']},ensure_ascii=False))
