"""Reproduce R15 public CPU/math/backend records without private assets/CUDA."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from probe_mean_contrast_representation import analytic_checks,mac_ledger

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/20261023'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9)


def validate_cpu(j,p):
    errors=[]
    def need(value,message):
        if not value:errors.append(message)
    need(j['new_original_W1_W2_calls']==j['new_native_FFN_calls']==j['new_model_generations']==j['new_fitted_parameters']==0,
         'Saved CPU study cannot execute/fit a new model')
    need(j['CUDA_initialized'] is False and j['native_registration'] is None and
         j['model_execution_eligibility_change'] is False,'CPU evidence cannot create native eligibility')
    need(j['existing_A_FP64_CPU_projection_calls']==4 and j['synthetic_reference_Q2_calls']==16,'CPU operation ledger changed')
    expected=analytic_checks(p['contrast_scales'])
    for a,b in zip(j['analytic_checks']['scales'],expected['scales']):
        need(all(close(a[k],b[k]) for k in b),'Smooth cancellation example does not reproduce')
    need(j['MAC_ledger']==mac_ledger(),'Both contrast factors/full four-row Q2 cost must be included')
    qr=j['quantizer_counterexample']['rows']
    need(len(qr)==8 and [r['contrast_scale'] for r in qr]==p['contrast_scales'],'Synthetic quantizer scale scope changed')
    for r in qr:
        need(r['true_anchor_Q2_values']==r['approx_anchor_Q2_values']==[7.]*4,'Quantizer anchor changed')
        measured=abs(statistics.mean(r['true_target_Q2_values'])-statistics.mean(r['approx_target_Q2_values']))
        need(measured==r['original_HiF4_mean_absolute_error'],'Pooled quantizer mean mismatch does not reproduce')
    need([r['original_HiF4_mean_absolute_error'] for r in qr]==[0.]+[.0625]*7,'Original quantizer counterexample changed')
    need({s['index'] for s in j['saved_scopes']}=={0,1,10,11},'Saved scope/source set changed')
    for s in j['saved_scopes']:
        raw=s['per_group']
        need(raw['group_ordinal']==list(range(64)) and all(len(v)==64 for v in raw.values()),'Missing/hidden saved group')
        need(all(0<=v<=1 for k in ('input_contrast_energy_fraction','input_contrast_top1_energy_fraction') for v in raw[k]),'Invalid energy fraction')
        need(close(math.fsum(raw['native_UB1_post_SSE']),s['native_UB1_post_SSE']),'Native baseline aggregation changed')
        need(close(math.fsum(raw['FP64_smooth_additive_common_post_floor_SSE']),s['FP64_smooth_additive_common_post_floor_SSE']),'Smooth reference aggregation changed')
        need(all(close(a-b,c) for a,b,c in zip(raw['native_UB1_post_SSE'],raw['FP64_smooth_additive_common_post_floor_SSE'],raw['FP64_smooth_reference_headroom'])),'Smooth headroom arithmetic changed')
        need(sum(v<0 for v in raw['FP64_smooth_reference_headroom'])==s['groups_smooth_floor_above_native_UB1'],'Negative smooth headroom groups hidden')
        need(close(sorted(raw['input_contrast_top1_energy_fraction'])[31],s['input_contrast_top1_energy_fraction_median']),'Lower-order torch median does not reproduce')
    for s in j['saved_summary']:
        ids={'old_training':{1,11},'old_development':{0,10},'all_exposed':{1,11,0,10}}[s['cohort']]
        chosen=[r for r in j['saved_scopes'] if r['index'] in ids]
        ub1=math.fsum(r['native_UB1_post_SSE'] for r in chosen)
        floor=math.fsum(r['FP64_smooth_additive_common_post_floor_SSE'] for r in chosen)
        need(close(ub1,s['native_UB1_post_SSE']) and close(floor,s['FP64_smooth_common_post_floor_SSE']) and close(floor/ub1,s['smooth_floor_over_native_UB1']),'Saved cohort denominator/reference changed')
        need(s['groups']==64*len(chosen) and sum(r['groups_smooth_floor_above_native_UB1'] for r in chosen)==s['groups_smooth_floor_above_native_UB1'],'Cohort group count changed')
    return errors


def validate_backend(j,p):
    errors=[]
    def need(value,message):
        if not value:errors.append(message)
    need(j['models_loaded']==j['new_model_generations']==j['fitted_parameters']==j['tensor_files_written']==0,
         'Synthetic backend cannot contain model weights/fits/videos')
    need(j['native_model_or_quality_acceptance'] is False and j['model_execution_eligibility_change'] is False,
         'Synthetic timing cannot become native method/video acceptance')
    names={'prepared_three_contrasts','prepared_mean','four_preactivation_rows','pooled_four_post_Q2_rows','inverse_gate_residual','full_pipeline'}
    need(len(j['fusion_controls'])==6 and {r['name'] for r in j['fusion_controls']}==names and
         all(r['BF16_bit_mismatches']==0 and r['finite'] for r in j['fusion_controls']),'Fused arithmetic controls failed/missing')
    need(j['all_controls_before_timing'] is True and len(j['original_QDQ_controls'])==12 and all(r['BF16_bit_mismatches']==0 for r in j['original_QDQ_controls']),'Original quantizer boundaries failed/missing')
    need(set(j['samples_ms'])==set(p['arms']) and len(p['arms'])==6,'Timing arm scope changed')
    for a in p['arms']:
        values=j['samples_ms'][a];med=statistics.median(values)
        need(len(values)==p['repetitions']==7 and all(v>0 for v in values),'Incomplete timing samples')
        need(j['median_ms'][a]==med and j['MAD_ms'][a]==statistics.median(abs(v-med) for v in values),'Timing/noise summary changed')
        need(j['speedup_over_same_run_dense'][a]==j['median_ms']['dense']/med and len(j['host_samples_ms'][a])==7,'Speed denominator/host samples changed')
    saving=j['median_ms']['mean_contrast_r256_unfused']-j['median_ms']['mean_contrast_r256_fused']
    margin=max(.5,3*max(j['MAD_ms'].values()))
    need(j['fused_saving_ms']==saving and j['noise_margin_ms']==margin and j['backend_fusion_effect']==(saving>=margin),'Frozen engineering decision changed')
    need(j['mean_contrast_MAC_ledger']==mac_ledger() and j['full_input_factor_MAC_fraction']==1536*(5120+13824)/(5120*13824),'Projection or nonlinear cost omitted')
    need(j['max_allocated_bytes']<=32*2**30,'Allocated memory cap exceeded')
    return errors


def main():
    errors=[]
    for stem,validator in [('CPU',validate_cpu),('backend',validate_backend)]:
        pp=BASE/f'R15_mean_contrast_{stem}_protocol.json'
        rp=BASE/f'R15_mean_contrast_{stem}_results.json'
        p,j=[json.loads(f.read_text()) for f in (pp,rp)]
        errors+=validator(j,p)
        if sha(pp)!=j['protocol_sha256']:errors.append('R15 frozen protocol changed: '+stem)
        bindings=p.get('source_sha256',{})
        if isinstance(bindings,str):bindings={'tools/probe_mean_contrast_representation.py':bindings}
        bindings.update({name:value for name,value in p.get('input_sha256',{}).items() if not name.startswith('.runtime/')})
        for name,value in bindings.items():
            if sha(ROOT/name)!=value:errors.append('R15 source/public input changed: '+name)
    record=json.loads((BASE/'R15_mechanism_record.json').read_text())
    for item in record['evidence']:
        if sha(ROOT/item['path'])!=item['sha256']:errors.append('R15 finding evidence changed: '+item['path'])
    current,triage=[json.loads((ROOT/p).read_text()) for p in ('research/CURRENT_DECISION.json','research/20261021/method_triage.json')]
    if current['latest_static_review']!=triage['latest_static_review']:errors.append('R15 live static pointers disagree')
    if record['id'] in current['eligible_model_method_ids'] or record['native_registration'] is not None:
        errors.append('R15 source/CPU/synthetic evidence cannot register itself as a model method')
    if errors:raise SystemExit('\n'.join(errors))
    print('R15: smooth cancellation, original HiF4 counterexample,256 saved groups and42 complete-chain synthetic timing samples CONSISTENT.')
    print('No new original-model FFN, fit, native policy or free-video claim; R14N0/D0 stay closed.')


if __name__=='__main__':main()
