"""Reproduce the public synthetic backend ledger without CUDA/private assets."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = 'research/20261022/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(result, protocol):
    errors = []
    def need(value, message):
        if not value: errors.append(message)
    need(result['models_loaded']==result['new_model_generations']==result['fitted_parameters']==0,
         'Synthetic benchmark cannot claim model execution or fitting')
    need(result['native_model_or_quality_acceptance'] is False, 'Synthetic timing is not video/model acceptance')
    need(result['all_controls_before_timing'] is True, 'All controls must precede timing')
    need(len(result['QDQ_controls'])==14 and all(r['bit_mismatches']==0 for r in result['QDQ_controls']),
         'Original quantizer boundary mismatch')
    expected = {'fixed/'+s for s in ('three_contrasts','mean','input_contrast','packed_points','pooled_V2','inverse_gate_writeback')}
    expected |= {'dynamic/'+s for s in ('three_contrasts','mean','input_contrast','score','selected_ids','packed_points','pooled_V2','inverse_gate_writeback')}
    expected |= {'full_pipeline/fixed','full_pipeline/dynamic'}
    need({c['name'] for c in result['fusion_controls']}==expected and len(result['fusion_controls'])==16,
         'Missing fusion control')
    for item in result['fusion_controls']:
        if item['name']=='dynamic/score':
            need(item['max_relative_difference_clamped_at1']<=5e-6, 'Norm reduction error exceeds declared tolerance')
        else:
            need(item['numeric_exact'] and item['max_absolute_difference']==0, 'Fusion changed declared arithmetic')
        need(item['pass'] is True, 'Failed control cannot be promoted')
    need(set(result['median_ms'])==set(protocol['arms']), 'Timing arm list changed')
    for arm in protocol['arms']:
        samples = result['samples_ms'][arm]
        median = statistics.median(samples)
        mad = statistics.median(abs(v-median) for v in samples)
        need(len(samples)==protocol['repetitions']==7 and all(v>0 for v in samples), 'Incomplete timing samples')
        need(median==result['median_ms'][arm] and mad==result['MAD_ms'][arm], 'Latency/noise summary does not reproduce')
        need(result['speedup_over_same_run_dense'][arm]==result['median_ms']['dense']/median, 'Speed denominator changed')
        need(len(result['host_samples_ms'][arm])==7, 'Missing host wall-time samples')
    saving = result['median_ms']['three_unfused_dynamic']-result['median_ms']['three_fused_dynamic']
    margin = max(.5,3*max(result['MAD_ms'].values()))
    need(result['same_operator_fusion_saving_ms']==saving and result['predeclared_noise_margin_ms']==margin,
         'Fusion effect or noise margin changed')
    need(result['backend_fusion_advance']==(saving>=margin), 'Frozen backend decision changed')
    m,d,h,g,k = [protocol[n] for n in ('M','D','H','stripes','two_point_stripes')]
    need(m==4*g and k==3958 and result['three_direction_projection_MAC_fraction']==.75*d/(2*h)+(2*g+k)/(2*m),
         'Reduced affine projection or extra W1 point omitted')
    need(result['max_allocated_bytes']<=32*2**30, 'Allocated resource cap exceeded')
    return errors


def main():
    path=ROOT/(BASE+'R14_three_direction_backend_results.json')
    protocol_path=ROOT/(BASE+'R14_three_direction_backend_protocol.json')
    result,protocol=[json.loads(p.read_text()) for p in (path,protocol_path)]
    errors=validate(result,protocol)
    if sha(protocol_path)!=result['protocol_sha256']:errors.append('Frozen protocol SHA changed')
    for name,expected in result['source_sha256'].items():
        if sha(ROOT/name)!=expected:errors.append('Executed source SHA changed: '+name)
    record_path=ROOT/(BASE+'R14_mechanism_record.json')
    record=json.loads(record_path.read_text())
    for item in record['evidence']:
        if sha(ROOT/item['path'])!=item['sha256']:errors.append('R14 evidence binding changed: '+item['path'])
    draft_path=ROOT/(BASE+'R14_native_value_DRAFT.json')
    ready_path=ROOT/(BASE+'R14_native_draft_CPU_readiness.json')
    draft,ready=[json.loads(p.read_text()) for p in (draft_path,ready_path)]
    if (draft['registered_model_route'] is not False or draft['execution_authority'] is not None
            or draft['new_model_generations']!=0 or record['native_registration'] is not None
            or record['model_execution_eligibility_change'] is not False):
        errors.append('Backend/draft cannot register or execute a model')
    if sha(ROOT/draft['analysis_source']['path'])!=draft['analysis_source']['sha256']:
        errors.append('Prospective training-only fixed-mask analysis binding changed')
    if (ready['torch_imported'] is not False or ready['new_model_generations']!=0
            or ready['current93_SHA_reverification'] is not False
            or ready['draft_sha256']!=sha(draft_path)
            or ready['source_sha256']!=sha(ROOT/'tools/check_r14_native_readiness.py')):
        errors.append('CPU readiness cannot be relabelled as fresh full-asset/model execution')
    current,triage=[json.loads((ROOT/p).read_text()) for p in
                    ('research/CURRENT_DECISION.json','research/20261021/method_triage.json')]
    if current['latest_static_review']!=triage['latest_static_review']:
        errors.append('R14 live evidence pointers disagree')
    if current['latest_static_review']['id']==record['id']:
        if current['latest_static_review']['result_sha256']!=sha(record_path):
            errors.append('R14 live mechanism binding changed')
    if errors:raise SystemExit('\n'.join(errors))
    print('R14:16 fusion controls,14 original QDQ checks,49 timing samples and noise-adjusted backend effect CONSISTENT.')
    print('Synthetic backend only; no Wan value, deployed policy, free video, or original-model FFN speed acceptance.')


if __name__=='__main__':
    main()
