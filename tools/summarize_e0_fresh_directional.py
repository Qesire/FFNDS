"""Close the existing six-pair native scope; JSON/hash-only, no new experiment."""
import argparse
import json
import math
from pathlib import Path

from ffnds_paired_video_strict_gate import inspect
from summarize_e0_multisource_directional import bands_join,local_summary,near,read,sha,video_vector

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='config/e0_fresh_official_directional_native_20261026_v1.json'
PROTOCOL_SHA='a0d666545296c1a2de4260fd6b3ceed9bd4239b12b689ac1d5ed94a9c0f2d813'
PASS='PASS_NATIVE_MULTISOURCE_AND_CORE_METRICS_NOT_QUALITY'


def summarize(hf):
    plan=read(ROOT/PROTOCOL,PROTOCOL_SHA)
    study=Path(plan['canonical_study_root'])
    if not study.is_relative_to(hf/'ffnds-derived'):
        raise ValueError('Frozen study root differs from HF cache')
    ledger=read(study/'STUDY.json')
    if (ledger['protocol_sha256']!=PROTOCOL_SHA or ledger['id']!=plan['id']
            or [r['pair'] for r in ledger['attempts']]!=[p['id'] for p in plan['pairs']]
            or any(r['status']!=PASS for r in ledger['attempts'])):
        raise ValueError('All six original fixed pairs must finish before consolidation')
    audit=read(ROOT/plan['source_audit'],plan['source_audit_sha256'])
    inputs=read(ROOT/plan['input_default'],plan['input_default_sha256'])
    pairs=[]
    all_reports=[]
    for entry,pair in zip(ledger['attempts'],plan['pairs']):
        d=study/pair['id']
        run=read(d/'RUN.json',entry['RUN_sha256'])
        if (run['status']!=PASS or run['protocol_sha256']!=PROTOCOL_SHA
                or run['pair']!=pair or run['source_id']!=pair['source_id']
                or run['seed']!=pair['seed'] or run['quality_accepted']
                or run['new_videos']!=2 or run['bank_RUN_sha256']!=plan['bank_RUN_sha256']
                or run['source_role']!=pair['role'] or run.get('original_dense_receipt') is not None
                or run['assigned_GPU_resources']['sampled_isolation_pass'] is not True):
            raise ValueError('Actual fixed source/seed/bank/receipt identity drift')
        original=next(r for r in inputs if r['source_index']==pair['source_index'])
        if (original['source_id']!=run['source_id'] or sha(original['image_path'])!=run['source_id']
                or original['official_metadata_row']!=pair['source_index']):
            raise ValueError('Original condition image identity drift')
        if inspect(d/'STRICT_BUNDLE.json'):
            raise ValueError('Actual strict native video/model gate failed')
        read(d/'NATIVE_RUN.json',run['native_RUN_sha256'])
        read(d/run['observation']['path'],run['observation']['sha256'])
        trace=read(d/'TRACE.json',run['trace_sha256'])
        if (len(trace['sites'])!=160 or len(trace['steps'])!=4
                or {(s['step'],s['block']) for s in trace['sites']}!={(i,j) for i in range(4) for j in range(40)}
                or trace['source_index']!=pair['source_index'] or trace['seed']!=pair['seed']):
            raise ValueError('Actual complete40x4 native pair coverage required')
        bands=read(d/run['disjoint_bands']['path'],run['disjoint_bands']['sha256'])
        steps=[]
        for step in range(4):
            flow=read(d/f'flow_s{step}.json')
            if flow!=run['counterfactual_records'][step] or flow['status']!='PASS_FLOW_AND_NATIVE_MAP_IDENTITIES':
                raise ValueError('Flow receipt/RUN identity drift')
            if sha(d/flow['saved_tensors']['path'])!=flow['saved_tensors']['sha256']:
                raise ValueError('Opaque native state payload checksum drift')
            n=flow['native_map_geometry']
            t=next(t for t in trace['steps'] if t['step']==step)
            near(t['delta_after_SSE'],n['actual_next_delta_SSE'])
            bc=next(c for c in bands['cases'] if c['step']==step)
            if bc['state_sha256']!=flow['saved_tensors']['sha256']:
                raise ValueError('Band/native state identity drift')
            near(sum(b['input_delta_SSE'] for b in bc['bands']),flow['geometry']['delta_SSE'])
            if step==0:
                if n['dense_discrete_map_energy_gain'] is not None or n['dense_map_response_SSE']!=0 or flow['geometry']['delta_SSE']!=0:
                    raise ValueError('Initial neutral state geometry drift')
            else:
                near(n['dense_discrete_map_energy_gain'],n['dense_map_response_SSE']/flow['geometry']['delta_SSE'])
            sites=[s for s in trace['sites'] if s['step']==step]
            steps.append({'step':step,'expert':sites[0]['expert'],
                'local_FFN':local_summary(sites,'FFN'),'native_writeback':local_summary(sites,'native_writeback'),
                'input_delta_SSE':flow['geometry']['delta_SSE'],
                'dense_native_map_energy_gain':n['dense_discrete_map_energy_gain'],
                'P_SSE':n['dense_map_response_SSE'],'N_SSE':n['candidate_effect_SSE'],
                'signed_cross':2*n['dense_map_candidate_effect_dot'],
                'candidate_net_energy_effect':n['candidate_effect_SSE']+2*n['dense_map_candidate_effect_dot'],
                'actual_next_delta_SSE':n['actual_next_delta_SSE'],
                'bands':bands_join(bc['bands'],n),
                'delta_flow_cosine':t['delta_flow_dot']/math.sqrt(t['delta_before_SSE']*t['flow_difference_SSE']) if t['delta_before_SSE'] else None})
        all_reports += [read(d/(b+'_metrics.json')) for b in ['teacher','student']]
        bound=['RUN.json','NATIVE_RUN.json','TRACE.json','OBSERVATION.json','DISJOINT_BANDS.json',
               'BUNDLE.json','STRICT_BUNDLE.json','teacher_metrics.json','student_metrics.json']
        pairs.append({'pair_id':pair['id'],'source_index':pair['source_index'],'seed':pair['seed'],
            'source_id':run['source_id'],'source_role':run['source_role'],
            'official_dimensions':original['official_dimensions'],'stratum':original['stratum'],
            'camera_instruction':original['camera_instruction'],'original_prompt':original['prompt'],
            'HF_relative':str(d.relative_to(hf)), 'files_sha256':{name:sha(d/name) for name in bound},
            'local_FFN':local_summary(trace['sites'],'FFN'),
            'native_writeback':local_summary(trace['sites'],'native_writeback'),
            'larger_local_error_sites':[{k:s[k] for k in ['step','block','expert']} for s in trace['sites'] if s['FFN']['student_operator_SSE']>s['FFN']['teacher_operator_SSE']],
            'steps':steps,
            'layer_summary':[{'block':b,'FFN':local_summary([s for s in trace['sites'] if s['block']==b],'FFN')} for b in range(40)],
            'video_function_vector':video_vector(d),'cost':run['cost'],'cost_scope':run['cost_scope'],
            'native_runtime_counts':run['native_runtime_counts'],'observer_counts':run['observer_counts'],
            'sampled_GPU_isolation_pass':run['assigned_GPU_resources']['sampled_isolation_pass'],
            'dense_control_scope':run['dense_control_scope'],'core_and_strict_gate_pass':True,'quality_accepted':False})
    output_bytes=sum(p.stat().st_size for p in study.rglob('*') if p.is_file())
    if output_bytes>plan['max_HF_output_gib']*1024**3:
        raise ValueError('Completed scope exceeds frozen output budget')
    qdq=sum(r['native_runtime_counts']['activation_qdq_calls']+r['observer_counts']['native_QDQ']
            -r['observer_counts']['dense_main_GEMM']-r['observer_counts']['dense_shadow_GEMM'] for r in pairs)
    gemm=sum(r['native_runtime_counts']['linear_forward_calls']+r['observer_counts']['candidate_main_GEMM']
             +r['observer_counts']['candidate_shadow_GEMM'] for r in pairs)
    if (qdq!=plan['predicted_native_budget']['native_QDQ_calls_all_target_linear']
            or gemm!=plan['predicted_native_budget']['native_GEMM_all_target_linear_including_affine']):
        raise ValueError('Actual native paid-operation ledger does not close')
    return {'id':'E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS_20261027','status':'COMPLETE_FIXED_NATIVE_AND_CORE_VIDEO_NOT_QUALITY',
        'protocol':PROTOCOL,'protocol_sha256':PROTOCOL_SHA,'STUDY_sha256':sha(study/'STUDY.json'),
        'analysis_source_sha256':sha(__file__),
        'shared_analysis_source_sha256':sha(ROOT/'tools/summarize_e0_multisource_directional.py'),
        'input_cohort_sha256':plan['input_default_sha256'],'source_exposure_audit_sha256':plan['source_audit_sha256'],
        'scope':'Four new-to-audited-frozen-project-input-cohort identities;6nested source×seed pairs.908/425have seed42/54;376/437seed42. No unknown-pretraining or whole-population generalization claim.',
        'pairs':pairs,
        'actual_budget':{'distinct_source_identities':4,'source_seed_pairs':6,'full_four_step_trajectories':12,
            'new_videos':12,'paired_sites':960,'additional_complete_dense_flows':24,'VAE_decodes':12,
            'decoded_metric_frames':sum(r['metrics']['artifact']['verified_full_decode_frames'] for r in all_reports),
            'frame_transitions':sum(r['metrics']['artifact']['verified_full_decode_frames']-1 for r in all_reports),
            'native_QDQ_calls_all_target_linear':qdq,'native_GEMM_calls_including_affine':gemm,
            'continuous_FP32_GEMM':sum(r['observer_counts']['continuous_GEMM'] for r in pairs),
            'peak_simultaneous_assigned_GPUs':1,'sampled_peak_physical_GPU_MiB':max(r['cost']['peak_gpu_memory_mib'] for r in pairs),
            'HF_output_bytes':output_bytes,'new_fits':0,'training_updates':0,'weight_decompositions':0,
            'official_quality_calls':0,'retries':0},
        'quality_accepted':False,'official_VBench_I2V':'NOT_RUN','human_full_video_review':'NOT_RUN',
        'analysis_new_model_calls':0,'analysis_new_tensor_numerical_work':0,'analysis_new_decodes':0,
        'comparison_limit':'First dense path for each fresh source-seed uses frozen native controller/step0neutrality and prior4source identity controls;no same-source uninstrumented historical byte-identity claim.',
        'posthoc_limit':'Finite response and disjoint latent bands are descriptive instruments;not semantic function masks,global stability bounds,selected new algorithm or deployment latency.',
        'stop_instruction':'User requested finish this existing scope,upload GitHub main,then pause;no subsequent experimental scope launched.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root',type=Path,default=Path.home()/'.cache/huggingface')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.hf_root.resolve())
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'source_identities':4,'source_seed_pairs':6,'paired_sites':960}))
