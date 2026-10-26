"""Join six completed native source units using JSON and artifact hashes only.

No tensor decoding, learned models, new metric evaluation or fitting. Old
two-source bands were obtained by differences of nested orthogonal masks;
four new sources use direct disjoint FFT masks. Their shared terms are joined
explicitly, while unavailable old input-band energies remain null.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from ffnds_paired_video_strict_gate import inspect

ROOT = Path(__file__).resolve().parents[1]
OLD_PAIRED = 'research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.json'
OLD_FLOW = 'research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.json'
OLD_BANDS = 'research/20261026/E0_DISJOINT_SIGNED_BANDS.json'
OLD_SPECTRUM = 'research/20261026/E0_SAVED_FLOW_SPECTRUM.json'
PHASE1_PROTOCOL = 'config/e0_multisource_directional_native_20261026_v1.json'
PHASE1_SHA = '931b8162f8379193697a00899b617f3851f70ef0182ca7126c2ea4511a7f4e77'
BANDS = dict(zip(
    ['DC_all_axes', 'static_spatial_low_excluding_DC', 'dynamic_spatial_low', 'spatial_high_remainder'],
    ['all_axis_DC', 'temporal_DC_spatial_low_nonDC', 'temporal_nonDC_spatial_low', 'remaining_spatial_high']))
DIMENSIONS = [16, 4608, 92480, 1929648]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path, digest=None):
    path = Path(path)
    if digest is not None and sha(path) != digest:
        raise ValueError('Artifact SHA drift: ' + str(path))
    return json.loads(path.read_text())


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('Native identity join differs: ' + str((a, b)))


def bands_join(rows, native, old=False):
    result = []
    for row in rows:
        result.append({'name': BANDS[row['name']] if old else row['name'],
            'real_dimensions': row['dimension'] if old else row['real_dimensions'],
            'P_SSE': row['P_SSE'], 'N_SSE': row['N_SSE'],
            'signed_cross': row['signed_cross'],
            'next_delta_SSE': row['next_SSE'] if old else row['next_delta_SSE'],
            'input_delta_SSE': None if old else row['input_delta_SSE']})
    if [r['name'] for r in result] != list(BANDS.values()) or [r['real_dimensions'] for r in result] != DIMENSIONS:
        raise ValueError('Exactly four fixed mutually disjoint latent bands required')
    terms = {'P_SSE': native['dense_map_response_SSE'],
             'N_SSE': native['candidate_effect_SSE'],
             'signed_cross': 2 * native['dense_map_candidate_effect_dot'],
             'next_delta_SSE': native['actual_next_delta_SSE']}
    for key, target in terms.items():
        scale = max(2*math.sqrt(terms['P_SSE']*terms['N_SSE']), 1) if key == 'signed_cross' else max(abs(target), 1)
        if abs(sum(r[key] for r in result)-target) > 1e-10*scale:
            raise ValueError('Disjoint band closure drift: ' + key)
    for r in result:
        near(r['P_SSE']+r['N_SSE']+r['signed_cross'], r['next_delta_SSE'])
        r['next_energy_fraction'] = r['next_delta_SSE']/terms['next_delta_SSE']
    return result


def local_summary(sites, boundary):
    values = [s[boundary] for s in sites]
    counts = {k: 0 for k in ['student_smaller', 'equal', 'student_larger']}
    for s in values:
        a, b = s['student_operator_SSE'], s['teacher_operator_SSE']
        counts['student_smaller' if a < b else 'student_larger' if a > b else 'equal'] += 1
    return dict(counts, sites=len(sites),
        student_teacher_operator_SSE_ratio_median=median(s['student_operator_SSE']/s['teacher_operator_SSE'] for s in values),
        teacher_operator_relative_dense_teacher_energy_median=median(s['teacher_operator_SSE']/s['reference_energy'] for s in values),
        student_operator_relative_dense_teacher_energy_median=median(s['student_operator_SSE']/s['reference_energy'] for s in values))


def video_vector(directory):
    reports = [read(directory/(x+'_metrics.json')) for x in ['teacher', 'student']]
    dense, candidate = [r['metrics'] for r in reports]
    for r in reports:
        if (r['metrics']['artifact']['verified_full_decode_frames'] != 81
                or [x['frame'] for x in r['per_frame']] != list(range(81)) or r['quality_accepted']):
            raise ValueError('Full81-frame unaccepted core evidence required')
    def m(branch, family, key):
        return branch[family][key]['mean']
    return {'video_SHA256': {label:r['candidate']['sha256'] for label,r in zip(['dense','candidate'],reports)},
        'RGB_RMSE_mean': m(candidate,'paired_spatial','RGB_RMSE_0to1'),
        'adjacent_RGB_delta_RMSE_mean': m(candidate,'paired_temporal','adjacent_delta_RGB_RMSE_0to1'),
        'paired_flow_L2_px_mean': m(candidate,'paired_motion','downsampled_Farneback_flow_L2_px'),
        'flow_magnitude_mean_of_transition_ratios': m(candidate,'paired_motion','candidate_to_reference_flow_magnitude_ratio'),
        'flow_magnitude_ratio_of_means': m(candidate,'temporal_motion','downsampled_Farneback_mean_flow_px')/m(dense,'temporal_motion','downsampled_Farneback_mean_flow_px'),
        'edge_energy_mean_of_frame_ratios': m(candidate,'paired_spatial','candidate_to_reference_laplacian_energy_ratio'),
        'edge_energy_ratio_of_means': m(candidate,'spatial_detail','laplacian_abs_energy_candidate')/m(dense,'spatial_detail','laplacian_abs_energy_candidate'),
        'adjacent_RGB_variation_ratio_of_means': m(candidate,'temporal_motion','adjacent_RGB_MAE_0to1')/m(dense,'temporal_motion','adjacent_RGB_MAE_0to1'),
        'signed_encoded_luma_drift_mean': m(candidate,'paired_spatial','signed_encoded_luma_drift_0to1'),
        'motion_compensated_paired_luma_difference_mean': m(candidate,'motion_diagnostics','paired_compensated_encoded_luma_delta_abs'),
        'low_reference_motion_luma_delta_mean': m(candidate,'static_region','uncompensated_low_reference_motion_luma_delta_abs'),
        'dense_and_candidate_temporal_luma_compensated_MAE': [m(r,'motion_diagnostics','compensated_encoded_luma_MAE') for r in [dense,candidate]],
        'windows16': {'dense':reports[0]['windows16'], 'candidate':reports[1]['windows16']},
        'official_VBench_I2V':'NOT_RUN', 'human_full_video_review':'NOT_RUN', 'quality_accepted':False,
        'scope':'Encoded RGB/luma and downsampled uint8 Farneback diagnostics; latent bands are not semantic masks and small motion may reflect texture loss.'}


def summarize(hf):
    derived = hf/'ffnds-derived'
    old_pair = read(ROOT/OLD_PAIRED)
    old_flow = read(ROOT/OLD_FLOW)
    old_bands = read(ROOT/OLD_BANDS)
    spectrum = read(ROOT/OLD_SPECTRUM)
    plan = read(ROOT/PHASE1_PROTOCOL, PHASE1_SHA)
    study = derived/'e0_multisource_directional_20261026/attempt1'
    ledger = read(study/'STUDY.json')
    if (ledger['protocol_sha256'] != PHASE1_SHA
            or [r['pair'] for r in ledger['attempts']] != [p['id'] for p in plan['pairs']]):
        raise ValueError('Complete fixed four-source phase1 ledger required')
    results = []
    for source in [0,4,8,12,16,20]:
        old = source in [0,12]
        if old:
            pr = next(r for r in old_pair['runs'] if r['source_index'] == source)
            fr = next(r for r in old_flow['runs'] if r['source_index'] == source)
            video_dir = derived/pr['raw']['HF_relative']
            for name,digest in pr['raw']['files_sha256'].items():
                read(video_dir/name,digest)
            flow_dir = derived/f'e0_whole_flow_counterfactual_20261026/attempt1/dc_source{source:02}'
            run = read(flow_dir/'RUN.json',fr['RUN_sha256'])
            paired_run = read(video_dir/'RUN.json')
            if run['protocol_sha256'] != old_flow['protocol_sha256'] or run['trajectory'] != paired_run['trajectory']:
                raise ValueError('Two old native scopes must have identical full trajectories')
        else:
            video_dir = flow_dir = study/f'dc_source{source:02}'
            entry = next(e for e in ledger['attempts'] if e['pair'] == video_dir.name)
            run = read(video_dir/'RUN.json',entry['RUN_sha256'])
            if (run['status'] != 'PASS_NATIVE_MULTISOURCE_AND_CORE_METRICS_NOT_QUALITY'
                    or run['status'] != entry['status'] or run['protocol_sha256'] != PHASE1_SHA):
                raise ValueError('Completed registered phase1 native result required')
            paired_run = run
        if (run['pair']['source_index'] != source or run['seed'] != source+42
                or run['bank_RUN_sha256'] != plan['bank_RUN_sha256'] or run['quality_accepted']):
            raise ValueError('Source/seed/fixed-bank/status identity drift')
        if inspect(video_dir/'STRICT_BUNDLE.json'):
            raise ValueError('Strict native media/model identity gate failed')
        read(video_dir/'NATIVE_RUN.json',paired_run['native_RUN_sha256'])
        read(video_dir/paired_run['observation']['path'],paired_run['observation']['sha256'])
        trace = read(video_dir/'TRACE.json',paired_run['trace_sha256'])
        if (len(trace['sites']) != 160 or len(trace['steps']) != 4
                or {(s['step'],s['block']) for s in trace['sites']} != {(i,j) for i in range(4) for j in range(40)}
                or trace['source_index'] != source or trace['seed'] != run['seed']):
            raise ValueError('Actual complete40x4 paired trace required')
        if not old:
            bands = read(flow_dir/run['disjoint_bands']['path'],run['disjoint_bands']['sha256'])['cases']
        steps = []
        for step in range(4):
            flow = read(flow_dir/f'flow_s{step}.json')
            if flow != run['counterfactual_records'][step] or flow['status'] != 'PASS_FLOW_AND_NATIVE_MAP_IDENTITIES':
                raise ValueError('Native saved flow JSON/RUN join failed')
            tensor_sha = sha(flow_dir/flow['saved_tensors']['path'])
            if tensor_sha != flow['saved_tensors']['sha256']:
                raise ValueError('Native state tensor identity drift')
            n = flow['native_map_geometry']
            ts = next(x for x in trace['steps'] if x['step'] == step)
            near(ts['delta_after_SSE'],n['actual_next_delta_SSE'])
            if old:
                bc = next(x for x in old_bands['cases'] if x['source_index'] == source and x['step'] == step)
                spec = next(x for x in spectrum['cases'] if x['source_index'] == source and x['step'] == step)
                if spec['tensor_sha256'] != tensor_sha:
                    raise ValueError('Historical band/state tensor identity drift')
            else:
                bc = next(x for x in bands if x['step'] == step)
                if bc['state_sha256'] != tensor_sha:
                    raise ValueError('Direct disjoint band/state tensor identity drift')
            rows = bands_join(bc['bands'],n,old)
            sites = [s for s in trace['sites'] if s['step'] == step]
            steps.append({'step':step,'expert':sites[0]['expert'],
                'local_FFN':local_summary(sites,'FFN'),'native_writeback':local_summary(sites,'native_writeback'),
                'input_delta_SSE':flow['geometry']['delta_SSE'],
                'dense_native_map_energy_gain':n['dense_discrete_map_energy_gain'],
                'P_SSE':n['dense_map_response_SSE'],'N_SSE':n['candidate_effect_SSE'],
                'signed_cross':2*n['dense_map_candidate_effect_dot'],
                'candidate_net_energy_effect':n['candidate_effect_SSE']+2*n['dense_map_candidate_effect_dot'],
                'actual_next_delta_SSE':n['actual_next_delta_SSE'], 'bands':rows,
                'delta_flow_cosine':ts['delta_flow_dot']/math.sqrt(ts['delta_before_SSE']*ts['flow_difference_SSE']) if ts['delta_before_SSE'] else None,
                'spatiotemporal_3x3_strata':[
                    {'frame_third':f,'vertical_third':v,
                     'FFN':local_summary([{'FFN':next(t['FFN'] for t in s['spatiotemporal_strata'] if t['frame_third']==f and t['vertical_third']==v)} for s in sites],'FFN')}
                    for f in range(3) for v in range(3)]})
        bound = ['RUN.json','TRACE.json','NATIVE_RUN.json','BUNDLE.json','STRICT_BUNDLE.json','teacher_metrics.json','student_metrics.json']
        results.append({'source_index':source,'seed':run['seed'],'source_id':run['source_id'],
            'role':'EXPOSED_NONCALIBRATION' if old else 'EXPOSED_CALIBRATION',
            'stratum':'subject' if source in [0,4,8] else 'background',
            'raw_HF_relative':str(flow_dir.relative_to(hf)),
            'video_HF_relative':str(video_dir.relative_to(hf)),
            'video_and_trace_bindings':{n:sha(video_dir/n) for n in bound},
            'flow_RUN_sha256':sha(flow_dir/'RUN.json'),
            'local_FFN':local_summary(trace['sites'],'FFN'),
            'native_writeback':local_summary(trace['sites'],'native_writeback'),
            'larger_local_error_sites':[{k:s[k] for k in ['step','block','expert']} for s in trace['sites'] if s['FFN']['student_operator_SSE']>s['FFN']['teacher_operator_SSE']],
            'steps':steps,'layer_summary':[{'block':b,'FFN':local_summary([s for s in trace['sites'] if s['block']==b],'FFN')} for b in range(40)],
            'video_function_vector':video_vector(video_dir),
            'native_diagnostic_cost':{k:run[k] for k in ['setup_seconds','observation_wall_seconds','native_runtime_counts','observer_counts','cost_scope']},
            'band_evidence_origin':'DIFFERENCES_OF_NESTED_ORTHOGONAL_MASKS_INPUT_ENERGY_UNAVAILABLE' if old else 'DIRECT_DISJOINT_UNITARY_FFT_WITH_INPUT_ENERGY',
            'core_strict_gates_pass':True,'quality_accepted':False})
    return {'id':'E0_SIX_SOURCE_DIRECTIONAL_RESULTS_20261027',
        'status':'PASS_SIX_EXPOSED_SOURCE_JSON_ARTIFACT_JOIN_NOT_QUALITY',
        'analysis_source_sha256':sha(__file__),
        'source_documents_sha256':{p:sha(ROOT/p) for p in [OLD_PAIRED,OLD_FLOW,OLD_BANDS,OLD_SPECTRUM,PHASE1_PROTOCOL]},
        'phase1_STUDY_sha256':sha(study/'STUDY.json'), 'sources':results,
        'source_count':6,'source_seed_paths':6,'paired_sites':960,
        'newly_added_native_sites':640,'newly_added_videos':8,
        'new_metric_evaluations_by_this_analysis':0,'new_tensor_numerical_work_by_this_analysis':0,
        'new_model_calls':0,'GPU_calls':0,'quality_accepted':False,
        'unit':'Six exposed source identities. All24CAL24 sources have historical exposure;4/8/16/20 calibrated the bank. Sites,steps,frames and two variants are nested observations, not independent source samples.',
        'mechanism_scope':'Finite response at actual displaced native states; energy gain is not a global Jacobian bound. P+N is a signed state identity, not FFN-only causal attribution or video quality. No ranking by global cross sign or SSE alone.',
        'missing':'Official model metrics and human complete-video review NOT_RUN; no new deployment timing or quality acceptance.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf-root',type=Path,default=Path.home()/'.cache/huggingface')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.hf_root.resolve())
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'source_units':6,'paired_sites':960,'new_model_calls':0}))
