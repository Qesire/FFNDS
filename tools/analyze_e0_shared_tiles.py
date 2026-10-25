"""Validate and summarize the completed contiguous-token mechanism study."""
import argparse
import json
from pathlib import Path
import statistics

from ffnds_observation_common import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]


def summary(cases):
    mean=lambda v:statistics.mean(v)
    out={'cases':len(cases), 'results':{}}
    for f,controls in cases[0]['results'].items():
        out['results'][f]={name:{k:mean(c['results'][f][name][k] for c in cases)
                               for k in values} for name,values in controls.items()}
    out['innovation_token_mean_energy_fraction']=mean(c['innovation_token_mean_energy_fraction'] for c in cases)
    out['innovation_over_current_output_energy']=mean(c['innovation_over_current_output_energy'] for c in cases)
    return out


def analyze(root):
    p=ROOT/'config/e0_shared_tile_saved_20261025.json';plan=json.loads(p.read_text())
    path=root/'saved_gpu_attempt1/RUN.json';gpu=json.loads(path.read_text())
    if gpu['status']!='PASS_SHARED_TILE_SAVED_DISCRIMINATION' or len(gpu['cases'])!=80:
        raise ValueError('Require complete frozen80 descriptions')
    if gpu['protocol_sha256']!=sha256(p) or gpu['diagnostic_projections']!=plan['diagnostic_projection_limits']:
        raise ValueError('Frozen analysis identity/budget mismatch')
    for name,digest in plan['bound_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Executed source changed')
    native=[];bindings=[{'HF_relative':'e0_shared_tiles_20261025/saved_gpu_attempt1/RUN.json','sha256':sha256(path)}]
    np=ROOT/'config/e0_contiguous_native_20261025.json';nplan=json.loads(np.read_text())
    for name,digest in nplan['driver_sources_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Native driver changed')
    for index in [0,12]:
        path=root/f'native_source{index:02d}_attempt1/RUN.json';d=json.loads(path.read_text())
        if d['status']!='PASS_NATIVE_CONTIGUOUS_OPERANDS_NOOP' or len(d['scopes'])!=20 or d['protocol_sha256']!=sha256(np):
            raise ValueError('Require both exact native controls')
        if not all(s['historical_uniform_all_five_fields_exact'] for s in d['scopes']):
            raise ValueError('Native sampled fields differ')
        native.append(d);bindings.append({'source_index':index,'HF_relative':f'e0_shared_tiles_20261025/native_source{index:02d}_attempt1/RUN.json','sha256':sha256(path)})
    if not all(x['assigned_GPU_resources']['sampled_isolation_pass'] for x in [gpu,*native]):
        raise ValueError('Assigned GPU isolation failed')
    cases=gpu['cases'];base=[c for c in cases if c['layout']=='contiguous' and c['row_tile']==128]
    return {'id':'E0_SHARED_TILE_PRE_W1_20261025','status':'COMPLETE',
            'analysis_sha256':sha256(Path(__file__)),'protocol_sha256':{'native':sha256(np),'saved':sha256(p)},
            'raw_bindings':bindings,'scope':{'sources':[0,12],'seed':42,'independent_source_clusters':2,
               'same_expert_pairs':20,'layout_tile_descriptions':80,'contiguous_rows_per_pair':384,
               'row_tiles':[16,64,128],'controls':'Other five original exposed sources per held source; shuffled positions are technical controls, not new independent sources.'},
            'actual_budget':{'new_full_generations':2,'failed_generations':0,'retries':0,
                 'extra_native_FFN_calls':0,'standalone_native_norm_calls':0,'new_decompositions':0,
                 'training_updates':0,'official_quality_calls':0,'new_model_downloads':0,
                 'diagnostic_projections':gpu['diagnostic_projections'],'peak_GPU_cards':2,'user_GPU_ceiling':4,
                 'new_HF_output_bytes':sum(f.stat().st_size for f in root.rglob('*') if f.is_file()),
                 'GPU_analysis_elapsed_seconds':gpu['elapsed_seconds']},
            'native_identity':{'latent_hashes_matched':8,'video_hashes_matched':2,
                'native_count_vectors_matched':2,'positions_with_five_fields_exact_on_original32_rows':40},
            'native_input_binding':gpu['native_bindings'],
            'max_relative_current_reconstruction_SSE':max(c['relative_current_reconstruction_SSE'] for c in base),
            'by_layout_tile':{f'{layout}_{tile}':summary([c for c in cases if c['layout']==layout and c['row_tile']==tile])
                              for layout,tile in [('contiguous',16),('contiguous',64),('contiguous',128),('shuffled_control',128)]},
            'by_block':{str(b):summary([c for c in base if c['block']==b]) for b in [0,10,20,30,39]},
            'by_source':{str(i):summary([c for c in base if c['source_index']==i]) for i in [0,12]},
            'by_expert':{str(s):summary([c for c in base if c['step']==s]) for s in [1,3]},
            'decomposition_by_block':{str(b):{
                'linear_nonlinear_rounding_energy_over_innovation':[statistics.mean(c['linear_nonlinear_rounding_energy_over_innovation'][k] for c in base if c['block']==b) for k in range(3)],
                'linear_nonlinear_cosine':statistics.mean(c['linear_nonlinear_cosine'] for c in base if c['block']==b)} for b in [0,10,20,30,39]},
            'cases':cases,'quality_or_speed_acceptance':False,'candidate_operator_selected':False,'stage_changed':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first validated summary')
    write_json(a.out,analyze(a.root))
