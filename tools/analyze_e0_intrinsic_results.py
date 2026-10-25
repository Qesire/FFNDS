"""Summarize completed full-hidden GPU and passive full-M native observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def summarize_cases(cases):
    out={'cases':len(cases),'neuron':{},'group':{}}
    fields=['net_over_neuron_diagonal','net_over_group_diagonal','group_cross_energy_over_net',
            'negative_aggregate_group_pair_fraction','coefficient_nonzero_fraction']
    out.update({k:float(np.mean([c['description'][k] for c in cases])) for k in fields})
    for unit in ['neuron','group']:
        for fraction,controls in cases[0]['description'][unit].items():
            out[unit][fraction]={}
            for name,metrics in controls.items():
                out[unit][fraction][name]={
                    k:float(np.mean([c['description'][unit][fraction][name][k] for c in cases]))
                    for k in metrics}
    return out


def summarize_support(scopes,field):
    selected=[s[field] for s in scopes if field in s]
    out={'scopes':len(selected),'fractions':{}}
    for fraction,item in selected[0]['fractions'].items():
        stats={k:float(np.mean([s['fractions'][fraction][k] for s in selected]))
               for k in ['retained_nonzero_element_fraction','union_column_fraction_full_M']}
        stats['row_tiles']={}
        for tile in item['row_tiles']:
            metrics=['hot_group_tile_fraction','retained_dense_tile_work_fraction']
            stats['row_tiles'][tile]={k:float(np.mean([s['fractions'][fraction]['row_tiles'][tile][k]
                                                     for s in selected])) for k in metrics}
            stats['row_tiles'][tile]['min_retained_dense_tile_work_fraction']=float(min(
                s['fractions'][fraction]['row_tiles'][tile]['retained_dense_tile_work_fraction'] for s in selected))
            stats['row_tiles'][tile]['max_retained_dense_tile_work_fraction']=float(max(
                s['fractions'][fraction]['row_tiles'][tile]['retained_dense_tile_work_fraction'] for s in selected))
        out['fractions'][fraction]=stats
    return out


def analyze(raw_root):
    path=raw_root/'saved_gpu_attempt1/RUN.json';gpu=json.loads(path.read_text())
    if gpu['status']!='PASS_FULL_HIDDEN_SAVED_GPU_MECHANISM_DESCRIPTION' or len(gpu['cases'])!=180:
        raise ValueError('Require complete180-case GPU description')
    configs={name:ROOT/'config'/f'e0_intrinsic_{name}_20261025.json' for name in ['saved_gpu','native_support']}
    for p in configs.values():
        protocol=json.loads(p.read_text())
        for name,digest in protocol['driver_sources_sha256'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Executed driver changed')
    if gpu['protocol_sha256']!=sha256(configs['saved_gpu']):raise ValueError('Saved protocol changed')
    native=[];bindings=[{'HF_relative':str(path.relative_to(raw_root.parent.parent)),
                        'sha256':sha256(path),'bytes':path.stat().st_size}]
    for index in [0,12]:
        path=raw_root/f'native_source{index:02d}_attempt1/RUN.json'
        r=json.loads(path.read_text())
        if r['status']!='PASS_NATIVE_FULL_M_SUPPORT_OBSERVATION_NOOP' or r['generations_started']!=1 or len(r['scopes'])!=20:
            raise ValueError('Require both original unchanged full-M controls')
        if r['protocol_sha256']!=sha256(configs['native_support']):raise ValueError('Native protocol changed')
        if not r['assigned_GPU_resources']['sampled_isolation_pass']:raise ValueError('Assigned GPU was shared')
        native.append(r);bindings.append({'source_index':index,'HF_relative':str(path.relative_to(raw_root.parent.parent)),
                                         'sha256':sha256(path),'bytes':path.stat().st_size,
                                         'video_sha256':r['video_sha256']})
    if not gpu['assigned_GPU_resources']['sampled_isolation_pass']:raise ValueError('GPU analysis isolation failed')
    cases=gpu['cases'];scopes=[{'source_index':r['source_index'],**s} for r in native for s in r['scopes']]
    mode_names=['absolute','same_expert_innovation'];blocks=[0,10,20,30,39]
    out={'id':'E0_INTRINSIC_FULL_HIDDEN_AND_NATIVE_SUPPORT_20261025','status':'COMPLETE',
         'analysis_sha256':sha256(Path(__file__)),
         'protocol_sha256':{name:sha256(p) for name,p in configs.items()},'raw_result_bindings':bindings,
         'scope':{'saved_source_clusters':6,'saved_absolute_cases':120,'saved_innovation_cases':60,
                  'native_source_clusters':2,'native_source_indices':[0,12],'full_dense_generations':2,
                  'native_full_M_scopes':40,'native_full_M_innovation_scopes':20,
                  'full_M':31668,'sampled_rows':32,'hidden_width':13824,'all_hidden_groups':216,
                  'blocks':blocks,'steps':[0,1,2,3],'source_exposure':'All source images previously exposed; all seed42, no camera.'},
         'actual_budget':{'full_model_generations':2,'failed_generations':0,
                          'GPU_saved_cases':180,'diagnostic_small_W2_subset_projections':gpu['diagnostic_small_W2_subset_projections'],
                          'diagnostic_full_reference_projections':gpu['diagnostic_full_reference_projections'],
                          'diagnostic_batched_group_projections':gpu['diagnostic_batched_group_projections'],
                          'extra_native_FFN_calls':0,'standalone_native_norm_calls':0,'training_updates':0,
                          'new_decompositions':0,'new_model_downloads':0,'official_quality_calls':0,
                          'peak_simultaneous_GPU_cards':1,'user_max_GPU_cards':4,
                          'saved_GPU_elapsed_seconds_including_assets':gpu['elapsed_seconds'],
                          'native_elapsed_including_observation_seconds':[r['elapsed_including_observation'] for r in native],
                          'new_HF_output_bytes':sum(p.stat().st_size for p in raw_root.rglob('*') if p.is_file())},
         'native_original_identity':{'four_step_latent_hashes_matched':8,'video_hashes_matched':2,
                                     'sampled_native_FFN_output_hashes_matched':40,
                                     'original_native_call_count_vectors_matched':2},
         'native_full_M_post_Q2_zero_fraction':{
             'mean':float(np.mean([s['post_Q2_zero_fraction'] for s in scopes])),
             'min':float(min(s['post_Q2_zero_fraction'] for s in scopes)),
             'max':float(max(s['post_Q2_zero_fraction'] for s in scopes))},
         'continuous_vs_native_full_W2_relative_SSE':{
             'median':float(np.median([c['continuous_vs_saved_native_relative_SSE'] for c in cases if c['mode']=='absolute'])),
             'max':float(max(c['continuous_vs_saved_native_relative_SSE'] for c in cases if c['mode']=='absolute'))},
         'by_block_mode':{str(b):{mode:summarize_cases([c for c in cases if c['block']==b and c['mode']==mode])
                                  for mode in mode_names} for b in blocks},
         'by_source_mode':{str(i):{mode:summarize_cases([c for c in cases if c['source_index']==i and c['mode']==mode])
                                  for mode in mode_names} for i in [0,4,8,12,16,20]},
         'by_block_step_mode':{str(b):{str(step):{mode:summarize_cases([c for c in cases if c['block']==b and
                                             c['step']==step and c['mode']==mode]) for mode in mode_names
                                             if any(c['block']==b and c['step']==step and c['mode']==mode for c in cases)}
                                             for step in [0,1,2,3]} for b in blocks},
         'full_M_support':{name:summarize_support(scopes,field) for name,field in
                           [('original_GELU','support'),('original_post_Q2','post_Q2_support'),
                            ('same_expert_post_Q2_innovation','same_expert_innovation_support')]},
         'full_M_support_by_source':{str(i):{name:summarize_support([s for s in scopes if s['source_index']==i],field)
                                    for name,field in [('original_post_Q2','post_Q2_support'),
                                     ('same_expert_post_Q2_innovation','same_expert_innovation_support')]} for i in [0,12]},
         'full_M_support_by_block':{str(b):{name:summarize_support([s for s in scopes if s['block']==b],field)
                                    for name,field in [('original_post_Q2','post_Q2_support'),
                                     ('same_expert_post_Q2_innovation','same_expert_innovation_support')]} for b in blocks},
         'quality_or_sparse_speed_acceptance':False,'candidate_operator_selected':False,'stage_changed':False}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve first accepted summary')
    write_json(args.out,analyze(args.raw_root))


if __name__=='__main__':main()
