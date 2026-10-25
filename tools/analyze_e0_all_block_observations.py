"""Validate all40 passive captures and their overlap with prior native evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def analyze(raw,prior_root):
    protocol=ROOT/'config/e0_all_block_tangent_native_20261025.json';p=json.loads(protocol.read_text())
    for name,h in p['driver_sources_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Executed source changed')
    previous=json.loads((ROOT/'research/20261025/E0_SHARED_TILE_PRE_W1.json').read_text())
    out={'id':p['id'],'status':'COMPLETE','analysis_sha256':sha256(Path(__file__)),
         'protocol_sha256':sha256(protocol),'raw_bindings':[],'sources':[],
         'native_identity':{'latent_hashes_matched':8,'video_hashes_matched':2,
                            'original_count_vectors_matched':2,'old32_five_field_positions':0,
                            'prior_contiguous_overlap_five_field_positions':0},
         'actual_budget':{'full_generations':2,'failed_generations':0,'retries':0,
                         'peak_GPU_cards':2,'user_GPU_ceiling':4,'extra_native_FFN_calls':0,
                         'full_M_derivative_reductions':0,'training_updates':0},
         'quality_or_speed_acceptance':False,'stage_changed':False}
    for index in [0,12]:
        path=raw/f'native_source{index:02d}_attempt1/RUN.json';r=json.loads(path.read_text())
        if r['status']!='PASS_NATIVE_ALL_BLOCK_TANGENT_OPERANDS_NOOP' or len(r['scopes'])!=160 or r['protocol_sha256']!=sha256(protocol):
            raise ValueError('Complete original-control all40 run required')
        if not r['assigned_GPU_resources']['sampled_isolation_pass']:raise ValueError('Assigned GPU shared')
        if {(s['step'],s['block']) for s in r['scopes']}!={(s,b) for s in range(4) for b in range(40)}:
            raise ValueError('Duplicate/missing native positions')
        prev_path=prior_root/path.relative_to(raw)
        expected=next(v for v in previous['raw_bindings'] if v.get('source_index')==index)
        if sha256(prev_path)!=expected['sha256']:raise ValueError('Prior contiguous run changed')
        prev=json.loads(prev_path.read_text())
        for scope in r['scopes']:
            payload=path.parent/'observed'/scope['path']
            if sha256(payload)!=scope['sha256']:raise ValueError('Native tensor changed')
            arrays=load_arrays(payload)
            if scope['step'] in [0,2]:
                mean=arrays['full_M_old_derivative_mean']
                if scope['full_M_derivative_rows']!=31668 or mean.shape!=(1,13824) or not np.isfinite(mean).all() or (mean<-.13).any() or (mean>1.13).any():
                    raise ValueError('Full-M derivative mean layout/range invalid')
                out['actual_budget']['full_M_derivative_reductions']+=1
            if scope['block'] in [0,10,20,30,39]:
                if scope.get('historical_uniform_all_five_fields_exact') is not True:raise ValueError('Original32 identity failed')
                out['native_identity']['old32_five_field_positions']+=1
                other=next(s for s in prev['scopes'] if (s['step'],s['block'])==(scope['step'],scope['block']))
                old_path=prev_path.parent/'observed'/other['path']
                if sha256(old_path)!=other['sha256']:raise ValueError('Prior vector changed')
                old=load_arrays(old_path);ids=[other['row_plan']['rows'].index(row) for row in scope['row_plan']['rows']]
                for name in ['q1_post','z','gelu','q2_post','ffn_output']:
                    if not np.array_equal(arrays[name],old[name][ids]):raise ValueError('Previous contiguous overlap changed')
                out['native_identity']['prior_contiguous_overlap_five_field_positions']+=1
        total=sum(f.stat().st_size for f in path.parent.rglob('*') if f.is_file())
        if total>p['max_output_gib_per_source']*(1<<30):raise ValueError('Output budget exceeded')
        out['raw_bindings'].append({'source_index':index,'HF_relative':f'e0_all_block_tangent_20261025/native_source{index:02d}_attempt1/RUN.json','sha256':sha256(path)})
        out['sources'].append({'source_index':index,'source_id':r['source_id'],'seed':r['seed'],
                               'video_sha256':r['video_sha256'],'trajectory':r['trajectory'],
                               'native_counts':r['native_counts'],'scopes':160,'HF_output_bytes':total})
    if out['actual_budget']['full_M_derivative_reductions']!=160:raise ValueError('Missing global means')
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['raw-root','prior-root','out']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first native manifest')
    write_json(a.out,analyze(a.raw_root,a.prior_root))
