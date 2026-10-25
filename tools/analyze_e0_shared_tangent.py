"""Validate saved shared-tangent results and retain source-level scope."""
import argparse
import json
from pathlib import Path
import statistics

from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def analyze(raw):
    p=ROOT/'config/e0_shared_tangent_20261025.json';plan=json.loads(p.read_text())
    path=raw/'tangent_gpu_attempt1/RUN.json';r=json.loads(path.read_text())
    if r['status']!='PASS_SHARED_TANGENT_DISCRIMINATION' or len(r['cases'])!=20:
        raise ValueError('Require complete20-case tangent study')
    if r['protocol_sha256']!=sha256(p) or r['diagnostic_projections']!=plan['projection_limits']:
        raise ValueError('Frozen budget/protocol mismatch')
    for f,h in plan['bound_sources'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Bound source changed')
    if not r['assigned_GPU_resources']['sampled_isolation_pass']:raise ValueError('GPU isolation failed')
    prior=json.loads((ROOT/'research/20261025/E0_SHARED_TILE_PRE_W1.json').read_text())
    for c in r['cases']:
        match=next(v for v in prior['cases'] if (v['source_index'],v['step'],v['block'],v['layout'],v['row_tile'])==
                   (c['source_index'],c['step'],c['block'],'contiguous',128))
        c['innovation_over_current_output_energy']=match['innovation_over_current_output_energy']
        for values in c['results'].values():
            values['relative_current_output_SSE']=values['relative_innovation_SSE']*c['innovation_over_current_output_energy']
    def mean(cases):
        return {'cases':len(cases),'results':{n:{k:statistics.mean(c['results'][n][k] for c in cases)
                    for k in cases[0]['results'][n]} for n in cases[0]['results']},
                'continuous_GELU_old_per_token_tangent_SSE':statistics.mean(c['continuous_GELU_old_per_token_tangent_SSE'] for c in cases),
                'Q2_rounding_residual_over_innovation':statistics.mean(c['Q2_rounding_residual_over_innovation'] for c in cases)}
    return {'id':plan['id'],'status':'COMPLETE','analysis_sha256':sha256(Path(__file__)),
            'protocol_sha256':sha256(p),'raw_binding':{'HF_relative':'e0_shared_tiles_20261025/tangent_gpu_attempt1/RUN.json','sha256':sha256(path)},
            'scope':{'exposed_sources':[0,12],'seed':42,'same_expert_pairs':20,'contiguous_rows_each':384},
            'actual_budget':{'new_generations':0,'native_QDQ_calls':0,'training_updates':0,
                  'new_weight_compositions':0,'diagnostic_projections':r['diagnostic_projections'],'peak_GPU_cards':1},
            'by_block':{str(b):mean([c for c in r['cases'] if c['block']==b]) for b in [0,10,20,30,39]},
            'by_source':{str(i):mean([c for c in r['cases'] if c['source_index']==i]) for i in [0,12]},
            'max_linear_preactivation_reconstruction_SSE':max(c['linear_preactivation_reconstruction_SSE'] for c in r['cases']),
            'hypothetical_composition_MAC_ratio':r['hypothetical_composition_MAC_ratio'],
            'cases':r['cases'],'candidate_operator_selected':False,'quality_or_speed_acceptance':False,'stage_changed':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first validated summary')
    write_json(a.out,analyze(a.raw_root))
