"""Validate the original64 dependency followthrough without exporting private receipts."""
import argparse
import json
from pathlib import Path
import statistics

from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def analyze(raw):
    path=raw/'closure_gpu_attempt1/RUN.json';d=json.loads(path.read_text())
    pp=ROOT/'config/e0_shared_closure_20261025.json';p=json.loads(pp.read_text())
    if d['status']!='PASS_ORIGINAL_GROUP_CLOSURE' or len(d['cases'])!=20:
        raise ValueError('Complete20 original group cases required')
    if d['diagnostic_projections']!=p['projection_limits'] or d['protocol_sha256']!=sha256(pp):
        raise ValueError('Scope or protocol mismatch')
    if not d['assigned_GPU_resources']['sampled_isolation_pass']:raise ValueError('GPU shared')
    for f,h in p['bound_sources'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Frozen source changed')
    stats={}
    for fraction,methods in d['cases'][0]['results'].items():
        stats[fraction]={}
        for method in methods:
            stats[fraction][method]={'group64_closure':{k:fun(v for c in d['cases'] for v in c['results'][fraction][method]['group_closure']['64'])
                                                      for k,fun in [('mean',statistics.mean),('min',min),('max',max)]},
                                    'mean_relative_innovation_SSE':statistics.mean(c['results'][fraction][method]['relative_innovation_SSE'] for c in d['cases'])}
    return {'id':p['id'],'status':'COMPLETE','analysis_sha256':sha256(Path(__file__)),
            'protocol_sha256':sha256(pp),'raw_binding':{'HF_relative':'e0_shared_tiles_20261025/closure_gpu_attempt1/RUN.json','sha256':sha256(path)},
            'actual_budget':{'new_generations':0,'native_QDQ_calls':0,'training_updates':0,
                'diagnostic_projections':d['diagnostic_projections'],'peak_GPU_cards':1,
                'all_current_study_HF_bytes':sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())},
            'summary':stats,'cases':d['cases'],
            'interpretation':'Closure is the original groupwise quantizer operand requirement for direct native evaluation. It is not a universal lower bound on certified-max, cached-scale or alternative representations.',
            'candidate_operator_selected':False,'quality_or_speed_acceptance':False,'stage_changed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first accepted summary')
    write_json(a.out,analyze(a.raw_root))
