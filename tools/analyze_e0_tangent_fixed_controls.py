"""Apply preexisting other-source scalar/affine controls to new adjacent rows.

CPU saved operands only; no model, GPU, optimizer or checkpoint operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from analyze_ffnds_observations import load_arrays
from analyze_e0_temporal_calibration import load_cases,fit_scale,fit_affine,score
from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def analyze(raw,old_root):
    plan=ROOT/'config/e0_temporal_saved_calibration_20261025.json'
    original=ROOT/'research/20261025/E0_TEMPORAL_CALIBRATION.json'
    prior=json.loads(original.read_text());cases=load_cases(old_root,json.loads(plan.read_text()))
    current_summary=ROOT/'research/20261025/E0_SHARED_TILE_PRE_W1.json'
    accepted=json.loads(current_summary.read_text());records=[]
    for index in [0,12]:
        path=raw/f'native_source{index:02d}_attempt1/RUN.json'
        if sha256(path)!=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)['sha256']:
            raise ValueError('Native run changed')
        run=json.loads(path.read_text())
        for block in [0,10,20,30,39]:
            for step in [1,3]:
                train=[c for c in cases if c['source_index']!=index and (c['block'],c['target_step'])==(block,step)]
                if len(train)!=5:raise ValueError('Other five source controls required')
                scale=fit_scale(train,'fixed_scale');alpha,beta=fit_affine(train)
                old_record=next(v for v in prior['records'] if (v['source_index'],v['block'],v['target_step'])==(index,block,step))
                if scale!=old_record['predictors']['fixed_scale']['calibration']['scale'] or alpha!=old_record['predictors']['fixed_affine']['calibration']['scale']:
                    raise ValueError('Frozen old scalar did not reproduce')
                if hashlib.sha256(beta.astype('<f8').tobytes()).hexdigest()!=old_record['predictors']['fixed_affine']['calibration']['channel_shift_F64_sha256']:
                    raise ValueError('Frozen old channel shift did not reproduce')
                pair=[]
                for t in [step-1,step]:
                    s=next(v for v in run['scopes'] if (v['step'],v['block'])==(t,block))
                    f=path.parent/'observed'/s['path']
                    if sha256(f)!=s['sha256']:raise ValueError('Native payload changed')
                    a=load_arrays(f);ids=[s['row_plan']['rows'].index(row) for row in sum(s['row_plan']['contiguous_tiles'],[])]
                    pair.append({'y':a['ffn_output'][ids].astype(np.float64),'gate':a['gate'].astype(np.float64)})
                case={'previous':pair[0]['y'],'current':pair[1]['y'],'gate':pair[1]['gate']}
                records.append({'source_index':index,'block':block,'step':step,
                    'training_source_indices':sorted(c['source_index'] for c in train),
                    'old_frozen_parameters_reproduced':True,
                    'results':{n:score(case,pred)['relative_SSE'] for n,pred in
                               [('reuse',case['previous']),('fixed_scale',scale*case['previous']),
                                ('fixed_affine',alpha*case['previous']+beta)]}})
    return {'status':'COMPLETE_CPU_ADJACENT_FIXED_REFERENCE','analysis_sha256':sha256(Path(__file__)),
        'inputs':{'old_plan':sha256(plan),'old_result':sha256(original),'new_contiguous_summary':sha256(current_summary),
                  'frozen_old_fitter':sha256(ROOT/'tools/analyze_e0_temporal_calibration.py')},
        'scope':'20 adjacent384-row cases, original other-five-source calibration reconstructed exactly. Native saved FFN labels/current gate; no parameter tuned to new contiguous labels.',
        'new_generations':0,'new_GPU_calls':0,'optimizer_updates':0,
        'by_block':{str(b):{n:statistics.mean(c['results'][n] for c in records if c['block']==b) for n in records[0]['results']} for b in [0,10,20,30,39]},
        'records':records}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw-root','old-root','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first fixed-control result')
    write_json(a.out,analyze(a.raw_root,a.old_root))
