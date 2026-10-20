"""Describe paired trajectory drift; this is separate from video quality acceptance."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json,digest


def cases(path):
    suite=json.loads(path.read_text())
    if suite['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY':raise RuntimeError('Need a completed full20 suite')
    result={}
    for entry in suite['completed']:
        if digest(entry['case_report'])!=entry['case_report_sha256']:raise RuntimeError('Case report changed')
        case=json.loads(Path(entry['case_report']).read_text())
        if case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or case['index'] in result:
            raise RuntimeError('Incomplete or duplicate formal case')
        result[case['index']]=case
    if set(result)!=set(range(20)):raise RuntimeError('All20 distinct formal inputs required')
    return suite,result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dense',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8)
    ds,dc=cases(args.dense);cs,cc=cases(args.candidate)
    if (ds['identity']['arm']!='D-Q' or ds['identity']['assets']!=cs['identity']['assets']
            or ds['identity']['inputs_sha256']!=cs['identity']['inputs_sha256']):
        raise RuntimeError('Paired dense/candidate assets or inputs differ')
    rows=[];started=time.perf_counter()
    for index in range(20):
        d,c=dc[index],cc[index]
        if d['input']!=c['input']:raise RuntimeError('Paired input differs')
        if [v['step'] for v in d['trajectory']]!=list(range(4)) or [v['step'] for v in c['trajectory']]!=list(range(4)):
            raise RuntimeError('Four native steps required')
        for a,b in zip(d['trajectory'],c['trajectory']):
            for value in [a,b]:
                if digest(value['path'])!=value['file_sha256']:raise RuntimeError('Latent file changed')
            with safe_open(a['path'],framework='pt',device='cpu') as da,safe_open(b['path'],framework='pt',device='cpu') as ca:
                x,y=da.get_tensor('latents'),ca.get_tensor('latents')
            if x.shape!=y.shape or x.dtype!=y.dtype:raise RuntimeError('Paired latent geometry/precision differs')
            x,y=x.double().flatten(),y.double().flatten()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):raise RuntimeError('Nonfinite latent')
            den=float(x.square().sum());cand=float(y.square().sum());error=float((y-x).square().sum())
            centered_x=x-x.mean();centered_y=y-y.mean()
            rows.append({'index':index,'source_group':d['input']['source_group'],'dimension':d['input']['dimension'],
                'step':a['step'],'elements':x.numel(),'dense_squared_norm':den,'candidate_squared_norm':cand,
                'squared_error':error,'relative_RMSE':float(np.sqrt(error/den)),
                'norm_ratio':float(np.sqrt(cand/den)),'cosine':float((x@y)/np.sqrt(cand*den)),
                'centered_variance_ratio':float(centered_y.square().sum()/centered_x.square().sum())})
    summaries=[]
    for step in range(4):
        selected=[r for r in rows if r['step']==step]
        summaries.append({'step':step,'input_count':20,'pooled_relative_RMSE':float(np.sqrt(
            sum(r['squared_error'] for r in selected)/sum(r['dense_squared_norm'] for r in selected))),
            **{f'{key}_mean':float(np.mean([r[key] for r in selected])) for key in
                ['relative_RMSE','norm_ratio','cosine','centered_variance_ratio']}})
    if torch.cuda.is_initialized():raise RuntimeError('CPU trajectory audit initialized CUDA')
    args.out.mkdir(parents=True,exist_ok=True)
    csv_path=args.out/'per_source_step.csv'
    with csv_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(args.out/'RUN.json',{'status':'COMPLETED_PAIRED_FORMAL20_LATENT_DRIFT_DESCRIPTION',
        'dense_suite_sha256':digest(args.dense),'candidate_suite_sha256':digest(args.candidate),
        'candidate_policy_sha256':cs['identity']['packed_policy_sha256'],
        'source_sha256':digest(Path(__file__)),'per_source_csv_sha256':digest(csv_path),
        'summary':summaries,'elapsed_seconds':time.perf_counter()-started,
        'interpretation':'Trajectory difference is descriptive, does not establish a causal state-distribution mechanism or video-quality acceptance',
        'quality_or_speed_acceptance':False})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
