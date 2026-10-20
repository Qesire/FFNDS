"""Describe frozen support/gain changes across calibration sets without comparing unpaired risks."""
import argparse
import json
from pathlib import Path

import numpy as np

from ar2_checkpoint import CheckpointStore,atomic_json,digest


def load(path):
    report=json.loads(path.read_text());root=Path(report['checkpoint_root'])
    store=CheckpointStore(root,json.loads((root/'identity.json').read_text())['identity'])
    if (report['status']!='COMPLETED_CALIBRATION_PARAMETER_FIT' or len(report['positions'])!=80
            or store.fingerprint!=report['fingerprint']):raise RuntimeError('Completed80-position expert fit required')
    return report,store


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--new-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();rows=[]
    for old_path,new_path in zip(args.old_fits,args.new_fits):
        old,os=load(old_path);new,ns=load(new_path)
        if old['expert']!=new['expert']:raise RuntimeError('Expert order differs')
        if os.fingerprint==ns.fingerprint:raise RuntimeError('This audit requires distinct calibration fits')
        for step in ([0,1] if old['expert']=='high_noise_model' else [2,3]):
            for block in range(40):
                name=f'step{step}_block{block:02d}';a=os.load('positions',name);b=ns.load('positions',name)
                if a is None or b is None:raise RuntimeError('Fit position missing')
                am,bm=a['metadata'],b['metadata'];aa,ba=a['arrays'],b['arrays']
                if any(am[k]!=bm[k] for k in ['weight_sha256','gate_sha256','original_b2_sha256']):
                    raise RuntimeError('Quantized weight, gate or original bias changed')
                if any(not np.array_equal(aa[k],ba[k]) for k in ['gate','original_b2']):
                    raise RuntimeError('Fixed numerical gate/bias arrays differ')
                support_a,support_b=set(aa['C_groups'].tolist()),set(ba['C_groups'].tolist())
                if len(support_a)!=108 or len(support_b)!=108:raise RuntimeError('Main half-width support changed')
                alphas=[m['policies']['C']['modes']['centered_gain_bias']['alpha'] for m in [am,bm]]
                delta=(ba['C_centered_gain_bias_b2'].astype(np.float64)-aa['C_centered_gain_bias_b2'])*aa['gate'].reshape(-1)
                rows.append({'step':step,'block':block,'expert':old['expert'],
                    'shared_retained_groups':len(support_a&support_b),'support_Jaccard':len(support_a&support_b)/len(support_a|support_b),
                    'old_alpha':alphas[0],'new_alpha':alphas[1],'gated_bias_change_RMS':float(np.sqrt(np.mean(delta**2)))})
    args.out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for step in range(4):
        selected=[r for r in rows if r['step']==step]
        summaries.append({'step':step,**{key:float(np.mean([r[key] for r in selected])) for key in
            ['shared_retained_groups','support_Jaccard','old_alpha','new_alpha','gated_bias_change_RMS']}})
    atomic_json(args.out/'RUN.json',{'status':'COMPLETED_FROZEN_CALIBRATION_PARAMETER_CHANGE_DESCRIPTION',
        'old_fits':[{'path':str(p.resolve()),'sha256':digest(p)} for p in args.old_fits],
        'new_fits':[{'path':str(p.resolve()),'sha256':digest(p)} for p in args.new_fits],
        'source_sha256':digest(Path(__file__)),'positions':rows,'summary':summaries,
        'unpaired_training_risks_not_compared':True,'quality_or_speed_acceptance':False})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
