"""Test the fixed calibration-only centered C policy on two unseen sources."""
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import NativeProjection, support_mask
from run_ar2_restart import ROOT, key_name, load_scopes, tensor_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrices', type=Path, nargs='+', required=True)
    ap.add_argument('--fits', type=Path, nargs=2, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    protocol_path = ROOT / 'config/cstatic_centered_fit.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['centered_vs_uncentered_max_ratio'] != .9 or protocol['required_passes_per_source'] != 3:
        raise RuntimeError('Frozen centered-policy gate changed')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    models, fit_bindings = {}, []
    for path in args.fits:
        report = json.loads(path.read_text())
        if report['status'] != 'COMPLETED_CALIBRATION_PARAMETER_FIT' or len(report['positions']) != 80:
            raise RuntimeError('Complete all-position training fits required')
        root = Path(report['checkpoint_root'])
        identity = json.loads((root / 'identity.json').read_text())
        model = CheckpointStore(root, identity['identity'])
        if model.fingerprint != report['fingerprint'] or identity['identity']['protocol_sha256'] != digest(protocol_path):
            raise RuntimeError('Learned parameter identity changed')
        models[report['expert']] = model
        fit_bindings.append({'report':str(path.resolve()), 'sha256':digest(path), 'fingerprint':model.fingerprint})
    if set(models) != {'high_noise_model','low_noise_model'}:
        raise RuntimeError('Both experts required')
    scopes, inputs = load_scopes(args.matrices)
    parent_meta = json.loads((args.parent / 'run_meta.json').read_text())
    if parent_meta['status'] != 'COMPLETED_LOCAL_DIAGNOSTIC' or digest(args.parent/'scope_policy_summary.csv') != parent_meta['final_csv_sha256']:
        raise RuntimeError('Frozen AR2 controls changed')
    parent = list(csv.DictReader((args.parent/'scope_policy_summary.csv').open()))
    store = CheckpointStore(args.out, {'protocol_sha256':digest(protocol_path), 'fit_bindings':fit_bindings,
        'heldout_inputs':inputs, 'parent_fingerprint':parent_meta['fingerprint'],
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/ar2_core.py']}, 'torch':torch.__version__, 'cpu_threads':8})
    meta = {'status':'RUNNING','fingerprint':store.fingerprint,'fixed_candidate':'C_centered_gain_bias',
            'validation_used_for_fit_or_selection':False,'quality_or_speed_acceptance':False}
    atomic_json(args.out/'run_meta.json',meta)
    started=time.perf_counter()
    summary=[]
    try:
        for key,data in scopes.items():
            scope=key_name(key)
            part=models[key.expert].load('positions',f'step{key.step}_block{key.block:02d}')
            if part is None:
                raise RuntimeError('Missing trained position')
            arrays,info=part['arrays'],part['metadata']
            if tensor_sha(data['projection'].weight_t)!=info['weight_sha256'] or tensor_sha(data['gate'])!=info['gate_sha256'] or tensor_sha(data['projection'].bias)!=info['original_b2_sha256']:
                raise RuntimeError('Held-out effective weight/gate/original bias differs from training contract')
            controls={r['policy']:float(r['native_normalized_SSE']) for r in parent if r['scope']==scope and r['gain']=='policy_crossfit'}
            den=float(next(r for r in parent if r['scope']==scope)['denominator_T'])
            for policy in ['C','R']:
                mask=support_mask(arrays[f'{policy}_groups'],rows=512,groups=216,group_size=64)
                for mode in protocol['modes']:
                    name=f'{scope}_{policy}_{mode}'
                    risk=store.load('risks',name)
                    parameters=info['policies'][policy]['modes'][mode]
                    if risk is None:
                        bias=torch.from_numpy(arrays[f'{policy}_{mode}_b2']).to(torch.bfloat16)
                        if tensor_sha(bias)!=parameters['b2_sha256']:
                            raise RuntimeError('Folded native BF16 bias encoding changed')
                        projection=NativeProjection(data['projection'].weight_t.T,bias)
                        native=projection.native(data['a'],data['gate'],mask,parameters['alpha'])
                        fp32=projection.fp32_candidate(data['a'],data['gate'],mask,parameters['alpha'])
                        target=data['target'].double()
                        result={'native_SSE':(native.double()-target).square().sum(1).numpy(),
                                'fp32_SSE':(fp32.double()-target).square().sum(1).numpy(),'weights':data['weights']}
                        if any(not np.isfinite(x).all() for x in result.values()):
                            raise RuntimeError('Nonfinite held-out policy risk')
                        store.save('risks',name,result,{'policy':policy,'mode':mode,'training_only':True})
                        risk=store.load('risks',name)
                    values=risk['arrays']
                    summary.append({'scope':scope,**asdict(key),'policy':policy,'mode':mode,'alpha':parameters['alpha'],
                        'native_risk':float(values['weights']@values['native_SSE']/den),
                        'FP32_risk':float(values['weights']@values['fp32_SSE']/den),
                        'AR2_R_mean':controls['R_mean'],'AR2_C_dynamic':controls['C_dynamic']})
            print(f'EVALUATED {scope}',flush=True)
        decisions=[]
        for key in scopes:
            rows=[r for r in summary if r['scope']==key_name(key) and r['policy']=='C']
            risks={r['mode']:r['native_risk'] for r in rows}
            decisions.append({'scope':key_name(key),**asdict(key),'centered_risk':risks['centered_gain_bias'],
                'uncentered_risk':risks['uncentered_gain'],'ratio':risks['centered_gain_bias']/risks['uncentered_gain'],
                'pass':risks['centered_gain_bias']<=.9*risks['uncentered_gain']})
        gates={}
        for task in [0,6]:
            passed=[r for r in decisions if r['task']==task and r['pass']]
            gates[str(task)]={'passed_scopes':len(passed),'gate_pass':len(passed)>=3 and {r['seed'] for r in passed}=={20261001,20261002} and {r['step'] for r in passed}=={0,2}}
        atomic_json(args.out/'decisions.json',{'scopes':decisions,'source_gates':gates,'fixed_candidate_pass':all(v['gate_pass'] for v in gates.values())})
        with (args.out/'scope_summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU projection evaluator initialized CUDA')
        meta.update(status='COMPLETED_HELDOUT_CENTERED_POLICY_DIAGNOSTIC',elapsed_seconds=time.perf_counter()-started,
                    source_gates=gates,final_csv_sha256=digest(args.out/'scope_summary.csv'))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
