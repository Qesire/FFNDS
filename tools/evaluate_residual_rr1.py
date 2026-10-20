"""Test only the calibration-selected RR1 correction on eight frozen scopes."""
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from reduced_rank_residual import predict_y
from run_ar2_restart import ROOT, key_name, load_scopes, tensor_sha


def open_verified(report_path, status):
    report = json.loads(report_path.read_text())
    root = Path(report['checkpoint_root'])
    identity = json.loads((root / 'identity.json').read_text())
    store = CheckpointStore(root, identity['identity'])
    if report['status'] != status or store.fingerprint != report['fingerprint']:
        raise RuntimeError('Incomplete or changed training fit')
    return store, report, identity['identity']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrices', type=Path, nargs=2, required=True)
    ap.add_argument('--base-fits', type=Path, nargs=2, required=True)
    ap.add_argument('--correction', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = ROOT / 'config/residual_rr1.json'
    correction, correction_report, correction_id = open_verified(
        args.correction, 'COMPLETED_RR1_CV_AND_SELECTED_FINAL_FIT')
    if correction_id['protocol_sha256'] != digest(protocol_path):
        raise RuntimeError('Frozen RR1 protocol changed')
    base_models, bindings = {}, []
    for path in args.base_fits:
        model, report, identity = open_verified(path, 'COMPLETED_CALIBRATION_PARAMETER_FIT')
        if len(report['positions']) != 80:
            raise RuntimeError('All-position base required')
        base_models[report['expert']] = model
        bindings.append({'path': str(path.resolve()), 'sha256': digest(path), 'fingerprint': model.fingerprint})
    if bindings != correction_id['base_fits']:
        raise RuntimeError('Correction was trained on a different base')
    parent_meta = json.loads((args.parent / 'run_meta.json').read_text())
    if (parent_meta['status'] != 'COMPLETED_LOCAL_DIAGNOSTIC'
            or digest(args.parent / 'scope_policy_summary.csv') != parent_meta['final_csv_sha256']):
        raise RuntimeError('AR2 normalization reference changed')
    parent = list(csv.DictReader((args.parent / 'scope_policy_summary.csv').open()))
    scopes, inputs = load_scopes(args.matrices)
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol_path),
        'correction_report_sha256': digest(args.correction), 'correction_fingerprint': correction.fingerprint,
        'base_fits': bindings, 'inputs': inputs, 'AR2_parent': parent_meta['fingerprint'],
        'torch': torch.__version__, 'cpu_threads': 8,
        'sources': {str(p): digest(p) for p in [Path(__file__), ROOT/'tools/reduced_rank_residual.py',
            ROOT/'tools/ar2_core.py', ROOT/'tools/run_ar2_restart.py']}})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint,
        'selected': correction_report['selected'], 'test_used_for_fit_or_selection': False,
        'quality_or_speed_acceptance': False}
    atomic_json(args.out / 'run_meta.json', meta)
    started = time.perf_counter()
    summary = []
    try:
        for key, data in scopes.items():
            scope = key_name(key)
            name = f'step{key.step}_block{key.block:02d}'
            base_part = base_models[key.expert].load('positions', name)
            residual_part = correction.load('final_models', name)
            if base_part is None or residual_part is None:
                raise RuntimeError('Missing frozen learned parameters')
            arrays, info = base_part['arrays'], base_part['metadata']
            if (tensor_sha(data['projection'].weight_t) != info['weight_sha256']
                    or tensor_sha(data['gate']) != info['gate_sha256']
                    or tensor_sha(data['projection'].bias) != info['original_b2_sha256']):
                raise RuntimeError('Held-out weight/gate/original bias differs')
            parameters = {k: torch.from_numpy(v).to(torch.bfloat16) for k,v in residual_part['arrays'].items()}
            if {k: tensor_sha(v) for k,v in parameters.items()} != residual_part['metadata']['parameter_sha256']:
                raise RuntimeError('Native BF16 correction encoding changed')
            result = store.load('risks', scope)
            if result is None:
                with safe_open(data['scope']['tensor_file'], framework='pt', device='cpu') as handle:
                    x = handle.get_tensor('V1')
                if x.shape != (512, 5120) or x.dtype != torch.bfloat16:
                    raise RuntimeError('Expected actual post-Q1 input rows')
                groups = arrays['C_groups']
                channels = torch.tensor((groups[:,None]*64 + np.arange(64)).reshape(-1))
                w = data['projection'].weight_t.index_select(1, channels).contiguous()
                bias = torch.from_numpy(arrays['C_centered_gain_bias_b2']).to(torch.bfloat16)
                base_info = info['policies']['C']['modes']['centered_gain_bias']
                if tensor_sha(bias) != base_info['b2_sha256']:
                    raise RuntimeError('Base folded bias encoding changed')
                effective = data['a'].index_select(1, channels).float() * base_info['alpha']
                base_y = F.linear(effective.to(torch.bfloat16), w, bias)
                fp32_y = F.linear(effective, w.float(), bias.float())
                native = (base_y + predict_y(x, parameters)) * data['gate']
                control = (fp32_y + predict_y(x, parameters, fp32=True)) * data['gate'].float()
                base = base_y * data['gate']
                fp32_base = fp32_y * data['gate'].float()
                target = data['target'].double()
                values = {f'{n}_SSE': (v.double()-target).square().sum(1).numpy() for n,v in
                    [('native',native),('FP32',control),('base_native',base),('base_FP32',fp32_base)]}
                values['weights'] = data['weights']
                if any(not np.isfinite(v).all() for v in values.values()):
                    raise FloatingPointError('Nonfinite held-out correction risk')
                store.save('risks', scope, values, {'uses_only_V1_for_correction': True})
                result = store.load('risks', scope)
            values = result['arrays']
            den = float(next(r for r in parent if r['scope']==scope)['denominator_T'])
            risks = {k: float(values['weights']@v/den) for k,v in values.items() if k.endswith('_SSE')}
            ratio = risks['native_SSE']/risks['base_native_SSE']
            row = {'scope': scope, **asdict(key), **risks, 'ratio': ratio, 'pass': ratio <= .75}
            summary.append(row)
            print(f'EVALUATED {scope}: corrected/base={ratio:.6g}', flush=True)
        gates = {}
        for task in [0,6]:
            passed = [r for r in summary if r['task']==task and r['pass']]
            gates[str(task)] = {'passed_scopes': len(passed), 'gate_pass': len(passed)>=3
                and {r['seed'] for r in passed}=={20261001,20261002} and {r['step'] for r in passed}=={0,2}}
        atomic_json(args.out/'decisions.json', {'scopes': summary, 'source_gates': gates,
            'route_pass': all(v['gate_pass'] for v in gates.values())})
        with (args.out/'scope_summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU evaluator initialized CUDA')
        meta.update(status='COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC', source_gates=gates,
            route_pass=all(v['gate_pass'] for v in gates.values()), elapsed_seconds=time.perf_counter()-started,
            final_csv_sha256=digest(args.out/'scope_summary.csv'))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
