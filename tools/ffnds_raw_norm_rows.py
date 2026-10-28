"""Registered, saved-row-only raw BF16 norm diagnostic. No model or Torch imports."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_RAW_NORM_ROWS_20261028'
PARENT_ID = 'E0_ORIGINAL_BF16_LOW39_R512_FOUR_CORNER_20261028'
PROTOCOL = 'config/e0_original_bf16_low39_four_corner_20261028_v1.json'
PSHA = 'fcd50986dc650e80abc0a1a665fca973b92af940fa6b34bf6615e1c78f322d4f'
RSHA = '768a5218da2ac17914ffd7e6a3e1623adf848be679a71a941634596c2d1c0088'
ESHA = '49e536c4725c4f525ffaece08d5daf23ac0d1781def107c1df8c9325613ae35b'
BASE = 'ffnds-derived/e0_original_bf16_low39_r512_four_corner_20261028/attempt1/worker/'
KEYS = ('rows', 'u', 'r', 'Y', 'V1', 'gate', 'head_scale', 'head_shift')
PRE_SOURCE = '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py'
PRE_SHA = 'f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5'


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, digest):
    data = path.read_bytes()
    need(sha(data) == digest, 'SHA mismatch: ' + str(path))
    return data


def parent(hf):
    protocol = json.loads(checked(ROOT / PROTOCOL, PSHA))
    run = json.loads(checked(hf / BASE / 'results/RUN.json', RSHA))
    exit_ = json.loads(checked(hf / BASE / 'EXIT_RECEIPT.json', ESHA))
    need(protocol['id'] == run['id'] == exit_['id'] == PARENT_ID, 'parent id')
    need(run['protocol_sha256'] == exit_['protocol_sha256'] == PSHA and exit_['run_sha256'] == RSHA, 'parent binding')
    need(run['status'] == 'COMPLETE_RAW_BF16_LOW39_R512_TWELVE_FULL_FLOWS_NO_VIDEO' and exit_['process_exited'] is True and type(exit_['child_returncode']) is int and exit_['child_returncode'] == 0 and exit_['timeout'] is False and exit_['automatic_retry'] is False, 'parent actual exit')
    return protocol, run


def draft(hf):
    protocol, run = parent(hf)
    cases = []
    for source in (0, 12):
        case = next(c for c in protocol['cases'] if c['index'] == source)
        for step in (2, 3):
            for op in ('DD', 'SD'):
                rid = f'raw_low39_source{source:02d}_{op}{step}'
                f = next(x for x in run['flows'] if x['run_id'] == rid)
                need(f['source_index'] == source and f['step'] == step and f['operation'] == f'{op}{step}' and f['status'] == 'COMPLETE_FULL40_FINITE_FLOW_NOT_VIDEO', 'parent flow identity')
                cases.append({'run_id': rid, 'source': source, 'step': step, 'operation': op, 'capture': f['capture'], 'tensor_metadata': {k: f['tensor_metadata'][k] for k in KEYS}, 'rows': case['rows'][str(step)]})
    return {'id': ID, 'status': 'DRAFT_NOT_REGISTERED', 'parent_protocol': {'path': PROTOCOL, 'sha256': PSHA}, 'parent_RUN': {'path': BASE+'results/RUN.json', 'sha256': RSHA}, 'parent_exit': {'path': BASE+'EXIT_RECEIPT.json', 'sha256': ESHA}, 'captures': cases, 'resources': {'GPU': 0, 'CPU_threads': 2, 'affinity_cpus': 2, 'wall_seconds': 180, 'new_hf_bytes_max': 16777216}, 'limits': {'capture_loads': 8, 'pairs': 4, 'sigma': 8, 'fixed_scalar_fits': 4, 'fold_evaluations': 8}, 'estimand': 'two exposed development image clusters; 256 nested rows each; ratio proxy not full-M head SSE, video quality or generalized calibration', 'grid': [21,29,52], 'epsilon': 1e-6, 'output_HF_relative': 'ffnds-derived/e0_raw_norm_rows_20261028/attempt1/worker/results', 'source_files': {}}


def decode_bf16(raw, shape, np):
    return (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape).astype(np.float64)


def load_capture(hf, case, np):
    binding = case['capture']; path = Path(binding['path'])
    need(binding['root'] == 'hf' and not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'ffnds-derived', 'capture path')
    data = checked(hf / path, binding['sha256'])
    need(len(data) == binding['bytes'], 'capture size')
    n = struct.unpack('<Q', data[:8])[0]
    need(0 < n < 65536, 'header length')
    header = json.loads(data[8:8+n]); start = 8+n
    out = {}
    for k in KEYS:
        spec = header[k]; meta = case['tensor_metadata'][k]
        shape = [256] if k == 'rows' else ([5120] if k in ('gate','head_scale','head_shift') else [256,5120])
        dtype = 'I64' if k == 'rows' else 'BF16'
        need(spec['shape'] == meta['shape'] == shape and spec['dtype'] == dtype and meta['dtype'] == ('int64' if k == 'rows' else 'bfloat16'), 'tensor header ' + k)
        a,b = spec['data_offsets']; need(0 <= a <= b <= len(data)-start and b-a == (8 if k == 'rows' else 2)*__import__('math').prod(shape), 'tensor offsets')
        raw = data[start+a:start+b]; need(sha(raw) == meta['raw_tensor_sha256'], 'tensor hash ' + k)
        out[k] = np.frombuffer(raw, dtype='<i8').copy() if k == 'rows' else decode_bf16(raw, shape, np)
        need(np.isfinite(out[k]).all(), 'nonfinite ' + k)
    need(out['rows'].tolist() == case['rows'] and len(set(case['rows'])) == 256 and all(0 <= i < 31668 for i in case['rows']), 'native row coordinates')
    return out


def stats(x, np):
    q = np.percentile(x, [5,50,95])
    return {'mean': float(np.mean(x)), 'std': float(np.std(x)), 'p05': float(q[0]), 'p50': float(q[1]), 'p95': float(q[2]), 'min': float(np.min(x)), 'max': float(np.max(x))}


def pair(d, s, np, ledger=None):
    for k in ('rows','V1','r','gate','head_scale','head_shift'):
        need(np.array_equal(d[k], s[k]), 'same-state binding ' + k)
    dc = d['u'] - d['u'].mean(axis=1, keepdims=True)
    sc = s['u'] - s['u'].mean(axis=1, keepdims=True)
    delta = sc-dc
    vd = np.mean(dc*dc,axis=1); vs = np.mean(sc*sc,axis=1)
    cross = 2*np.mean(dc*delta,axis=1); square = np.mean(delta*delta,axis=1)
    need(np.allclose(vs-vd, cross+square, rtol=1e-9, atol=1e-10), 'variance decomposition')
    if ledger is None:
        sd, ss = np.sqrt(vd+1e-6), np.sqrt(vs+1e-6)
    else:
        with ledger.work('sigma'): sd = np.sqrt(vd+1e-6)
        with ledger.work('sigma'): ss = np.sqrt(vs+1e-6)
    tau = sd/ss
    write = s['u']-d['u']; ideal = d['gate']*(s['Y']-d['Y']); rem = write-ideal
    return tau, {'tau': stats(tau,np), 'delta_variance': stats(vs-vd,np), 'signed_cross': stats(cross,np), 'delta_centered_square': stats(square,np), 'variance_closure_max_abs': float(np.max(np.abs(vs-vd-cross-square))), 'write_delta_sse': float(np.sum(write*write)), 'gate_y_delta_sse': float(np.sum(ideal*ideal)), 'actual_write_remainder_sse': float(np.sum(rem*rem)), 'write_remainder_cross': float(2*np.sum(ideal*rem)), 'write_closure_residual': float(np.sum(write*write)-np.sum(ideal*ideal)-np.sum(rem*rem)-2*np.sum(ideal*rem))}


def evaluate(tau, kappa, np):
    return {'kappa': float(kappa), 'relative_ratio_mse': float(np.mean((tau/kappa-1)**2)), 'identity_ratio_mse': float(np.mean((tau-1)**2))}


def analyze(captures, np, ledger):
    pairs, taus = [], {}
    for source in (0,12):
        for step in (2,3):
            d,s = (captures[(source,step,op)] for op in ('DD','SD'))
            with ledger.work('pairs'): tau, detail = pair(d,s,np,ledger)
            taus[(source,step)] = tau
            rows = d['rows']; slots = rows//(29*52)
            detail.update(source=source, step=step, row_indices=rows.tolist(), time_slots=slots.tolist(), time_slot_counts=np.bincount(slots,minlength=21).tolist())
            pairs.append(detail)
    folds = []
    for step in (2,3):
        for train,test in ((0,12),(12,0)):
            with ledger.work('fixed_scalar_fits'): k = np.exp(np.mean(np.log(taus[(train,step)])))
            with ledger.work('fold_evaluations'): heldout = evaluate(taus[(test,step)],k,np)
            with ledger.work('fold_evaluations'): optimistic = evaluate(taus[(train,step)],k,np)
            folds.append({'step': step, 'train_source': train, 'heldout_source': test, 'heldout': heldout, 'same_source_optimistic': optimistic})
    return {'pairs': pairs, 'folds': folds, 'evidence': 'NATIVE_SAVED_ROWS_DEVELOPMENT', 'limitations': '2 exposed development image clusters; 256 rows nested per source and step; ratio proxy is not full-M native head SSE, video quality, or generalized calibration.'}


class Ledger:
    def __init__(self, out, cfg, protocol_sha):
        self.path = out/'RUN.json'
        self.limits = cfg['limits']
        self.report = {'id': ID, 'status': 'STARTED', 'protocol_sha256': protocol_sha,
                       'parent_RUN_sha256': RSHA, 'limits': self.limits,
                       'predebited': {k: 0 for k in self.limits},
                       'observed': {k: 0 for k in self.limits}, 'automatic_retry': False}
        self.save()

    def save(self):
        data=(json.dumps(self.report,indent=2,allow_nan=False)+'\n').encode()
        need(len(data)<=16777216, 'output budget')
        tmp=self.path.with_suffix('.tmp')
        tmp.write_bytes(data)
        os.replace(tmp,self.path)

    def fail(self, exc, elapsed):
        self.report.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(exc).__name__, error=str(exc), wall_seconds=elapsed)
        self.save()

    @contextmanager
    def work(self, key):
        need(self.report['predebited'][key] < self.limits[key], 'operation budget: '+key)
        self.report['predebited'][key] += 1
        self.save()
        yield
        self.report['observed'][key] += 1
        self.save()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('draft','inspect','run')); ap.add_argument('--protocol',required=True); ap.add_argument('--hf-root',required=True); ap.add_argument('--out'); args=ap.parse_args()
    start=time.monotonic(); hf=Path(args.hf_root).resolve(); path=Path(args.protocol).resolve()
    if args.command == 'draft':
        obj=draft(hf)
        for name in ('tools/ffnds_raw_norm_rows.py','tests/test_ffnds_raw_norm_rows.py',PRE_SOURCE):
            obj['source_files'][name]=sha((ROOT/name).read_bytes())
        path.write_text(json.dumps(obj,indent=2)+'\n'); return
    raw=path.read_bytes(); cfg=json.loads(raw)
    expected=draft(hf); expected['source_files']=cfg['source_files']; expected['status']=cfg['status']
    need(cfg == expected and cfg['status'] in ('DRAFT_NOT_REGISTERED','FROZEN_REGISTERED'), 'protocol differs from closed parent')
    need(set(cfg['source_files']) == {'tools/ffnds_raw_norm_rows.py','tests/test_ffnds_raw_norm_rows.py',PRE_SOURCE} and cfg['source_files'][PRE_SOURCE] == PRE_SHA, 'exact source closure')
    for name,digest in cfg['source_files'].items(): need(sha((ROOT/name).read_bytes()) == digest, 'source hash')
    if args.command == 'inspect':
        print(json.dumps({'id':ID, 'protocol_sha256':sha(raw), 'parent_verified':True, 'captures':len(cfg['captures']), 'status':cfg['status']})); return
    need(cfg['status'] == 'FROZEN_REGISTERED', 'freeze protocol before registered run')
    active_doc=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()); active=active_doc['experiment_authority']
    standing=active_doc['user_continuation_authorization_20261027']
    matches=[x for x in active['current_user_authorized_scopes'] if x.get('id') == ID]
    need(active['all_cpu_gpu_model_experiments'] == 'USER_RESUMED_EXACT_REGISTERED_SCOPES_ONLY' and active['current_user_authorization'] == 'user_continuation_authorization_20261027' and len(matches)==1 and active_doc['status'] == 'E0_OPEN_DISCOVERY_USER_RESUMED_REGISTERED_SCOPES_ONLY' and standing['status'] == 'ACTIVE_CURRENT_USER_DIRECTED_SESSION' and standing['new_scope_registration_allowed'] is True and standing['automatic_retry'] is False, 'standing user authorization')
    scope=matches[0]; res=scope['resources']
    need(scope.get('protocol') == str(path.relative_to(ROOT)) and scope.get('protocol_sha256') == sha(raw) and scope.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and res == cfg['resources'] and scope.get('limits') == cfg['limits'], 'ACTIVE scope binding')
    need(args.out == cfg['output_HF_relative'], 'output must be frozen HF relative path')
    out=(hf/args.out).resolve(); need(out.is_relative_to(hf) and not out.exists(), 'fresh HF output directory')
    need(len(os.sched_getaffinity(0)) == 2, 'launch with exactly two allowed CPUs')
    out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger(out,cfg,sha(raw))
    try:
        for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[var]='2'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import numpy as np
        captures={}
        for c in cfg['captures']:
            need(time.monotonic()-start < 180, 'wall budget exceeded')
            with ledger.work('capture_loads'):
                captures[(c['source'],c['step'],c['operation'])]=load_capture(hf,c,np)
        result=analyze(captures,np,ledger)
        need(ledger.report['predebited'] == ledger.report['observed'] == cfg['limits'], 'incomplete operation counts')
        need(time.monotonic()-start < 180, 'wall budget exceeded')
        ledger.report.update(result, status='COMPLETE_SAVED_ROWS_DEVELOPMENT', wall_seconds=time.monotonic()-start)
        ledger.save()
    except BaseException as exc:
        ledger.fail(exc,time.monotonic()-start)
        raise


if __name__ == '__main__': main()
