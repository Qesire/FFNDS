#!/usr/bin/env python3
"""Bounded, saved-state-only trajectory comparison; no model execution."""
import argparse
import hashlib
import json
import os
import resource
import struct
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

SHAPE = [16, 21, 58, 104]
MAX_READ = 256 * 1024 * 1024
MAX_OUTPUT = 1024 * 1024


def digest(b):
    return hashlib.sha256(b).hexdigest()


def cosine(a, b):
    aa = float(np.sum(a * a, dtype=np.float64))
    bb = float(np.sum(b * b, dtype=np.float64))
    return None if aa == 0 or bb == 0 else float(np.sum(a * b, dtype=np.float64) / (aa * bb) ** .5)


def diagnostic(dense, candidate, previous):
    d = np.asarray(dense, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    e = c - d
    prev = np.asarray(previous, dtype=np.float64)
    delta = e - prev
    sq = lambda v: float(np.sum(v * v, dtype=np.float64))
    err, prior, inc, cross = sq(e), sq(prev), sq(delta), float(2 * np.sum(prev * delta, dtype=np.float64))
    denom = sq(d)
    per_slot = []
    for i in range(d.shape[1]):
        de, ce = d[:, i], c[:, i]
        energy = sq(de)
        per_slot.append({'slot': i, 'relative_sse': sq(ce-de)/energy if energy else None, 'cosine': cosine(de, ce)})
    valid = [v['relative_sse'] for v in per_slot if v['relative_sse'] is not None]
    return {'relative_sse': err / denom if denom else None, 'cosine': cosine(d,c),
            'time_slots': per_slot, 'time_slot_equal_mean_relative_sse': sum(valid)/len(valid) if len(valid)==len(per_slot) else None,
            'error_norm_squared': err, 'previous_error_norm_squared': prior,
            'increment_norm_squared': inc, 'signed_twice_inner_product': cross,
            'identity_residual': err-prior-inc-cross,
            'error_norm_ratio_to_previous': (err/prior)**.5 if prior else None,
            'increment_cosine_previous_error': cosine(delta,prev)}, e


def load_state(hf, b):
    p = (hf / b['path']).resolve()
    if not p.is_relative_to(hf.resolve()) or b['root'] != 'hf': raise ValueError('asset path')
    raw = p.read_bytes()
    if len(raw) != b['bytes'] or digest(raw) != b['sha256']: raise ValueError('file identity')
    n = struct.unpack('<Q', raw[:8])[0]
    if n > 65536 or n + 8 > len(raw): raise ValueError('header length')
    header = json.loads(raw[8:8+n])
    if set(header) != {'latents'}: raise ValueError('tensor keys')
    t = header['latents']
    if t != {'dtype':'F32','shape':SHAPE,'data_offsets':[0, int(np.prod(SHAPE))*4]}: raise ValueError('header contract')
    payload = raw[8+n:]
    if len(payload) != int(np.prod(SHAPE))*4 or digest(payload) != b['raw_tensor_sha256']: raise ValueError('tensor identity')
    a = np.frombuffer(payload, dtype='<f4').reshape(SHAPE)
    if not np.isfinite(a).all(): raise ValueError('nonfinite tensor')
    # Widened native BF16 must have zero low sixteen bits in each FP32 word.
    if np.any(np.frombuffer(payload, dtype='<u4') & 65535): raise ValueError('not widened BF16')
    return a, {'header_sha256':digest(raw[8:8+n]), 'header_bytes':n, 'file_sha256':digest(raw), 'raw_tensor_sha256':digest(payload)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',required=True); ap.add_argument('--hf-root',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    start=time.monotonic(); cfg=json.loads(Path(args.protocol).read_bytes())
    if cfg['status'] != 'FROZEN_REGISTERED': raise ValueError('protocol not frozen and registered')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only scope requires hidden GPUs')
    repo = Path(__file__).resolve().parents[1]
    for item in cfg['repo_sources']:
        source = (repo / item['path']).resolve()
        content = source.read_bytes()
        if not source.is_relative_to(repo) or len(content) != item['bytes'] or digest(content) != item['sha256']:
            raise ValueError('frozen source identity')
    hf=Path(args.hf_root).resolve(); out=(hf/args.output).resolve()
    if not out.is_relative_to(hf) or args.output != cfg['output_HF_relative']: raise ValueError('output path')
    if out.exists(): raise FileExistsError(out)
    rp=hf/cfg['parent']['RUN']['path']; ep=hf/cfg['parent']['EXIT_RECEIPT']['path']; rb=rp.read_bytes(); eb=ep.read_bytes()
    if digest(rb)!=cfg['parent']['RUN']['sha256'] or digest(eb)!=cfg['parent']['EXIT_RECEIPT']['sha256']: raise ValueError('parent identity')
    run=json.loads(rb); receipt=json.loads(eb)
    if receipt['child_returncode']!=0 or not receipt['process_exited'] or receipt['timeout'] or receipt['run_sha256']!=digest(rb) or receipt['protocol_sha256']!=run['protocol_sha256'] or not run['status'].startswith('COMPLETE_'): raise ValueError('parent exit')
    phases={(p['source_index'],p['mode']):p for p in run['phases']}
    if len(run['phases']) != 6 or set(phases)!={(s,m) for s in (0,12) for m in ('dense','energy94','random94')}: raise ValueError('cohort')
    expected={(x['source'],x['mode'],x['step']):x['binding'] for x in cfg['states']}
    if len(expected)!=24 or sum(b['bytes'] for b in expected.values())+len(rb)+len(eb)>MAX_READ: raise ValueError('read budget')
    states={}; evidence=[]
    for (source,mode),p in phases.items():
        if p['seed'] != 42 or len(p['trajectory'])!=4: raise ValueError('phase')
        for b in p['trajectory']:
            key=(source,mode,b['step'])
            if key not in expected or expected[key]!=b or b['shape']!=SHAPE or b['native_dtype_before_FP32_storage']!='torch.bfloat16': raise ValueError('binding')
            states[key],ev=load_state(hf,b); evidence.append({'source':source,'mode':mode,'step':b['step'],**ev})
    results=[]
    for source in (0,12):
        for mode in ('energy94','random94'):
            previous=np.zeros(SHAPE,dtype=np.float64)
            for step in range(4):
                values,previous=diagnostic(states[source,'dense',step],states[source,mode,step],previous)
                results.append({'source':source,'candidate':mode,'step':step,**values})
    elapsed=time.monotonic()-start
    if len(results)!=16 or elapsed>180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>512*1024*1024: raise RuntimeError('budget exceeded')
    report={'schema':'ffnds.saved_trajectory_drift.v1','id':cfg['id'],
            'protocol_sha256':digest(Path(args.protocol).read_bytes()),
            'status':'COMPLETE_SAVED_STATE_DESCRIPTIVE_DIAGNOSTIC',
            'NUMERIC_PATH':'HIF4_FAKE_QDQ_DIAGNOSTIC',
            'parent':cfg['parent'],'states':evidence,'pairs':results,
            'counts':{'state_reads':24,'candidate_dense_pairs':16,'model_calls':0,'GPU_calls':0},
            'wall_seconds':elapsed,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'NumPy_version':np.__version__,
            'read_bytes':sum(b['bytes'] for b in expected.values())+len(rb)+len(eb),
            'interpretation':'Two exposed source trajectories; accumulated free-trajectory difference and path-dependent increments, not same-input local FFN error, flow causation or video quality.'}
    data=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>MAX_OUTPUT: raise RuntimeError('output budget')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f: f.write(data)

if __name__ == '__main__': main()
