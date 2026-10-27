"""Bounded saved-operand Q2 intervention; no full model or residual execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time

ID = 'E0_Q2_DIRECT_COLLATERAL_20261027'
ROOT = Path(__file__).resolve().parents[1]
LIMITS = {'qdq': 32, 'bf16_linear': 56, 'fp32_logical': 72, 'fp32_batches': 24}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def check_authorization(plan, digest, active):
    if plan['id'] != ID or plan['limits'] != LIMITS:
        raise ValueError('Protocol ID or exact accounting mismatch')
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    if not any(x.get('id') == ID and x.get('protocol_sha256') == digest and
               x.get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE') for x in scopes):
        raise ValueError('Exact protocol SHA not authorized')
    expected = {(s, t, b) for s in (0, 12) for t in (0, 2) for b in (0, 39)}
    if {(x['source'], x['step'], x['block']) for x in plan['cases']} != expected or len(plan['cases']) != 8:
        raise ValueError('Wrong case coverage')
    if len(plan['shards']) != 4 or {(x['expert'], x['block']) for x in plan['shards']} != {(e,b) for e in ('high_noise_model','low_noise_model') for b in (0,39)}:
        raise ValueError('Wrong shard coverage')


def verify_assets(plan, hf, weight_hf=None):
    weight_hf = hf if weight_hf is None else weight_hf
    for item, cache in [(c, hf) for c in plan['cases']] + [(s, weight_hf) for s in plan['shards']]:
        path = (cache / item['path']).resolve()
        if not path.is_relative_to(cache) or not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('Asset SHA/path mismatch')
    for item in plan['sources']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item['sha256']:
            raise ValueError('Source SHA/path mismatch')
    required = {'tools/e0_q2_direct_collateral.py', 'tools/worker_identity.py', '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py', '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu', 'tools/startup_gpu_reservation.py', 'tools/assigned_gpu_resources.py', '.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py', '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QTensor.py', '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QType.py'}
    if not required <= {x['path'] for x in plan['sources']}:
        raise ValueError('Required source SHA bindings missing')
    if not any(x['path'].endswith('.so') and 'hifxg_quant' in x['path'] for x in plan['sources']): raise ValueError('Native compiled hifxg extension SHA required')


def masks(h, smooth):
    import torch
    x = (h.float() / smooth.float()).to(h.dtype)
    z = x[:, 6912:6976].abs()
    masks_out = []
    for indices in (z.argmax(dim=1), z.argmin(dim=1)):
        m = torch.zeros_like(h, dtype=torch.bool)
        m.scatter_(1, (indices + 6912).unsqueeze(1), True)
        masks_out.append(m)
    m = torch.zeros_like(h, dtype=torch.bool); m[:, 6912:6976] = True
    return x, masks_out + [m]


def terms(a, q, mask, whole=False):
    import torch
    if torch.count_nonzero(q[mask]).item(): raise ValueError('Q target not zero')
    outside = torch.ones_like(mask); outside[:, 6912:6976] = False
    if not torch.equal(q[outside], a[outside]): raise ValueError('HiF4 group64 closure violated')
    direct = -torch.where(mask, a, 0).float()
    collateral = torch.where(mask, 0, q.float() - a.float())
    total = q.float() - a.float()
    return direct, collateral, total


def stats(x, y):
    import torch
    x, y = x.double().flatten(), y.double().flatten()
    nx, ny = x.norm().item(), y.norm().item(); cross = torch.dot(x, y).item()
    return {'norm_x': nx, 'norm_y': ny, 'cross': cross,
            'cos': cross / (nx * ny) if nx and ny else None,
            'x_over_y': nx / ny if ny else None}


def write(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); os.replace(tmp, path)


def run(args):
    plan = json.loads(args.protocol.read_text()); digest = sha(args.protocol)
    check_authorization(plan, digest, json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
    hf = args.hf_root.resolve(); out = (hf / plan['output']).resolve()
    if not out.is_relative_to(hf) or out == hf: raise ValueError('Output must be within HF root')
    weight_hf = (args.weight_hf_root or hf).resolve()
    verify_assets(plan, hf, weight_hf)
    from worker_identity import require_worker
    require_worker()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f: json.dump({'id': ID, 'protocol_sha256': digest, 'attempt': 1}, f)
    report = {'id': ID, 'status': 'STARTED', 'protocol_sha256': digest, 'counts': dict.fromkeys(LIMITS,0), 'cases': [], 'scope': 'saved operands only; no residual, suffix, video, quality or speed claim'}
    started = time.monotonic(); monitor = None
    def check_resources():
        if monitor is not None: monitor.check()
        if shutil.disk_usage(out).free < (10 << 30): raise RuntimeError('Free disk reserve below 10 GiB')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()) > (1 << 30): raise RuntimeError('Output budget exceeded')
    def alarm(*_): raise TimeoutError('15-minute budget')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(900)
    def count(key, n=1):
        report['counts'][key] += n
        if report['counts'][key] > LIMITS[key]: raise RuntimeError('Call budget exceeded')
    try:
        import torch
        import torch.nn.functional as F
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        report['reservation'] = receipt
        check_resources()
        snapshots=out/'EXECUTED_SOURCES'
        for item in plan['sources']:
            target=snapshots/item['path']; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(args.protocol,snapshots/'protocol.json')
        check_resources()
        from safetensors.torch import load_file
        sys.path.insert(0, str(ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4'))
        sys.path.insert(0, str(ROOT/'.runtime/code/HiF4_Sp'))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
        ext=importlib.import_module('HiFloat4.hif4_gpu.quant_cy.base.cusrc.hifxg_quant')
        bound_so={str((ROOT/x['path']).resolve()) for x in plan['sources'] if x['path'].endswith('.so') and 'hifxg_quant' in x['path']}
        if str(Path(ext.__file__).resolve()) not in bound_so: raise ValueError('Loaded native extension is not bound')
        if torch.cuda.device_count() != 1: raise ValueError('Exactly one visible GPU required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_grad_enabled(False)
        cached = {}
        for shard in plan['shards']:
            rec = torch.load(weight_hf / shard['path'], map_location='cpu', weights_only=False)[shard['expert']][f"blocks.{shard['block']}.ffn.2"]
            w, smooth = rec['weight_qdq'], rec['smooth_scale']
            if tuple(w.shape)!=(5120,13824) or not w.is_floating_point() or smooth.numel()!=13824 or not smooth.is_floating_point() or not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(smooth).all()) or not bool((smooth>0).all()) or rec.get('activation_qdq') is not True or rec.get('force_fp32') is not True: raise ValueError('Shard tensor layout/flags/values mismatch')
            report.setdefault('shard_storage',[]).append({'expert':shard['expert'],'block':shard['block'],'weight_dtype':str(w.dtype),'smooth_dtype':str(smooth.dtype)})
            cached[(shard['expert'],shard['block'])] = (w.to(device='cuda',dtype=torch.bfloat16), smooth.reshape(1,-1).to(device='cuda',dtype=torch.float32))
            check_resources()
        panels = []
        for c in plan['cases']:
            expert = 'high_noise_model' if c['step'] == 0 else 'low_noise_model'
            if c['expert'] != expert: raise ValueError('Expert mismatch')
            d = load_file(str(hf/c['path']), device='cpu')
            shapes = {'gelu':(160,13824),'q2_post':(160,13824),'gate':(1,5120),'ffn_output':(160,5120),'b2':(5120,)}
            if any(tuple(d[k].shape)!=shape or d[k].dtype!=torch.bfloat16 for k,shape in shapes.items()): raise ValueError('Panel layout mismatch')
            d = {k:d[k].cuda() for k in shapes}; w,smooth = cached[(expert,c['block'])]
            pre, ms = masks(d['gelu'],smooth)
            count('qdq'); no = native_hif4_activation_qdq(pre,force_fp32=True).to(torch.bfloat16)
            if not torch.equal(no,d['q2_post']): raise ValueError('Original Q2 replay differs from saved')
            if len(c['row_plan']['rows']) != 160: raise ValueError('Row plan mismatch')
            panels.append((c,d,w,smooth,ms)); check_resources()
        for c,d,w,smooth,ms in panels:
            a=d['q2_post']; gate=d['gate'].float(); bias=d['b2']; count('bf16_linear'); check_resources()
            original_pre_q2=(d['gelu'].float()/smooth).to(torch.bfloat16)
            base=F.linear(a,w,bias); result={'source':c['source'],'step':c['step'],'block':c['block'],'base_replay':stats(base.float()-d['ffn_output'].float(), d['ffn_output']), 'arms':{}}
            for name,mask in zip(('max','min','whole'),ms):
                modified=d['gelu'].masked_fill(mask,0); pre=(modified.float()/smooth.float()).to(torch.bfloat16)
                count('qdq'); q=native_hif4_activation_qdq(pre,force_fp32=True).to(torch.bfloat16)
                direct, collateral, total = terms(a,q,mask,name=='whole')
                ad=a.masked_fill(mask,0); count('bf16_linear',2); yd=F.linear(ad,w,bias); yq=F.linear(q,w,bias)
                count('fp32_logical',3); count('fp32_batches'); projected=F.linear(torch.cat([direct,collateral,total]),w.float()).reshape(3,160,5120)*gate
                p0,p1,p2=projected.unbind(); error=p0+p1-p2
                # Conservative per-coordinate absolute dot bound using max |W| and max row L1.
                u=2**-24; gamma=(13824*u)/(1-13824*u)
                bound=gamma * float(w.float().abs().max().item()) * max(float(v.double().abs().sum(1).max().item()) for v in (direct,collateral,total)) * float(gate.abs().max().item())
                bound=3*bound + 4*u*max(float(p.abs().max().item()) for p in (p0,p1,p2))
                outside=torch.ones_like(mask);outside[:,6912:6976]=False
                rows=[]
                selected=mask.nonzero()[:,1].reshape(160,-1)
                for j in range(160):
                    v0,v1,v2=(v[j].double() for v in (p0,p1,p2))
                    rows.append({'row_id':c['row_plan']['rows'][j], 'selected_indices':selected[j].tolist(), 'selected_pre_q2':original_pre_q2[j,selected[j]].float().tolist(), 'direct_norm':v0.norm().item(),'collateral_norm':v1.norm().item(),'total_norm':v2.norm().item(),'direct_collateral_cross':torch.dot(v0,v1).item(),'direct_total_cross':torch.dot(v0,v2).item(),'collateral_total_cross':torch.dot(v1,v2).item()})
                result['arms'][name]={'rows':rows,'direct_vs_total':stats(p0,p2),'collateral_vs_total':stats(p1,p2),'direct_vs_collateral':stats(p0,p1),'decomposition_error':stats(error,p2),'bf16_D_vs_Q':stats(yd.float()-base.float(),yq.float()-base.float()),'target_nonzero':int(torch.count_nonzero(q[mask]).item()),'changed_inside_group':int(torch.count_nonzero((q!=a)&~outside).item()),'changed_outside_group':int(torch.count_nonzero((q!=a)&outside).item()),'max_abs_decomposition_error':float(error.abs().max().item()),'conservative_roundoff_bound':bound,'roundoff_note':'Standard FP32 model gamma_13824; FP64 row L1, max weight; includes gate and additions, excludes underflow/backend deviations; not a backend certificate.'}
            report['cases'].append(result); write(out/'RUN.json',report)
            check_resources()
        if report['counts'] != LIMITS: raise RuntimeError('Incomplete call accounting')
        report['status']='COMPLETE_LOCAL_DIAGNOSTIC'; report['elapsed_seconds']=time.monotonic()-started
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally:
        signal.alarm(0)
        if monitor is not None:
            monitor.close(); report['assigned_gpu_resources']=monitor.report()
            if report['assigned_gpu_resources']['violations']: report['status']='FAILED_GPU_MONITOR'
        write(out/'RUN.json',report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',type=Path,required=True); parser.add_argument('--hf-root',type=Path,required=True); parser.add_argument('--weight-hf-root',type=Path); run(parser.parse_args())
