"""Synthetic A100 2:4 activation left-operand probe; exact frozen scope only."""
import argparse, hashlib, json, os, shutil, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_ACTIVATION24_SPMM_PROBE_20261027'
SHAPES = ((31668, 5120, 13824), (31668, 13824, 5120))
LIMITS = {'random_tensor_init': 4, 'top2_select': 16, 'compress': 16,
          'dense_gemm': 18, 'sparse_gemm': 30, 'model_calls': 0,
          'video_generations': 0, 'fits': 0, 'official_scores': 0}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def charge(counts, key):
    if key not in LIMITS or counts.get(key, 0) >= LIMITS[key]:
        raise RuntimeError('Budget exceeded before ' + key)
    counts[key] = counts.get(key, 0) + 1


def padded_rows(m, align=16):
    if type(m) is not int or m <= 0 or align <= 0: raise ValueError('Invalid padding')
    return ((m + align - 1) // align) * align


def select_top2(x):
    import torch
    if x.ndim != 2 or x.shape[1] % 4: raise ValueError('K must consist of four-tuples')
    q = x.reshape(x.shape[0], -1, 4)
    indices = torch.argsort(q.abs(), dim=-1, descending=True, stable=True)[..., :2]
    mask = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, indices, True)
    return torch.where(mask, q, torch.zeros_like(q)).reshape_as(x)


def authorize(plan, digest, active):
    if plan.get('id') != ID or plan.get('limits') != LIMITS or plan.get('shapes') != [list(s) for s in SHAPES] or plan.get('warmups') != 2 or plan.get('repetitions') != 5 or plan.get('wall_seconds') != 300 or not plan.get('assets_finalized'):
        raise ValueError('Unfrozen or changed scope')
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    if not any(s.get('id') == ID and s.get('protocol_sha256') == digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for s in scopes):
        raise ValueError('Exact protocol not authorized')


def verify(items):
    if len({x['path'] for x in items}) != len(items): raise ValueError('Duplicate binding')
    for x in items:
        p = (ROOT / x['path']).resolve()
        if not p.is_file() or sha(p) != x['sha256']: raise ValueError('Source binding mismatch')


def verify_loaded_backends(plan, maps_text=None):
    if maps_text is None:
        maps_text = Path('/proc/self/maps').read_text()
    expected = {}
    for item in plan['bindings']:
        path = (ROOT / item['path']).resolve()
        name = path.name
        kind = 'torch_cuda' if name == 'libtorch_cuda.so' else ('cusparselt' if name.startswith('libcusparseLt.so.') else None)
        if kind:
            if kind in expected: raise ValueError('Multiple bound backend libraries: ' + kind)
            expected[kind] = (path, item)
    if set(expected) != {'torch_cuda', 'cusparselt'}: raise ValueError('Missing bound backend libraries')
    loaded = {'torch_cuda': set(), 'cusparselt': set()}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'): continue
        raw = fields[5]
        deleted = raw.endswith(' (deleted)')
        path = Path(raw[:-10] if deleted else raw).resolve(); name = path.name
        kind = 'torch_cuda' if name == 'libtorch_cuda.so' else ('cusparselt' if name.startswith('libcusparseLt.so.') else None)
        if kind:
            if deleted: raise ValueError('Deleted mapped backend: ' + kind)
            loaded[kind].add(path)
    result = {}
    for kind, (path, item) in expected.items():
        if loaded[kind] != {path}: raise ValueError('Loaded backend differs from binding: ' + kind)
        if Path(item['path']).is_absolute(): raise ValueError('Backend binding path must be relative')
        result[kind] = {'binding_path': item['path'], 'sha256': item['sha256']}
    return result


def write(p, obj):
    temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(obj, allow_nan=False, separators=(',', ':'))+'\n'); os.replace(temp,p)


def run(args):
    plan = json.loads(args.protocol.read_text()); digest = sha(args.protocol)
    authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
    required = {'tools/e0_activation24_spmm_probe.py','tests/test_e0_activation24_spmm_probe.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py'}
    paths = {v['path'] for v in plan['bindings']}
    if not required <= paths or not any(x.endswith('torch/sparse/semi_structured.py') for x in paths) or not any(x.endswith('torch/sparse/_semi_structured_ops.py') for x in paths) or not any(x.endswith('torch/sparse/_semi_structured_conversions.py') for x in paths) or not any(x.endswith('libtorch_cuda.so') for x in paths) or not any(x.endswith('libcusparseLt.so.0') for x in paths): raise ValueError('Missing backend/source binding')
    verify(plan['bindings']); sys.path.insert(0,str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker()
    hf = args.hf_root.resolve(); out = (hf/plan['output_HF_relative']).resolve()
    if out == hf or not out.is_relative_to(hf): raise ValueError('Output escaped HF')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f: json.dump({'id':ID,'protocol_sha256':digest},f)
    report = {'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'cases':[],'scope':'synthetic BF16 only; not video or model quality','random_seeds':[20261027,20261028,20261029,20261030]}
    monitor=None; started=time.monotonic()
    def check_storage():
        if shutil.disk_usage(out).free < 10 << 30: raise RuntimeError('HF free space below 10 GiB')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()) > 32 << 20: raise RuntimeError('Output exceeds 32 MiB')
    def timeout(*_): raise TimeoutError('300-second bound')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(300)
    try:
        check_storage()
        snap=out/'EXECUTED_SOURCES';snap.mkdir()
        shutil.copyfile(args.protocol,snap/'protocol.json')
        for index,item in enumerate(plan['bindings']):
            source=(ROOT/item['path']).resolve()
            if source.suffix not in ('.py','.sh'): continue
            if source.stat().st_size > 1 << 20: raise RuntimeError('Source snapshot file exceeds 1 MiB')
            target=snap/(str(index)+'_'+source.name);shutil.copyfile(source,target)
            check_storage()
        check_storage()
        import torch
        import torch.nn.functional as F
        from startup_gpu_reservation import reserve_before_assets
        from torch.sparse.semi_structured import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUSPARSELT
        import torch.sparse.semi_structured as sparse_module
        bound = {str((ROOT / v['path']).resolve()) for v in plan['bindings']}
        if str(Path(sparse_module.__file__).resolve()) not in bound: raise RuntimeError('Imported sparse source differs from binding')
        reservation,monitor,receipt=reserve_before_assets(torch); report['reservation']=receipt
        if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or not torch.backends.cusparselt.is_available() or SparseSemiStructuredTensor._FORCE_CUTLASS: raise RuntimeError('A100 and default cuSPARSELt required')
        torch.set_num_threads(2); torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32=False
        report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'cusparselt':torch.backends.cusparselt.version(),'gpu':torch.cuda.get_device_name(0),'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats()
        def rand(shape,seed):
            charge(report['counts'],'random_tensor_init'); g=torch.Generator(device=device);g.manual_seed(seed);return torch.randn(shape,device=device,dtype=torch.bfloat16,generator=g)
        def select(a): charge(report['counts'],'top2_select');return select_top2(a)
        def compress(a):
            charge(report['counts'],'compress'); v=torch.sparse.to_sparse_semi_structured(a)
            if not isinstance(v,SparseSemiStructuredTensorCUSPARSELT): raise RuntimeError('Unexpected backend')
            return v
        def dense(a,w): charge(report['counts'],'dense_gemm');return F.linear(a,w)
        def sparse(a,w,m): charge(report['counts'],'sparse_gemm');return torch.mm(a,w.t())[:m]
        def timer(fn):
            monitor.check();torch.cuda.synchronize(); baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();st=time.monotonic();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();value=fn();end.record();torch.cuda.synchronize();return value,{'cuda_ms':start.elapsed_time(end),'wall_ms':(time.monotonic()-st)*1000,'allocated_before_bytes':baseline,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        for idx,(m,k,n) in enumerate(SHAPES):
            monitor.check(); check_storage(); mp=padded_rows(m); a=rand((m,k),20261027+2*idx);w=rand((n,k),20261028+2*idx)
            masked=select(a); padded=F.pad(masked,(0,0,0,mp-m)); packed=compress(padded)
            if idx == 0:
                report['loaded_backends']=verify_loaded_backends(plan);write(out/'RUN.json',report)
            dense_ref=dense(a,w); masked_ref=dense(masked,w); sparse_ref=sparse(packed,w,m);torch.cuda.synchronize()
            err=(sparse_ref.float()-masked_ref.float());den=masked_ref.float().norm().item(); diff=err.norm().item()/den if den else None
            if not bool(torch.isfinite(sparse_ref).all()) or diff is None or diff>plan['relative_l2_tolerance']:raise RuntimeError('Sparse vs masked numerical check failed')
            case={'M':m,'M_padded':mp,'K':k,'N':n,'sparse_vs_masked_relative_l2':diff,'sparse_vs_masked_max_abs':err.abs().max().item(),'masked_vs_dense_relative_l2':(masked_ref.float()-dense_ref.float()).norm().item()/dense_ref.float().norm().item(),'timing':{}}
            del dense_ref,masked_ref,sparse_ref,err
            def online():
                q=select(a); q=F.pad(q,(0,0,0,mp-m)); return sparse(compress(q),w,m)
            for name,fn in [('dense',lambda:dense(a,w)),('prepacked',lambda:sparse(packed,w,m)),('online',online)]:
                times=[]
                for it in range(7):
                    val,t=timer(fn);del val
                    if it>=2:times.append(t)
                case['timing'][name]=times
            case['last_timer_peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated();case['last_timer_peak_torch_reserved_bytes']=torch.cuda.max_memory_reserved()
            report['cases'].append(case);write(out/'RUN.json',report)
            check_storage()
            del a,w,masked,padded,packed
        check_storage()
        if report['counts']!=LIMITS:raise RuntimeError('Incomplete exact ledger')
        report.update(status='COMPLETE_SYNTHETIC_ONLY',elapsed_seconds=time.monotonic()-started)
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0)
        if monitor:
            monitor.close();report['assigned_gpu_resources']=monitor.report()
            if report['assigned_gpu_resources']['violations']:report['status']='FAILED_GPU_MONITOR'
        write(out/'RUN.json',report)
        check_storage()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);run(p.parse_args())
