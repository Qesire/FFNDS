"""Draft: raw BF16 post-GELU capture transaction for the fixed Wan study.

Execution requires an independently reviewed frozen protocol and ACTIVE scope.
Draft protocols cannot authorize a model call.
"""
from contextlib import ExitStack, contextmanager
import math

SITES = frozenset(((0, 0), (0, 20), (2, 0), (2, 20)))
SHAPE = (31668, 13824)
RAW_BYTES = 2 * math.prod(SHAPE)
SINGLE_FILE_LIMIT = 1 << 30


@contextmanager
def single_file_bound(maximum=SINGLE_FILE_LIMIT):
    import resource, signal
    old_limit=resource.getrlimit(resource.RLIMIT_FSIZE); old_signal=signal.getsignal(signal.SIGXFSZ)
    need(type(maximum) is int and maximum > 0 and (old_limit[1] == resource.RLIM_INFINITY or old_limit[1] >= maximum), 'single-file hard limit too small')
    def exceeded(*_): raise OSError('single-file output limit reached; partial output retained')
    signal.signal(signal.SIGXFSZ,exceeded)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE,(maximum,old_limit[1]))
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE,old_limit)
        signal.signal(signal.SIGXFSZ,old_signal)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save_reserved_packet(counter, path, payload_bytes, save):
    """Reserve before serializer, retain and account partial files on error."""
    counter.reserve_payload(payload_bytes)
    try:
        return save()
    finally:
        if path.exists():
            counter.observe_external(path)


class CaptureTransaction:
    """Enforce one input, gate and actual grid per selected native FFN.

    `copy_h` must synchronously snapshot its argument before the original W2
    consumes it; `commit` persists that snapshot and original gate without
    applying any masking or arithmetic. Both are charged by the caller.
    """
    def __init__(self, copy_h, commit, sites=SITES, expected_shape=SHAPE):
        self.copy_h, self.commit = copy_h, commit
        self.sites, self.expected_shape = frozenset(sites), tuple(expected_shape)
        self.pending, self.seen = None, set()

    def begin(self, step, block, expert, gate):
        site = (step, block)
        if site not in self.sites:
            need(self.pending is None, 'unexpected nested FFN')
            return False
        need(self.pending is None and site not in self.seen and expert == (0 if step < 2 else 1)
             and gate is not None, 'duplicate or invalid FFN transaction')
        self.pending = {'site': site, 'expert': expert, 'gate': gate, 'h': None, 'h_stride': None}
        return True

    def operand(self, step, block, expert, value):
        site = (step, block)
        if site not in self.sites:
            need(self.pending is None, 'unexpected W2 transaction')
            return
        p = self.pending
        need(p is not None and p['site'] == site and p['expert'] == expert and p['h'] is None
             and tuple(value.shape) == self.expected_shape, 'W2 input transaction differs')
        p['h_stride'] = tuple(value.stride()) if callable(getattr(value, 'stride', None)) else None
        p['h'] = self.copy_h(value)

    def finish(self, step, block, expert, gate, grid):
        site = (step, block)
        if site not in self.sites:
            need(self.pending is None, 'unexpected writeback transaction')
            return
        p = self.pending
        need(p is not None and p['site'] == site and p['expert'] == expert and p['h'] is not None
             and gate is p['gate'] and len(grid) == 3 and all(type(x) is int and x > 0 for x in grid)
             and math.prod(grid) == self.expected_shape[0], 'writeback identity or raster differs')
        self.commit(p, tuple(grid))
        self.seen.add(site)
        self.pending = None

    def verify_complete(self):
        need(self.pending is None and self.seen == self.sites, 'incomplete capture')


def make_tap_class(vm, native):
    """Create native tap with original inherited FFN/geometry accounting."""
    class FullMTap(vm.LightTap):
        def __init__(self, *args, capture, **kwargs):
            super().__init__(*args, **kwargs)
            self.capture = capture

        def mm_call(self, mm, value, meta, original):
            slot = self.mapping.get(id(mm))
            if slot is not None and slot[0] == 'block' and slot[2] == 'ffn.2':
                expert, block = self.block
                native.need(value.dtype == self.guard.torch.bfloat16, 'post-GELU dtype differs')
                self.capture.operand(self.step(expert), block, expert, value)
            return super().mm_call(mm, value, meta, original)

        def install(self):
            super().install()
            for expert, model in enumerate(self.runner.model.model):
                infer = model.transformer_infer
                old_ffn, old_post = infer.infer_ffn, infer.post_process
                def ffn(phase, x, attn, shift, scale, gate=None, *, _expert=expert, _infer=infer, _old=old_ffn):
                    step, block = self.step(_expert), int(_infer.block_idx)
                    native.need(self.block == (_expert, block), 'FFN block identity differs')
                    self.capture.begin(step, block, _expert, gate)
                    return _old(phase, x, attn, shift, scale, gate)
                def post(x, y, gate, pre_infer_out=None, *, _expert=expert, _infer=infer, _old=old_post):
                    step, block = self.step(_expert), int(_infer.block_idx)
                    if (step, block) in self.capture.sites:
                        native.need(pre_infer_out is not None, 'actual grid missing')
                        grid = tuple(pre_infer_out.grid_sizes.tuple)
                        self.capture.finish(step, block, _expert, gate, grid)
                    return _old(x, y, gate, pre_infer_out)
                self.stack.enter_context(native.patch_method(infer, 'infer_ffn', ffn))
                self.stack.enter_context(native.patch_method(infer, 'post_process', post))

        def finish(self):
            super().finish()
            self.capture.verify_complete()
    return FullMTap

# The scope is explicitly draft: the runner refuses it until a reviewer and
# the coordinator bind the real worker/launcher/source provenance.
ID = 'E0_TEAL_WAN_FULLM_H_CAPTURE_20261030'
OUTPUT = 'ffnds-derived/e0_teal_wan_fullm_h_capture_20261030/attempt1/worker/results'
CPU_RESOURCES = {'GPU':0, 'CPU_threads':4, 'wall_seconds':600, 'outer_seconds':900, 'new_HF_bytes':8 << 20, 'source_sha_read_bytes_max':1 << 30, 'automatic_retry':False}
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
             'host_available_min_bytes': 128 << 30, 'free_disk_min_bytes': 16 << 30,
             'wall_seconds': 3600, 'outer_seconds': 3900, 'new_HF_bytes': 9 << 30,
             'session_max_GPUs': 4, 'automatic_retry': False, 'single_file_bytes_max': SINGLE_FILE_LIMIT}


def rid(case, mode):
    need(mode == 'dense' and case['native_source_ordinal'] in (1040, 979), 'unexpected video')
    return 'teal_fullm_ord%d_seed42_dense' % case['native_source_ordinal']


def limits(vm, setup, latent_shape):
    result = {k: (v if k in setup else v // 2 * 2) for k, v in vm.native_original_limits.items()}
    result.update(core_CPU_reports=0, BF16_factor_GEMM=0, original_W2_H2D_skips=0,
        original_W2_H2D_skipped_bytes=0, factor_CPU_file_loads=0, factor_CPU_tensor_bytes=0,
        factor_GPU_family_preloads=0, factor_GPU_H2D_tensors=0, factor_GPU_H2D_bytes=0,
        prefetch_transfer_observations=324, stage_CUDA_event_pairs=14,
        state_tensor_D2H=20, state_tensor_D2H_bytes=2*math.prod(latent_shape)*30,
        state_output_writes=20, video_geometry_checks=2, historical_state_byte_checks=20,
        source_registration_writes=0, capture_H_D2H=8, capture_H_D2H_bytes=8*RAW_BYTES,
        capture_gate_D2H=8, capture_packets=8, capture_grids=8, historical_video_byte_checks=2)
    return result


def authorize(plan, digest, native, root, preflight=False):
    need(plan['id'] == ID and plan['resources'] == RESOURCES and plan['assets_finalized'] is True
         and plan['status'] == 'FROZEN_PENDING_REGISTRATION', 'draft scope is not executable')
    active = native.read(root / 'research/ACTIVE_RESEARCH.json')
    need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,
         'continuation revoked')
    scopes = [x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id') == ID]
    need(len(scopes) == 1 and scopes[0].get('protocol_sha256') == digest
         and scopes[0].get('execution_status') in (('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') if preflight else ('RUNNING_FIXED_SCOPE',))
         and scopes[0].get('limits') == plan['limits'] and scopes[0].get('resources') == RESOURCES
         and scopes[0].get('run_ids') == plan['run_ids'] and scopes[0].get('max_gpu_cards') == 1
         and scopes[0].get('session_max_GPUs') == 4, 'scope authorization differs')
    review = native.read(native.bound(root, scopes[0]['review']))
    fingerprint = native.canonical({k:v for k,v in plan.items() if k not in ('status','assets_finalized','review')})
    need(review.get('state') == 'REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition') == 'APPROVE'
         and review.get('scope_id') == ID and review.get('approval_scope_fingerprint') == fingerprint
         and review.get('cpu_preflight_spec_sha256')==scopes[0].get('cpu_preflight_spec_sha256')
         and review.get('gpu_launch_spec_sha256')==scopes[0].get('gpu_launch_spec_sha256')
         and review.get('responses_to_review') and review.get('adversarial_counterexample')
         and review.get('author_agent_id') and review.get('reviewer_agent_id')
         and review['author_agent_id'] != review['reviewer_agent_id']
         and review.get('critical_source_hashes') == {x['path']:x['sha256'] for x in plan['repo_sources']},
         'independent recheck missing or differs')
    for k in ('author_review', 'independent_review', 'reviewer_recheck'):
        native.bound(root, review[k])
    return scopes[0]


def verify_job(args, plan, digest, scope, root, worker_path, kind):
    import os
    from pathlib import Path
    import shutil
    need(kind in ('cpu','gpu'), 'job kind differs')
    expected_sha=scope['cpu_preflight_spec_sha256' if kind=='cpu' else 'gpu_launch_spec_sha256']
    need(args.job_spec is not None and __import__('hashlib').sha256(args.job_spec.read_bytes()).hexdigest()==expected_sha, 'job SHA differs')
    import json
    job=json.loads(args.job_spec.read_text()); hf=args.hf_root.resolve()
    affinity=scope['cpu_affinity']; need(isinstance(affinity,list) and len(affinity)==4 and len(set(affinity))==4 and all(type(x) is int and x>=0 for x in affinity), 'CPU affinity contract differs')
    env=['env','FFNDS_WORKER_CONFIG='+str(worker_path),'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin']
    script=str(root/'tools/e0_teal_wan_fullm_capture.py')
    child=['taskset','-c',','.join(str(x) for x in affinity),str(root/'.venv/bin/python'),'-I','-B',script]
    tail=['--protocol',str(args.protocol.resolve()),'--hf-root',str(hf),'--local-root',str(args.local_root.resolve()),'--job-spec',str(args.job_spec.resolve())]
    if kind=='cpu':
        result=(hf/OUTPUT).parent.parent/'cpu_preflight/worker/results'
        expected=env+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(result)]+tail
        need(args.out.resolve()==result and Path(job['output_dir']).resolve()==result.parent and Path(job['run_json']).resolve()==result/'RUN.json'
             and job['external_timeout_seconds']==CPU_RESOURCES['outer_seconds'], 'CPU launch paths/budget differ')
    else:
        gpu=os.environ.get('FFNDS_PHYSICAL_GPU',''); uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
        need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'), 'GPU assignment differs')
        expected=env+['bash',str(root/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid]+child+['run']+tail
        output=hf/OUTPUT
        need(Path(job['output_dir']).resolve()==output.parent and Path(job['run_json']).resolve()==output/'RUN.json'
             and job['external_timeout_seconds']==RESOURCES['outer_seconds'] and shutil.which('nvidia-smi')==plan['execution_monitor_binary']['path'], 'GPU launch paths/budget differ')
        binary=plan['execution_monitor_binary']; need(Path(binary['path']).stat().st_size==binary['bytes'] and __import__('hashlib').sha256(Path(binary['path']).read_bytes()).hexdigest()==binary['sha256'], 'monitor binary differs')
    need(job['argv']==expected and sorted(os.sched_getaffinity(0))==sorted(affinity)
         and job['id']==ID and job['protocol_sha256']==digest and Path(job['cwd']).resolve()==root
         and Path(job['hf_root']).resolve()==hf and Path(job['protocol_path']).resolve()==args.protocol.resolve(), 'exact job argv/identity differs')


def verify_static_assets(plan, native, root, hf, local_root, report, persist):
    """Precharge byte inspections; partial reads stay explicitly unknown."""
    import importlib.metadata
    source_bytes=sum(b['bytes'] for b in plan['repo_sources'])
    package_bytes=plan['local_package_inventory_payload_bytes']
    inventory_bytes=4*plan['local_transformers']['inventory']['bytes']
    model_metadata_bytes=sum(x['bytes'] for x in plan['model_view_metadata'])
    case_asset_bytes=sum(x['image']['bytes'] for x in plan['cases'])+sum(x['generation']['bytes'] for x in plan['original_dense_references'])
    need(source_bytes+package_bytes+inventory_bytes+model_metadata_bytes+case_asset_bytes<=CPU_RESOURCES['source_sha_read_bytes_max'], 'source/package byte ceiling exceeded')
    budget={'source_files':{'predebited_bytes':source_bytes,'completed_bytes':'UNKNOWN'},
            'package_inventory':{'predebited_bytes':inventory_bytes,'completed_bytes':'UNKNOWN'},
            'package_files':{'predebited_bytes':package_bytes,'completed_bytes':'UNKNOWN'},
            'model_metadata':{'predebited_bytes':model_metadata_bytes,'completed_bytes':'UNKNOWN'},
            'case_assets':{'predebited_bytes':case_asset_bytes,'completed_bytes':'UNKNOWN'},
            'small_metadata_read_bytes':'UNKNOWN_NOT_INCLUDED_IN_SOURCE_PACKAGE_BUDGET'}
    report['static_asset_reads']=budget; persist()
    for item in plan['repo_sources']: native.bound(root,item)
    budget['source_files']['completed_bytes']=source_bytes; persist()
    for case in plan['cases']: native.bound(hf,case['image'])
    for ref in plan['original_dense_references']: native.bound(hf,ref['generation'])
    budget['case_assets']['completed_bytes']=case_asset_bytes; persist()
    import e0_s2_causal_worker_proof_v5 as proof
    proof.require_closed_proof(plan,hf); native.model_metadata(plan)
    budget['model_metadata']['completed_bytes']=model_metadata_bytes; persist()
    inventory=native.read(native.bound(hf,plan['local_transformers']['inventory']))['files']
    need(sum(b['bytes'] for b in inventory)==package_bytes, 'package inventory byte identity differs')
    local,rows=native.verify_local_package(plan,{'repo':root,'hf':hf},local_root)
    budget['package_inventory']['completed_bytes']=inventory_bytes
    budget['package_files']['completed_bytes']=package_bytes; persist()
    need({k:importlib.metadata.version(k) for k in plan['runtime_versions']}==plan['runtime_versions'], 'package versions differ')
    return local,rows


def run(args):
    import importlib.metadata
    import importlib.util
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import sys
    import time
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tools'))
    spec = importlib.util.spec_from_file_location('_teal_capture_parent', root / 'tools/e0_teal_wan_all160_video_v3.py')
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    vm, native, io, base = parent.vm, parent.native, parent.io, parent.base
    from raw_bf16_factor_w2_runtime import FactorW2Runtime
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    hf = args.hf_root.resolve(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    scope = authorize(plan, digest, native, root)
    import worker_identity
    worker_identity.require_worker()
    worker_path = worker_identity.private_config_path()
    need(native.sha(worker_path) == plan['execution_private_worker_config_sha256'], 'private worker profile differs')
    need(os.environ.get('FFNDS_PHYSICAL_GPU') is not None and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'), 'assigned physical GPU and UUID required')
    need(plan['limits'] == limits(vm, base.acquisition.SETUP, native.VIDEO['latent_shape'])
         and plan['run_ids'] == [rid(c, 'dense') for c in plan['cases']]
         and [c['native_source_ordinal'] for c in plan['cases']] == [1040, 979]
         and plan['output_HF_relative'] == OUTPUT, 'fixed workload differs')
    need(not {'numpy', 'torch'} & set(sys.modules), 'fresh process required')
    verify_job(args,plan,digest,scope,root,worker_path,'gpu')
    cpu_proof=scope['cpu_preflight']; cpu_run=native.read(native.bound(hf,cpu_proof['RUN']))
    cpu_exit=native.read(native.bound(hf,cpu_proof['exit'])); vm.exited(cpu_exit,cpu_proof['RUN'],digest,ID)
    need(cpu_run['status']=='PASS_FULLM_CAPTURE_BYTE_PREFLIGHT' and cpu_run['id']==ID
         and cpu_run['protocol_sha256']==digest and cpu_run['source_closure_sha256']==native.canonical(plan['repo_sources'])
         and cpu_run['model_calls']==0 and cpu_run['GPU_calls']==0
         and cpu_exit.get('spec_sha256')==scope['cpu_preflight_spec_sha256'], 'closed CPU preflight differs')
    out = (hf / OUTPUT).resolve()
    need(out.is_relative_to(hf) and not out.exists(), 'exclusive HF output required')
    out.mkdir(parents=True)
    started = time.monotonic(); monitor = None; finder = None; no_hif4 = native.NoHiF4()
    report = {'id':ID, 'protocol_sha256':digest, 'status':'STARTED_NO_RETRY', 'phases':[], 'captures':[],
              'numeric_path':'BF16_ORIGINAL', 'automatic_retry':False, 'observer_time_is_not_speed':True}
    counter = io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    for name in plan['run_ids']: counter.hold(str(counter.path(out/name/'output.mp4')), 32 << 20)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    ledger = vm.Ledger(report.setdefault('operation_ledger', {}), lambda x:journal.write(json.dumps(x,separators=(',',':'))+'\n'))
    persist = lambda:counter.write_json(out/'RUN.json', report)
    def check():
        if monitor: monitor.check()
        counter.check()
        if 'torch' in sys.modules:
            need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU memory budget exceeded')
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes']
             and time.monotonic()-started <= RESOURCES['wall_seconds'], 'resource budget exceeded')
    def timeout(*_): raise TimeoutError('capture wall exceeded')
    persist(); previous_handler = signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    file_guard = single_file_bound(); entered_file_guard=False; old_tap = vm.LightTap
    old_id, old_limits, old_ids, old_resource = native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE
    old_vm = (vm.run_id, vm.RESOURCES, vm.transfer_geometry)
    try:
        file_guard.__enter__(); entered_file_guard=True
        local,package_rows=verify_static_assets(plan,native,root,hf,args.local_root,report,persist)
        finder=CheckedPackageFinder(local,package_rows)
        sys.dont_write_bytecode=True; sys.path[:0]=[str(local), str(root/'.runtime/code/LightX2V')]; sys.meta_path[:0]=[no_hif4,finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        from safetensors.torch import save_file
        from ub1_consumer_runtime import tensor_sha
        reservation, monitor, report['CUDA_reservation'] = vm.reserved_startup.reserve_before_assets(torch)
        need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0)
             and 'A100' in torch.cuda.get_device_name(0), 'one assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE = ID, plan['limits'], tuple(plan['run_ids']), RESOURCES
        vm.run_id, vm.RESOURCES, vm.transfer_geometry = rid, RESOURCES, base.transfer_geometry
        snap = out/'EXECUTED_SOURCES'; snap.mkdir()
        copies = [(root/b['path'],snap/b['path'],b) for b in plan['repo_sources']]
        copies.append((args.protocol,snap/'protocol.json',native.binding(args.protocol,root)))
        snapshot_bytes=sum(b['bytes'] for _,_,b in copies)
        report['static_asset_reads']['source_snapshot']={'predebited_bytes':snapshot_bytes,'completed_bytes':'UNKNOWN'}; persist()
        counter.hold('frozen_metadata',snapshot_bytes)
        for src,dst,b in copies: dst.parent.mkdir(parents=True,exist_ok=True); counter.copy_bound(src,dst,b)
        need(counter.holds.pop('frozen_metadata')==0,'snapshot bytes differ')
        report['static_asset_reads']['source_snapshot']['completed_bytes']=snapshot_bytes; persist()
        with native.OriginalGuard(torch,plan,ledger) as guard:
            pipe=native.build_pipeline(plan); torch.cuda.synchronize(); check()
            runtime_args=dict(factor_forward=vm.core.factor_forward,meter=vm.FactorMeter(ledger),synchronize=torch.cuda.synchronize,
                              record_stream=lambda t:t.record_stream(torch.cuda.current_stream()))
            with FactorW2Runtime(pipe,guard,{},**runtime_args) as runtime:
                for case, ref in zip(plan['cases'], plan['original_dense_references']):
                    check(); authorize(plan,digest,native,root); runtime.set_mode('dense'); name=rid(case,'dense')
                    previous=native.read(native.bound(hf,ref['generation']))
                    need(previous['source_id']==case['source_id'] and previous['image_sha256']==case['image']['sha256']
                         and previous['prompt_sha256']==case['prompt_sha256'] and previous['seed']==42
                         and previous['mode']=='dense' and len(previous['states'])==10, 'Dense reference identity differs')
                    controls={case['native_source_ordinal']:previous['states']}
                    def copy_h(value):
                        need(value.dtype==torch.bfloat16 and tuple(value.shape)==SHAPE,'actual H shape/dtype differs')
                        return ledger.call(name,{'capture_H_D2H':1,'capture_H_D2H_bytes':RAW_BYTES},
                                           lambda:value.detach().to('cpu',copy=True).contiguous())
                    def commit(p,grid):
                        def save():
                            check(); gate=p['gate']; need(gate.dtype==torch.bfloat16,'gate dtype differs')
                            gate_cpu=ledger.call(name,{'capture_gate_D2H':1},lambda:gate.detach().to('cpu',copy=True).contiguous())
                            step, block=p['site']; path=out/name/f'H_step{step}_block{block:02}.safetensors'
                            save_reserved_packet(counter,path,RAW_BYTES+gate_cpu.numel()*gate_cpu.element_size(),
                                lambda:save_file({'h':p['h'],'gate':gate_cpu},str(path)))
                            record={'source_ordinal':case['native_source_ordinal'],'step':step,'block':block,'expert':p['expert'],
                                    'h_shape':list(p['h'].shape),'h_dtype':'bfloat16','h_source_stride':list(p['h_stride']),'h_raw_sha256':tensor_sha(p['h']),
                                    'gate_shape':list(gate.shape),'gate_stride':list(gate.stride()),'gate_dtype':'bfloat16',
                                    'gate_raw_sha256':tensor_sha(gate_cpu),'actual_grid':list(grid),'file':counter.binding(path,hf)}
                            report['captures'].append(record); persist()
                        return ledger.call(name,{'capture_packets':1,'capture_grids':1},save)
                    transaction=CaptureTransaction(copy_h,commit)
                    tap_class=make_tap_class(vm,native)
                    vm.LightTap=lambda *a, **kw:tap_class(*a,capture=transaction,**kw)
                    phase=vm.native_phase(torch,pipe,guard,runtime,plan,case,'dense',out,hf,digest,report,persist,check,counter,controls)
                    vm.LightTap=old_tap
                    ledger.call(name,{'historical_video_byte_checks':1},lambda:need(
                        phase['video']['sha256']==ref['video_sha256'],'historical Dense video differs'))
                    persist()
            ledger.finish(); need(len(report['captures'])==8 and len(report['phases'])==2,'capture coverage differs')
            report.update(status='COMPLETE_FULLM_OBSERVATIONAL_CAPTURE',checked_transformers_imports=finder.origins)
    except BaseException as error:
        report.update(status='FAILED_CAPTURE_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous_handler); vm.LightTap=old_tap
        native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=old_id,old_limits,old_ids,old_resource
        vm.run_id,vm.RESOURCES,vm.transfer_geometry=old_vm
        monitor_error=None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources']=monitor.report()
                need(not report['assigned_GPU_resources']['violations'],'final monitor violation')
            except BaseException as error:
                monitor_error=error; report.update(status='FAILED_CAPTURE_PRESERVED_NO_RETRY',resource_error=str(error))
        for item in (finder,no_hif4):
            if item in sys.meta_path: sys.meta_path.remove(item)
        report['wall_seconds']=time.monotonic()-started; report['output_accounting']=counter.summary(); persist()
        journal.close()
        if entered_file_guard: file_guard.__exit__(None,None,None)
        if monitor_error: raise monitor_error


def preflight(args):
    import importlib.metadata
    import importlib.util
    import os
    from pathlib import Path
    import signal
    import sys
    import time
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'tools'))
    spec=importlib.util.spec_from_file_location('_teal_preflight_parent',root/'tools/e0_teal_wan_all160_video_v3.py')
    parent=importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    native,vm,base=parent.native,parent.vm,parent.base
    hf=args.hf_root.resolve(); plan=native.read(args.protocol); digest=native.sha(args.protocol)
    scope=authorize(plan,digest,native,root,preflight=True)
    import worker_identity
    worker_identity.require_worker()
    need(native.sha(worker_identity.private_config_path())==plan['execution_private_worker_config_sha256'], 'private worker differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'} & set(sys.modules), 'CPU-only preflight required')
    verify_job(args,plan,digest,scope,root,worker_identity.private_config_path(),'cpu')
    need(plan['CPU_preflight_resources']==CPU_RESOURCES and sum(b['bytes'] for b in plan['repo_sources'])<=CPU_RESOURCES['source_sha_read_bytes_max'], 'CPU proof budget differs')
    need(plan['limits']==limits(vm,base.acquisition.SETUP,native.VIDEO['latent_shape'])
         and plan['run_ids']==[rid(c,'dense') for c in plan['cases']]
         and [c['native_source_ordinal'] for c in plan['cases']]==[1040,979], 'fixed cases/ledger differ')
    out=args.out.resolve(); need(out.is_relative_to(hf) and not out.exists(), 'fresh CPU output required'); out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_CPU_PREFLIGHT_NO_RETRY'}
    native.write(out/'RUN.json',report); begun=time.monotonic()
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight wall'))); signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        local,package_rows=verify_static_assets(plan,native,root,hf,args.local_root,report,lambda:native.write(out/'RUN.json',report))
        need(not {'torch','numpy'} & set(sys.modules),'numerical module imported')
        report.update(status='PASS_FULLM_CAPTURE_BYTE_PREFLIGHT',source_closure_sha256=native.canonical(plan['repo_sources']),
                      model_calls=0,GPU_calls=0,numerical_imports=False)
    except BaseException as error:
        report.update(status='FAILED_CPU_PREFLIGHT_NO_RETRY',error_type=type(error).__name__,error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old); report['wall_seconds']=time.monotonic()-begun
        import json
        need(len(json.dumps(report).encode())<=CPU_RESOURCES['new_HF_bytes'], 'CPU proof output exceeds budget')
        native.write(out/'RUN.json',report)


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','run')); parser.add_argument('--out',type=Path); parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--hf-root',type=Path,required=True); parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--job-spec',type=Path,required=True)
    args=parser.parse_args();
    if args.action=='preflight':
        need(args.out is not None,'CPU output required'); preflight(args)
    else: run(args)

if __name__=='__main__': main()
