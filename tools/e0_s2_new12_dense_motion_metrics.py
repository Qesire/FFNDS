"""Bound, single-axis RAFT observations of twelve existing S2 Dense videos.

Drafts are NOT authorization. Each shard requires its own frozen protocol, ACTIVE
scope and separately exited byte-only CPU preflight. No generation is performed.
"""
import argparse
import copy
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('_new12_raft_quality_helpers', ROOT / 'tools/e0_raw_bf16_temporal_video20_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
base, v4 = quality.base, quality.v4
need = base.require
TEMPLATE = 'config/e0_raw_bf16_temporal_video20_quality_20261029_v1.json'
PARENTS = 'config/e0_s2_new12_dense_baseline_part{}_20261029_v2.json'
PARENT_HF = 'ffnds-derived/e0_s2_new12_dense_baseline_v2_20261029/part{}/attempt1/worker'
SHARDS = ((911, 912, 978, 979), (1039, 1040, 922, 923), (616, 624, 456, 472))
WEIGHT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'
WEIGHT_SUFFIX = '/raft_model/models/raft-things.pth'
LIMITS = {'dimension_calls': 1, 'video_results': 4, 'decoded_frames': 324,
          'RAFT_forwards': 160, 'RAFT_iterations': 3200, 'video_generations': 0,
          'training_updates': 0, 'decompositions': 0}
RESOURCES = {'GPU': 1, 'CPU_threads': 1, 'wall_seconds': 3600, 'outer_seconds': 3900,
             'new_HF_bytes': 256 << 20, 'GPU_reserved_bytes': 8 << 30,
             'free_disk_bytes': 10 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 1, 'wall_seconds': 900, 'outer_seconds': 1020,
                 'automatic_retry': False}
CPU_STATUS = 'PASS_NEW12_RAFT_BYTE_PREFLIGHT_NO_NUMPY_TORCH_GPU_MODEL'
STATUS = 'COMPLETE_NEW12_DENSE_CONTINUOUS_RAFT_SHARD'
SELF = 'tools/e0_s2_new12_dense_motion_metrics.py'
TEST = 'tests/test_e0_s2_new12_dense_motion_metrics.py'


def identity(shard):
    return 'E0_S2_NEW12_DENSE_CONTINUOUS_RAFT_20261029_SHARD{}'.format(shard)


def parent_data(hf, part):
    parent = ROOT / PARENTS.format(part)
    worker = Path(hf) / PARENT_HF.format(part)
    return {'protocol': base.binding(parent, ROOT),
            'RUN': base.binding(worker / 'results/RUN.json', hf),
            'exit': base.binding(worker / 'EXIT_RECEIPT.json', hf)}


def check_parents(parents, hf):
    phases = {}
    for part, proof in enumerate(parents):
        need(proof['protocol']['path'] == PARENTS.format(part), 'Wrong parent protocol')
        p = base.read_bound(ROOT, proof['protocol'])
        r = base.read_bound(hf, proof['RUN'])
        base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], p['id'])
        need(p['part'] == r['part'] == part and p['id'] == r['id']
             and r['protocol_sha256'] == proof['protocol']['sha256']
             and r['status'] == 'COMPLETE_NEW12_DENSE_BASELINE_PART'
             and r['actual_full160_videos'] == 7 and len(r['phases']) == 7
             and r['historical_control_all_ten_states_passed'] is True
             and not r['assigned_GPU_resources']['violations']
             and not r['operation_ledger']['forbidden_attempts']
             and all(r['operation_ledger'][side].get(k, 0) == n for side in ('predebited', 'observed')
                     for k, n in p['limits'].items()), 'Parent generation not closed')
        cases = {c['native_source_ordinal']: c for c in p['cases']}
        need(len(cases) == len(p['cases']) == 7 and len(set(p['run_ids'])) == 7
             and {x['run_id'] for x in r['phases']} == set(p['run_ids']), 'Parent case/run collision')
        for phase in r['phases']:
            ordinal = phase['native_source_ordinal']
            c = cases[ordinal]
            need(ordinal not in phases and phase['id'] == p['id']
                 and phase['protocol_sha256'] == proof['protocol']['sha256']
                 and phase['run_id'] in p['run_ids'] and phase['mode'] == 'dense'
                 and phase['numeric_path'] == 'BF16_ORIGINAL'
                 and phase['status'] == 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
                 and phase['all160_FFN_covered'] is True and phase['completed_native_flow_steps'] == 4
                 and phase['scheduler_generator_reset'] is True
                 and phase['seed'] == c['seed'] == 42
                 and phase['source_id'] == c['source_id']
                 and phase['image_sha256'] == c['image']['sha256']
                 and phase['prompt_sha256'] == c['prompt_sha256'], 'Parent phase/case mismatch')
            gen = base.read_bound(hf, phase['generation_receipt'])
            need(gen == {k:v for k,v in phase.items() if k != 'generation_receipt'}, 'GENERATION payload differs from parent phase')
            video = phase['video']
            need(video['frames'] == 81 and video['size'] == [832, 464] and video['fps'] == 16,
                 'Parent video geometry differs')
            phases[ordinal] = {'ordinal': ordinal, 'case': c, 'parent_part': part,
                               'generation': phase['generation_receipt'], 'video': base.clip_binding(video, hf)}
    return phases


def draft(hf, shard):
    parents = [parent_data(hf, part) for part in (0, 1)]
    phases = check_parents(parents, hf)
    template = base.read(ROOT / TEMPLATE)
    rows = [phases[o] for o in SHARDS[shard]]
    repo_paths = {b['path'] for b in template['repo_sources']} | {SELF, TEST, TEMPLATE, *[PARENTS.format(p) for p in (0, 1)]}
    # Preserve original evaluator environment and successful import receipt verbatim;
    # only source/checkpoint assets needed here are independently byte-proved.
    plan = {k: copy.deepcopy(template[k]) for k in ('environment', 'import_environment', 'hf_bindings',
            'reused_import_proof', 'directory_fd_audit_policy', 'activation_bootstrap_policy',
            'directory_fd_filelock_probe_source')}
    selected = [b for b in template['hf_bindings'] if '/installed_source_view/' in b['path'] or b['path'].endswith(WEIGHT_SUFFIX)]
    need(len([b for b in selected if b['path'].endswith(WEIGHT_SUFFIX) and b['sha256'] == WEIGHT_SHA]) == 1,
         'Original actual RAFT weights not bound')
    plan.update(id=identity(shard), shard=shard, status='DRAFT_NOT_AUTHORIZED', assets_finalized=False,
                binding_status='BOUND_DRAFT_NOT_REGISTERED', automatic_retry=False,
                execution_monitor_binary=None, execution_private_worker_config_sha256=None,
                limits=copy.deepcopy(LIMITS), resources=copy.deepcopy(RESOURCES),
                CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES), parents=parents, videos=rows,
                raft_assets=selected, repo_sources=[base.binding(ROOT / p, ROOT) for p in sorted(repo_paths)],
                template=base.binding(ROOT / TEMPLATE, ROOT),
                output_HF_relative='ffnds-derived/e0_s2_new12_dense_motion_metrics_20261029/shard{}/attempt1/worker/results'.format(shard),
                estimand='DEVELOPMENT single seed, per-source full-frame top-5-percent flow magnitude, not subject motion or official full quality',
                sampling={'videos': 4, 'frames_per_video': 81, 'fps': 16, 'frame_stride': 2,
                          'pairs_per_video': 40, 'RAFT_iterations_per_pair': 20})
    return plan


def validate(plan, frozen=False):
    shard = plan.get('shard')
    need(type(shard) is int and shard in (0, 1, 2) and plan.get('id') == identity(shard), 'Shard identity differs')
    need(plan.get('limits') == LIMITS and plan.get('resources') == RESOURCES
         and plan.get('CPU_preflight_resources') == CPU_RESOURCES and plan.get('automatic_retry') is False,
         'Fixed scope differs')
    need([v['ordinal'] for v in plan['videos']] == list(SHARDS[shard]) and len(plan['parents']) == 2,
         'Source coverage differs')
    need(plan.get('sampling') == {'videos': 4, 'frames_per_video': 81, 'fps': 16, 'frame_stride': 2,
         'pairs_per_video': 40, 'RAFT_iterations_per_pair': 20}, 'Sampling differs')
    prior = base.read_bound(ROOT, plan['template'])
    need(plan['template']['path'] == TEMPLATE and all(plan[k] == prior[k] for k in ('environment', 'import_environment',
         'hf_bindings', 'reused_import_proof', 'directory_fd_audit_policy', 'activation_bootstrap_policy',
         'directory_fd_filelock_probe_source')), 'Original evaluator contract differs')
    selected = [b for b in prior['hf_bindings'] if '/installed_source_view/' in b['path'] or b['path'].endswith(WEIGHT_SUFFIX)]
    need(plan['raft_assets'] == selected and len([b for b in selected if b['path'].endswith(WEIGHT_SUFFIX)
         and b['sha256'] == WEIGHT_SHA]) == 1, 'RAFT source/weights differ')
    required = {x['path'] for x in prior['repo_sources']} | {SELF, TEST, TEMPLATE, *[PARENTS.format(p) for p in (0,1)]}
    paths = [x['path'] for x in plan['repo_sources']]
    need(len(paths) == len(set(paths)) and set(paths) == required, 'Repository source closure differs')
    need(plan['output_HF_relative'] == 'ffnds-derived/e0_s2_new12_dense_motion_metrics_20261029/shard{}/attempt1/worker/results'.format(shard), 'Output identity differs')
    monitor = plan.get('execution_monitor_binary')
    private_sha = plan.get('execution_private_worker_config_sha256')
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    need(monitor is None or (isinstance(monitor, dict) and set(monitor) == {'path', 'bytes', 'sha256'}
         and isinstance(monitor['path'], str) and Path(monitor['path']).is_absolute()
         and type(monitor['bytes']) is int and monitor['bytes'] > 0 and valid_sha(monitor['sha256'])),
         'Invalid execution monitor binary binding')
    need(private_sha is None or valid_sha(private_sha), 'Invalid execution private worker config SHA')
    if frozen:
        need(monitor is not None and private_sha is not None, 'Execution worker/monitor bindings required')
        need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
             and plan['binding_status'] == 'BOUND_READY_FOR_REGISTRATION', 'Draft is not frozen')


def verify_execution_monitor(plan):
    binding = plan['execution_monitor_binary']
    found = shutil.which('nvidia-smi')
    need(found is not None, 'Execution nvidia-smi unavailable')
    path = Path(found).resolve()
    need(path == Path(binding['path']).resolve() and path.is_file()
         and path.stat().st_size == binding['bytes'] and base.sha(path) == binding['sha256'],
         'Execution monitor binary path/bytes/SHA differs')
    return path


def verify_execution_worker(plan):
    from worker_identity import require_worker, private_config_path
    require_worker()
    need(base.sha(private_config_path()) == plan['execution_private_worker_config_sha256'],
         'Execution private worker config SHA differs')


def authorize(plan, digest, active=None):
    validate(plan, True)
    active = base.read(ROOT / 'research/ACTIVE_RESEARCH.json') if active is None else active
    need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True,
         'Current continuation not authorized')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    need(len(scopes) == 1 and scopes[0].get('protocol_sha256') == digest
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
         and scopes[0].get('limits') == LIMITS and scopes[0].get('resources') == RESOURCES
         and scopes[0].get('max_gpu_cards') == 1, 'ACTIVE exact scope/protocol/resources required')
    return scopes[0]


def all_bindings(plan):
    rows = [('REPO', b) for b in plan['repo_sources']]
    rows += [('HF', b) for b in plan['raft_assets']]
    rows += [('HF', p[k]) for p in plan['parents'] for k in ('RUN', 'exit')]
    rows += [('HF', v[k]) for v in plan['videos'] for k in ('generation', 'video')]
    unique = {}
    for root, b in rows:
        key = (root, b['path'])
        item = {k:b[k] for k in ('path', 'bytes', 'sha256')}
        need(key not in unique or unique[key] == item, 'Conflicting asset binding')
        unique[key] = item
    return [(r, b) for (r, _), b in sorted(unique.items())]


def verify_lineage(plan, hf):
    phases = check_parents(plan['parents'], hf)
    need([phases[o] for o in SHARDS[plan['shard']]] == plan['videos'], 'Bound parent/video lineage changed')
    paths = [base.bound(hf, row['video']) for row in plan['videos']]
    need(len(set(paths)) == len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 4,
         'Aliased video files')
    need(len({r['case']['image']['sha256'] for r in plan['videos']}) == 4, 'Source image cluster collision')


def stat_record(path):
    s = Path(path).stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def cpu_expected(plan, digest):
    return {'id': plan['id'], 'protocol_sha256': digest, 'status': CPU_STATUS,
            'sources_sha256': base.state_sha(plan['repo_sources']), 'parents': plan['parents'],
            'execution_monitor_binary': plan['execution_monitor_binary'],
            'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
            'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False,
            'model_calls': 0, 'GPU_calls': 0}


def preflight(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
         and len(os.sched_getaffinity(0)) <= 1, 'One CPU CUDA-hidden byte preflight required')
    plan = base.read(args.protocol); digest = base.sha(args.protocol); authorize(plan, digest)
    verify_execution_worker(plan); verify_execution_monitor(plan)
    need(args.out.resolve().is_relative_to(args.hf_root.resolve()) and not args.out.exists(), 'Fresh confined CPU report required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'status': 'STARTED_BYTE_PREFLIGHT', 'byte_checks': []}
    base.write(args.out, report); start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU preflight budget exhausted')))
    signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for root, b in all_bindings(plan):
            path = base.bound(ROOT if root == 'REPO' else args.hf_root, b)
            report['byte_checks'].append({'root': root, 'binding': b, 'stat': stat_record(path)})
        verify_lineage(plan, args.hf_root)
        quality.verify_import_proof(plan, args.hf_root)
        quality.verify_environment(plan, args.hf_root)
        need(not {'torch', 'numpy'} & set(sys.modules), 'Numerical imports occurred in byte preflight')
        report.update(cpu_expected(plan, digest)); report['byte_checks_sha256'] = base.state_sha(report['byte_checks'])
    except BaseException as e:
        report.update(status='FAILED_CPU_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic() - start; base.write(args.out, report)


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight')
    need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Closed CPU proof required')
    report = base.read_bound(hf, proof['RUN'])
    base.exited(base.read_bound(hf, proof['launch_exit']), proof['RUN'], digest, plan['id'])
    need(all(report.get(k) == v for k,v in cpu_expected(plan, digest).items())
         and 0 <= report.get('wall_seconds', -1) <= CPU_RESOURCES['wall_seconds']
         and report.get('byte_checks_sha256') == base.state_sha(report['byte_checks'])
         and [(r['root'], r['binding']) for r in report['byte_checks']] == all_bindings(plan), 'CPU proof mismatch')
    for row in report['byte_checks']:
        need(stat_record((ROOT if row['root'] == 'REPO' else hf) / row['binding']['path']) == row['stat'], 'Preflight asset changed')
    return proof


def install_decoder_counter(module, ledger):
    """Count actual successful OpenCV reads while preserving original decoding."""
    original = module.cv2.VideoCapture
    class Capture:
        def __init__(self, path):
            self.inner = original(path); self.count = 0
        def __getattr__(self, key): return getattr(self.inner, key)
        def read(self):
            success, frame = self.inner.read()
            if success:
                with ledger('decoded_frames'): self.count += 1
            return success, frame
        def release(self):
            self.inner.release()
            need(self.count == 81, 'Expected 81 successfully decoded frames')
    module.cv2.VideoCapture = Capture
    return original


def validate_result(plan, report, hf):
    need(report.get('id') == plan['id'] and report.get('status') == STATUS
         and report.get('predebited') == report.get('invoked') == report.get('completed') == LIMITS,
         'Incomplete observed scope')
    rows = report.get('RAFT_observations', [])
    need(len(rows) == 4 and [r.get('ordinal') for r in rows] == list(SHARDS[plan['shard']]), 'RAFT source coverage differs')
    for row, video in zip(rows, plan['videos']):
        need(row.get('video_sha256') == video['video']['sha256'] and row.get('status') == 'COMPLETE'
             and row.get('video_path') == str((Path(hf) / video['video']['path']).resolve()), 'RAFT video identity differs')
        need(row.get('summary') == base.validate_raft(row), 'RAFT scores or summary differ')
    need(report.get('source_equal_mean_top5percent_flow_px') == statistics.fmean(
         row['summary']['mean_top5percent_flow_px_per_2frame_pair'] for row in rows), 'Source aggregate differs')
    result = report.get('official_dynamic_degree_only')
    need(isinstance(result, (list, tuple)) and len(result) == 2 and len(result[1]) == 4
         and {r['video_path']:r['video_results'] for r in result[1]} ==
             {r['video_path']:r['official_binary'] for r in rows}, 'Dynamic degree/observer disagreement')
    weight = next(b for b in plan['raft_assets'] if b['path'].endswith(WEIGHT_SUFFIX))
    need(report.get('actual_RAFT_checkpoint_loads') == [weight], 'Actual RAFT weight identity missing')
    expected_source = next(b for b in plan['raft_assets'] if b['path'].endswith('/vbench/third_party/RAFT/core/raft.py'))
    need(report.get('actual_RAFT_source') == expected_source, 'Actual RAFT class source identity missing')
    need(not report.get('assigned_GPU_resources', {}).get('violations')
         and report.get('assigned_GPU_resources', {}).get('sampled_isolation_pass') is True,
         'GPU resource evidence missing/violated')


def run(args):
    need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh isolated worker required')
    hf = args.hf_root.resolve(); plan = base.read(args.protocol); digest = base.sha(args.protocol)
    scope = authorize(plan, digest); proof = require_cpu(scope, plan, digest, hf)
    verify_execution_worker(plan); verify_execution_monitor(plan); base.precision_environment()
    out = (hf / plan['output_HF_relative']).resolve()
    need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined attempt required')
    out.mkdir(parents=True); base.write(out / 'ATTEMPT.json', {'id': plan['id'], 'protocol_sha256': digest, 'automatic_retry': False})
    report = {'id': plan['id'], 'protocol_sha256': digest, 'status': 'STARTED_NEW12_RAFT',
              'prior_CPU_preflight': proof, 'parents': plan['parents'], 'RAFT_observations': [],
              'interpretation': plan['estimand'], 'official_full_quality': 'NOT_RUN', 'automatic_retry': False}
    def persist_ledger():
        base.write(out / 'RUN.json', report)
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
             <= RESOURCES['new_HF_bytes'], 'New output budget exceeded during operation')
    ledger = base.Ledger(report, limits=LIMITS, persist=persist_ledger)
    started = time.monotonic(); monitor = None
    def check():
        authorize(plan, digest)
        if monitor: monitor.check()
        need(time.monotonic() - started <= RESOURCES['wall_seconds'], 'Wall budget exceeded')
        need(shutil.disk_usage(out).free >= RESOURCES['free_disk_bytes'], 'Free disk reserve exceeded')
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()) <= RESOURCES['new_HF_bytes'], 'New output budget exceeded')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed RAFT budget exhausted')))
    signal.alarm(RESOURCES['wall_seconds'])
    try:
        verify_lineage(plan, hf); quality.verify_import_proof(plan, hf)
        helper, v8 = quality.verify_environment(plan, hf)
        need(len(os.sched_getaffinity(0)) <= 1, 'One CPU affinity required')
        base.scope_cache_environment(out)
        os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        report['isolation'] = base.configure_import_environment(plan, hf)
        helper.disable_local_ipv6_probe(hf / plan['environment']['site'], v8)
        numpy = importlib.import_module('numpy'); report['numpy'] = base.module_observation(numpy, 'numpy', plan, hf)
        torch = importlib.import_module('torch'); report['torch'] = base.module_observation(torch, 'torch', plan, hf)
        from startup_gpu_reservation import reserve_before_assets
        report['execution_monitor_binary'] = plan['execution_monitor_binary']
        verify_execution_monitor(plan)
        reservation, monitor, receipt = reserve_before_assets(torch); report['reservation'] = receipt
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        name = torch.cuda.get_device_name(0)
        need(torch.cuda.device_count() == 1 and ('A100' in name or 'RTX PRO 6000' in name), 'Assigned GPU must be A100 or RTX PRO 6000')
        report['GPU_name'] = name
        snapshot = out / 'EXECUTED_SOURCES'; snapshot.mkdir()
        for item in plan['repo_sources']:
            target = snapshot / item['path']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / item['path'], target)
        shutil.copyfile(args.protocol, snapshot / 'protocol.json')
        common = importlib.import_module('e0_ub1_consumer_official_quality')
        from offline_i2v_metric_loaders import submodules
        from assigned_gpu_resources import AssignedGPUResources
        binary = verify_execution_monitor(plan)
        def no_network(*_a, **_kw): raise RuntimeError('Implicit network access forbidden')
        import socket
        socket.socket.connect = no_network; socket.create_connection = no_network
        torch.hub.download_url_to_file = no_network; torch.hub.load = no_network; torch.hub.load_state_dict_from_url = no_network
        imports = {'denied_attempts': []}; _, _, checked_font = helper.import_audit_guard(out, imports)
        import subprocess
        subprocess.check_output = checked_font(subprocess.check_output, hf / plan['environment']['site'] / 'matplotlib/font_manager.py')
        module, observation = v4.metric_module('dynamic_degree', plan, hf)
        report['official_function'] = observation; report['native_metric_import_audit'] = imports
        need(not imports['denied_attempts'], 'Native import audit violation')
        raft_source = Path(inspect.getfile(module.RAFT)).resolve()
        expected_raft = (hf / plan['environment']['source_root'] / 'vbench/third_party/RAFT/core/raft.py').resolve()
        source_binding = next(b for b in plan['raft_assets'] if (hf / b['path']).resolve() == expected_raft)
        need(raft_source == base.bound(hf, source_binding), 'Actual RAFT class origin differs')
        report['actual_RAFT_source'] = source_binding
        audit_record = {'denied_attempts': []}; report['native_dir_fd_audit'] = audit_record
        def denied(reason):
            def fail(): audit_record['denied_attempts'].append(reason); raise RuntimeError(reason)
            return fail
        audit = common.audit_scope(out, AssignedGPUResources.query.__code__, binary, os.environ['FFNDS_PHYSICAL_GPU'])
        sys.addaudithook(v4.v3.normalize_dir_fd_audit(audit, out, audit_record, denied))
        assets = base.read(ROOT / plan['environment']['asset_manifest'])
        device = torch.device('cuda:0')
        expected = [str(base.bound(hf, v['video'])) for v in plan['videos']]
        inp = out / 'dynamic_degree_INPUT.json'
        base.write(inp, [{'prompt_en': v['case']['prompt'], 'dimension': ['dynamic_degree'],
                          'video_list': [p]} for v,p in zip(plan['videos'], expected)])
        check(); original_capture = install_decoder_counter(module, ledger)
        weight_binding = next(b for b in plan['raft_assets'] if b['path'].endswith(WEIGHT_SUFFIX))
        weight_path = base.bound(hf, weight_binding)
        old_load = torch.load
        report['actual_RAFT_checkpoint_loads'] = []
        def checked_load(path, *args_, **kwargs_):
            need(isinstance(path, (str, os.PathLike)) and Path(path).resolve() == weight_path
                 and not report['actual_RAFT_checkpoint_loads'], 'Unexpected or repeated checkpoint load')
            need(base.sha(weight_path) == WEIGHT_SHA, 'Actual RAFT checkpoint bytes changed')
            report['actual_RAFT_checkpoint_loads'].append(weight_binding)
            base.write(out / 'RUN.json', report)
            return old_load(path, *args_, **kwargs_)
        torch.load = checked_load
        try:
            with ledger('dimension_calls'), ledger('video_results', 4), base.raft_observer(module, report['RAFT_observations'], ledger, expected), torch.inference_mode():
                result = module.compute_dynamic_degree(str(inp), device, submodules('dynamic_degree', assets, hf))
                torch.cuda.synchronize(); result = common.plain(result)
                base.check_result(result, expected, 'dynamic_degree')
        finally:
            module.cv2.VideoCapture = original_capture
            torch.load = old_load
        need(report['actual_RAFT_checkpoint_loads'] == [weight_binding], 'Exactly one actual RAFT checkpoint load required')
        need(len(report['RAFT_observations']) == 4, 'Four observations required')
        bypath = {r['video_path']: r for r in report['RAFT_observations']}
        need(set(bypath) == set(expected), 'Observed video coverage differs')
        report['RAFT_observations'] = [bypath[p] for p in expected]
        for row, video in zip(report['RAFT_observations'], plan['videos']):
            row.update(ordinal=video['ordinal'], video_sha256=video['video']['sha256'], summary=base.validate_raft(row))
        report['source_equal_mean_top5percent_flow_px'] = statistics.fmean(r['summary']['mean_top5percent_flow_px_per_2frame_pair'] for r in report['RAFT_observations'])
        report['official_dynamic_degree_only'] = result
        ledger.finish(); check()
        need(not audit_record['denied_attempts'] and torch.cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_bytes'], 'GPU memory or audit budget exceeded')
        report.update(status=STATUS, peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        report.update(status='FAILED_NEW12_RAFT_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally:
        original_error = sys.exc_info()[1]; signal.alarm(0); report['wall_seconds'] = time.monotonic()-started
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                need(not report['assigned_GPU_resources']['violations'], 'Assigned GPU monitor violation')
            except BaseException as e:
                report.update(status='FAILED_GPU_MONITOR_PRESERVED_NO_RETRY', monitor_error=str(e))
                if original_error is None:
                    base.write(out/'RUN.json', report); raise
        base.write(out/'RUN.json', report)
    try:
        validate_result(plan, report, hf)
    except BaseException as e:
        report.update(status='FAILED_RESULT_VERIFICATION_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e))
        base.write(out / 'RUN.json', report); raise


def verify_closed(args):
    plan = base.read(args.protocol); validate(plan, True); digest = base.sha(args.protocol)
    need(args.run.resolve() == (args.hf_root / plan['output_HF_relative'] / 'RUN.json').resolve()
         and args.exit.resolve() == (args.hf_root / plan['output_HF_relative']).parent.joinpath('EXIT_RECEIPT.json').resolve(),
         'Closed RUN/EXIT paths differ from shard attempt')
    run_binding = base.binding(args.run, args.hf_root); report = base.read(args.run)
    base.exited(base.read(args.exit), run_binding, digest, plan['id'])
    need(report.get('protocol_sha256') == digest and 0 <= report.get('wall_seconds', -1) <= RESOURCES['wall_seconds'], 'Closed RUN protocol/wall differs')
    require_cpu({'cpu_preflight': report.get('prior_CPU_preflight')}, plan, digest, args.hf_root)
    verify_lineage(plan, args.hf_root)
    validate_result(plan, report, args.hf_root)
    return {'id': plan['id'], 'status': 'PASS_CLOSED_RAFT_SHARD', 'RUN': run_binding,
            'exit': base.binding(args.exit, args.hf_root), 'model_calls': 0, 'GPU_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('draft','inspect','preflight','run','verify-closed'))
    p.add_argument('--shard', type=int, choices=(0,1,2)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--out', type=Path); p.add_argument('--run', type=Path); p.add_argument('--exit', type=Path)
    a = p.parse_args()
    if a.action == 'draft':
        need(a.shard is not None and a.out is not None and not a.out.exists(), 'Fresh draft/shard required')
        a.out.parent.mkdir(parents=True, exist_ok=True); base.write(a.out, draft(a.hf_root, a.shard))
    elif a.action == 'inspect':
        plan = base.read(a.protocol); validate(plan); print(json.dumps({'id':plan['id'], 'limits':LIMITS, 'resources':RESOURCES}, indent=2))
    elif a.action == 'preflight': preflight(a)
    elif a.action == 'run': run(a)
    else: print(json.dumps(verify_closed(a), indent=2))

if __name__ == '__main__': main()
