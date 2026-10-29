"""Bound, single-axis RAFT observations of twelve SM120 matched videos per shard.

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
_video_spec = importlib.util.spec_from_file_location('_sm120_metric_video_parent', ROOT / 'tools/e0_s2_new12_sm120_fixed_depth_videos.py')
video_parent = importlib.util.module_from_spec(_video_spec); _video_spec.loader.exec_module(video_parent)
TEMPLATE = 'config/e0_raw_bf16_temporal_video20_quality_20261029_v1.json'
PARENTS = 'config/e0_s2_new12_sm120_fixed_depth_part{}_20261029_v1.json'
PARENT_HF = 'ffnds-derived/e0_s2_new12_sm120_fixed_depth_videos_20261029/part{}/attempt1/worker'
SHARDS = video_parent.SHARDS
MODES = video_parent.MODES
WEIGHT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'
WEIGHT_SUFFIX = '/raft_model/models/raft-things.pth'
LIMITS = {'dimension_calls': 1, 'video_results': 12, 'decoded_frames': 972,
          'RAFT_forwards': 480, 'RAFT_iterations': 9600, 'video_generations': 0,
          'training_updates': 0, 'decompositions': 0}
RESOURCES = {'GPU': 1, 'CPU_threads': 1, 'wall_seconds': 3600, 'outer_seconds': 3900,
             'new_HF_bytes': 256 << 20, 'GPU_reserved_bytes': 8 << 30,
             'free_disk_bytes': 10 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 1, 'wall_seconds': 900, 'outer_seconds': 1020,
                 'automatic_retry': False}
CPU_STATUS = 'PASS_SM120_FOUR_ARM_RAFT_BYTE_PREFLIGHT_NO_NUMPY_TORCH_GPU_MODEL'
STATUS = 'COMPLETE_SM120_FOUR_ARM_CONTINUOUS_RAFT_SHARD'
SELF = 'tools/e0_s2_sm120_fixed_depth_motion_metrics.py'
TEST = 'tests/test_e0_s2_sm120_fixed_depth_motion_metrics.py'


def identity(shard):
    return 'E0_S2_SM120_FIXED_DEPTH_CONTINUOUS_RAFT_20261029_SHARD{}'.format(shard)


def parent_data(hf, part):
    parent = ROOT / PARENTS.format(part)
    worker = Path(hf) / PARENT_HF.format(part)
    return {'protocol': base.binding(parent, ROOT),
            'RUN': base.binding(worker / 'results/RUN.json', hf),
            'exit': base.binding(worker / 'EXIT_RECEIPT.json', hf)}


def check_parent(proof, hf, shard):
    from types import SimpleNamespace
    need(proof['protocol']['path'] == PARENTS.format(shard), 'Wrong SM120 parent protocol')
    parent = base.read_bound(ROOT, proof['protocol']); result = base.read_bound(hf, proof['RUN'])
    base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], parent['id'])
    need(parent['part'] == shard and parent['id'] == result['id'] and result['part'] == shard
         and result['protocol_sha256'] == proof['protocol']['sha256']
         and result['status'] == 'COMPLETE_SM120_NEW12_FIXED_DEPTH_MATCHED_PART'
         and result['actual_full160_videos'] == 14 and result['actual_neutral_videos'] == 2
         and result['actual_new_source_videos'] == 12 and result['actual_states'] == 140
         and result['actual_flow_steps'] == 56 and result['actual_FFN_positions'] == 2240
         and result['SM120_runtime_neutral_all10_and_video_passed'] is True
         and result.get('GPU_compute_capability') == [12, 0]
         and not result['assigned_GPU_resources']['violations']
         and not result['operation_ledger']['forbidden_attempts']
         and all(result['operation_ledger'][side].get(k,0)==n for side in ('predebited','observed')
                 for k,n in parent['limits'].items()), 'SM120 full native parent is not closed')
    video_parent.verify_closed(SimpleNamespace(protocol=ROOT/proof['protocol']['path'], hf_root=Path(hf),
        run=Path(hf)/proof['RUN']['path'], exit=Path(hf)/proof['exit']['path']))
    need([c['native_source_ordinal'] for c in parent['cases']] == list(SHARDS[shard])
         and len(result['phases']) == 14 and [p['run_id'] for p in result['phases']] == parent['run_ids'],
         'Wrong SM120 three-source ordered parent')
    rows=[]
    for case in parent['cases']:
        ordinal=case['native_source_ordinal']
        phases=[p for p in result['phases'][2:] if p['native_source_ordinal']==ordinal]
        need([p['mode'] for p in phases] == list(video_parent.order(case)) and phases[0]['mode']=='dense',
             'Fresh Dense must precede three candidate arms')
        dense=phases[0]
        for p in phases:
            gen=base.read_bound(hf,p['generation_receipt'])
            need(p['id']==parent['id'] and p['protocol_sha256']==proof['protocol']['sha256']
                 and p['numeric_path']=='BF16_ORIGINAL' and p['seed']==case['seed']==42
                 and p['source_id']==case['source_id'] and p['image_sha256']==case['image']['sha256']
                 and p['prompt_sha256']==case['prompt_sha256'] and p['all160_FFN_covered'] is True
                 and p['completed_native_flow_steps']==4 and p['scheduler_generator_reset'] is True
                 and p['status']=='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
                 and gen['states']==p['states'] and gen['video']==p['video'] and gen['run_id']==p['run_id']
                 and gen['mode']==p['mode'] and set(p['states']) ==
                     {'raw_noise_FP32','initial_noise',*[f'flow_step{i}' for i in range(4)],*[f'latent_step{i}' for i in range(4)]}
                 and p['video']['frames']==81 and p['video']['size']==[832,464] and p['video']['fps']==16,
                 'SM120 generation identity differs')
            for name in ('raw_noise_FP32','initial_noise'):
                need(video_parent.vm.same_state(p['states'][name],dense['states'][name]), 'Same-source paired noise differs')
            rows.append({'ordinal':ordinal,'mode':p['mode'],'case':case,'generation':p['generation_receipt'],
                         'video':base.clip_binding(p['video'],hf),'states':p['states']})
    need(len(rows)==12 and len({(r['ordinal'],r['mode']) for r in rows})==12
         and len({r['case']['image']['sha256'] for r in rows})==3
         and len({r['video']['sha256'] for r in rows})==12, 'Twelve distinct clips from three images required')
    return rows


def draft(hf, shard):
    parents = [parent_data(hf, shard)]
    rows = check_parent(parents[0], hf, shard)
    template = base.read(ROOT / TEMPLATE)
    repo_paths = {b['path'] for b in template['repo_sources']} | {SELF, TEST, TEMPLATE, PARENTS.format(shard), 'tools/e0_s2_new12_sm120_fixed_depth_videos.py'}
    repo_paths |= {b['path'] for b in base.read_bound(ROOT, parents[0]['protocol'])['repo_sources']}
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
                output_HF_relative='ffnds-derived/e0_s2_sm120_fixed_depth_motion_metrics_20261029/shard{}/attempt1/worker/results'.format(shard),
                estimand='DEVELOPMENT single seed, three development image sources, same SM120 fresh Dense and three fixed policies; full-frame top-5-percent flow magnitude, not subject motion or official full quality',
                sampling={'videos': 12, 'frames_per_video': 81, 'fps': 16, 'frame_stride': 2,
                          'pairs_per_video': 40, 'RAFT_iterations_per_pair': 20})
    return plan


def validate(plan, frozen=False):
    shard = plan.get('shard')
    need(type(shard) is int and shard in (0, 1, 2, 3) and plan.get('id') == identity(shard), 'Shard identity differs')
    need(plan.get('limits') == LIMITS and plan.get('resources') == RESOURCES
         and plan.get('CPU_preflight_resources') == CPU_RESOURCES and plan.get('automatic_retry') is False,
         'Fixed scope differs')
    need(len(plan['parents']) == 1 and len(plan['videos']) == 12
         and [v['ordinal'] for v in plan['videos']] == [o for o in SHARDS[shard] for _ in MODES]
         and all({v['mode'] for v in plan['videos'] if v['ordinal']==o} == set(MODES) for o in SHARDS[shard]),
         'Source/arm coverage differs')
    need(plan.get('estimand') == 'DEVELOPMENT single seed, three development image sources, same SM120 fresh Dense and three fixed policies; full-frame top-5-percent flow magnitude, not subject motion or official full quality', 'Estimand differs')
    need(plan.get('sampling') == {'videos': 12, 'frames_per_video': 81, 'fps': 16, 'frame_stride': 2,
         'pairs_per_video': 40, 'RAFT_iterations_per_pair': 20}, 'Sampling differs')
    prior = base.read_bound(ROOT, plan['template'])
    need(plan['template']['path'] == TEMPLATE and all(plan[k] == prior[k] for k in ('environment', 'import_environment',
         'hf_bindings', 'reused_import_proof', 'directory_fd_audit_policy', 'activation_bootstrap_policy',
         'directory_fd_filelock_probe_source')), 'Original evaluator contract differs')
    selected = [b for b in prior['hf_bindings'] if '/installed_source_view/' in b['path'] or b['path'].endswith(WEIGHT_SUFFIX)]
    need(plan['raft_assets'] == selected and len([b for b in selected if b['path'].endswith(WEIGHT_SUFFIX)
         and b['sha256'] == WEIGHT_SHA]) == 1, 'RAFT source/weights differ')
    required = {x['path'] for x in prior['repo_sources']} | {SELF, TEST, TEMPLATE, PARENTS.format(shard), 'tools/e0_s2_new12_sm120_fixed_depth_videos.py'}
    required |= {b['path'] for b in base.read_bound(ROOT, plan['parents'][0]['protocol'])['repo_sources']}
    paths = [x['path'] for x in plan['repo_sources']]
    need(len(paths) == len(set(paths)) and set(paths) == required, 'Repository source closure differs')
    need(plan['output_HF_relative'] == 'ffnds-derived/e0_s2_sm120_fixed_depth_motion_metrics_20261029/shard{}/attempt1/worker/results'.format(shard), 'Output identity differs')
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
    rows += [('HF', b) for v in plan['videos'] for b in v['states'].values()]
    unique = {}
    for root, b in rows:
        key = (root, b['path'])
        item = {k:b[k] for k in ('path', 'bytes', 'sha256')}
        need(key not in unique or unique[key] == item, 'Conflicting asset binding')
        unique[key] = item
    return [(r, b) for (r, _), b in sorted(unique.items())]


def verify_lineage(plan, hf):
    rows = check_parent(plan['parents'][0], hf, plan['shard'])
    need(rows == plan['videos'], 'Bound SM120 parent/video/state lineage changed')
    paths = [base.bound(hf, row['video']) for row in rows]
    need(len(set(paths)) == len({(p.stat().st_dev,p.stat().st_ino) for p in paths}) == 12,
         'Aliased videos rejected')


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


def summarize(plan, rows):
    lookup={(r['ordinal'],r['mode']):r for r in rows}
    need(len(lookup)==12, 'Duplicate source/arm observation')
    def ratio(value, denominator):
        return {'value':value/denominator,'status':'DEFINED'} if denominator>0 else {'value':None,'status':'ZERO_DENSE_MOTION'}
    sources=[]
    for ordinal in SHARDS[plan['shard']]:
        values={mode:lookup[ordinal,mode]['summary']['mean_top5percent_flow_px_per_2frame_pair'] for mode in MODES}
        sources.append({'ordinal':ordinal,'means':values,
            'candidate_to_same_SM120_Dense_ratios':{mode:ratio(values[mode],values['dense']) for mode in MODES[1:]},
            'window_candidate_to_same_SM120_Dense_ratios':{mode:[ratio(x['mean_top5percent_flow_px'],y['mean_top5percent_flow_px'])
                 for x,y in zip(lookup[ordinal,mode]['summary']['five_fixed_8pair_windows'],lookup[ordinal,'dense']['summary']['five_fixed_8pair_windows'])]
                 for mode in MODES[1:]}})
    means={mode:statistics.fmean(row['means'][mode] for row in sources) for mode in MODES}
    return {'independent_image_sources':3,'seed_repetitions_per_source':1,'per_source':sources,
        'source_equal_means':means,'ratio_of_source_equal_means':{mode:ratio(means[mode],means['dense']) for mode in MODES[1:]},
        'mean_of_source_ratios':{mode:(statistics.fmean(row['candidate_to_same_SM120_Dense_ratios'][mode]['value'] for row in sources)
            if all(row['candidate_to_same_SM120_Dense_ratios'][mode]['status']=='DEFINED' for row in sources) else None) for mode in MODES[1:]},
        'scope':'Three source development descriptive comparisons; full-frame top5% flow, not subject motion, human review or acceptance'}


def validate_result(plan, report, hf):
    need(report.get('id') == plan['id'] and report.get('status') == STATUS
         and report.get('predebited') == report.get('invoked') == report.get('completed') == LIMITS,
         'Incomplete observed scope')
    rows = report.get('RAFT_observations', [])
    need(len(rows) == 12 and [(r.get('ordinal'),r.get('mode')) for r in rows] ==
         [(v['ordinal'],v['mode']) for v in plan['videos']], 'RAFT source/arm coverage differs')
    for row, video in zip(rows, plan['videos']):
        need(row.get('video_sha256') == video['video']['sha256'] and row.get('status') == 'COMPLETE'
             and row.get('video_path') == str((Path(hf) / video['video']['path']).resolve()), 'RAFT video identity differs')
        need(row.get('summary') == base.validate_raft(row), 'RAFT scores or summary differ')
    need(report.get('source_summary') == summarize(plan,rows), 'Source-level aggregation differs')
    result = report.get('official_dynamic_degree_only')
    need(isinstance(result, (list, tuple)) and len(result) == 2 and len(result[1]) == 12
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
    report = {'id': plan['id'], 'protocol_sha256': digest, 'status': 'STARTED_SM120_FOUR_ARM_RAFT',
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
            with ledger('dimension_calls'), ledger('video_results', 12), base.raft_observer(module, report['RAFT_observations'], ledger, expected), torch.inference_mode():
                result = module.compute_dynamic_degree(str(inp), device, submodules('dynamic_degree', assets, hf))
                torch.cuda.synchronize(); result = common.plain(result)
                base.check_result(result, expected, 'dynamic_degree')
        finally:
            module.cv2.VideoCapture = original_capture
            torch.load = old_load
        need(report['actual_RAFT_checkpoint_loads'] == [weight_binding], 'Exactly one actual RAFT checkpoint load required')
        need(len(report['RAFT_observations']) == 12, 'Twelve observations required')
        bypath = {r['video_path']: r for r in report['RAFT_observations']}
        need(set(bypath) == set(expected), 'Observed video coverage differs')
        report['RAFT_observations'] = [bypath[p] for p in expected]
        for row, video in zip(report['RAFT_observations'], plan['videos']):
            row.update(ordinal=video['ordinal'], mode=video['mode'], video_sha256=video['video']['sha256'], summary=base.validate_raft(row))
        report['source_summary'] = summarize(plan, report['RAFT_observations'])
        report['official_dynamic_degree_only'] = result
        ledger.finish(); check()
        need(not audit_record['denied_attempts'] and torch.cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_bytes'], 'GPU memory or audit budget exceeded')
        report.update(status=STATUS, peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        report.update(status='FAILED_SM120_RAFT_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
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
    p.add_argument('--shard', type=int, choices=(0,1,2,3)); p.add_argument('--protocol', type=Path)
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
