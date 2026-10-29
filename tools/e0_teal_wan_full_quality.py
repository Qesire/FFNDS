"""Preparation helpers for eight-axis TEAL Wan matched-video evaluation.

A draft requires the closed native parent; it does not authorize execution.
Each shard requires frozen ACTIVE registration and an exited CPU byte preflight.
"""
import argparse
import copy
import gc
import importlib
import inspect
import json
import os
import shutil
import signal
import time
import importlib.util
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
_spec = importlib.util.spec_from_file_location('_s2_full_quality_motion', ROOT / 'tools/e0_teal_wan_motion_metrics.py')
motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motion)
base = motion.base
need = base.require
exec_spec = importlib.util.spec_from_file_location('_teal_quality_private_execution',ROOT/'tools/e0_teal_quality_execution.py')
execution = importlib.util.module_from_spec(exec_spec);exec_spec.loader.exec_module(execution)

DIMENSIONS = tuple(motion.quality.DIMENSIONS)
MODES, SHARDS = motion.MODES, motion.SHARDS
SELF = 'tools/e0_teal_wan_full_quality.py'
TEST = 'tests/test_e0_teal_wan_full_quality.py'
SOURCE_BANK = 'config/e0_s2_new12_development_source_bank_20261029_v1.json'
LIMITS = {'official_dimension_calls': 8, 'video_dimension_results': 24,
          'RAFT_forwards': 120, 'RAFT_iterations': 2400, 'decoded_frames': 243,
          'video_generations': 0, 'training_updates': 0, 'decompositions': 0}
COSTS = {'imaging_quality': {'MUSIQ': 243},
         'aesthetic_quality': {'CLIP_L14': 9, 'aesthetic_head': 9},
         'motion_smoothness': {'AMT': 120},
         'dynamic_degree': {'RAFT': 120, 'RAFT_iterations': 2400},
         'subject_consistency': {'DINO': 243},
         'background_consistency': {'CLIP_B32': 3},
         'i2v_subject': {'DINO': 246}, 'i2v_background': {'DreamSim_embed': 246}}
RESOURCES = {'GPU': 1, 'CPU_threads': 2, 'wall_seconds': 3600, 'outer_seconds': 3900,
             'new_HF_bytes': 1 << 30, 'GPU_reserved_bytes': 8 << 30,
             'free_disk_bytes': 10 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 900,
                 'outer_seconds': 1020, 'automatic_retry': False}


def identity(shard):
    need(type(shard) is int and shard in range(4), 'Invalid shard')
    return 'E0_TEAL_WAN_EIGHT_AXIS_20261029_SHARD{}'.format(shard)


def draft(hf, shard):
    """Bind only after actual EXIT0 and parent verify_closed (including 2 controls)."""
    identity(shard)
    parent = motion.parent_data(hf, shard)
    rows = motion.check_parent(parent, hf, shard)
    template = base.read(ROOT / motion.TEMPLATE)
    env_fields = ('environment', 'import_environment', 'hf_bindings', 'reused_import_proof',
                  'directory_fd_audit_policy', 'activation_bootstrap_policy',
                  'directory_fd_filelock_probe_source')
    plan = {k: copy.deepcopy(template[k]) for k in env_fields}
    paths = {b['path'] for b in template['repo_sources']} | {SELF, TEST, motion.SELF, motion.TEST,
             motion.TEMPLATE, SOURCE_BANK, motion.PARENTS.format(shard), 'tools/e0_teal_wan_quality_lineage.py', 'tests/test_e0_teal_wan_quality_lineage.py', 'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py', 'tests/test_e0_s2_sm120_fixed_depth_motion_metrics_v2.py'}
    paths |= execution.HELPERS
    paths |= {b['path'] for b in base.read_bound(ROOT, parent['protocol'])['repo_sources']}
    bank = base.read(ROOT / SOURCE_BANK)
    by_ordinal = {s['native_source_ordinal']: s for s in bank['sources']}
    need(len(by_ordinal) == len(bank['sources']) == 12, 'Source bank identity differs')
    images = []
    for ordinal in SHARDS[shard]:
        case = next(v['case'] for v in rows if v['ordinal'] == ordinal)
        source = by_ordinal[ordinal]
        need(source['source_id'] == case['source_id'] and source['image']['sha256'] == case['image']['sha256']
             and source['prompt'] == case['prompt'], 'Frozen bank and parent source differ')
        images.append({'ordinal': ordinal, 'name': str(ordinal) + '.png', 'binding': case['image'],
                       'scene_stratum': source['scene_stratum']})
    plan.update(id=identity(shard), shard=shard, status='DRAFT_NOT_AUTHORIZED', binding_status='BOUND_DRAFT_NOT_REGISTERED',
                assets_finalized=False, automatic_retry=False, limits=copy.deepcopy(LIMITS),
                costs=copy.deepcopy(COSTS), resources=copy.deepcopy(RESOURCES),
                CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES), parents=[parent], videos=rows,
                images=images, source_bank=base.binding(ROOT / SOURCE_BANK, ROOT), template=base.binding(ROOT / motion.TEMPLATE, ROOT),
                repo_sources=[base.binding(ROOT / p, ROOT) for p in sorted(paths)],
                execution_monitor_binary=execution.monitor_binding(base), execution_private_worker_config_sha256=execution.PROFILE_SHA,
                execution_device={'name_contains':'A100','cuda_capability':[8,0],'session_max_GPUs':4},
                output_HF_relative='ffnds-derived/e0_teal_wan_full_quality_20261029/shard{}/attempt1/worker/results'.format(shard),
                policy={'all_eight_dimensions_each_clip': True, 'I2V_axes': 'DEVELOPMENT_DIAGNOSTIC_NO_OFFICIAL_STRATUM',
                        'continuous_RAFT': 'SEPARATE_REGISTERED_PRODUCER', 'temporal_flickering': 'NOT_RUN_NINTH_AXIS',
                        'aggregation': 'SOURCE_EQUAL_PAIRED_FRESH_SAME_DEVICE_DENSE', 'official_five_independent_samples': False},
                dimensions=list(DIMENSIONS))
    return plan


def all_bindings(plan):
    rows = [('REPO', b) for b in plan['repo_sources']]
    rows += [('HF', b) for b in plan['hf_bindings']]
    rows += [(b['root'], b) for b in plan['import_environment']['modules'].values()]
    rows += [('HF', p[k]) for p in plan['parents'] for k in ('RUN', 'exit')]
    rows += [('REPO', plan['reused_import_proof']['protocol'])]
    rows += [('HF', plan['reused_import_proof'][k]) for k in ('RUN', 'exit')]
    rows += [('HF', v[k]) for v in plan['videos'] for k in ('generation', 'video')]
    rows += [('HF', b) for v in plan['videos'] for b in v['states'].values()]
    rows += [('HF', image['binding']) for image in plan['images']]
    unique = {}
    for root, b in rows:
        item = {k: b[k] for k in ('path', 'bytes', 'sha256')}
        key = (root, item['path'])
        need(key not in unique or unique[key] == item, 'Conflicting asset identity')
        unique[key] = item
    return [(root, b) for (root, _), b in sorted(unique.items())]


def summarize(plan, hf, dimensions):
    """Source-equal paired scalars; no fabricated overall official score."""
    need(len(dimensions) == len(DIMENSIONS) and {x['dimension'] for x in dimensions} == set(DIMENSIONS), 'Eight unique dimensions required')
    videos = plan['videos']
    lookup = {str((Path(hf) / v['video']['path']).resolve()): v for v in videos}
    need(len(lookup) == len(videos) == 3 and {(v['ordinal'], v['mode']) for v in videos} ==
         {(o, m) for o in SHARDS[plan['shard']] for m in MODES}, 'Exact source/arm coverage required')
    scores = {(v['ordinal'], v['mode']): {} for v in videos}
    for entry in dimensions:
        dimension = entry['dimension']
        for row in base.check_result(entry['result'], list(lookup), dimension):
            v = lookup[row['video_path']]
            if dimension.startswith('i2v_'):
                need(row.get('image_path') == 'vbench2_beta_i2v/data/crop/16-9/' + str(v['ordinal']) + '.png', 'I2V image pairing differs')
            scores[v['ordinal'], v['mode']][dimension] = float(row['video_results'])
    sources = []
    for ordinal in SHARDS[plan['shard']]:
        values = {m: scores[ordinal, m] for m in MODES}
        differences = {m: {d: values[m][d] - values['dense'][d] for d in DIMENSIONS} for m in MODES[1:]}
        sources.append({'ordinal': ordinal, 'scores': values, 'candidate_minus_same_fresh_Dense': differences})
    axes = {d: {'source_count': 1, 'source_equal_means': {m: statistics.fmean(s['scores'][m][d] for s in sources) for m in MODES},
                'source_equal_paired_differences': {m: statistics.fmean(s['candidate_minus_same_fresh_Dense'][m][d] for s in sources) for m in MODES[1:]}}
            for d in DIMENSIONS}
    return {'per_source': sources, 'per_axis': axes, 'independent_image_sources': 1,
            'samples_per_source_per_arm': 1, 'official_five_independent_samples': False,
            'quality_accepted': False, 'I2V_axes': 'DEVELOPMENT_DIAGNOSTIC_NO_OFFICIAL_STRATUM'}

CPU_STATUS = 'PASS_TEAL_WAN_EIGHT_AXIS_BYTE_PREFLIGHT_NO_NUMPY_TORCH_GPU_MODEL'
STATUS = 'COMPLETE_TEAL_WAN_EIGHT_AXIS_SHARD_QUALITY_NOT_ACCEPTED'


def validate(plan, frozen=False):
    shard = plan['shard']
    need(plan['id'] == identity(shard) and plan['dimensions'] == list(DIMENSIONS)
         and plan['limits'] == LIMITS and plan['costs'] == COSTS
         and plan['resources'] == RESOURCES and plan['CPU_preflight_resources'] == CPU_RESOURCES
         and plan['automatic_retry'] is False, 'Fixed quality scope differs')
    need(plan['policy'] == {'all_eight_dimensions_each_clip': True,
         'I2V_axes': 'DEVELOPMENT_DIAGNOSTIC_NO_OFFICIAL_STRATUM',
         'continuous_RAFT': 'SEPARATE_REGISTERED_PRODUCER', 'temporal_flickering': 'NOT_RUN_NINTH_AXIS',
         'aggregation': 'SOURCE_EQUAL_PAIRED_FRESH_SAME_DEVICE_DENSE', 'official_five_independent_samples': False}, 'Policy differs')
    need(len(plan['parents']) == 1 and plan['parents'][0]['protocol']['path'] == motion.PARENTS.format(shard), 'Wrong parent')
    prior = base.read_bound(ROOT, plan['template'])
    need(plan['template']['path'] == motion.TEMPLATE and all(plan[k] == prior[k] for k in
         ('environment', 'import_environment', 'hf_bindings', 'reused_import_proof',
          'directory_fd_audit_policy', 'activation_bootstrap_policy', 'directory_fd_filelock_probe_source')), 'Original environment differs')
    required = {b['path'] for b in prior['repo_sources']} | {SELF, TEST, motion.SELF, motion.TEST,
        motion.TEMPLATE, SOURCE_BANK, motion.PARENTS.format(shard), 'tools/e0_teal_wan_quality_lineage.py', 'tests/test_e0_teal_wan_quality_lineage.py', 'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py', 'tests/test_e0_s2_sm120_fixed_depth_motion_metrics_v2.py'}
    required |= execution.HELPERS
    required |= {b['path'] for b in base.read_bound(ROOT, plan['parents'][0]['protocol'])['repo_sources']}
    paths = [b['path'] for b in plan['repo_sources']]
    need(len(paths) == len(set(paths)) and set(paths) == required, 'Repository closure differs')
    need(plan['source_bank']['path'] == SOURCE_BANK, 'Wrong source bank')
    bank = base.read_bound(ROOT, plan['source_bank'])
    sources = {s['native_source_ordinal']: s for s in bank['sources']}
    need(len(sources) == 12 and len(plan['videos']) == 3 and len(plan['images']) == 1, 'Source count differs')
    need([(v['ordinal'], v['mode']) for v in plan['videos']] ==
         [(o, mode) for o in SHARDS[shard] for mode in MODES], 'Source/arm order differs')
    for image in plan['images']:
        o = image['ordinal']; source = sources[o]
        case = next(v['case'] for v in plan['videos'] if v['ordinal'] == o)
        need(o in SHARDS[shard] and image['name'] == str(o) + '.png'
             and image['binding'] == case['image'] and source['image']['sha256'] == case['image']['sha256']
             and source['source_id'] == case['source_id'] and source['prompt'] == case['prompt']
             and image['scene_stratum'] == source['scene_stratum'], 'Source metadata differs')
    need({x['ordinal'] for x in plan['images']} == set(SHARDS[shard]), 'Image coverage differs')
    need(plan['output_HF_relative'] == 'ffnds-derived/e0_teal_wan_full_quality_20261029/shard{}/attempt1/worker/results'.format(shard), 'Output identity differs')
    def valid_sha(s): return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)
    b = plan.get('execution_monitor_binary'); private = plan.get('execution_private_worker_config_sha256')
    need(b is None or (set(b) == {'path','bytes','sha256'} and Path(b['path']).is_absolute()
         and type(b['bytes']) is int and b['bytes'] > 0 and valid_sha(b['sha256'])), 'Bad execution monitor')
    need(private is None or valid_sha(private), 'Bad private worker binding')
    need(b==execution.monitor_binding(base) and private==execution.PROFILE_SHA and plan.get('execution_device')=={'name_contains':'A100','cuda_capability':[8,0],'session_max_GPUs':4},'Execution environment differs')
    if frozen:
        need(plan['status'] == 'FROZEN_PENDING_REGISTRATION' and plan['binding_status'] == 'BOUND_READY_FOR_REGISTRATION'
             and plan['assets_finalized'] is True and b is not None and private is not None, 'Draft not frozen')


def authorize(plan, digest, active=None):
    validate(plan, True)
    active = base.read(ROOT / 'research/ACTIVE_RESEARCH.json') if active is None else active
    need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True, 'Continuation inactive')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    need(len(scopes) == 1 and scopes[0].get('protocol_sha256') == digest
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
         and scopes[0].get('limits') == LIMITS and scopes[0].get('resources') == RESOURCES
         and scopes[0].get('max_gpu_cards') == 1, 'ACTIVE exact registration required')
    execution.require_review(plan,scopes[0],base,lambda x:base.state_sha({k:v for k,v in x.items() if k not in ('status','assets_finalized','binding_status')}),('cpu_preflight_spec_sha256','gpu_launch_spec_sha256'))
    return scopes[0]


def verify_lineage(plan, hf):
    need(motion.check_parent(plan['parents'][0], hf, plan['shard']) == plan['videos'], 'Parent lineage differs')
    paths = [base.bound(hf, v['video']) for v in plan['videos']]
    need(len(set(paths)) == len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 3, 'Aliased clips')


def cpu_expected(plan, digest):
    return {'id': plan['id'], 'protocol_sha256': digest, 'status': CPU_STATUS,
            'sources_sha256': base.state_sha(plan['repo_sources']), 'parents': plan['parents'],
            'execution_monitor_binary': plan['execution_monitor_binary'],
            'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
            'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False, 'model_calls': 0, 'GPU_calls': 0}


def preflight(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch','numpy'} & set(sys.modules)
         and len(os.sched_getaffinity(0)) <= 2, 'Two CPU CUDA-hidden byte worker required')
    plan = base.read(args.protocol); digest = base.sha(args.protocol); scope=authorize(plan, digest)
    execution.verify_job(args,plan,scope,base,'quality','preflight',SELF,CPU_RESOURCES)
    motion.verify_execution_worker(plan); motion.verify_execution_monitor(plan)
    need(args.out.resolve().is_relative_to(args.hf_root.resolve()) and not args.out.exists(), 'Fresh confined CPU output required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'status': 'STARTED_BYTE_PREFLIGHT', 'byte_checks': []}
    base.write(args.out, report); started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU byte proof budget exceeded')))
    signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for root, b in all_bindings(plan):
            path = base.bound(ROOT if root == 'REPO' else args.hf_root, b)
            report['byte_checks'].append({'root': root, 'binding': b, 'stat': motion.stat_record(path)})
        verify_lineage(plan, args.hf_root)
        motion.quality.verify_import_proof(plan, args.hf_root)
        motion.quality.verify_environment(plan, args.hf_root)
        need(not {'torch','numpy'} & set(sys.modules), 'Numeric import in CPU preflight')
        report.update(cpu_expected(plan, digest)); report['job_spec_sha256']=scope['cpu_preflight_spec_sha256']; report['byte_checks_sha256'] = base.state_sha(report['byte_checks'])
    except BaseException as e:
        report.update(status='FAILED_CPU_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started; base.write(args.out, report)


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight'); need(isinstance(proof, dict) and {'RUN','launch_exit'} <= set(proof), 'Closed CPU proof required')
    report = base.read_bound(hf, proof['RUN']); receipt=base.read_bound(hf, proof['launch_exit']);base.exited(receipt, proof['RUN'], digest, plan['id'])
    need(receipt.get('spec_sha256')==scope['cpu_preflight_spec_sha256'] and report.get('job_spec_sha256')==scope['cpu_preflight_spec_sha256'],'Preflight spec differs')
    need(all(report.get(k) == v for k,v in cpu_expected(plan, digest).items())
         and 0 <= report.get('wall_seconds', -1) <= CPU_RESOURCES['wall_seconds']
         and report.get('byte_checks_sha256') == base.state_sha(report['byte_checks'])
         and [(x['root'], x['binding']) for x in report['byte_checks']] == all_bindings(plan), 'CPU proof differs')
    for row in report['byte_checks']:
        need(motion.stat_record((ROOT if row['root'] == 'REPO' else hf) / row['binding']['path']) == row['stat'], 'Preflight asset changed')
    return proof


def install_model_audit(torch, plan, hf, report, persist):
    """Observe concrete loaders and instantiated nn.Module classes, not opaque forwards."""
    import safetensors.torch as st
    allowed = {str((hf / b['path']).resolve()): b for b in plan['hf_bindings']}
    old_load, old_jit, old_safe, old_init = torch.load, torch.jit.load, st.load_file, torch.nn.Module.__init__
    classes = {}
    def wrap(loader, original):
        def load(path, *a, **kw):
            name = path if isinstance(path, (str, os.PathLike)) else getattr(path, 'name', None)
            if isinstance(name, (str, os.PathLike)):
                resolved = str(Path(name).resolve()); need(resolved in allowed, 'Unregistered model loader path')
                binding = allowed[resolved]; base.bound(hf, binding)
                item = {'dimension': report.get('active_dimension'), 'loader': loader, 'binding': binding}
            else:
                item = {'dimension': report.get('active_dimension'), 'loader': loader, 'binding': None,
                        'status': 'OPAQUE_IN_MEMORY_STREAM_OR_OBJECT'}
            report['actual_checkpoint_loads'].append(item); persist()
            return original(path, *a, **kw)
        return load
    def init(instance, *a, **kw):
        cls = type(instance); key = (report.get('active_dimension'), cls.__module__, cls.__qualname__)
        if key not in classes:
            try:
                path = Path(inspect.getfile(cls)).resolve()
                if path.is_file():
                    root = 'REPO' if path.is_relative_to(ROOT) else ('HF' if path.is_relative_to(hf) else 'OTHER')
                    origin = {'root': root, 'path': str(path.relative_to(ROOT if root == 'REPO' else hf)) if root != 'OTHER' else str(path),
                              'bytes': path.stat().st_size, 'sha256': base.sha(path)}
                else: origin = None
            except (OSError, TypeError): origin = None
            classes[key] = {'dimension': key[0], 'module': key[1], 'class': key[2], 'origin': origin}
            report['actual_model_classes'] = list(classes.values())
        return old_init(instance, *a, **kw)
    torch.load, torch.jit.load, st.load_file, torch.nn.Module.__init__ = wrap('torch.load', old_load), wrap('torch.jit.load', old_jit), wrap('safetensors.torch.load_file', old_safe), init
    def restore():
        torch.load, torch.jit.load, st.load_file, torch.nn.Module.__init__ = old_load, old_jit, old_safe, old_init
    return restore


def validate_result(plan, report, hf):
    need(report.get('GPU_compute_capability')==[8,0] and 'A100' in report.get('GPU_name',''),'Actual SM80 A100 identity differs')
    need(report.get('id') == plan['id'] and report.get('status') == STATUS
         and report.get('predebited') == report.get('invoked') == report.get('completed') == LIMITS,
         'Completed operation ledger differs')
    need(report.get('predebited_outer_model_work') == COSTS, 'Frame/model-work budget differs')
    dims = report.get('completed_dimensions', [])
    need(report.get('source_summary') == summarize(plan, hf, dims), 'Source aggregation differs')
    need([d['dimension'] for d in dims] == list(DIMENSIONS), 'Eight ordered calls required')
    need(len(report.get('official_module_functions', [])) == 8, 'Official function provenance incomplete')
    obs = report.get('RAFT_observations', [])
    paths = [str((hf / v['video']['path']).resolve()) for v in plan['videos']]
    need(len(obs) == 3 and {o['video_path'] for o in obs} == set(paths), 'RAFT coverage differs')
    for o in obs: base.validate_raft(o)
    dynamic = next(d for d in dims if d['dimension'] == 'dynamic_degree')['result'][1]
    need({r['video_path']:r['video_results'] for r in dynamic} == {r['video_path']:r['official_binary'] for r in obs}, 'RAFT result differs')
    allowed = {(b['path'], b['bytes'], b['sha256']) for b in plan['hf_bindings']}
    loads = report.get('actual_checkpoint_loads', [])
    need(loads and all(x.get('binding') is None or tuple(x['binding'][k] for k in ('path','bytes','sha256')) in allowed for x in loads), 'Actual loader path not bound')
    raft_weight = next(b for b in plan['hf_bindings'] if b['path'].endswith(motion.WEIGHT_SUFFIX))
    need(raft_weight['sha256'] == motion.WEIGHT_SHA and any(x.get('dimension') == 'dynamic_degree' and x.get('binding') == raft_weight for x in loads), 'Actual RAFT checkpoint load missing')
    raft_class = next(b for b in plan['hf_bindings'] if b['path'].endswith('/vbench/third_party/RAFT/core/raft.py'))
    need(report.get('actual_RAFT_class') == raft_class, 'Actual RAFT class source differs')
    classes = report.get('actual_model_classes', [])
    need(classes and all(c.get('dimension') in DIMENSIONS for c in classes), 'Actual class audit empty or unscoped')
    for c in classes:
        origin = c.get('origin')
        if origin is not None:
            root = origin['root']
            path = (ROOT / origin['path'] if root == 'REPO' else hf / origin['path'] if root == 'HF' else Path(origin['path']))
            need(path.is_file() and path.stat().st_size == origin['bytes'] and base.sha(path) == origin['sha256'], 'Class source changed')
    need(not report.get('native_dir_fd_audit', {}).get('denied_attempts') and
         not report.get('native_metric_import_audit', {}).get('denied_attempts'), 'Audit violation')
    need(not report.get('assigned_GPU_resources', {}).get('violations') and
         report.get('assigned_GPU_resources', {}).get('sampled_isolation_pass') is True, 'GPU monitor evidence absent')


def _run_original(args):
    base.precision_environment()
    need(not {'torch','numpy'} & set(sys.modules), 'Fresh isolated worker required')
    hf = args.hf_root.resolve(); plan = base.read(args.protocol); digest = base.sha(args.protocol)
    scope = authorize(plan, digest); execution.verify_job(args,plan,scope,base,'quality','run',SELF,RESOURCES); cpu = require_cpu(scope, plan, digest, hf)
    motion.verify_execution_worker(plan); motion.verify_execution_monitor(plan)
    out = (hf / plan['output_HF_relative']).resolve()
    need(out.is_relative_to(hf) and out != hf and not out.exists(), 'Fresh confined attempt required')
    out.mkdir(parents=True); base.write(out / 'ATTEMPT.json', {'id':plan['id'], 'protocol_sha256':digest, 'automatic_retry':False})
    report = {'id':plan['id'], 'protocol_sha256':digest, 'status':'STARTED_SM120_EIGHT_AXIS',
              'prior_CPU_preflight':cpu, 'job_spec_sha256':scope['gpu_launch_spec_sha256'], 'parents':plan['parents'], 'automatic_retry':False,
              'declared_outer_model_work_budgets':COSTS, 'predebited_outer_model_work':{},
              'model_work_accounting':'Frame budgets predebited; RAFT forward/iterations and dynamic decode observed; other internal forwards/construction counts UNKNOWN',
              'checkpoint_observation_scope':'Wrapped torch.load, torch.jit.load, and safetensors.torch.load_file lookups after metric imports; pre-import aliases including PEFT safe_load_file, opaque streams, safe_open and other native/internal loaders UNKNOWN',
              'actual_checkpoint_loads':[], 'actual_model_classes':[], 'official_module_functions':[],
              'completed_dimensions':[], 'RAFT_observations':[], 'native_dir_fd_audit':{'denied_attempts':[]},
              'temporal_flickering':'NOT_RUN_NINTH_AXIS', 'official_five_independent_samples':False,
              'quality_accepted':False}
    def persist():
        base.write(out / 'RUN.json', report)
        need(output_bytes(out) <= RESOURCES['new_HF_bytes'], 'Output byte budget exceeded')
    ledger = base.Ledger(report, limits=LIMITS, persist=persist)
    started = time.monotonic(); monitor = None; restore_audit = None
    def check():
        authorize(plan, digest)
        if monitor: monitor.check()
        need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Wall budget exceeded')
        need(shutil.disk_usage(out).free >= RESOURCES['free_disk_bytes'], 'Free disk budget exceeded')
        need(output_bytes(out) <= RESOURCES['new_HF_bytes'], 'Output byte budget exceeded')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed quality wall budget exhausted')))
    signal.alarm(RESOURCES['wall_seconds'])
    try:
        verify_lineage(plan, hf); motion.quality.verify_import_proof(plan, hf)
        helper, v8 = motion.quality.verify_environment(plan, hf)
        need(len(os.sched_getaffinity(0)) <= 2, 'Two CPU affinity required')
        base.scope_cache_environment(out)
        os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        report['isolation'] = base.configure_import_environment(plan, hf)
        helper.disable_local_ipv6_probe(hf / plan['environment']['site'], v8)
        numpy = importlib.import_module('numpy'); report['numpy'] = base.module_observation(numpy, 'numpy', plan, hf)
        torch = importlib.import_module('torch'); report['torch'] = base.module_observation(torch, 'torch', plan, hf)
        from startup_gpu_reservation import reserve_before_assets
        from assigned_gpu_resources import AssignedGPUResources
        binary = motion.verify_execution_monitor(plan)
        _, monitor, receipt = reserve_before_assets(torch); report['reservation'] = receipt
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0)==(8,0), 'Assigned A100 SM80 required')
        report['GPU_name'] = torch.cuda.get_device_name(0); report['GPU_compute_capability'] = list(torch.cuda.get_device_capability(0));report['numeric_policy']={'TF32':False,'model_tensor_dtype':'MODEL_INTERNAL_DTYPE_NOT_FULLY_OBSERVED'}
        snapshot = out / 'EXECUTED_SOURCES'; snapshot.mkdir()
        for item in plan['repo_sources']:
            target = snapshot / item['path']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / item['path'], target)
        shutil.copyfile(args.protocol, snapshot / 'protocol.json')
        common = importlib.import_module('e0_ub1_consumer_official_quality')
        source = (hf / plan['environment']['source_root']).resolve(); sys.path.insert(0, str(source))
        from offline_i2v_metric_loaders import submodules
        assets = base.read(ROOT / plan['environment']['asset_manifest']); modelbase = hf / assets['HF_relative_root']; device = torch.device('cuda:0')
        oldhub = torch.hub.load
        def no_network(*_a, **_kw): raise RuntimeError('Implicit network access forbidden')
        torch.hub.load = common.make_local_hub_loader(oldhub, modelbase / 'source/dino', {'facebookresearch/dino:main',str(modelbase / 'source/dino')})
        def state_load(url, *_a, **kw):
            need(url == 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', 'Unexpected weight URL')
            return torch.load(modelbase / 'dino_model/dino_vitbase16_pretrain.pth', map_location=kw.get('map_location'), weights_only=kw.get('weights_only',False))
        torch.hub.load_state_dict_from_url = state_load; torch.hub.download_url_to_file = no_network
        import socket
        socket.socket.connect = no_network; socket.create_connection = no_network
        import subprocess
        imports = {'denied_attempts':[]}; _, _, checked_font = helper.import_audit_guard(out, imports)
        subprocess.check_output = checked_font(subprocess.check_output, hf / plan['environment']['site'] / 'matplotlib/font_manager.py')
        modules = {}
        for d in DIMENSIONS:
            modules[d], observation = motion.v4.metric_module(d, plan, hf)
            report['official_module_functions'].append(observation)
        report['actual_module_origins'] = base.actual_import_origins(plan, hf)
        report['actual_distribution_versions'] = helper.effective_distributions(base.read(ROOT / v8['lock_path']), hf / plan['environment']['overlay'])
        report['native_metric_import_audit'] = imports; need(not imports['denied_attempts'], 'Import audit violation')
        raft_class = Path(inspect.getfile(modules['dynamic_degree'].RAFT)).resolve()
        expected_raft = (source / 'vbench/third_party/RAFT/core/raft.py').resolve()
        need(raft_class == expected_raft, 'Actual RAFT class differs')
        report['actual_RAFT_class'] = next(b for b in plan['hf_bindings'] if (hf / b['path']).resolve() == expected_raft)
        def denied(reason):
            def fail(): report['native_dir_fd_audit']['denied_attempts'].append(reason); raise RuntimeError(reason)
            return fail
        audit = common.audit_scope(out, AssignedGPUResources.query.__code__, binary, os.environ['FFNDS_PHYSICAL_GPU'])
        sys.addaudithook(motion.v4.v3.normalize_dir_fd_audit(audit, out, report['native_dir_fd_audit'], denied))
        restore_audit = install_model_audit(torch, plan, hf, report, persist)
        work = out / 'work'; images = work / 'vbench2_beta_i2v/data/crop/16-9'; images.mkdir(parents=True)
        for img in plan['images']:
            (images / img['name']).symlink_to(base.bound(hf, img['binding']))
            need((images / img['name']).resolve() == base.bound(hf, img['binding']), 'I2V source symlink differs')
        os.chdir(work)
        expected = [str(base.bound(hf, v['video'])) for v in plan['videos']]
        for dimension in DIMENSIONS:
            check(); inp = out / (dimension + '_INPUT.json')
            base.write(inp, [{'prompt_en':v['case']['prompt'], 'dimension':[dimension],
                'image_name':str(v['ordinal']) + '.png', 'video_list':[path]} for v,path in zip(plan['videos'],expected)])
            report['active_dimension'] = dimension; need(dimension not in report['predebited_outer_model_work'], 'Duplicate dimension')
            report['predebited_outer_model_work'][dimension] = COSTS[dimension].copy(); persist()
            module = modules[dimension]
            need(Path(module.__file__).resolve() == source / ('vbench2_beta_i2v' if dimension.startswith('i2v_') else 'vbench') / (dimension + '.py'), 'Official module origin differs')
            if dimension == 'i2v_background':
                original_dream = module.dreamsim
                def dream(*a, **kw): kw.update(cache_dir=str(modelbase/'dreamsim'), device=device); return original_dream(*a, **kw)
                module.dreamsim = dream
            original_capture = None
            try:
                with ledger('official_dimension_calls'), ledger('video_dimension_results', 3):
                    if dimension == 'dynamic_degree':
                        original_capture = motion.install_decoder_counter(module, ledger)
                        with base.raft_observer(module, report['RAFT_observations'], ledger, expected), torch.inference_mode():
                            result = module.compute_dynamic_degree(str(inp), device, submodules(dimension, assets, hf))
                    elif dimension == 'i2v_background':
                        from peft import PeftModel, LoraConfig
                        adapter = modelbase / 'dreamsim/ensemble_lora'
                        adapter_sha = next(x['sha256'] for x in plan['hf_bindings'] if x['path'].endswith('/dreamsim/ensemble_lora/adapter_config.json'))
                        with common.dreamsim_adapter_compat(PeftModel, LoraConfig, adapter, adapter_sha), torch.inference_mode():
                            result = module.compute_i2v_background(str(inp), device, submodules(dimension, assets, hf))
                    else:
                        with torch.inference_mode(): result = getattr(module, 'compute_'+dimension)(str(inp), device, submodules(dimension, assets, hf))
                    torch.cuda.synchronize(); result = common.plain(result); base.check_result(result, expected, dimension)
                    report['completed_dimensions'].append({'dimension':dimension, 'result':result, 'input_sha256':base.sha(inp)})
                gc.collect(); torch.cuda.empty_cache()
                need(torch.cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_bytes'], 'GPU reservation budget exceeded')
            finally:
                if original_capture is not None: module.cv2.VideoCapture = original_capture
                if dimension == 'i2v_background': module.dreamsim = original_dream
        ledger.finish(); check(); report['source_summary'] = summarize(plan, hf, report['completed_dimensions'])
        report.update(active_dimension=None, status=STATUS, peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        report.update(status='FAILED_SM120_EIGHT_AXIS_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally:
        original_error = sys.exc_info()[1]; signal.alarm(0)
        if restore_audit: restore_audit()
        report['wall_seconds'] = time.monotonic()-started
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                need(not report['assigned_GPU_resources']['violations'], 'GPU monitor violation')
            except BaseException as e:
                report.update(status='FAILED_GPU_MONITOR_PRESERVED_NO_RETRY', monitor_error=str(e)); base.write(out/'RUN.json', report)
                if original_error is None: raise
        base.write(out/'RUN.json', report)
    try: validate_result(plan, report, hf)
    except BaseException as e:
        report.update(status='FAILED_RESULT_VERIFICATION_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e)); base.write(out/'RUN.json', report); raise


_scope_budget = None

def output_bytes(out):
    """Only this worker's target output is optimized; no Path/global patch."""
    global _scope_budget
    if _scope_budget is None or Path(out).resolve()!=_scope_budget.out:
        return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and not p.is_symlink())
    if _scope_budget.initial is None:
        if not (_scope_budget.snapshot/'protocol.json').is_file():
            return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and not p.is_symlink())
        _scope_budget.initialize()
    return _scope_budget.total()['total_bytes']


def run(args):
    from e0_s2_sm120_fixed_depth_motion_metrics_v2 import SnapshotBudget
    global _scope_budget
    need(_scope_budget is None,'Nested budget scope forbidden')
    plan=base.read(args.protocol);out=(args.hf_root/plan['output_HF_relative']).resolve()
    budget=SnapshotBudget(out,plan,args.protocol,RESOURCES['new_HF_bytes']);_scope_budget=budget;completed=False
    try:
        _run_original(args);completed=True;final=budget.final();path=out/'RUN.json';report=base.read(path)
        report.update(snapshot_audit=budget.initial,output_budget_audit=final);base.write(path,report);budget.final()
    except BaseException as error:
        path=out/'RUN.json'
        if path.is_file():
            report=base.read(path)
            if completed:report['status']='FAILED_OUTPUT_AUDIT_PRESERVED_NO_RETRY'
            report.update(output_audit_error=type(error).__name__);base.write(path,report)
        raise
    finally:_scope_budget=None


def verify_closed(args):
    plan = base.read(args.protocol); validate(plan, True); digest = base.sha(args.protocol); hf = args.hf_root.resolve()
    need(args.run.resolve() == (hf / plan['output_HF_relative'] / 'RUN.json').resolve()
         and args.exit.resolve() == (hf / plan['output_HF_relative']).parent.joinpath('EXIT_RECEIPT.json').resolve(), 'Closed receipt paths differ')
    binding = base.binding(args.run, hf); report = base.read(args.run)
    receipt=base.read(args.exit);base.exited(receipt, binding, digest, plan['id'])
    need(receipt.get('spec_sha256')==report.get('job_spec_sha256'),'GPU spec receipt differs')
    need(report.get('protocol_sha256') == digest and 0 <= report.get('wall_seconds', -1) <= RESOURCES['wall_seconds'], 'Closed protocol/wall differs')
    from e0_s2_sm120_fixed_depth_motion_metrics_v2 import SnapshotBudget
    budget=SnapshotBudget(hf / plan['output_HF_relative'],plan,args.protocol,RESOURCES['new_HF_bytes'])
    need(budget.initialize()==report['snapshot_audit'], 'Snapshot audit differs')
    now=budget.final();prior=report['output_budget_audit']
    need(prior['snapshot_bytes']==now['snapshot_bytes'] and prior['limit_bytes']==now['limit_bytes'] and now['total_bytes']<=now['limit_bytes'], 'Output budget differs')
    require_cpu({'cpu_preflight':report.get('prior_CPU_preflight'), 'cpu_preflight_spec_sha256':base.read_bound(hf,report['prior_CPU_preflight']['RUN'])['job_spec_sha256']}, plan, digest, hf)
    verify_lineage(plan, hf); validate_result(plan, report, hf)
    out = hf / plan['output_HF_relative']
    for img in plan['images']:
        link = out / 'work/vbench2_beta_i2v/data/crop/16-9' / img['name']
        need(link.is_symlink() and link.resolve() == base.bound(hf, img['binding']), 'Closed I2V image binding differs')
    for entry in report['completed_dimensions']:
        dimension = entry['dimension']; inp = out / (dimension + '_INPUT.json')
        expected = [{'prompt_en':v['case']['prompt'], 'dimension':[dimension],
            'image_name':str(v['ordinal']) + '.png', 'video_list':[str((hf / v['video']['path']).resolve())]} for v in plan['videos']]
        need(base.sha(inp) == entry['input_sha256'] and base.read(inp) == expected, 'Metric input identity differs')
    return {'id':plan['id'], 'status':'PASS_CLOSED_EIGHT_AXIS_SHARD', 'RUN':binding,
            'exit':base.binding(args.exit, hf), 'model_calls':0, 'GPU_calls':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft','inspect','preflight','run','verify-closed'))
    parser.add_argument('--shard', type=int, choices=range(4)); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--job-spec',type=Path); parser.add_argument('--out', type=Path); parser.add_argument('--run', type=Path); parser.add_argument('--exit', type=Path)
    args = parser.parse_args()
    if args.action == 'draft':
        need(args.shard is not None and args.out is not None and not args.out.exists(), 'Fresh draft and shard required')
        args.out.parent.mkdir(parents=True, exist_ok=True); base.write(args.out, draft(args.hf_root, args.shard))
    elif args.action == 'inspect':
        plan = base.read(args.protocol); validate(plan); print(json.dumps({'id':plan['id'], 'limits':LIMITS, 'resources':RESOURCES}, indent=2))
    elif args.action == 'preflight': preflight(args)
    elif args.action == 'run': run(args)
    else: print(json.dumps(verify_closed(args), indent=2))

if __name__ == '__main__': main()
