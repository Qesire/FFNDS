"""Complete only the unfinished calibration sources of each failed V2 part.

The old development source is replayed first as the same four-step byte neutral.
The capture, original FFNs, runner, observer, I/O and single-file guards are the
frozen V2 functions. Completed V2 prefixes remain separately audited assets.
"""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))


def private_v2(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One unchanged namespace verifies the completed/failed parent identity. The
# other private namespace supplies execution classes with this smaller ledger.
parent_v2 = private_v2('_ffnds_full_depth_v3_parent_proof')
execution_v2 = private_v2('_ffnds_full_depth_v3_execution')
legacy, native = execution_v2.legacy, execution_v2.native
FROZEN_V2_SHA = 'e9bc695f5db69f900869c0efaaa47a815f8adab4ef247147080a820a8abbe075'
RESOURCES = copy.deepcopy(parent_v2.RESOURCES)
RESOURCES.update(wall_seconds=1800, outer_seconds=2100)
REPAIR = copy.deepcopy(parent_v2.REPAIR)
COHORT = legacy.COHORT
CASE_INDICES = {0: [0, 2, 3], 1: [0, 3]}
COMPLETION = {'version': 3, 'case_selection': 'EXACT_PARENT_CASE_INDICES_PART0_0_2_3_PART1_0_3_NO_RESAMPLING',
    'remaining_calibration_native_ordinals_by_part': {'0': [896, 240], '1': [304]},
    'sources_per_part': {'0': 3, '1': 2},
    'old_development_replays_per_part': 1, 'new_calibration_videos_per_part': {'0': 2, '1': 1},
    'full_model_flows_per_part': {'0': 12, '1': 8}, 'raw_FFN_sites_per_part': {'0': 480, '1': 320},
    'frozen_numeric_observer_and_IO': 'IDENTICAL_V2', 'completed_V2_prefixes': 'REUSED_ONLY_AFTER_SEPARATE_BYTE_AUDIT',
    'cohort_claim': 'UNIQUE_SOURCES_REQUIRE_PARENT_PREFIX_AUDIT_AND_BOTH_ACTUAL_V3_EXIT0', 'automatic_retry': False}
EXTRA_SOURCES = {'tools/e0_original_bf16_full_depth_collect_v3.py', 'tests/test_e0_original_bf16_full_depth_collect_v3.py'}
run_id = legacy.run_id
WrittenBytes = execution_v2.WrittenBytes
counted_io = execution_v2.counted_io
single_file_bound = execution_v2.single_file_bound
execution_worker_sha = execution_v2.execution_worker_sha


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two V3 completion parts')
    return legacy.BASE_ID + '_PART' + str(part) + '_V3'


def output_relative(part):
    return f'ffnds-derived/e0_original_bf16_full_depth_collect_v3_20261029/part{part}/attempt1/worker/results'


def limits(part):
    native.need(type(part) is int and part in CASE_INDICES, 'Exact asymmetric completion part required')
    count = len(CASE_INDICES[part])
    result = dict(parent_v2.limits())
    # Scale actual per-source work from V2's4 sources to this part's3/2.
    # Setup still loads both complete experts/T5; dev byte checks stay10.
    setup = {'original_expert_loads', 'raw_Default_MM_loads', 'buffer_Default_MM_loads',
        'T5_raw_Default_MM_loads', 'T5_buffer_Default_MM_loads', 'native_prefix_replays_no_decode', 'neutral_state_byte_checks'}
    thirds = {'video_generations', 'VAE_decodes', 'core_CPU_reports', 'video_geometry_checks', 'source_registration_writes'}
    for key, value in result.items():
        if key in setup:
            continue
        if key in thirds:
            native.need(value == 3, 'Frozen V2 three-calibration counter differs')
            result[key] = count-1
        else:
            native.need(value % 4 == 0, 'Frozen V2 per-source counter not divisible by4')
            result[key] = value // 4 * count
    return result


# Change only this private execution namespace, never the frozen/public modules.
CURRENT_PART = None
def execution_limits():
    return limits(CURRENT_PART)
legacy.limits = execution_limits


def parent_protocol(plan):
    path = native.bound(ROOT, plan['failed_predecessor']['protocol'])
    protocol = native.read(path)
    parent_v2.validate(protocol, True)
    native.need(protocol['part'] == plan['part'] and protocol['id'] == parent_v2.scope_id(plan['part']), 'Exact V2 part parent differs')
    return protocol


def validate(plan, frozen=False):
    native.need(native.sha(ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py') == FROZEN_V2_SHA,
        'Frozen V2 execution source changed')
    native.need(plan.get('id') == scope_id(plan.get('part')) and plan.get('numeric_path') == 'BF16_ORIGINAL'
        and plan.get('limits') == limits(plan['part']) and plan.get('resources') == RESOURCES
        and plan.get('repair') == REPAIR and plan.get('completion_scope') == COMPLETION
        and plan.get('automatic_retry') is False and plan.get('output_HF_relative') == output_relative(plan['part']),
        'Fixed V3 completion scope differs')
    old = parent_protocol(plan)
    native.need(plan['cases'] == [old['cases'][i] for i in CASE_INDICES[plan['part']]]
        and [c['role'] for c in plan['cases']] == ['development']+['calibration']*(len(CASE_INDICES[plan['part']])-1)
        and [c['native_source_ordinal'] for c in plan['cases'][1:]] == COMPLETION['remaining_calibration_native_ordinals_by_part'][str(plan['part'])]
        and plan['run_ids'] == [run_id(c) for c in plan['cases']], 'V3 source/rows must be exact old dev plus final calibration')
    inherited = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
        'private_worker_config_sha256', 'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof',
        'runtime_overlay', 'parent', 'raw_master_CPU_proof', 'source_cohort', 'tensor_contract', 'row_draw_identity')
    native.need(all(plan[k] == old[k] for k in inherited), 'V3 native environment/model/source contract changed')
    paths = [b['path'] for b in plan['repo_sources']]
    native.need(EXTRA_SOURCES | {b['path'] for b in old['repo_sources']} <= set(paths)
        and len(paths) == len(set(paths)) and all(b['bytes'] <= REPAIR['maximum_single_video_file_bytes'] for b in plan['repo_sources']),
        'V3 frozen closure/single-file bound differs')
    if frozen:
        previous = plan['failed_predecessor']
        native.need(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION'
            and all(isinstance(previous.get(k), dict) for k in ('RUN', 'launch_exit'))
            and isinstance(previous.get('completed_prefix_generations'), list) and len(previous['completed_prefix_generations']) == 2,
            'Actual V2 terminal/prefix receipts NOT_BOUND')


def verify_predecessor(plan, hf):
    old = parent_protocol(plan); previous = plan['failed_predecessor']
    run = native.read(native.bound(hf, previous['RUN']))
    exit_record = native.read(native.bound(hf, previous['launch_exit']))
    native.need(run.get('id') == old['id'] and run.get('protocol_sha256') == previous['protocol']['sha256']
        and run.get('status') == 'FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY'
        and exit_record.get('id') == old['id'] and exit_record.get('protocol_sha256') == previous['protocol']['sha256']
        and exit_record.get('run_sha256') == previous['RUN']['sha256'] and exit_record.get('process_exited') is True
        and type(exit_record.get('child_returncode')) is int and exit_record['child_returncode'] != 0
        and exit_record.get('automatic_retry') is False, 'Actual failed V2 RUN/exit required; no completed retry')
    ledger = run['operation_ledger']
    native.need(not ledger['forbidden_attempts'] and all(0 <= ledger['observed'].get(k, 0)
        <= ledger['predebited'].get(k, 0) <= n for k, n in old['limits'].items()), 'Failed V2 actual ledger invalid')
    refs = previous['completed_prefix_generations']
    expected = []
    for case in old['cases'][:2]:
        matches = [p for p in run['phases'] if p.get('native_source_ordinal') == case['native_source_ordinal']]
        native.need(len(matches) == 1 and isinstance(matches[0].get('generation_receipt'), dict), 'V2 complete prefix generation missing')
        phase = matches[0]
        expected.append({'native_source_ordinal': case['native_source_ordinal'], 'role': case['role'], 'GENERATION': phase['generation_receipt']})
    native.need(refs == expected, 'Exact V2 completed-prefix receipt order differs')
    for case, item in zip(old['cases'][:2], refs):
        generation = native.read(native.bound(hf, item['GENERATION']))
        native.need(generation.get('id') == old['id'] and generation.get('protocol_sha256') == previous['protocol']['sha256']
            and generation.get('status') == 'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160'
            and generation.get('numeric_path') == 'BF16_ORIGINAL' and generation.get('source_state') == 'teacher_input'
            and generation.get('native_source_ordinal') == case['native_source_ordinal']
            and generation.get('source_id') == case['source_id'] and generation.get('role') == case['role']
            and generation.get('completed_native_flow_steps') == 4 and generation.get('native_ffn_sites') == 160
            and {(p['step'], p['block']) for p in generation['packets']} == {(s, b) for s in range(4) for b in range(40)}
            and len(generation['packets']) == 160 and len(generation['states']) == 10, 'V2 completed prefix metadata incomplete')
    remaining = {c['native_source_ordinal'] for c in plan['cases'][1:]}
    native.need(not any(p.get('native_source_ordinal') in remaining and 'generation_receipt' in p for p in run['phases']),
        'V2 selected calibration already complete; do not recapture')
    return previous


def verify_lineage(plan, hf):
    old = parent_protocol(plan)
    parent_v2.verify_lineage(old, hf)
    verify_predecessor(plan, hf)
    return old


def cpu_identity(plan, digest):
    return {'id': plan['id'], 'protocol_sha256': digest,
        'status': 'PASS_V3_FINAL_CAL_AND_DEV_ROWS_V2_FAILURE_PREFIX_CURRENT_RAW_STAT_NO_NUMERIC_IMPORTS',
        'part': plan['part'], 'sources_sha256': native.canonical(plan['repo_sources']),
        'source_cohort_sha256': plan['source_cohort']['sha256'], 'raw_master_CPU_proof': plan['raw_master_CPU_proof'],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'],
        'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
        'failed_predecessor': plan['failed_predecessor'], 'repair': REPAIR, 'completion_scope': COMPLETION,
        'controls': 10, 'sources': len(plan['cases']), 'native_row_sites_planned': 160*len(plan['cases']), 'all_CUDA_devices_hidden': True,
        'Torch_or_NumPy_imported': False, 'model_calls': 0, 'GPU_calls': 0, 'raw_master_payload_bytes_rehashed': 0}


def preflight(args):
    started = time.monotonic()
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules), 'Fresh hidden-CUDA stdlib proof required')
    plan = native.read(args.protocol); validate(plan, True)
    native.need(len(os.sched_getaffinity(0)) <= 4, 'Preflight affinity exceeds4')
    verify_lineage(plan, args.hf_root)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    for case in plan['cases']:
        native.bound(args.hf_root, case['image'])
    for item in plan['cases'][0]['controls'].values():
        native.bound(args.hf_root, item)
    native.model_metadata(plan)
    native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    result = cpu_identity(plan, native.sha(args.protocol)); result['wall_seconds'] = time.monotonic()-started
    native.need(result['wall_seconds'] <= 300 and not {'torch', 'numpy'} & set(sys.modules), 'CPU proof wall/import gate differs')
    return result


def authorize(plan, digest):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    policy = active.get('resource_scheduling_policy_20261028', {})
    native.need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
        and active.get('current_session_resource_constraint_20261028', {}).get('max_simultaneously_assigned_idle_GPUs') == 2,
        'Current repository/session authority differs')
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(matches) == 1 and matches[0].get('protocol_sha256') == digest
        and matches[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and matches[0].get('limits') == limits(plan['part']) and matches[0].get('resources') == RESOURCES
        and matches[0].get('run_ids') == plan['run_ids'] and matches[0].get('max_gpu_cards') == 1,
        'Unique independent V3 registration required')
    return matches[0]


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight')
    native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Independent V3 part proof NOT_BOUND')
    run = native.read(native.bound(hf, proof['RUN']))
    native.need(all(run.get(k) == v for k, v in cpu_identity(plan, digest).items()) and 0 <= run.get('wall_seconds', -1) <= 300,
        'Independent V3 byte proof differs')
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, plan['id'])
    return proof


def registration(plan,hf):
    from ffnds_cohort_preflight import audit
    registered_sources,samples=[],[]
    for c in plan['cases'][1:]:
        rid=run_id(c)
        registered_sources.append({'source_id':c['source_id'],'image_id':c['image_sha256'],'prompt_id':c['prompt_sha256'],'prompt':c['prompt'],
          'image_path':str((hf/c['image']['path']).resolve()),'image_name':Path(c['image']['path']).name,'conditioning_id':c['conditioning_id'],
          'role':'CALIBRATION','exposure_history':{'fitting':False,'method_selection':True,'prior_evaluation':True}})
        samples.append({'source_id':c['source_id'],'variant':'dense','run_id':rid,'asset_id':'video:'+rid,'seed':42,'sample_index':0,
          'conditioning_id':c['conditioning_id'],'initial_noise_id':plan['id']+':'+c['source_id'],
          'scheduler_id':'Wan22StepDistillScheduler:shift5:sigma1,.9375,.8333333134651184,.625:high,high,low,low',
          **{k:native.VIDEO[k] for k in ('frames','fps','width','height')},'status':'PLANNED'})
    result={'schema':'ffnds.cohort.v1','mode':'CUSTOM_PAIRED','experiment':plan['id'],'phase':'REGISTERED_REAL_GENERATION_PLAN',
      'sources':registered_sources,'samples':samples,'calibration':[],'required_variants':['dense']}
    checked=audit(result);native.need(not checked['errors'] and checked['summary']['independent_image_clusters']==len(plan['cases'])-1,'Raw calibration video registration invalid')
    return result

def run(args):
    global CURRENT_PART
    started = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    CURRENT_PART = plan['part']
    scope = authorize(plan, digest); cpu = require_cpu(scope, plan, digest, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numerical worker required')
    verify_lineage(plan, args.hf_root)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    native.model_metadata(plan)
    local, rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Actual package versions differ')
    native.need(execution_worker_sha() == plan['execution_private_worker_config_sha256'], 'Private execution identity differs')
    native.need(len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1' for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'CPU affinity/TF32 override differs')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined V3 output required; no retry')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'numeric_path': 'BF16_ORIGINAL', 'part': plan['part'],
        'status': 'STARTED_RAW_FULL_DEPTH_PART_V3_NO_RETRY', 'automatic_retry': False, 'CPU_preflight': cpu, 'phases': [],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'], 'failed_predecessor': plan['failed_predecessor'],
        'repair': REPAIR, 'completion_scope': COMPLETION, 'source_cohort': plan['source_cohort'], 'native_part_image_clusters': len(plan['cases']),
        'new_calibration_image_clusters': len(plan['cases'])-1, 'old_development_image_clusters': 1, 'complete_cohort_claimed': False,
        'cohort_completion': 'PART_COMPLETION_REQUIRES_SEPARATE_V2_PREFIX_AUDIT_AND_BOTH_V3_EXIT0',
        'state_identity': 'Teacher states only; no student states captured', 'source_history': 'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT',
        'official_quality': 'NOT_RUN', 'training_updates': 0, 'decompositions': 0, 'quality_or_speed_acceptance': False,
        'capture_runtime_identity': 'Native generation plus row capture/D2H/serialization/CPU checks; not a throughput measurement'}
    monitor = None
    counter = WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    # Reserve all opaque native video writers before the first model call.
    for case in plan['cases'][1:]:
        counter.hold(str(counter.path(out/run_id(case)/'output.mp4')), REPAIR['maximum_single_video_file_bytes'])
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = legacy.Ledger(report.setdefault('operation_ledger', {}), plan['run_ids'], lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    persist()

    def check():
        authorize(plan, digest)
        if monitor:
            monitor.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Host/disk resource reserve exceeded')
        counter.check()
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Full-depth completion wall1800 exhausted')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU reserved32GiB exceeded')

    def timeout(*_):
        raise TimeoutError('Fixed V3 full-depth wall1800 exhausted; no retry')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, rows), native.NoHiF4()
    file_guard = single_file_bound()
    try:
        file_guard.__enter__()
        check(); snapshot = out/'EXECUTED_SOURCES'; snapshot.mkdir()
        copies = [(ROOT/item['path'], snapshot/item['path'], item) for item in plan['repo_sources']]
        copies += [(args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)),
            (ROOT/plan['source_cohort']['path'], snapshot/'source_cohort.json', plan['source_cohort'])]
        counter.hold('frozen_metadata', sum(item['bytes'] for _, _, item in copies))
        for source, destination, item in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            counter.copy_bound(source, destination, item)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Frozen snapshot byte reserve not exhausted exactly')
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, report['CUDA_reservation'] = reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'Single assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'matmul_TF32': False, 'float32_matmul_precision': 'highest', 'bf16_reduced_precision_reduction': True}
        with counted_io(counter), native.OriginalGuard(torch, plan, ledger) as guard:
            begun = time.perf_counter(); pipe = native.build_pipeline(plan); setup_seconds = time.perf_counter()-begun
            report.update(raw_model_setup_seconds=setup_seconds, original_loader_paths=guard.loaded,
                BF16_matrix_residency='Original raw CPU masters and four CUDA block buffers; original offload including W2 unchanged')
            registered = registration(plan, args.hf_root)
            native.write(snapshot/'registered_video_plan.json', registered); persist(); check()
            for index, case in enumerate(plan['cases']):
                native.need(index == 0 or report.get('local_development_neutral_passed') is True, 'Calibration before local neutral')
                phase = legacy.native_phase(torch, pipe, guard, plan, case, out, args.hf_root, digest, report, persist, check)
                if index == 0:
                    native.need(phase['all_four_original_flow_and_post_bytes_passed'] is True, 'Local neutral did not pass')
                    report['local_development_neutral_passed'] = True; persist(); check(); continue
                from ffnds_video_metrics import evaluate, validate as validate_core
                import cv2
                cv2.setNumThreads(4)
                identity = {'source_id': case['image_sha256'], 'seed': 42, 'expected_frames': 81,
                    'input_prompt_digest': case['prompt_sha256'], 'model_digest': native.canonical(plan['models']),
                    'protocol_sha256': digest, 'run_sha256': phase['generation_receipt']['sha256']}
                cost = {'generation_wall_seconds': phase['source_capture_wall_seconds'], 'ffn_boundary_seconds': phase['FFN_boundary_CUDA_seconds'],
                    'setup_seconds': setup_seconds, 'peak_gpu_memory_mib': phase['peak_cuda_allocated_bytes']/(1 << 20)}
                core = ledger.call(run_id(case), {'core_CPU_reports': 1}, evaluate, Path(phase['video']['path']), None, {'identity': identity, 'cost': cost})
                native.need(not validate_core(core), 'New81frame CORE incomplete')
                core_path = out/run_id(case)/'CORE_METRICS.json'; native.write(core_path, core)
                phase['core_metrics'] = native.binding(core_path, args.hf_root)

                def save_registration():
                    folder = out/f'job{index-1:06d}'; folder.mkdir()
                    native.write(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
                        'job': registered['samples'][index-1], 'source': registered['sources'][index-1], 'native_ffn_sites': 160,
                        'scheduler_generator_reset': True, 'native_ffn_counters': phase['native_counter_delta'],
                        'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL'})
                ledger.call(run_id(case), {'source_registration_writes': 1}, save_registration); persist(); check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); native.write(out/'VERIFIED_GENERATIONS.json', verified)
            report.update(status='COMPLETE_RAW_BF16_FULL_DEPTH_V3_REMAINING_CAL_WITH_DEV_NEUTRAL',
                verified_cohort=native.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root), checked_transformers_imports=finder.origins,
                captured_native_sites=sum(len(p['packets']) for p in report['phases']), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            native.need(report['captured_native_sites'] == 160*len(plan['cases']) and not counter.holds, 'Incomplete part sites/output reservations')
            check()
    except BaseException as error:
        report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources'].get('violations'), 'Final GPU monitor violations')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY', resource_error=str(error))
        for item in (finder, no_hif4):
            if item in sys.meta_path:
                sys.meta_path.remove(item)
        # A failed native video writer may leave one named partial mp4. Account
        # those three possible files explicitly, never scan any directory.
        for case in plan['cases'][1:]:
            path = counter.path(out/run_id(case)/'output.mp4')
            if path not in counter.files and path.exists():
                counter.observe_external(path, hold=str(path))
        try:
            report['wall_seconds'] = time.monotonic()-started
            journal.close()
            inventory = {str(path.relative_to(out)): value for path, value in counter.files.items() if path != out/'RUN.json'}
            audit = {'schema': 'ffnds.raw_bf16.output_written_bytes.v2', 'id': plan['id'], 'protocol_sha256': digest,
                'files_excluding_top_level_RUN_and_this_manifest': inventory, 'incremental_checkpoint': counter.summary(),
                'complete_live_bytes_formula': 'sum(inventory bytes) + actual RUN.json bytes + actual OUTPUT_ACCOUNTING.json bytes',
                'scope': 'Owned fresh results directory; source snapshots, journals, completed/partial outputs; directories have no payload bytes'}
            counter.write_json(out/'OUTPUT_ACCOUNTING.json', audit)
            report['output_accounting'] = {'receipt': counter.binding(out/'OUTPUT_ACCOUNTING.json', args.hf_root),
                'checkpoint_before_final_RUN_write': counter.summary(), 'recursive_output_walks': 0}
            persist()
        finally:
            file_guard.__exit__(None, None, None)
        if monitor_error is not None and sys.exc_info()[0] is None:
            raise monitor_error
    return report


def bind_actual_predecessor(plan, hf):
    """Bind only terminal actual receipts; never reinterpret a live V2 prefix."""
    parent = parent_protocol(plan)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == parent['id']]
    native.need(len(matches) == 1, 'Unique V2 registered predecessor required')
    record = matches[0]
    if not {'RUN', 'actual_worker_exit'} <= set(record):
        return False
    previous = plan['failed_predecessor']
    previous.update(RUN=record['RUN'], launch_exit=record['actual_worker_exit'])
    run = native.read(native.bound(hf, previous['RUN']))
    previous['completed_prefix_generations'] = []
    for case in parent['cases'][:2]:
        phases = [p for p in run['phases'] if p.get('native_source_ordinal') == case['native_source_ordinal']]
        native.need(len(phases) == 1 and 'generation_receipt' in phases[0], 'V2 complete dev-plus-first-cal prefix NOT_BOUND')
        previous['completed_prefix_generations'].append({'native_source_ordinal': case['native_source_ordinal'],
            'role': case['role'], 'GENERATION': phases[0]['generation_receipt']})
    verify_predecessor(plan, hf)
    plan.update(status='DRAFT_V3_ACTUAL_PREDECESSOR_BOUND_REVIEW_AND_FREEZE_PENDING', binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING')
    return True


def draft(hf, part):
    source = ROOT/f'config/e0_original_bf16_full_depth_collect_part{part}_20261029_v2.json'
    parent = native.read(source); parent_v2.validate(parent, True)
    plan = copy.deepcopy(parent)
    cases = [copy.deepcopy(parent['cases'][i]) for i in CASE_INDICES[part]]
    plan.update(id=scope_id(part), status='DRAFT_V3_WAIT_ACTUAL_V2_TERMINAL_AND_PREFIX_RECEIPTS',
        assets_finalized=False, binding_status='NOT_BOUND_ACTUAL_V2_TERMINAL', cases=cases,
        limits=limits(part), resources=RESOURCES, run_ids=[run_id(c) for c in cases],
        completion_scope=COMPLETION, output_HF_relative=output_relative(part),
        failed_predecessor={'protocol': native.binding(source, ROOT), 'RUN': None, 'launch_exit': None, 'completed_prefix_generations': None},
        source_split='Parent6cal+2dev; part0 replays1dev+completes2cal, part1 replays1dev+completes1cal. V2 completed first cal per part and source288 raw capture are separately audited.',
        row_tensor_payload_bytes_per_part=160*len(cases)*legacy.PACKET_BYTES,
        state_tensor_payload_bytes_per_part=parent['state_tensor_payload_bytes_per_part']//4*len(cases),
        neutral_policy='This part first reproduces old dev raw/initial and all4flow+post bytes; new cal only after neutral. Completed V2 prefixes are not resampled.',
        cohort_completion='Unique8-source dataset requires separate V2 prefix/source288 rescue audit plus both V3 actual exit0; V2 remains failed.',
        historical_outputs_byte_budget='Failed V1/V2 outputs retained separately; this NEW12GiB per-part budget is additional, not a24GiB historical total',
        V2_source288_salvage='EXTERNAL_CPU_AUDIT_PENDING_NOT_AN_EXECUTION_INPUT_NOT_CLAIMED_SUCCESS')
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted({b['path'] for b in parent['repo_sources']} | EXTRA_SOURCES | {str(source.relative_to(ROOT))})]
    bind_actual_predecessor(plan, hf)
    validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--part', type=int); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--hf-root', type=Path, required=True); parser.add_argument('--local-root', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); args.hf_root = args.hf_root.resolve()
    if args.command == 'draft':
        native.need(args.part in (0, 1) and args.out is not None and not args.out.exists(), 'Part/fresh draft required')
        value = draft(args.hf_root, args.part); native.write(args.out, value)
        print(json.dumps({'id': value['id'], 'status': value['status'], 'model_calls': 0}))
    elif args.command == 'inspect':
        value = native.read(args.protocol); validate(value)
        print(json.dumps({'id': value['id'], 'part': value['part'], 'limits': limits(value['part']), 'resources': RESOURCES,
            'completion_scope': COMPLETION, 'model_calls': 0}, indent=2))
    elif args.command == 'preflight':
        native.need(args.out is not None and args.local_root is not None, 'Fresh CPU output/local package required')
        out = args.out.resolve(); native.need(out.is_relative_to(args.hf_root) and not out.exists(), 'Fresh confined CPU output required')
        out.mkdir(parents=True)
        value = {'id': native.read(args.protocol)['id'], 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_V3_BYTE_PREFLIGHT_NO_RETRY'}
        native.write(out/'RUN.json', value)
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_V3_BYTE_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
        finally:
            native.write(out/'RUN.json', value)
        print(json.dumps({'status': value['status'], 'model_calls': 0}))
    else:
        native.need(args.local_root is not None, 'Exact local package required')
        print(json.dumps({'status': run(args)['status']}))


if __name__ == '__main__':
    main()
