"""One source240 completion, with a previously proved development neutral."""
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


def private_v3(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools/e0_original_bf16_full_depth_collect_v3.py')
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


parent_v3 = private_v3('_ffnds_ord240_v4_parent')
execution_v3 = private_v3('_ffnds_ord240_v4_execution')
legacy, native = execution_v3.legacy, execution_v3.native
from raw_bf16_assigned_gpu_monitor_v4 import MONITOR, startup as reserved_startup

ID = 'E0_ORIGINAL_BF16_FULL_DEPTH_ORD240_20261029_V4'
PARENT = 'config/e0_original_bf16_full_depth_collect_part0_20261029_v3.json'
FROZEN_V3_SHA = '13bc1ccdfb5b746e802f6324dac211f54e034e5abfc7acd70cbd883b15bb5cec'
REPAIR = copy.deepcopy(parent_v3.REPAIR)
RESOURCES = copy.deepcopy(parent_v3.RESOURCES)
RESOURCES.update(wall_seconds=900, outer_seconds=1200, new_HF_bytes=4 << 30)
COMPLETION = {'version': 4, 'native_source_ordinal': 240, 'cases': 1, 'role': 'calibration',
    'development': 'REUSE_V3_ORD894_GEN_AND_TEN_STATES_NO_NEW_DEV_REPLAY',
    'full_model_flows': 4, 'raw_FFN_sites': 160, 'new_video_and_CORE': 1,
    'numeric_observer_and_IO': 'IDENTICAL_V2_V3', 'monitor': MONITOR, 'automatic_retry': False}
EXTRA_SOURCES = {'tools/e0_original_bf16_full_depth_collect_v4.py', 'tools/raw_bf16_assigned_gpu_monitor_v4.py',
    'tests/test_e0_original_bf16_full_depth_collect_v4.py'}
run_id = legacy.run_id
WrittenBytes, counted_io = execution_v3.WrittenBytes, execution_v3.counted_io
single_file_bound, execution_worker_sha = execution_v3.single_file_bound, execution_v3.execution_worker_sha


def limits():
    old = parent_v3.parent_v2.limits(); result = dict(old)
    setup = {'original_expert_loads', 'raw_Default_MM_loads', 'buffer_Default_MM_loads', 'T5_raw_Default_MM_loads', 'T5_buffer_Default_MM_loads'}
    one_video = {'video_generations', 'VAE_decodes', 'core_CPU_reports', 'video_geometry_checks', 'source_registration_writes'}
    zero = {'native_prefix_replays_no_decode', 'neutral_state_byte_checks'}
    for key, value in old.items():
        if key in setup:
            continue
        if key in zero:
            result[key] = 0
        elif key in one_video:
            native.need(value == 3, 'Frozen V2 calibration counters changed'); result[key] = 1
        else:
            native.need(value % 4 == 0, 'Frozen V2 source counters changed'); result[key] = value//4
    return result


legacy.limits = limits


def output_relative():
    return 'ffnds-derived/e0_original_bf16_full_depth_collect_ord240_v4_20261029/attempt1/worker/results'


def parent_protocol(plan):
    previous = plan['failed_predecessor']
    native.need(previous['protocol']['path'] == PARENT, 'Exact V3p0 parent required')
    result = native.read(native.bound(ROOT, previous['protocol'])); parent_v3.validate(result, True)
    return result


def validate(plan, frozen=False):
    native.need(native.sha(ROOT/'tools/e0_original_bf16_full_depth_collect_v3.py') == FROZEN_V3_SHA
        and plan.get('id') == ID and plan.get('numeric_path') == 'BF16_ORIGINAL'
        and plan.get('limits') == limits() and plan.get('resources') == RESOURCES
        and plan.get('repair') == REPAIR and plan.get('completion_scope') == COMPLETION
        and plan.get('output_HF_relative') == output_relative() and plan.get('automatic_retry') is False,
        'Exact single-source V4 scope differs')
    old = parent_protocol(plan)
    native.need(plan['cases'] == [old['cases'][-1]] and plan['cases'][0]['native_source_ordinal'] == 240
        and plan['cases'][0]['role'] == 'calibration' and plan['run_ids'] == [run_id(plan['cases'][0])],
        'Only original source240 calibration may run; no dev replay')
    for key in ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
        'private_worker_config_sha256', 'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof',
        'runtime_overlay', 'parent', 'raw_master_CPU_proof', 'source_cohort', 'tensor_contract', 'row_draw_identity'):
        native.need(plan[key] == old[key], 'Frozen raw execution/source contract differs: '+key)
    paths = [b['path'] for b in plan['repo_sources']]
    native.need(EXTRA_SOURCES | {b['path'] for b in old['repo_sources']} <= set(paths)
        and len(paths) == len(set(paths)), 'V4 immutable source closure incomplete')
    if frozen:
        native.need(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION'
            and isinstance(plan.get('reused_development_neutral'), dict), 'V4 draft NOT_BOUND')


def verify_predecessor(plan, hf, *, bind_state_bytes=False):
    old = parent_protocol(plan); previous = plan['failed_predecessor']
    run = native.read(native.bound(hf, previous['RUN'])); exit_record = native.read(native.bound(hf, previous['launch_exit']))
    native.need(run.get('id') == old['id'] and run.get('protocol_sha256') == previous['protocol']['sha256']
        and run.get('status') == 'FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY' and run.get('local_development_neutral_passed') is True
        and exit_record.get('id') == old['id'] and exit_record.get('protocol_sha256') == previous['protocol']['sha256']
        and exit_record.get('run_sha256') == previous['RUN']['sha256'] and exit_record.get('process_exited') is True
        and type(exit_record.get('child_returncode')) is int and exit_record['child_returncode'] == 1
        and exit_record.get('automatic_retry') is False, 'Actual failed V3p0 and prior passed neutral required')
    phase = next(p for p in run['phases'] if p['native_source_ordinal'] == 894)
    reused = plan['reused_development_neutral']
    native.need(reused == {'native_source_ordinal': 894, 'GENERATION': phase['generation_receipt'], 'states': phase['states']},
        'Original V3 dev GEN/ten states differ')
    generation = native.read(native.bound(hf, reused['GENERATION']))
    native.need(generation.get('id') == old['id'] and generation.get('protocol_sha256') == previous['protocol']['sha256']
        and generation.get('status') == 'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160'
        and generation.get('native_source_ordinal') == 894 and generation.get('role') == 'development'
        and generation.get('all_four_original_flow_and_post_bytes_passed') is True
        and generation.get('completed_native_flow_steps') == 4 and generation.get('native_ffn_sites') == 160
        and generation['states'] == reused['states'] and len(reused['states']) == 10,
        'Reused development neutral not complete')
    for name, item in reused['states'].items():
        expected = old['cases'][0]['controls'][name]
        native.need(all(item[k] == expected[k] for k in ('raw_tensor_sha256', 'shape', 'dtype')), 'Reused dev state not original byte neutral')
        if bind_state_bytes:
            native.bound(hf, item)
    native.need(not any(p.get('native_source_ordinal') == 240 and 'generation_receipt' in p for p in run['phases']),
        'Source240 already complete; no repeated completed generation')
    return previous


def verify_lineage(plan, hf):
    old = parent_protocol(plan); parent_v3.verify_lineage(old, hf)
    verify_predecessor(plan, hf)
    return old


def cpu_identity(plan, digest):
    return {'id': ID, 'protocol_sha256': digest, 'status': 'PASS_V4_SOURCE240_REUSED_V3_DEV_TEN_STATES_CURRENT_RAW_STAT_NO_NUMERIC_IMPORTS',
        'sources_sha256': native.canonical(plan['repo_sources']), 'source_cohort_sha256': plan['source_cohort']['sha256'],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'],
        'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
        'failed_predecessor': plan['failed_predecessor'], 'reused_development_neutral': plan['reused_development_neutral'],
        'controls_reused': 10, 'controls_newly_generated': 0, 'sources': 1, 'native_row_sites_planned': 160,
        'reused_development_state_bytes_bound': sum(x['bytes'] for x in plan['reused_development_neutral']['states'].values()),
        'all_CUDA_devices_hidden': True, 'Torch_or_NumPy_imported': False, 'model_calls': 0, 'GPU_calls': 0,
        'raw_master_payload_bytes_rehashed': 0, 'completion_scope': COMPLETION}


def preflight(args):
    started = time.monotonic()
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules), 'Fresh hidden-CUDA stdlib CPU proof required')
    plan = native.read(args.protocol); validate(plan, True)
    native.need(len(os.sched_getaffinity(0)) <= 4, 'Preflight affinity exceeds4')
    verify_lineage(plan, args.hf_root); verify_predecessor(plan, args.hf_root, bind_state_bytes=True)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    native.bound(args.hf_root, plan['cases'][0]['image'])
    native.model_metadata(plan); native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    result = cpu_identity(plan, native.sha(args.protocol)); result['wall_seconds'] = time.monotonic()-started
    native.need(result['wall_seconds'] <= 300 and not {'torch', 'numpy'} & set(sys.modules), 'CPU proof wall/import gate')
    return result


def authorize(plan, digest):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json'); policy = active.get('resource_scheduling_policy_20261028', {})
    native.need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
        and active.get('current_session_resource_constraint_20261028', {}).get('max_simultaneously_assigned_idle_GPUs') == 2,
        'Current repository/session authority differs')
    found = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    native.need(len(found) == 1 and found[0].get('protocol_sha256') == digest and found[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and found[0].get('limits') == limits() and found[0].get('resources') == RESOURCES
        and found[0].get('run_ids') == plan['run_ids'] and found[0].get('max_gpu_cards') == 1, 'Exact V4 registered scope required')
    return found[0]


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight')
    native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Independent V4 CPU proof NOT_BOUND')
    run = native.read(native.bound(hf, proof['RUN']))
    native.need(all(run.get(k) == v for k, v in cpu_identity(plan, digest).items()) and 0 <= run.get('wall_seconds', -1) <= 300, 'V4 actual CPU proof differs')
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, ID)
    return proof


def registration(plan,hf):
    from ffnds_cohort_preflight import audit
    registered_sources,samples=[],[]
    for c in plan['cases']:
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
    checked=audit(result);native.need(not checked['errors'] and checked['summary']['independent_image_clusters']==1,'Raw calibration video registration invalid')
    return result

def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
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
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined V4 output required; no retry')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'numeric_path': 'BF16_ORIGINAL', 'part': plan['part'],
        'status': 'STARTED_RAW_FULL_DEPTH_ORD240_V4_NO_RETRY', 'automatic_retry': False, 'CPU_preflight': cpu, 'phases': [],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'], 'failed_predecessor': plan['failed_predecessor'],
        'repair': REPAIR, 'completion_scope': COMPLETION, 'source_cohort': plan['source_cohort'], 'native_part_image_clusters': len(plan['cases']),
        'new_calibration_image_clusters': 1, 'old_development_image_clusters': 0, 'reused_development_neutral': plan['reused_development_neutral'], 'local_development_neutral_passed': True, 'neutral_identity': 'REUSED_V3_DEV894_NO_NEW_REPLAY', 'complete_cohort_claimed': False,
        'cohort_completion': 'SOURCE240_ONLY_REQUIRES_SEPARATE_FINAL_UNIQUE8_AUDIT',
        'state_identity': 'Teacher states only; no student states captured', 'source_history': 'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT',
        'official_quality': 'NOT_RUN', 'training_updates': 0, 'decompositions': 0, 'quality_or_speed_acceptance': False,
        'capture_runtime_identity': 'Native generation plus row capture/D2H/serialization/CPU checks; not a throughput measurement'}
    monitor = None
    counter = WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    # Reserve all opaque native video writers before the first model call.
    for case in plan['cases']:
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
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Full-depth source240 wall900 exhausted')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU reserved32GiB exceeded')

    def timeout(*_):
        raise TimeoutError('Fixed V4 full-depth wall900 exhausted; no retry')

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
        reserve_before_assets = reserved_startup.reserve_before_assets
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
                native.need(report.get('local_development_neutral_passed') is True and case['role'] == 'calibration', 'Source240 requires reused neutral and calibration role')
                phase = legacy.native_phase(torch, pipe, guard, plan, case, out, args.hf_root, digest, report, persist, check)
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
                    folder = out/f'job{index:06d}'; folder.mkdir()
                    native.write(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
                        'job': registered['samples'][index], 'source': registered['sources'][index], 'native_ffn_sites': 160,
                        'scheduler_generator_reset': True, 'native_ffn_counters': phase['native_counter_delta'],
                        'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL'})
                ledger.call(run_id(case), {'source_registration_writes': 1}, save_registration); persist(); check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); native.write(out/'VERIFIED_GENERATIONS.json', verified)
            report.update(status='COMPLETE_RAW_BF16_FULL_DEPTH_SOURCE240_V4_REUSED_DEV_NEUTRAL',
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
        for case in plan['cases']:
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


def draft(hf):
    old = native.read(ROOT/PARENT); parent_v3.validate(old, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    found = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == old['id']]
    native.need(len(found) == 1 and {'RUN', 'actual_worker_exit'} <= set(found[0]), 'Actual failed V3p0 bindings NOT_BOUND')
    previous = {'protocol': native.binding(ROOT/PARENT, ROOT), 'RUN': found[0]['RUN'], 'launch_exit': found[0]['actual_worker_exit']}
    run = native.read(native.bound(hf, previous['RUN'])); dev = next(p for p in run['phases'] if p['native_source_ordinal'] == 894)
    plan = copy.deepcopy(old); case = copy.deepcopy(old['cases'][-1])
    plan.update(id=ID, status='DRAFT_V4_SOURCE240_PRIOR_DEV_REUSE_BOUND_REVIEW_PENDING', assets_finalized=False,
        binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING', limits=limits(), resources=RESOURCES,
        completion_scope=COMPLETION, cases=[case], run_ids=[run_id(case)], failed_predecessor=previous,
        reused_development_neutral={'native_source_ordinal': 894, 'GENERATION': dev['generation_receipt'], 'states': dev['states']},
        output_HF_relative=output_relative(), row_tensor_payload_bytes_per_part=160*legacy.PACKET_BYTES,
        state_tensor_payload_bytes_per_part=old['state_tensor_payload_bytes_per_part']//3,
        source_split='Only cal240 generated here; dev894 complete V3 neutral reused, no extra independent source',
        neutral_policy='Reuse exact V3p0 dev894 GEN+10 original-byte-matched states; do not replay dev or misclassify first cal as development',
        historical_outputs_byte_budget='Failed V1/V2/V3 and completed source outputs retained; this NEW4GiB budget is additional',
        cohort_completion='Only source240 completion; final unique8 union requires separate closed-scope audit')
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted({b['path'] for b in old['repo_sources']} | EXTRA_SOURCES | {PARENT})]
    validate(plan); verify_predecessor(plan, hf)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--protocol', type=Path); parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--local-root', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); args.hf_root = args.hf_root.resolve()
    if args.command == 'draft':
        native.need(args.out is not None and not args.out.exists(), 'Fresh draft required')
        value = draft(args.hf_root); native.write(args.out, value); print(json.dumps({'id': ID, 'status': value['status'], 'model_calls': 0}))
    elif args.command == 'inspect':
        value = native.read(args.protocol); validate(value); print(json.dumps({'id': ID, 'limits': limits(), 'resources': RESOURCES, 'completion_scope': COMPLETION}, indent=2))
    elif args.command == 'preflight':
        native.need(args.out is not None and args.local_root is not None, 'Fresh CPU output/local package required')
        out = args.out.resolve(); native.need(out.is_relative_to(args.hf_root) and not out.exists(), 'Fresh confined CPU output')
        out.mkdir(parents=True); value = {'id': ID, 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_V4_CPU_PREFLIGHT_NO_RETRY'}
        native.write(out/'RUN.json', value)
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_V4_BYTE_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
        finally:
            native.write(out/'RUN.json', value)
        print(json.dumps({'status': value['status'], 'model_calls': 0}))
    else:
        native.need(args.local_root is not None, 'Exact local package required'); print(json.dumps({'status': run(args)['status']}))


if __name__ == '__main__':
    main()
