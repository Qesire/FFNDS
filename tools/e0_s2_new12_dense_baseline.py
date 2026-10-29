"""New12 full-depth original-BF16 Dense acquisition; independently versioned.

Draft and CPU proof do not import numerical libraries. Runtime is authorization
gated. Historical Dense neutral precedes the six new development sources.
"""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))
_spec = importlib.util.spec_from_file_location('_ffnds_new12_private_video', ROOT / 'tools/e0_raw_bf16_full160_video20_v2.py')
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)
native = vm.native
BANK = 'config/e0_s2_new12_development_source_bank_20261029_v1.json'
ORDINALS = (911, 912, 978, 979, 1039, 1040, 922, 923, 616, 624, 456, 472)
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
    'host_available_min_bytes': 128 << 30, 'free_disk_min_bytes': 6 << 30, 'wall_seconds': 2400, 'outer_seconds': 2700,
    'new_HF_bytes': 4 << 30, 'session_max_GPUs': 2, 'automatic_retry': False}
SETUP = ('original_expert_loads', 'raw_Default_MM_loads', 'buffer_Default_MM_loads',
         'T5_raw_Default_MM_loads', 'T5_buffer_Default_MM_loads')


def limits(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts')
    original = vm.native_original_limits
    native.need(all(n % 2 == 0 for k, n in original.items() if k not in SETUP),
                'Original two-video limits not divisible')
    result = {k: (n if k in SETUP else n // 2 * 7) for k, n in original.items()}
    result.update(core_CPU_reports=0, prefetch_transfer_observations=7*162,
        stage_CUDA_event_pairs=7*7, state_tensor_D2H=70,
        state_tensor_D2H_bytes=7*math.prod(native.VIDEO['latent_shape'])*30,
        state_output_writes=70, video_geometry_checks=7,
        historical_state_byte_checks=10, source_registration_writes=7)
    return result


def validate_bank(bank):
    sources = bank['sources']
    native.need(bank['status'] == 'REVIEWED_NEW_DEVELOPMENT_INPUTS_NO_MODEL_SCOPE'
        and [c['native_source_ordinal'] for c in sources] == list(ORDINALS)
        and len({c['image']['sha256'] for c in sources}) == 12
        and len({c['source_id'] for c in sources}) == 12, 'Fixed source identity differs')
    for index, c in enumerate(sources):
        native.need(c['cohort_index'] == index and c['part'] == index % 2
            and c['role'] == 'DEVELOPMENT' and c['prompt'] == c['original_prompt']
            and hashlib.sha256(c['prompt'].encode()).hexdigest() == c['prompt_id']
            and c['image_id'] == c['image']['sha256'], 'Source condition differs')
    return copy.deepcopy(sources)


def cases(bank, part, neutral):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts')
    rows = validate_bank(bank)
    native.need(neutral['native_source_ordinal'] == (898 if part == 0 else 96)
        and neutral['seed'] == 42 and neutral['part'] == part
        and hashlib.sha256(neutral['prompt'].encode()).hexdigest() == neutral['prompt_sha256'],
        'Historical neutral identity differs')
    result = [copy.deepcopy(neutral)]
    result[0]['case_role'] = 'HISTORICAL_DENSE_REPEAT_CONTROL_NOT_NEW_SOURCE'
    for c in rows:
        if c['part'] == part:
            result.append({'index': c['cohort_index'], 'part': part,
                'native_source_ordinal': c['native_source_ordinal'], 'source_id': c['source_id'],
                'image': copy.deepcopy(c['image']), 'prompt': c['prompt'],
                'prompt_sha256': c['prompt_id'], 'seed': 42,
                'prior_history': 'DEVELOPMENT_VISUALLY_REVIEWED_NOT_BLIND',
                'case_role': 'NEW_DEVELOPMENT_IMAGE_NOT_CONFIRMED_MOTION'})
    return result


def run_id(case, mode):
    native.need(mode == 'dense', 'Only original Dense is in scope')
    return f'new12_dense_ord{case["native_source_ordinal"]}_seed42'


def registration(part_cases, part, hf, bank_sha256, protocol_id):
    """Build a single-arm manifest; the role override is explicitly descriptive."""
    from ffnds_cohort_preflight import audit
    expected_control = 898 if part == 0 else 96
    native.need(len(part_cases) == 7 and part_cases[0]['native_source_ordinal'] == expected_control
        and [c['native_source_ordinal'] for c in part_cases[1:]] == list(ORDINALS[part::2]),
        'Seven ordered cases including historical control required')
    sources, jobs = [], []
    for c in part_cases:
        condition = c['source_id'] + ':' + c['prompt_sha256']
        control = c['case_role'] == 'HISTORICAL_DENSE_REPEAT_CONTROL_NOT_NEW_SOURCE'
        sources.append({'source_id': c['source_id'], 'image_id': c['image']['sha256'],
            'prompt_id': c['prompt_sha256'], 'prompt': c['prompt'],
            'image_path': str((hf / c['image']['path']).resolve()),
            'image_name': Path(c['image']['path']).name, 'conditioning_id': condition,
            'role': 'DEVELOPMENT', 'acquisition_role': c['case_role'],
            'exposure_history': {'fitting': None if control else False, 'method_selection': control,
                                 'prior_evaluation': control, 'input_visual_review': not control}})
        rid = run_id(c, 'dense')
        jobs.append({'source_id': c['source_id'], 'variant': 'dense', 'sample_index': 0,
            'seed': 42, 'run_id': rid, 'asset_id': rid + ':actual_video',
            'initial_noise_id': rid + ':actual_noise', 'conditioning_id': condition,
            'scheduler_id': 'Wan22StepDistill_shift5_1000_750_500_250',
            'frames': 81, 'fps': 16, 'width': 832, 'height': 464})
    manifest = {'schema': 'ffnds.cohort.v1', 'phase': 'REGISTERED_REAL_GENERATION_PLAN',
        'mode': 'CUSTOM_PAIRED', 'purpose': 'DEVELOPMENT', 'sources': sources, 'samples': jobs,
        'required_variants': ['dense'], 'expected_samples_per_source_per_variant': 1,
        'source_bank_id': bank_sha256, 'native_protocol_id': protocol_id,
        'pairing_note': 'SINGLE_DENSE_ARM_NO_PAIRED_COMPARISON'}
    native.need(not audit(manifest)['errors'], 'Single Dense registration manifest invalid')
    return manifest


def verify_source_provenance(bank, hf):
    """Verify actual producer receipts and twelve bound images; never open old images."""
    rows = validate_bank(bank)
    extract_protocol = ROOT/'config/e0_s2_source18_archive_cpu_20261029_v2.json'
    extract_binding = bank['all_candidate_parent']
    extract = native.read(native.bound(hf, extract_binding['RUN']))
    native.need(native.sha(extract_protocol) == extract['protocol_sha256'], 'Extraction protocol SHA differs')
    vm.exited(native.read(native.bound(hf, extract_binding['exit'])), extract_binding['RUN'],
              extract['protocol_sha256'], extract['id'])
    native.need(extract['id'] == 'E0_S2_SOURCE18_PROVISIONAL_ARCHIVE_CPU_V2_20261029'
        and extract['status'] == 'COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_COHORT_REVIEW'
        and extract['archive_payload_verified'] is True and extract['model_calls'] == extract['GPU_calls'] == 0
        and extract['archive_streams_predebited'] == extract['archive_streams_observed'] == 1
        and extract['predebited_archive_bytes'] == extract['actual_archive_bytes'] == 2607670436
        and extract['image_files_predebited'] == extract['image_files_observed'] == len(extract['sources']) == 18
        and extract['predebited_image_bytes'] == extract['actual_image_bytes'], 'Extraction proof differs')
    history_binding = bank['history_screen']
    history = native.read(native.bound(hf, history_binding['RUN']))
    vm.exited(native.read(native.bound(hf, history_binding['exit'])), history_binding['RUN'],
              history['protocol_sha256'], history['id'])
    native.need(history['id'] == 'E0_S2_SOURCE18_NONCONFIRMATION_HISTORY_DHASH_CPU_20261029'
        and history['status'] == 'COMPLETE_BOUNDED_NONCONFIRMATION_HISTORY_SCREEN'
        and history['image_decodes_predebited'] == history['image_decodes_observed'] == 74
        and history['image_bytes_predebited'] == history['image_bytes_observed']
        and history['reserved_image_opens'] == history['GPU_calls'] == history['model_calls'] == 0,
        'History receipt differs')
    eidx = {r['native_source_ordinal']: r for r in extract['sources']}
    hidx = {r['native_source_ordinal']: r for r in history['candidates']}
    native.need(len(eidx) == len(hidx) == 18, 'Candidate producer duplicates')
    for c in rows:
        n = c['native_source_ordinal']; e = eidx[n]; h = hidx[n]
        native.need(c['source_audit'] == extract_binding and c['history_screen']['RUN'] == history_binding['RUN']
            and c['history_screen']['exit'] == history_binding['exit']
            and c['history_screen']['protocol']['sha256'] == history['protocol_sha256']
            and e['image'] == c['image'] and e['source_id'] == c['source_id']
            and e['original_prompt'] == c['prompt'] and e['original_prompt_sha256'] == c['prompt_id']
            and e['exclusion'] is False and h['recommended'] is True and not h['flagged_pairs'],
            'Candidate source proof differs')
        native.bound(ROOT, c['history_screen']['protocol'])
        native.bound(ROOT, c['input_review'])
        native.bound(hf, c['image'])
    return {'extraction_RUN': extract_binding['RUN'], 'history_RUN': history_binding['RUN'],
            'verified_new_images': len(rows), 'reserved_confirmation_images_opened': 0}


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts')
    return f'E0_S2_NEW12_DENSE_BASELINE_20261029_PART{part}_V1'


def output_relative(part):
    scope_id(part)
    return f'ffnds-derived/e0_s2_new12_dense_baseline_20261029/part{part}/attempt1/worker/results'


def draft(hf, part):
    scope_id(part)
    envpath = ROOT / f'config/e0_s2_spatial_student_collect_part{part}_20261029_v2.json'
    env = native.read(envpath)
    bank = native.read(ROOT/BANK)
    selected = cases(bank, part, env['cases'][0])
    keys = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions',
        'local_transformers', 'private_worker_config_sha256', 'execution_private_worker_config_sha256',
        'execution_raw_master_CPU_proof', 'runtime_overlay')
    plan = {k: copy.deepcopy(env[k]) for k in keys}
    paths = {x['path'] for x in env['repo_sources']} | vm.REQUIRED | {
        BANK, 'config/e0_s2_source18_archive_cpu_20261029_v2.json', str(envpath.relative_to(ROOT)), 'tools/e0_s2_new12_dense_baseline.py',
        'tests/test_e0_s2_new12_dense_baseline.py', 'tools/portable_raw_bf16_worker_proof.py',
        'tools/e0_s2_spatial_student_collect_v2.py', 'tools/e0_s2_spatial_student_collect.py',
        'tools/e0_raw_bf16_temporal_video20_quality.py', 'tools/e0_raw_bf16_temporal_video20.py',
        'tools/ffnds_stage_registered_i2v.py', 'tools/ffnds_cohort_preflight.py'}
    plan.update(id=scope_id(part), part=part, status='DRAFT_NOT_AUTHORIZED', assets_finalized=False,
        numeric_path='BF16_ORIGINAL', automatic_retry=False, resources=copy.deepcopy(RESOURCES),
        limits=limits(part), CPU_preflight_resources=copy.deepcopy(CPU_RESOURCES), source_bank=native.binding(ROOT/BANK, ROOT),
        neutral_parent=native.binding(envpath, ROOT), cases=selected,
        run_ids=[run_id(c, 'dense') for c in selected], output_HF_relative=output_relative(part),
        core_CPU_reports='NOT_RUN_SEPARATE_SCOPE', quality='NOT_RUN',
        scope_meaning='SIX_NEW_DEVELOPMENT_IMAGES_PLUS_ONE_HISTORICAL_CONTROL',
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(paths)])
    capture_out = hf/env['output_HF_relative']
    plan['neutral_capture'] = {'RUN': native.binding(capture_out/'RUN.json', hf),
        'exit': native.binding(capture_out.parent/'EXIT_RECEIPT.json', hf)}
    return plan


def validate(plan, frozen=False):
    part = plan['part']
    bank = native.read(native.bound(ROOT, plan['source_bank']))
    env = native.read(native.bound(ROOT, plan['neutral_parent']))
    native.need(plan['id'] == scope_id(part) and plan['resources'] == RESOURCES
        and plan['limits'] == limits(part) and plan['CPU_preflight_resources'] == CPU_RESOURCES and plan['numeric_path'] == 'BF16_ORIGINAL'
        and plan['automatic_retry'] is False and plan['output_HF_relative'] == output_relative(part)
        and plan['cases'] == cases(bank, part, env['cases'][0])
        and plan['run_ids'] == [run_id(c, 'dense') for c in plan['cases']]
        and plan['core_CPU_reports'] == 'NOT_RUN_SEPARATE_SCOPE' and plan['quality'] == 'NOT_RUN',
        'Fixed scope differs')
    inherited = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions',
        'local_transformers', 'private_worker_config_sha256', 'execution_private_worker_config_sha256',
        'execution_raw_master_CPU_proof', 'runtime_overlay')
    native.need(all(plan[k] == env[k] for k in inherited), 'Portable environment differs')
    required = {x['path'] for x in env['repo_sources']} | vm.REQUIRED | {
        BANK, 'config/e0_s2_source18_archive_cpu_20261029_v2.json', plan['neutral_parent']['path'], 'tools/e0_s2_new12_dense_baseline.py',
        'tests/test_e0_s2_new12_dense_baseline.py', 'tools/portable_raw_bf16_worker_proof.py',
        'tools/e0_s2_spatial_student_collect_v2.py', 'tools/e0_s2_spatial_student_collect.py',
        'tools/e0_raw_bf16_temporal_video20_quality.py', 'tools/e0_raw_bf16_temporal_video20.py',
        'tools/ffnds_stage_registered_i2v.py', 'tools/ffnds_cohort_preflight.py'}
    paths = [x['path'] for x in plan['repo_sources']]
    native.need(required <= set(paths) and len(paths) == len(set(paths)), 'Source closure incomplete')
    if frozen:
        native.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True,
                    'Scope not frozen')


CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 600, 'outer_seconds': 900,
                 'new_HF_bytes': 8 << 20, 'automatic_retry': False}


def old_control(plan, hf):
    """Reuse the closed, byte-neutral capture chain, opening only this source.

    The capture has already verified its prior three-video lineage. Here both
    its ten saved Dense states and the ten original video states are rehashed;
    the other 59 historical videos are outside this finite verification scope.
    """
    spec = importlib.util.spec_from_file_location('_ffnds_new12_old_proof', ROOT/'tools/e0_s2_spatial_student_collect_v2.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    env = native.read(native.bound(ROOT, plan['neutral_parent']))
    old.core.validate(env, True)
    proof = plan['neutral_capture']
    report = native.read(native.bound(hf, proof['RUN']))
    vm.exited(native.read(native.bound(hf, proof['exit'])), proof['RUN'],
              plan['neutral_parent']['sha256'], env['id'])
    expected = {k: v for k, v in env['limits'].items() if v}
    native.need(report['status'] == 'COMPLETE_ONE_SOURCE_THREE_ARMS_FULL160_SPATIAL_STUDENT_CAPTURE'
        and report['part'] == plan['part'] and report['actual_native_packets'] == 480
        and report['all_thirty_prior_video_state_byte_neutrals'] is True
        and report['operation_ledger']['predebited'] == report['operation_ledger']['observed'] == expected
        and not report['operation_ledger']['forbidden_attempts']
        and not report['assigned_GPU_resources']['violations'], 'Historical capture is not complete/exited/neutral')
    phase = next(x for x in report['phases'] if x['mode'] == 'dense')
    observed = native.read(native.bound(hf, phase['observation_receipt']))
    native.need(all(phase.get(k) == v for k, v in observed.items())
        and phase['all_ten_prior_video_state_bytes_equal'] is True
        and len(phase['states']) == 10, 'Actual Dense observation identity differs')
    result = phase['original_video_control']
    native.need(set(phase['states']) == set(result['states']), 'Ten state names differ')
    for name, state in result['states'].items():
        native.need(vm.same_state(phase['states'][name], state), 'Captured/prior neutral bytes differ')
        native.bound(hf, state); native.bound(hf, phase['states'][name])
    native.bound(hf, result['reused_video'])
    gen = native.read(native.bound(hf, result['generation']))
    case = plan['cases'][0]
    native.need(gen['mode'] == 'dense' and gen['seed'] == 42 and gen['states'] == result['states']
        and gen['source_id'] == case['source_id'] and gen['image_sha256'] == case['image']['sha256']
        and gen['prompt_sha256'] == case['prompt_sha256'] and gen['all160_FFN_covered'] is True
        and gen['completed_native_flow_steps'] == 4, 'Historical Dense generation differs')
    native.bound(hf, case['image'])
    return result


def portable_master():
    import portable_raw_bf16_worker_proof as proof
    from types import SimpleNamespace
    vm.io.mirror = SimpleNamespace(ID=proof.ID, TOTAL=proof.TOTAL, validate=proof.validate)


def authorize(plan, digest):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and active['current_session_resource_constraint_20261028']['max_simultaneously_assigned_idle_GPUs'] == 2,
        'Required current two-GPU authority differs')
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(matches) == 1 and matches[0].get('protocol_sha256') == digest
        and matches[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and matches[0].get('limits') == plan['limits'] and matches[0].get('resources') == RESOURCES
        and matches[0].get('run_ids') == plan['run_ids'] and matches[0].get('max_gpu_cards') == 1,
        'Exact scope not registered')
    return matches[0]


def preflight(args):
    import os, signal, time
    start = time.monotonic()
    plan = native.read(args.protocol); digest = native.sha(args.protocol)
    authorize(plan, digest)
    native.need(args.out is not None and not args.out.exists()
        and args.out.resolve().is_relative_to(args.hf_root.resolve()), 'Fresh confined CPU output required')
    args.out.mkdir(parents=True)
    report = {'status': 'STARTED_METADATA_AND_BYTE_CPU_PREFLIGHT', 'model_calls': 0, 'GPU_calls': 0}
    native.write(args.out/'RUN.json', report)
    def timeout(*_): raise TimeoutError('CPU preflight wall exceeded')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        validate(plan, True)
        native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
            and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA standard-library worker required')
        report.update(id=plan['id'], protocol_sha256=digest)
        for b in plan['repo_sources']: native.bound(ROOT, b)
        portable_master(); vm.io.require_execution_master(plan, args.hf_root)
        native.model_metadata(plan)
        native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
        bank = native.read(native.bound(ROOT, plan['source_bank']))
        source_proof = verify_source_provenance(bank, args.hf_root)
        control = old_control(plan, args.hf_root)
        native.need(not {'torch', 'numpy'} & set(sys.modules), 'Numerical import during CPU proof')
        report.update(status='PASS_NEW12_DENSE_SOURCE_MASTER_PACKAGE_AND_HISTORICAL_CONTROL',
            source_closure_sha256=native.canonical(plan['repo_sources']), source_proof=source_proof,
            neutral=control, all_CUDA_devices_hidden=True, numerical_imports=False)
    except BaseException as error:
        report.update(status='FAILED_CPU_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start
        native.write(args.out/'RUN.json', report)
    return report


def require_cpu(scope, plan, digest, hf):
    proof = scope['cpu_preflight']
    report = native.read(native.bound(hf, proof['RUN']))
    vm.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, plan['id'])
    native.need(report.get('id') == plan['id'] and report.get('protocol_sha256') == digest
        and report.get('status') == 'PASS_NEW12_DENSE_SOURCE_MASTER_PACKAGE_AND_HISTORICAL_CONTROL'
        and report.get('source_closure_sha256') == native.canonical(plan['repo_sources'])
        and report.get('model_calls') == report.get('GPU_calls') == 0
        and report.get('all_CUDA_devices_hidden') is True and report.get('numerical_imports') is False
        and 0 <= report.get('wall_seconds', -1) <= CPU_RESOURCES['wall_seconds'], 'Actual exited CPU proof required')
    return proof, report['neutral']


def require_neutral(phase):
    native.need(phase.get('matched_olddev_all_ten_states') is True
        and len(phase.get('states', {})) == 10, 'Historical neutral failed; no new source may start')


def run(args):
    import importlib.metadata, json, os, shutil, signal, time
    from types import SimpleNamespace
    start = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    scope = authorize(plan, digest); proof, control = require_cpu(scope, plan, digest, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numerical worker required')
    portable_master(); vm.io.require_execution_master(plan, args.hf_root)
    for b in plan['repo_sources']: native.bound(ROOT, b)
    native.model_metadata(plan)
    local, package_rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions']
        and len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1'
        for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'Runtime or precision differs')
    bank = native.read(native.bound(ROOT, plan['source_bank']))
    verify_source_provenance(bank, args.hf_root)
    for b in control['states'].values(): native.bound(args.hf_root, b)
    native.bound(args.hf_root, control['generation']); native.bound(args.hf_root, control['reused_video'])
    out = (args.hf_root / plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined output only')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'part': plan['part'], 'status': 'STARTED_NO_RETRY',
        'numeric_path': 'BF16_ORIGINAL', 'CPU_preflight': proof, 'phases': [], 'automatic_retry': False,
        'new_development_image_clusters': 6, 'historical_neutral_repeat_controls': 1,
        'official_quality': 'NOT_RUN', 'core_CPU_reports': 'NOT_RUN_SEPARATE_SCOPE', 'motion': 'NOT_MEASURED',
        'quality_or_speed_acceptance': False}
    native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE = plan['id'], limits(plan['part']), tuple(plan['run_ids']), RESOURCES
    vm.run_id, vm.RESOURCES = run_id, RESOURCES
    monitor = None
    counter = vm.io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    for rid in plan['run_ids']: counter.hold(str(counter.path(out/rid/'output.mp4')), 128 << 20)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = vm.Ledger(report.setdefault('operation_ledger', {}), lambda e: journal.write(json.dumps(e, separators=(',', ':'))+'\n'))
    persist()
    def check():
        if monitor: monitor.check()
        counter.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'],
                    'Host or disk reserve exceeded')
        native.need(time.monotonic()-start <= RESOURCES['wall_seconds'], 'Fixed wall budget exceeded')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU40GiB bound exceeded')
    def timeout(*_): raise TimeoutError('Fixed Dense wall2400 exceeded')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, package_rows), native.NoHiF4()
    file_guard = vm.io.single_file_bound()
    try:
        file_guard.__enter__()
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        reservation, monitor, report['CUDA_reservation'] = vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'TF32': False, 'FP32_matmul': 'highest', 'BF16_reduced_precision_reduction': True}
        check()
        snapshot = out/'EXECUTED_SOURCES'; snapshot.mkdir()
        copies = [(ROOT/b['path'], snapshot/b['path'], b) for b in plan['repo_sources']]
        copies.append((args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)))
        counter.hold('frozen_metadata', sum(b['bytes'] for _, _, b in copies))
        for source, dest, binding in copies:
            dest.parent.mkdir(parents=True, exist_ok=True); counter.copy_bound(source, dest, binding)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Source snapshot accounting differs')
        manifest = registration(plan['cases'], plan['part'], args.hf_root, plan['source_bank']['sha256'], plan['id'])
        counter.write_json(snapshot/'registered_video_plan.json', manifest)
        with native.OriginalGuard(torch, plan, ledger) as guard:
            at = time.perf_counter(); pipe = native.build_pipeline(plan); torch.cuda.synchronize()
            report['raw_model_setup_seconds'] = time.perf_counter()-at; report['original_loader_paths'] = guard.loaded
            runtime = SimpleNamespace(mode='dense')
            controls = {plan['cases'][0]['native_source_ordinal']: control['states']}
            for index, case in enumerate(plan['cases']):
                check(); authorize(plan, digest)
                phase = vm.native_phase(torch, pipe, guard, runtime, plan, case, 'dense', out, args.hf_root,
                                        digest, report, persist, check, counter, controls)
                if index == 0:
                    require_neutral(phase)
                folder = out/f'job{index:06d}'; folder.mkdir()
                def write_registration():
                    counter.write_json(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1',
                        'status': 'COMPLETE_NATIVE_GENERATION_ONLY', 'job': manifest['samples'][index],
                        'source': manifest['sources'][index], 'native_ffn_sites': 160, 'scheduler_generator_reset': True,
                        'native_ffn_counters': phase['native_counter_delta'],
                        'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL'})
                ledger.call(phase['run_id'], {'source_registration_writes': 1}, write_registration)
                persist(); check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(manifest, out); counter.write_json(out/'VERIFIED_GENERATIONS.json', verified)
            native.need(len(report['phases']) == 7 and not counter.holds, 'Incomplete part output')
            report.update(status='COMPLETE_NEW12_DENSE_BASELINE_PART', actual_full160_videos=7,
                actual_flow_steps=28, actual_FFN_positions=1120, actual_states=70,
                historical_control_all_ten_states_passed=True,
                verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root),
                checked_transformers_imports=finder.origins)
            check()
    except BaseException as error:
        report.update(status='FAILED_DENSE_BASELINE_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources'].get('violations'), 'Final monitor violation')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_DENSE_BASELINE_PRESERVED_NO_RETRY', resource_error=str(error))
        for finder_item in (finder, no_hif4):
            if finder_item in sys.meta_path: sys.meta_path.remove(finder_item)
        for rid in plan['run_ids']:
            path = counter.path(out/rid/'output.mp4')
            if path not in counter.files and path.exists():
                try: counter.observe_external(path, hold=str(path))
                except BaseException as error: counter.accounting_errors.append(str(error))
        report.update(wall_seconds=time.monotonic()-start, output_accounting_checkpoint_before_final_RUN=counter.summary(),
            operation_journal=counter.binding(out/'OPERATIONS.jsonl', args.hf_root))
        persist(); journal.close(); file_guard.__exit__(None, None, None)
        if monitor_error is not None: raise monitor_error
    return report


def main():
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('draft', 'inspect', 'preflight', 'run'))
    p.add_argument('--part', type=int, choices=(0, 1)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--local-root', type=Path); p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.action == 'draft':
        native.need(args.part is not None and args.out is not None and not args.out.exists(), 'Fresh draft path and part required')
        args.out.parent.mkdir(parents=True, exist_ok=True); native.write(args.out, draft(args.hf_root, args.part))
    elif args.action == 'inspect':
        plan = native.read(args.protocol); validate(plan)
        print(json.dumps({'id': plan['id'], 'limits': plan['limits'], 'resources': RESOURCES, 'model_calls': 0, 'GPU_calls': 0}, indent=2))
    elif args.action == 'preflight': preflight(args)
    else: run(args)


if __name__ == '__main__': main()
