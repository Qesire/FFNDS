"""Eight original official functions on 60 completed raw-BF16 paired clips.

Twenty exposed image clusters, three full-depth arms, one seed. Preparation is
stdlib-only. Actual generation/exit receipts remain NOT_BOUND until both parts
finish. Original v4/v8 protected metric code is reused without source edits.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
_spec = importlib.util.spec_from_file_location('_ffnds_raw_two_quality_for_video20', ROOT/'tools/e0_original_bf16_consumer_quality.py')
raw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(raw)
base, v4 = raw.base, raw.v4
ID = 'E0_RAW_BF16_FULL160_VIDEO20_OFFICIAL_QUALITY_20261029'
COHORT = 'config/e0_raw_bf16_full160_video20_sources_20261029_draft.json'
GENERATION_V1 = 'config/e0_raw_bf16_full160_video20_part{part}_20261029_v1.json'
GENERATION = 'config/e0_raw_bf16_full160_video20_part{part}_20261029_v2.json'
PREFIX_PROTOCOL = 'config/e0_raw_bf16_video20_prefix_core_repair_cpu_20261029_v1.json'
PREFIX_OUTPUT = 'ffnds-derived/e0_raw_bf16_video20_prefix_core_repair_cpu_20261029/attempt1/worker/results'
METRIC_PARENT = 'config/e0_original_bf16_consumer_quality_20261028_v1.json'
MODES = ('dense', 'shared', 'step')
DIMENSIONS, COMMON = raw.DIMENSIONS, raw.COMMON
COUNTS = {d: 60 if d in COMMON else 30 for d in DIMENSIONS}
COSTS = {'imaging_quality': {'MUSIQ': 4860}, 'aesthetic_quality': {'CLIP_L14': 180, 'aesthetic_head': 180},
    'motion_smoothness': {'AMT': 2400}, 'dynamic_degree': {'RAFT': 2400, 'RAFT_iterations': 48000},
    'subject_consistency': {'DINO': 4860}, 'background_consistency': {'CLIP_B32': 60},
    'i2v_subject': {'DINO': 2460}, 'i2v_background': {'DreamSim_embed': 2460}}
LIMITS = {'official_dimension_calls': 8, 'new_video_dimension_results': 420, 'reused_core_reports': 60,
    'RAFT_forwards': 2400, 'RAFT_iterations': 48000, 'video_generations': 0, 'training_updates': 0,
    'decompositions': 0, 'CPU_metric_recomputations': 0}
RESOURCES = {'GPU': 1, 'session_max_GPUs': 2, 'repository_GPU_policy': 'DYNAMIC_MAX_ACTUALLY_CALLABLE',
    'CPU_threads': 2, 'wall_seconds': 7200, 'outer_seconds': 7500, 'new_HF_bytes': 1 << 29,
    'GPU_reserved_bytes': 8 << 30, 'free_disk_bytes': 10 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 1200, 'outer_seconds': 1500,
    'model_calls': 0, 'numeric_imports': 0, 'automatic_retry': False}
CPU_STATUS = 'PASS_VIDEO20_60_GENERATIONS_METRIC_BYTES_REUSED_EXACT_IMPORTS_CURRENT_WORKER_NO_NUMERIC_IMPORTS'
POLICY = {'I2V_subject': 'TEN_SUBJECT_IMAGE_CLUSTERS_THREE_ARMS',
    'I2V_background': 'SEVEN_BACKGROUND_PLUS_THREE_CAMERA_IMAGE_CLUSTERS_THREE_ARMS',
    'camera_conditioning': 'NEW_CAMERA_PROMPT_PRESERVED_IMAGE_HISTORY_EXPOSED',
    'temporal_flickering': 'NOT_RUN_NINTH_AXIS_NO_STATIC_ELIGIBILITY_OR_IMPORT_PROOF',
    'semantic_camera_action': 'NOT_CERTIFIED_BY_RAFT_OR_I2V',
    'statistical_unit': 'TWENTY_IMAGE_CLUSTERS_SEED42_ARMS_FRAMES_PAIRS_NESTED',
    'quality_acceptance': 'PENDING_FULL_VECTOR_HUMAN_REVIEW_NO_ADAPTIVE_THRESHOLDS',
    'official_five_independent_samples': False}
REQUIRED = {'tools/e0_raw_bf16_video20_quality.py', 'tests/test_e0_raw_bf16_video20_quality.py',
    'tools/e0_original_bf16_consumer_quality.py', 'tools/raw_bf16_video20_prefix_core_repair.py',
    'tests/test_raw_bf16_video20_prefix_core_repair.py', COHORT}


def stat_record(path):
    x = Path(path).stat()
    return {'device': x.st_dev, 'inode': x.st_ino, 'bytes': x.st_size, 'mtime_ns': x.st_mtime_ns, 'ctime_ns': x.st_ctime_ns}


def rows_for_dimension(videos, dimension):
    base.require(dimension in DIMENSIONS, 'Unknown fixed metric dimension')
    return [v for v in videos if dimension in COMMON or v['i2v_dimension'] == dimension]


def video_identity(case, mode):
    return {'source': case['native_source_ordinal'], 'source_id': case['source_id'], 'source_index': case['index'],
        'mode': mode, 'seed': 42, 'image_sha256': case['image']['sha256'], 'prompt': case['prompt'],
        'prompt_sha256': case['prompt_sha256'], 'image_name': Path(case['image']['path']).name,
        'stratum': case['stratum'], 'conditioning_identity': case['conditioning_identity'],
        'i2v_dimension': 'i2v_subject' if case['stratum'] == 'subject' else 'i2v_background', 'part': case['part']}


def validate(plan, frozen=False):
    base.require(plan.get('id') == ID and plan.get('numeric_path') == 'BF16_ORIGINAL'
        and plan.get('dimensions') == list(DIMENSIONS) and plan.get('limits') == LIMITS
        and plan.get('costs') == COSTS and plan.get('resources') == RESOURCES
        and plan.get('CPU_preflight_resources') == CPU_RESOURCES and plan.get('policy') == POLICY
        and plan.get('automatic_retry') is False, 'Fixed video20 quality scope differs')
    cohort = base.read_bound(ROOT, plan['source_cohort'])
    cases = cohort['sources']; expected = [video_identity(c, m) for c in cases for m in MODES]
    base.require(len(cases) == 20 and len({c['image']['sha256'] for c in cases}) == 20
        and len(plan['videos']) == 60 and len(plan['images']) == 20
        and {s: sum(c['stratum'] == s for c in cases) for s in ('subject', 'background', 'camera')}
            == {'subject': 10, 'background': 7, 'camera': 3}, 'Twenty fixed image clusters/strata required')
    base.require(all(all(v.get(k) == value for k, value in identity.items()) for v, identity in zip(plan['videos'], expected)),
        'Sixty source/arm/new-prompt identities differ')
    base.require(plan['images'] == [{'source': c['native_source_ordinal'], 'name': Path(c['image']['path']).name,
        'binding': {k: c['image'][k] for k in ('path', 'bytes', 'sha256')}} for c in cases],
        'Exact original I2V reference images/source mapping required')
    base.require(all(len(rows_for_dimension(plan['videos'], d)) == COUNTS[d] for d in DIMENSIONS), 'Fixed scene-aware metric eligibility differs')
    base.require([(p['kind'], p['part']) for p in plan['parents']] == [('FAILED_V1_PREFIX', 0), ('FAILED_V1_PREFIX', 1), ('COMPLETE_V2_REMAINING9', 0), ('COMPLETE_V2_REMAINING9', 1)]
        and all(p['protocol'] is None or p['protocol']['path'] == (GENERATION_V1 if p['kind'] == 'FAILED_V1_PREFIX' else GENERATION).format(part=p['part']) for p in plan['parents']),
        'Four explicit native execution parents required')
    proof = plan['reused_import_proof']
    base.require(plan['prefix_repair'].get('protocol') is None or plan['prefix_repair']['protocol']['path'] == PREFIX_PROTOCOL,
        'Exact separate prefix CPU repair protocol required')
    base.require(proof['protocol']['path'] == raw.METRIC_PROTOCOL and proof['protocol']['sha256'] == raw.METRIC_SHA
        and proof['RUN']['sha256'] == raw.IMPORT_RUN_SHA and proof['exit']['sha256'] == raw.IMPORT_EXIT_SHA,
        'Successful exact v4 imports proof required')
    base.require(plan['directory_fd_audit_policy'] == v4.v3.POLICY and plan['activation_bootstrap_policy'] == v4.POLICY,
        'Successful protected environment policy differs')
    paths = [b['path'] for b in plan['repo_sources']]
    base.require(REQUIRED <= set(paths) and len(paths) == len(set(paths))
        and not any(k.startswith('failed_') or k == 'legacy' for k in plan), 'Code closure/lineage differs')
    out = Path(plan['output_HF_relative'])
    base.require(not out.is_absolute() and '..' not in out.parts and out.parts[:1] == ('ffnds-derived',), 'Quality output escaped HF')
    if frozen:
        base.require(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION'
            and all(all(p.get(k) is not None for k in ('protocol', 'RUN', 'exit')) for p in plan['parents'])
            and all(plan['prefix_repair'].get(k) is not None for k in ('protocol', 'RUN', 'exit'))
            and all(all(v.get(k) is not None for k in ('video', 'generation', 'core', 'native_evidence', 'states'))
                and (v['source_index'] < 2 or v.get('registered') is not None) for v in plan['videos']),
            'Actual completed generation proof NOT_BOUND')


def all_byte_bindings(plan):
    rows = [('REPO', b) for b in plan['repo_sources']]
    rows += [('REPO', plan['source_cohort'])]
    rows += [('HF', b) for b in plan['hf_bindings']]
    for parent in plan['parents']:
        rows += [('REPO', parent['protocol']), ('HF', parent['RUN']), ('HF', parent['exit'])]
    repair = plan['prefix_repair']
    rows += [('REPO', repair['protocol']), ('HF', repair['RUN']), ('HF', repair['exit'])]
    for video in plan['videos']:
        rows += [('HF', video[k]) for k in ('video', 'generation', 'core', 'native_evidence')]
        if video.get('registered') is not None: rows.append(('HF', video['registered']))
        rows += [('HF', b) for b in video['states'].values()]
    rows += [('HF', image['binding']) for image in plan['images']]
    unique = {}
    for root, item in rows:
        key = root, item['path']; minimal = {k: item[k] for k in ('path', 'bytes', 'sha256')}
        base.require(key not in unique or unique[key] == minimal, 'Conflicting reused byte binding')
        unique[key] = minimal
    return [(root, b) for (root, _), b in sorted(unique.items())]


def generation_metadata(plan, hf):
    """Compose six genuine failed-parent clips and 54 new clips, never relabel V1."""
    spec = importlib.util.spec_from_file_location('_ffnds_quality_prefix_evidence', ROOT/'tools/raw_bf16_video20_prefix_core_repair.py')
    prefix = importlib.util.module_from_spec(spec); spec.loader.exec_module(prefix)
    cohort = base.read_bound(ROOT, plan['source_cohort'])
    phase_index, parent_sha, parent_protocols, parents = {}, {}, {}, []
    common_identity = None
    repair_proof = plan['prefix_repair']; repair_protocol = base.read_bound(ROOT, repair_proof['protocol'])
    prefix.validate_plan(repair_protocol, True)
    repair_run = base.read_bound(hf, repair_proof['RUN'])
    base.exited(base.read_bound(hf, repair_proof['exit']), repair_proof['RUN'], repair_proof['protocol']['sha256'], prefix.ID)
    base.require(repair_protocol['id'] == repair_run['id'] == prefix.ID
        and repair_run['protocol_sha256'] == repair_proof['protocol']['sha256']
        and repair_run['status'] == 'COMPLETE_CPU_PREFIX_CORE_REPAIR_SIX_VALID_GENERATIONS_PARENTS_REMAIN_FAILED'
        and repair_run['operation_ledger'] == {'predebited': 4, 'observed': 4}
        and all(repair_run.get(k) == 0 for k in ('GPU_calls', 'model_calls', 'new_generations', 'tensor_loads'))
        and repair_run['parents'] == repair_protocol['parents'] and len(repair_run['phases']) == 6,
        'Actual successful zero-model CPU prefix CORE repair required')
    repaired = {(p['native_source_ordinal'], p['mode']): p for p in repair_run['phases']}
    base.require(len(repaired) == 6, 'Duplicate repaired prefix phase')
    for proof in plan['parents']:
        protocol = base.read_bound(ROOT, proof['protocol']); run = base.read_bound(hf, proof['RUN'])
        complete = proof['kind'] == 'COMPLETE_V2_REMAINING9'
        if complete:
            base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], protocol['id'])
            base.require(protocol['id'] == 'E0_RAW_BF16_FULL160_VIDEO20_REMAINING18_20261029_PART'+str(proof['part'])+'_V2'
                and run.get('status') == 'COMPLETE_RAW_BF16_FULL160_VIDEO20_V2_REMAINING9_PART_NINE_TRIADS'
                and len(run['phases']) == 27 and len(run['triads']) == 9
                and all(run['operation_ledger'][side].get(k, 0) == n for side in ('predebited', 'observed') for k, n in protocol['limits'].items())
                and run.get('actual_full160_videos') == 27 and run.get('actual_flow_steps') == 108 and run.get('actual_FFN_positions') == 4320,
                'Actual remaining9-source V2 completion required')
            old = next(p for p in plan['parents'] if p['kind'] == 'FAILED_V1_PREFIX' and p['part'] == proof['part'])
            base.require(all(protocol['failed_predecessor'][k] == old[k if k != 'launch_exit' else 'exit'] for k in ('protocol', 'RUN', 'launch_exit')),
                'Remaining9 source exclusion failed-parent binding differs')
        else:
            old = next(p for p in repair_run['parents'] if p['part'] == proof['part'])
            base.require(all(old[k] == proof[k] for k in ('protocol', 'RUN', 'exit')), 'CPU repair reused a different failed native parent')
            prefix.validate_parent(old, hf, verify_files=False)
            ex = base.read_bound(hf, proof['exit'])
            base.require(type(ex['child_returncode']) is int and ex['child_returncode'] == 1
                and protocol['id'] == 'E0_RAW_BF16_FULL160_VIDEO20_20261029_PART'+str(proof['part']), 'Failed V1 parent must remain actual exit1')
        base.require(protocol['source_cohort'] == plan['source_cohort']
            and protocol['prior_development_criteria'] == plan['prior_development_criteria']
            and run.get('id') == protocol['id'] and run.get('protocol_sha256') == proof['protocol']['sha256']
            and run.get('numeric_path') == 'BF16_ORIGINAL'
            and not run.get('assigned_GPU_resources', {}).get('violations')
            and not run['operation_ledger']['forbidden_attempts'], 'Original full-depth source/precision lineage differs')
        identity = {k: protocol[k] for k in ('numeric_path', 'factor_parent', 'models', 'runtime_overlay', 'runtime_versions')}
        base.require(common_identity is None or identity == common_identity, 'Paired generation numeric/factor identity differs across parts')
        common_identity = identity
        expected_cases = [c for c in cohort['sources'] if c['part'] == proof['part'] and (not complete or c['index'] >= 2)]
        base.require(protocol['cases'] == expected_cases, 'Generation source/image/prompt/exclusion protocol differs')
        for phase in run['phases']:
            key = phase['native_source_ordinal'], phase['mode']
            base.require(key not in phase_index, 'A native completed source/arm was repeated')
            if not complete:
                repaired_phase = repaired[key]
                base.require(repaired_phase['parent_remains_failed'] is True and repaired_phase['generation_receipt'] == phase['generation_receipt']
                    and repaired_phase['states'] == phase['states'] and repaired_phase['native_operand_transfer_evidence'] == phase['native_operand_transfer_evidence']
                    and repaired_phase['failed_native_parent_RUN'] == proof['RUN'] and repaired_phase['failed_native_parent_exit'] == proof['exit'], 'CPU repair native provenance differs')
                phase = dict(phase, core_metrics=repaired_phase['core_metrics'])
            phase_index[key] = phase; parent_sha[key] = proof['protocol']['sha256']; parent_protocols[key] = protocol
        parents.append(run)
    return phase_index, parent_sha, parent_protocols, parents


def verify_generation(plan, hf):
    """Follow four real native exits and one CPU repair to all60 exact clips."""
    cohort = base.read_bound(ROOT, plan['source_cohort']); by_source = {c['native_source_ordinal']: c for c in cohort['sources']}
    phase_index, parent_sha, parent_protocols, parents = generation_metadata(plan, hf)
    base.require(set(phase_index) == {(s, m) for s in by_source for m in MODES}, 'All60 paired generation phases required')
    video_paths = []
    for v in plan['videos']:
        key = v['source'], v['mode']; case = by_source[v['source']]; phase = phase_index[key]
        generation = base.read_bound(hf, v['generation'])
        registered = base.read_bound(hf, v['registered']) if v.get('registered') is not None else None
        base.require(phase['generation_receipt'] == v['generation'] and phase['core_metrics'] == v['core']
            and phase['native_operand_transfer_evidence'] == v['native_evidence'] and phase['states'] == v['states']
            and generation['status'] == phase['status'] == 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
            and generation.get('all160_FFN_covered') is True and generation['completed_native_flow_steps'] == 4
            and generation['native_counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 0, 'weight_quantizations': 0}
            and all(generation[k] == phase[k] == v[k] for k in ('source_id', 'image_sha256', 'prompt_sha256', 'mode', 'seed'))
            and generation['native_source_ordinal'] == v['source'] and generation['protocol_sha256'] == parent_sha[key]
            and generation['numeric_path'] == 'BF16_ORIGINAL'
            and v['image_sha256'] == case['image']['sha256'] and v['prompt'] == case['prompt']
            and hashlib.sha256(v['prompt'].encode()).hexdigest() == v['prompt_sha256'] == case['prompt_sha256']
            and generation['states'] == v['states'] and len(v['states']) == 10,
            'Original source/new conditioning/full-depth generation identity differs')
        path = base.bound(hf, v['video']); video_paths.append(path)
        base.require(v['video'] == base.clip_binding(generation['video'], hf) == base.clip_binding(phase['video'], hf)
            and generation['video']['frames'] == 81 and generation['video']['size'] == [832, 464]
            and generation['video']['fps'] == 16, 'Actual complete81frame video binding differs')
        if registered is not None:
            base.require(registered['decoded_frames'] == 81 and registered['decoded_size'] == [832, 464] and registered['decoded_fps'] == 16
                and Path(registered['video_path']).resolve() == path and registered['status'] == 'COMPLETE_NATIVE_GENERATION_ONLY'
                and registered['numeric_path'] == 'BF16_ORIGINAL' and registered['job']['run_id'] == generation['run_id'] == phase['run_id']
                and registered['job']['variant'] == v['mode'] and registered['job']['source_id'] == v['source_id']
                and registered['job']['seed'] == 42 and registered['job']['conditioning_id'] == generation['conditioning_id']
                and registered['source']['prompt'] == v['prompt'] and registered['source']['image_id'] == v['image_sha256']
                and registered['immutable_generation_receipt'] == v['generation'] and registered['native_ffn_sites'] == 160
                and registered['native_ffn_counters'] == generation['native_counter_delta'] and registered['scheduler_generator_reset'] is True
                and Path(registered['actual_initial_noise_file']).resolve() == (hf/v['states']['initial_noise']['path']).resolve(),
                'Actual registered source/seed/conditioning differs')
        else:
            base.require(case['index'] in (0, 1) and v['registration_identity'] == 'DERIVED_FROM_ORIGINAL_GEN_AND_FROZEN_CASE_NO_JOB_WAS_CREATED',
                'Only genuine failed-prefix GEN may have no historical registered job')
        evidence = base.read_bound(hf, v['native_evidence'])
        base.require(evidence.get('all160_FFN_sites') == [[s, b] for s in range(4) for b in range(40)], 'Actual160 native site evidence differs')
        core = base.read_bound(hf, v['core'])
        from ffnds_video_metrics import validate as validate_core
        base.require(not validate_core(core) and core['candidate']['sha256'] == v['video']['sha256']
            and core['identity']['source_id'] == v['image_sha256'] and core['identity']['seed'] == 42
            and core['identity']['expected_frames'] == 81 and core['identity']['input_prompt_digest'] == v['prompt_sha256']
            and core['identity']['protocol_sha256'] == parent_sha[key] and core['identity']['run_sha256'] == v['generation']['sha256'],
            'Original all81-frame CORE/new-prompt binding differs')
        if v['mode'] != 'dense':
            dense = phase_index[v['source'], 'dense']
            base.require(core['reference']['sha256'] == dense['video']['sha256']
                and core['reference_identity']['run_sha256'] == dense['generation_receipt']['sha256'],
                'Paired CORE Dense reference provenance differs')
    base.require(len(set(video_paths)) == len({(p.stat().st_dev, p.stat().st_ino) for p in video_paths}) == 60,
        'Aliased generation videos rejected')
    for source in by_source:
        phases = [phase_index[source, m] for m in MODES]
        for name in ('raw_noise_FP32', 'initial_noise'):
            actual = [{k: p['states'][name][k] for k in ('bytes', 'sha256', 'raw_tensor_sha256', 'shape', 'dtype')} for p in phases]
            base.require(actual[0] == actual[1] == actual[2], 'Actual same-source triad noise bytes differ')
    return parents


verify_import_proof, verify_environment = raw.verify_import_proof, raw.verify_environment


def verify_worker(plan):
    from worker_identity import require_worker, private_config_path
    require_worker(); base.require(base.sha(private_config_path()) == plan['execution_private_worker_config_sha256'], 'Exact current A100 worker profile differs')


def cpu_report(plan, digest):
    return {'id': ID, 'protocol_sha256': digest, 'status': CPU_STATUS,
        'sources_sha256': base.state_sha(plan['repo_sources']), 'import_environment_sha256': base.state_sha(plan['import_environment']),
        'reused_import_RUN_sha256': raw.IMPORT_RUN_SHA, 'parents': plan['parents'], 'source_cohort': plan['source_cohort'],
        'prefix_repair': plan['prefix_repair'],
        'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
        'new_scored_videos': 60, 'independent_image_clusters': 20, 'reused_core_reports': 60,
        'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False, 'model_calls': 0, 'GPU_calls': 0}


def preflight(protocol, hf, output):
    base.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 2, 'Fresh CUDA-hidden two-CPU byte worker required')
    base.precision_environment(); plan = base.read(protocol); validate(plan, True); verify_worker(plan)
    report = cpu_report(plan, base.sha(protocol)); report.update(status='STARTED_FRESH_60_VIDEO_QUALITY_BYTE_PROOF',
        byte_checks=[], predebited_bytes=0, actual_verified_bytes=0)
    base.require(not output.exists(), 'Fresh CPU proof output required'); base.write(output, report)
    started = time.monotonic()
    def timeout(*_): raise TimeoutError('Fixed quality CPU wall1200 exhausted; no retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        for root, binding in all_byte_bindings(plan):
            report['predebited_bytes'] += binding['bytes']
            path = base.bound(ROOT if root == 'REPO' else hf, binding)
            report['actual_verified_bytes'] += binding['bytes']
            report['byte_checks'].append({'root': root, 'binding': binding, 'stat': stat_record(path)})
        verify_generation(plan, hf); verify_import_proof(plan, hf); verify_environment(plan, hf)
        base.require(not {'torch', 'numpy'} & set(sys.modules), 'Numerical import in byte proof')
        report.update(cpu_report(plan, base.sha(protocol))); report['byte_checks_sha256'] = base.state_sha(report['byte_checks'])
    except BaseException as error:
        report.update(status='FAILED_QUALITY_BYTE_PROOF_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started; base.write(output, report)
    return report


def bind_completed_generations(plan, hf):
    """Bind four native parents and one actual CPU-only six-clip repair."""
    parents = []
    for kind, pattern in (('FAILED_V1_PREFIX', GENERATION_V1), ('COMPLETE_V2_REMAINING9', GENERATION)):
        for part in (0, 1):
            path = ROOT/pattern.format(part=part); protocol = base.read(path)
            folder = hf/protocol['output_HF_relative']
            parents.append({'kind': kind, 'part': part, 'protocol': base.binding(path, ROOT),
                'RUN': base.binding(folder/'RUN.json', hf), 'exit': base.binding(folder.parent/'EXIT_RECEIPT.json', hf)})
    plan['parents'] = parents
    folder = hf/PREFIX_OUTPUT
    plan['prefix_repair'] = {'protocol': base.binding(ROOT/PREFIX_PROTOCOL, ROOT),
        'RUN': base.binding(folder/'RUN.json', hf), 'exit': base.binding(folder.parent/'EXIT_RECEIPT.json', hf)}
    phases, _, protocols, _ = generation_metadata(plan, hf)
    base.require(len(phases) == 60, 'Exactly six retained and54 new genuine generations required')
    for video in plan['videos']:
        key = video['source'], video['mode']; phase = phases[key]; protocol = protocols[key]
        registered = None
        if video['source_index'] >= 2:
            index = protocol['run_ids'].index(phase['run_id'])
            registered = base.binding(hf/protocol['output_HF_relative']/f'job{index:06d}'/'RUN.json', hf)
        video.update(video=base.clip_binding(phase['video'], hf), generation=phase['generation_receipt'],
            registered=registered, core=phase['core_metrics'], native_evidence=phase['native_operand_transfer_evidence'], states=phase['states'])
    plan.update(binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING', status='DRAFT_BOUND_60_COMPLETED_NATIVE_VIDEOS')
    return plan


def make_draft(hf, *, bind_completed=False):
    old = base.read(ROOT/METRIC_PARENT); cohort = base.read(ROOT/COHORT)
    generation_protocols = [base.read(ROOT/GENERATION_V1.format(part=p)) for p in (0, 1)]
    base.require(generation_protocols[0]['prior_development_criteria'] == generation_protocols[1]['prior_development_criteria'],
        'Original generation prior criteria differ across parts')
    plan = {k: copy.deepcopy(old[k]) for k in ('environment', 'import_environment', 'hf_bindings',
        'directory_fd_audit_policy', 'activation_bootstrap_policy', 'directory_fd_filelock_probe_source', 'reused_import_proof')}
    plan.update(id=ID, numeric_path='BF16_ORIGINAL', status='DRAFT_GENERATION_ACTUAL_RECEIPTS_NOT_BOUND',
        assets_finalized=False, binding_status='NOT_BOUND_GENERATIONS_PENDING', automatic_retry=False,
        dimensions=list(DIMENSIONS), limits=LIMITS, costs=COSTS, resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES,
        policy=POLICY, source_cohort=base.binding(ROOT/COHORT, ROOT),
        prior_development_criteria=copy.deepcopy(generation_protocols[0]['prior_development_criteria']),
        execution_private_worker_config_sha256=generation_protocols[0]['execution_private_worker_config_sha256'],
        parents=[{'kind': kind, 'part': p, 'protocol': base.binding(ROOT/pattern.format(part=p), ROOT)
            if (ROOT/pattern.format(part=p)).is_file() else None, 'RUN': None, 'exit': None}
            for kind, pattern in (('FAILED_V1_PREFIX', GENERATION_V1), ('COMPLETE_V2_REMAINING9', GENERATION)) for p in (0, 1)],
        prefix_repair={'protocol': base.binding(ROOT/PREFIX_PROTOCOL, ROOT) if (ROOT/PREFIX_PROTOCOL).is_file() else None, 'RUN': None, 'exit': None},
        videos=[{**video_identity(case, mode), **{k: None for k in ('video', 'generation', 'registered', 'core', 'native_evidence', 'states')},
            'registration_identity': 'DERIVED_FROM_ORIGINAL_GEN_AND_FROZEN_CASE_NO_JOB_WAS_CREATED' if case['index'] < 2 else 'ACTUAL_COMPLETE_V2_REGISTERED_JOB',
            'native_parent_role': 'FAILED_V1_COMPLETE_PREFIX' if case['index'] < 2 else 'COMPLETE_V2_REMAINING9'}
            for case in cohort['sources'] for mode in MODES],
        images=[{'source': c['native_source_ordinal'], 'name': Path(c['image']['path']).name,
            'binding': {k: c['image'][k] for k in ('path', 'bytes', 'sha256')}} for c in cohort['sources']],
        output_HF_relative='ffnds-derived/e0_raw_bf16_video20_official_quality_20261029/attempt1/worker/results',
        estimation={'independent_image_clusters': 20, 'scored_videos': 60, 'video_axis_results': 420,
            'official_five_independent_samples': False, 'source_history': cohort['source_history'],
            'human_fullplay_review': 'PENDING', 'quality_accepted': False, 'latent_SSE_quality_ranking': 'NOT_USED'},
        import_reuse='EXACT_SUCCESSFUL_V4_EIGHT_MODULE_IMPORTS_ONLY_PROOF_PLUS_NEW_WORKER_ORIGIN_SOURCE_STAT_CHECKS',
        pending_freeze='Bind two failed V1 triples+two complete V2 triples+actual six-clip CPU CORE repair and all60 artifacts; fresh independent review; byte CPU exit; one GPU scope')
    # These closed CPU/failed-native proofs may be bound while V2 is running.
    # No live V2 RUN is accepted or hashed into this draft.
    for parent in plan['parents'][:2]:
        protocol = generation_protocols[parent['part']]; folder = hf/protocol['output_HF_relative']
        parent.update(RUN=base.binding(folder/'RUN.json', hf), exit=base.binding(folder.parent/'EXIT_RECEIPT.json', hf))
    repair_folder = hf/PREFIX_OUTPUT
    if plan['prefix_repair']['protocol'] is not None and (repair_folder/'RUN.json').is_file() and (repair_folder.parent/'EXIT_RECEIPT.json').is_file():
        proof = {'protocol': plan['prefix_repair']['protocol'], 'RUN': base.binding(repair_folder/'RUN.json', hf),
            'exit': base.binding(repair_folder.parent/'EXIT_RECEIPT.json', hf)}
        repair = base.read_bound(hf, proof['RUN'])
        base.exited(base.read_bound(hf, proof['exit']), proof['RUN'], proof['protocol']['sha256'], 'E0_RAW_BF16_VIDEO20_PREFIX_CORE_REPAIR_CPU_20261029')
        base.require(repair['status'] == 'COMPLETE_CPU_PREFIX_CORE_REPAIR_SIX_VALID_GENERATIONS_PARENTS_REMAIN_FAILED', 'Prefix CPU repair is incomplete')
        plan['prefix_repair'] = proof
        old_phases = {(p['native_source_ordinal'], p['mode']): p for p in repair['phases']}
        for video in plan['videos']:
            if video['source_index'] < 2:
                phase = old_phases[video['source'], video['mode']]
                video.update(video=phase['video'], generation=phase['generation_receipt'], core=phase['core_metrics'],
                    native_evidence=phase['native_operand_transfer_evidence'], states=phase['states'])
    paths = {b['path'] for b in old['repo_sources']} | REQUIRED | {METRIC_PARENT}
    plan['repo_sources'] = [base.binding(ROOT/p, ROOT) for p in sorted(paths)]
    if bind_completed: bind_completed_generations(plan, hf)
    validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--protocol', type=Path); parser.add_argument('--hf-root', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--out', type=Path); parser.add_argument('--bind-completed', action='store_true')
    args = parser.parse_args()
    if args.command == 'draft':
        base.require(args.out is not None and not args.out.exists(), 'Fresh draft path required')
        plan = make_draft(args.hf_root.resolve(), bind_completed=args.bind_completed)
        base.write(args.out, plan)
        print(json.dumps({'id': ID, 'binding_status': plan['binding_status'], 'model_calls_executed': 0}))
    elif args.command == 'inspect':
        plan = base.read(args.protocol); validate(plan)
        print(json.dumps({'id': ID, 'binding_status': plan['binding_status'], 'limits': LIMITS, 'costs': COSTS, 'model_calls_executed': 0}, indent=2))
    elif args.command == 'preflight':
        base.require(args.protocol is not None and args.out is not None, 'Protocol and fresh CPU RUN path required')
        report = preflight(args.protocol, args.hf_root.resolve(), args.out)
        print(json.dumps({'id': ID, 'status': report['status'], 'model_calls': 0}))
    else:
        base.require(args.protocol is not None, 'Frozen protocol required')
        report = run(args); print(json.dumps({'status': report['status'], 'completed': report['completed']}))


def authorize(plan, digest):
    validate(plan, True); active = base.read(ROOT/'research/ACTIVE_RESEARCH.json')
    user = active.get('user_continuation_authorization_20261027', {}); policy = active.get('resource_scheduling_policy_20261028', {})
    base.require(user.get('new_scope_registration_allowed') is True and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE'
        and policy.get('global_gpu_card_cap') is None and active.get('current_session_resource_constraint_20261028', {}).get('max_simultaneously_assigned_idle_GPUs') == 2,
        'Current session authority differs')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    base.require(len(scopes) == 1 and scopes[0].get('protocol_sha256') == digest
        and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and scopes[0].get('limits') == LIMITS and scopes[0].get('resources') == RESOURCES and scopes[0].get('max_gpu_cards') == 1,
        'Unique exact future quality scope required')
    return scopes[0]


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight'); base.require(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Exited independent byte CPU proof NOT_BOUND')
    report = base.read_bound(hf, proof['RUN'])
    base.require(all(report.get(k) == v for k, v in cpu_report(plan, digest).items())
        and 0 <= report.get('wall_seconds', -1) <= CPU_RESOURCES['wall_seconds']
        and report.get('byte_checks_sha256') == base.state_sha(report['byte_checks']), 'New quality CPU identity differs')
    base.exited(base.read_bound(hf, proof['launch_exit']), proof['RUN'], digest, ID)
    expected = all_byte_bindings(plan)
    base.require([(r['root'], r['binding']) for r in report['byte_checks']] == expected, 'CPU byte closure incomplete')
    for row in report['byte_checks']:
        path = (ROOT if row['root'] == 'REPO' else hf)/row['binding']['path']
        base.require(stat_record(path) == row['stat'], 'Byte-proved asset changed after actual CPU exit')
    return proof


def verify_assets(plan, hf):
    # require_cpu already checked every current stat against independently
    # rehashed bytes before any numerical imports. Only small lineage records
    # and their video hashes are revisited here; metric checkpoints stay stat-bound.
    verify_worker(plan); verify_generation(plan, hf); verify_import_proof(plan, hf)


def summarize(plan, hf, dimensions, observations):
    """Keep axes separate; report all paired source tails without result selection."""
    base.require(len(dimensions) == 8 and {e['dimension'] for e in dimensions} == set(DIMENSIONS), 'Exactly eight completed axes required')
    lookup = {str((hf/v['video']['path']).resolve()): v for v in plan['videos']}
    base.require(len(lookup) == 60, 'Sixty distinct scored videos required')
    scores = {(v['source'], v['mode']): {} for v in plan['videos']}
    for entry in dimensions:
        d = entry['dimension']; paths = [str((hf/v['video']['path']).resolve()) for v in rows_for_dimension(plan['videos'], d)]
        for row in base.check_result(entry['result'], paths, d):
            v = lookup[row['video_path']]; scores[v['source'], v['mode']][d] = row['video_results']
    flows = {}
    for observation in observations:
        base.require(observation['video_path'] in lookup, 'Unknown RAFT video'); v = lookup[observation['video_path']]
        key = v['source'], v['mode']; base.require(key not in flows, 'Duplicate RAFT clip'); base.validate_raft(observation); flows[key] = observation
    base.require(len(flows) == 60 and sum(len(v) for v in scores.values()) == 420, 'Exact420 video-axis/60 RAFT coverage incomplete')
    sources = []
    for dense in [v for v in plan['videos'] if v['mode'] == 'dense']:
        s = dense['source']
        sources.append({'source': s, 'source_id': dense['source_id'], 'image_sha256': dense['image_sha256'],
            'stratum': dense['stratum'], 'conditioning_identity': dense['conditioning_identity'], 'seed': 42,
            'scores': {m: scores[s, m] for m in MODES},
            'new_minus_Dense': {m: {d: float(scores[s, m][d])-float(scores[s, 'dense'][d]) for d in scores[s, 'dense']} for m in MODES[1:]},
            'continuous_RAFT': {m: base.validate_raft(flows[s, m]) for m in MODES},
            'paired_motion': {m: base.motion_comparison(flows[s, m], flows[s, 'dense']) for m in MODES[1:]}})
    aggregates = {}
    for d in DIMENSIONS:
        eligible = [s for s in sources if d in s['scores']['dense']]
        means = {m: statistics.fmean(float(s['scores'][m][d]) for s in eligible) for m in MODES}
        aggregates[d] = {'source_count': len(eligible), 'source_equal_means': means,
            'paired_differences': {m: {'mean': statistics.fmean(s['new_minus_Dense'][m][d] for s in eligible),
                'median': statistics.median(s['new_minus_Dense'][m][d] for s in eligible),
                'all_sources_sorted_by_signed_difference': sorted([{'source': s['source'], 'difference': s['new_minus_Dense'][m][d]}
                    for s in eligible], key=lambda x: (x['difference'], x['source']))} for m in MODES[1:]}}
    motion_means = {m: statistics.fmean(s['continuous_RAFT'][m]['mean_top5percent_flow_px_per_2frame_pair'] for s in sources) for m in MODES}
    motion = {}
    for mode in MODES[1:]:
        ratios = [{'source': s['source'], **s['paired_motion'][mode]['candidate_to_Dense_mean_ratio']} for s in sources]
        defined = [r for r in ratios if r['status'] == 'DEFINED']
        motion[mode] = {'ratio_of_source_equal_means': motion_means[mode]/motion_means['dense'] if motion_means['dense'] > 0 else None,
            'mean_of_defined_source_ratios': statistics.fmean(r['value'] for r in defined) if defined else None,
            'undefined_source_ratios': [r['source'] for r in ratios if r['status'] != 'DEFINED'],
            'all_defined_sources_sorted_by_ratio': sorted(defined, key=lambda r: (r['value'], r['source']))}
    return {'per_source': sources, 'per_axis': aggregates, 'source_equal_continuous_RAFT_means': motion_means,
        'continuous_motion_comparisons': motion, 'independent_image_clusters': 20, 'samples_per_source_per_arm': 1,
        'source_exposure': 'HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND', 'axes_kept_separate': True,
        'continuous_motion_identity': 'RAFT top5percent image flow, not semantic action/camera proof; official binary separately retained',
        'official_five_independent_samples': False, 'quality_accepted': False, 'policy': POLICY}


def run(args):
    base.precision_environment()
    base.require('torch' not in sys.modules and 'numpy' not in sys.modules,'Fresh isolated quality worker required')
    hf,protocol=args.hf_root.resolve(),args.protocol.resolve();plan,digest=base.read(protocol),base.sha(protocol)
    scope=authorize(plan,digest);cpu=require_cpu(scope,plan,digest,hf);proof=verify_import_proof(plan,hf)
    from worker_identity import require_worker
    require_worker()
    out=(hf/plan['output_HF_relative']).resolve()
    base.require(out.is_relative_to(hf) and out!=hf and not out.exists(),'Fresh confined quality attempt required; no retry')
    out.mkdir(parents=True);base.write(out/'ATTEMPT.json',{'id':ID,'protocol_sha256':digest,'automatic_retry':False})
    started=time.monotonic();monitor=None
    report={'id':ID,'status':'STARTED_RAW_BF16_VIDEO20_60_CLIPS_QUALITY','protocol_sha256':digest,
      'numeric_path':'BF16_ORIGINAL','prior_independent_CPU_preflight':cpu,'reused_successful_imports_CPU_preflight':proof,
      'producers':plan['parents'],'prefix_CPU_CORE_repair':plan['prefix_repair'],
      'six_genuine_videos_from_failed_native_parents':6,'new_complete_V2_native_videos':54,
      'core_reports_reused':[v['core'] for v in plan['videos']],
      'declared_outer_model_work_budgets':COSTS,'predebited_outer_model_work':{},
      'model_work_accounting':'Original frame/batch budgets; RAFT forwards and iterations observed exactly; other internal work UNKNOWN',
      'opaque_internal_model_forwards':'UNKNOWN','opaque_model_constructions':'UNKNOWN','opaque_checkpoint_loads':'UNKNOWN',
      'completed_dimensions':[],'RAFT_observations':[],'official_module_functions':[],
      'new_scored_videos':60,'independent_image_clusters':20,'official_five_independent_samples':False,
      'temporal_flickering':'NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED','camera_motion':'NOT_RUN_NOT_IN_SCOPE',
      'human_fullplay_review':'NOT_RUN','quality_accepted':False,'native_dir_fd_audit':{'denied_attempts':[]}}
    ledger=base.Ledger(report,limits=LIMITS,persist=lambda:base.write(out/'RUN.json',report))
    def check():
        authorize(plan,digest)
        if monitor:monitor.check()
        base.require(time.monotonic()-started<=RESOURCES['wall_seconds'],'Quality wall budget exhausted')
        base.require(shutil.disk_usage(out).free>=RESOURCES['free_disk_bytes'],'Quality free disk reserve exceeded')
        base.require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Quality new-output byte budget exceeded')
    def timeout(*_):raise TimeoutError('Fixed7200second video20 quality attempt exhausted; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds'])
    try:
        verify_assets(plan,hf);helper,v8=verify_environment(plan,hf);helper.two_cpus();check()
        with ledger('reused_core_reports',60):report['core_reuse_status']='BOUND_ALL81FRAME_REPORTS_NO_RECOMPUTATION'
        env=plan['environment'];base.scope_cache_environment(out)
        report['isolation']=base.configure_import_environment(plan,hf)
        helper.disable_local_ipv6_probe(hf/env['site'],v8)
        numpy=importlib.import_module('numpy');report['pre_Torch_actual_NumPy']=base.module_observation(numpy,'numpy',plan,hf)
        import torch
        report['actual_Torch_origin']=base.module_observation(torch,'torch',plan,hf)
        from startup_gpu_reservation import reserve_before_assets
        from assigned_gpu_resources import AssignedGPUResources
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        report['runtime_precision']={'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
          'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        base.require(torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Single assigned A100 required')
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir()
        for item in plan['repo_sources']:
            target=snapshot/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(protocol,snapshot/'protocol.json')
        common=importlib.import_module('e0_ub1_consumer_official_quality')
        base.require(Path(common.__file__).resolve()==ROOT/'tools/e0_ub1_consumer_official_quality.py','Frozen helper origin differs')
        binary=Path(shutil.which('nvidia-smi') or '').resolve()
        base.require(str(binary)==env['nvidia_smi']['path'] and base.sha(binary)==env['nvidia_smi']['sha256'],'GPU monitor executable differs')
        source=(hf/env['source_root']).resolve();sys.path.insert(0,str(source))
        from offline_i2v_metric_loaders import submodules
        assets=base.read(ROOT/env['asset_manifest']);modelbase=hf/assets['HF_relative_root'];device=torch.device('cuda:0')
        oldhub=torch.hub.load
        def no_download(*_args,**_kwargs):raise RuntimeError('Network/implicit metric download forbidden')
        torch.hub.load=common.make_local_hub_loader(oldhub,modelbase/'source/dino',{'facebookresearch/dino:main',str(modelbase/'source/dino')})
        def state_load(url,*_args,**kwargs):
            base.require(url=='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth','Unregistered metric checkpoint URL')
            return torch.load(modelbase/'dino_model/dino_vitbase16_pretrain.pth',map_location=kwargs.get('map_location'),weights_only=kwargs.get('weights_only',False))
        torch.hub.load_state_dict_from_url=state_load;torch.hub.download_url_to_file=no_download
        import socket
        socket.socket.connect=no_download;socket.create_connection=no_download
        import subprocess
        native_imports={'denied_attempts':[]};_,_,checked_font=helper.import_audit_guard(out,native_imports)
        subprocess.check_output=checked_font(subprocess.check_output,hf/env['site']/'matplotlib/font_manager.py')
        modules={}
        for dimension in DIMENSIONS:
            modules[dimension],observation=v4.metric_module(dimension,plan,hf)
            report['official_module_functions'].append(observation)
        report['actual_module_origins']=base.actual_import_origins(plan,hf)
        report['actual_distribution_versions']=helper.effective_distributions(base.read(ROOT/v8['lock_path']),hf/env['overlay'])
        report['native_metric_import_audit']=native_imports
        base.require(not native_imports['denied_attempts'],'Protected native import denied an operation')
        def denied(reason):
            def fail():report['native_dir_fd_audit']['denied_attempts'].append(reason);raise RuntimeError(reason)
            return fail
        audit=common.audit_scope(out,AssignedGPUResources.query.__code__,binary,os.environ['FFNDS_PHYSICAL_GPU'])
        sys.addaudithook(v4.v3.normalize_dir_fd_audit(audit,out,report['native_dir_fd_audit'],denied))
        work=out/'work';images=work/'vbench2_beta_i2v/data/crop/16-9';images.mkdir(parents=True)
        for image in plan['images']:(images/image['name']).symlink_to(base.bound(hf,image['binding']))
        os.chdir(work)
        for dimension in DIMENSIONS:
            check();selected=rows_for_dimension(plan['videos'],dimension)
            expected_paths=[str(base.bound(hf,x['video'])) for x in selected]
            base.require(len(selected)==COUNTS[dimension],'Sixty-clip scene-aware dimension dispatch differs')
            inp=out/(dimension+'_INPUT.json')
            base.write(inp,[{'prompt_en':x['prompt'],'dimension':[dimension],'image_name':x['image_name'],'video_list':[path]} for x,path in zip(selected,expected_paths)])
            report['active_dimension']=dimension
            base.require(dimension not in report['predebited_outer_model_work'],'Duplicate dimension predebit')
            report['predebited_outer_model_work'][dimension]=COSTS[dimension].copy();base.write(out/'RUN.json',report)
            module=modules[dimension]
            base.require(Path(module.__file__).resolve()==source/('vbench2_beta_i2v' if dimension.startswith('i2v_') else 'vbench')/(dimension+'.py'),'Unexpected official module origin')
            if dimension=='i2v_background':
                original_dream=module.dreamsim
                def dream(*args_,**kwargs_):
                    kwargs_.update(cache_dir=str(modelbase/'dreamsim'),device=device);return original_dream(*args_,**kwargs_)
                module.dreamsim=dream
            try:
                # Both the outer call and every intended per-video result are
                # charged before invoking the original evaluator. Failures keep
                # this attempted work and never get a second invocation.
                with ledger('official_dimension_calls'),ledger('new_video_dimension_results',len(selected)):
                    if dimension=='dynamic_degree':
                        with base.raft_observer(module,report['RAFT_observations'],ledger,expected_paths),torch.inference_mode():
                            result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                    elif dimension=='i2v_background':
                        from peft import PeftModel,LoraConfig
                        adapter=modelbase/'dreamsim/ensemble_lora'
                        adapter_sha=next(x['sha256'] for x in plan['hf_bindings'] if x['path'].endswith('/dreamsim/ensemble_lora/adapter_config.json'))
                        with common.dreamsim_adapter_compat(PeftModel,LoraConfig,adapter,adapter_sha),torch.inference_mode():
                            result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                        report['dreamsim_adapter_compatibility']={'config_sha256':adapter_sha,'original_five_lora_keys_only':True,'inactive_extensions_verified':True,'checkpoint_bytes_changed':False}
                    else:
                        with torch.inference_mode():result=getattr(module,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
                    torch.cuda.synchronize();result=common.plain(result)
                    base.check_result(result,expected_paths,dimension)
                    entry={'dimension':dimension,'result':result,'input_sha256':base.sha(inp)}
                    if dimension=='dynamic_degree':
                        observations=report['RAFT_observations']
                        base.require(len(observations)==60 and {x['video_path'] for x in observations}==set(expected_paths),'Sixty complete raw RAFT observations required')
                        for row in observations:base.validate_raft(row)
                        base.require(report['completed']['RAFT_forwards']==2400 and report['completed']['RAFT_iterations']==48000,'Actual2400/48000 RAFT work differs')
                        entry['RAFT_observations']=copy.deepcopy(observations)
                    report['completed_dimensions'].append(entry)
                gc.collect();torch.cuda.empty_cache()
                base.require(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'Quality GPU reservation budget exceeded')
            finally:
                if dimension=='i2v_background':module.dreamsim=original_dream
        ledger.finish();base.require(report['predebited_outer_model_work']==COSTS,'Incomplete eight-axis outer budgets')
        base.require(not report['native_dir_fd_audit']['denied_attempts'],'Native directory-FD audit violation')
        report['source_summary']=summarize(plan,hf,report['completed_dimensions'],report['RAFT_observations'])
        report.update(active_dimension=None,status='COMPLETE_RAW_BF16_VIDEO20_60_CLIPS_EIGHT_DIMENSIONS_QUALITY_NOT_ACCEPTED',peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
        check()
    except BaseException as error:
        report.update(status='FAILED_RAW_QUALITY_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        original_error=sys.exc_info()[1];signal.alarm(0);report['wall_seconds']=time.monotonic()-started
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report()
                base.require(not report['assigned_GPU_resources'].get('violations'),'Quality GPU monitor violation')
            except BaseException as error:
                report.update(status='FAILED_GPU_MONITOR_PRESERVED_NO_RETRY',monitor_error=str(error));base.write(out/'RUN.json',report)
                if original_error is None:raise
        base.write(out/'RUN.json',report)
    return report


if __name__ == '__main__': main()
