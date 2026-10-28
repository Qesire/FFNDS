"""Passive original-BF16 full-depth 21-tau trajectories at eight fixed xy anchors.

This is a new acquisition, not a retry or an edited historical collector.
Only private legacy namespaces are reused. New images and human-reviewed source
history remain NOT_BOUND until the coordinator freezes the complete cohort.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))


def private_execution():
    spec = importlib.util.spec_from_file_location('_ffnds_full21tau_execution', ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


execution = private_execution()
legacy, native = execution.legacy, execution.native
from raw_bf16_assigned_gpu_monitor_v4 import MONITOR, startup as reserved_startup

BASE_ID = 'E0_RAW_BF16_FULL21TAU_NEW10_DENSE_ACQUISITION_20261029'
BASE_PROTOCOL = 'config/e0_original_bf16_full_depth_collect_ord240_20261029_v4.json'
SOURCE_ID = 'E0_RAW_BF16_FRESH10_SOURCE_ARCHIVE_CPU_V2_20261029'
COHORT_SCHEMA = 'ffnds.raw_bf16.full21tau.source_bank.v1'
M, D, H, ROWS = 31668, 5120, 13824, 168
GRID = [21, 29, 52]
SHAPE = native.VIDEO['latent_shape']
# Four disjoint 2x2 geometric strata, each with a horizontal adjacent pair.
# The lower integer midpoint is used in each half-open native spatial bin.
ANCHORS = [(6, 12), (6, 13), (6, 38), (6, 39), (21, 12), (21, 13), (21, 38), (21, 39)]
NEIGHBORS = [[0, 1], [2, 3], [4, 5], [6, 7]]
ROW_SHAPES = {'V1': [ROWS, D], 'h': [ROWS, H], 'Y': [ROWS, D], 'r': [ROWS, D], 'u': [ROWS, D],
              'gate': [D], 'shift': [D], 'scale': [D], 'rows': [ROWS]}
PACKET_BYTES = ROWS*(4*D+H)*2+3*D*2+ROWS*8
STATE_BYTES = math.prod(SHAPE)*(5*4+5*2)
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
             'host_available_min_bytes': 128 << 30, 'wall_seconds': 1800, 'outer_seconds': 2100,
             'new_HF_bytes': 16 << 30, 'free_disk_min_bytes': 20 << 30,
             'repository_GPU_policy': 'DYNAMIC_MAX_ACTUALLY_CALLABLE', 'automatic_retry': False}
DESIGN = {'grid_tau_y_x': GRID, 'anchors_y_x': [list(x) for x in ANCHORS], 'horizontal_neighbor_anchor_ids': NEIGHBORS,
          'anchor_selection': 'FOUR_2X2_GEOMETRIC_STRATA_INTEGER_MIDPOINT_AND_X_PLUS_ONE_BEFORE_TENSOR_VALUES',
          'row_order': 'SORTED_NATIVE_ROW_ID_EQUALS_TAU_MAJOR_ANCHOR_ID_MINOR',
          'coordinate_identity': 'IDENTICAL_XY_TAU_FOR_ALL_SOURCE_STEP_BLOCK',
          'latent_times_per_anchor': 21, 'anchors': 8, 'rows_per_packet': ROWS,
          'complete_21tau_trajectories_per_packet': 8, 'temporal_pairs_per_packet': 1680,
          'horizontal_neighbor_pairs_per_packet': 84, 'steps': 4, 'blocks': 40,
          'numeric_path': 'BF16_ORIGINAL_NO_FP4_NO_QUANTIZATION_NO_MODEL_REPLACEMENT',
          'state': 'DENSE_TEACHER_ONLY', 'tensors': list(ROW_SHAPES), 'preactivation_a': 'NOT_COLLECTED',
          'motion_correspondence': 'NOT_MEASURED_FIXED_XY_IS_NOT_MOTION_ALIGNMENT',
          'source_role': 'TEN_ADDITIONAL_IMAGE_CLUSTERS_WITH_AUDITED_HISTORY_EXPLICIT_REVIEW_NOT_BLIND_CONFIRMATION',
          'neutral': 'PART0_REPLAY_OLD_DEV894_TEN_STATE_BYTES_THEN_FIVE_NEW_PART1_REUSES_CLOSED_PART0_NEUTRAL',
          'fitting': 0, 'decompositions': 0, 'candidates': 0, 'official_quality': 0, 'automatic_retry': False}
REPAIR = copy.deepcopy(execution.REPAIR)
REQUIRED = {'tools/e0_raw_bf16_full_trajectory_collect.py', 'tests/test_e0_raw_bf16_full_trajectory_collect.py',
            'tools/raw_bf16_assigned_gpu_monitor_v4.py', 'tools/e0_original_bf16_full_depth_collect_v2.py',
            'tools/e0_original_bf16_full_depth_collect.py', 'tools/e0_original_bf16_videos.py'}
FROZEN_SOURCES = {
    'tools/e0_original_bf16_full_depth_collect.py': '3079c7ce5cc40747dbedcb64419026d708973b1e723e558bf292ebaa7d8e4dcf',
    'tools/e0_original_bf16_full_depth_collect_v2.py': 'e9bc695f5db69f900869c0efaaa47a815f8adab4ef247147080a820a8abbe075'}


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two full-trajectory parts')
    return BASE_ID+'_PART'+str(part)


def output_relative(part):
    return f'ffnds-derived/e0_raw_bf16_full21tau_new10_20261029/part{part}/attempt1/worker/results'


def anchor_rows():
    return sorted(t*29*52+y*52+x for t in range(21) for y, x in ANCHORS)


def drawings():
    rows = anchor_rows()
    return {str(s): {str(b): list(rows) for b in range(40)} for s in range(4)}


def row_geometry(rows):
    native.need(rows == anchor_rows() and len(rows) == len(set(rows)) == ROWS, 'Fixed full21tau anchor rows differ')
    return {'grid_tau_y_x': GRID, 'anchors_y_x': [list(x) for x in ANCHORS],
            'horizontal_neighbor_anchor_ids': NEIGHBORS, 'row_anchor_ids': [i for _ in range(21) for i in range(8)],
            'row_taus': [t for t in range(21) for _ in range(8)], 'trajectory_shape_tau_anchor': [21, 8],
            'row_order': DESIGN['row_order'], 'motion_correspondence': DESIGN['motion_correspondence']}


def run_id(case):
    return 'raw_full21tau_ord'+str(case['native_source_ordinal'])+'_seed42_dense'


def limits(part):
    total, new = (6, 5) if part == 0 else (5, 5)
    native.need(type(part) is int and part in (0, 1), 'Exact trajectory part')
    old = execution.limits()
    setup = {'original_expert_loads', 'raw_Default_MM_loads', 'buffer_Default_MM_loads', 'T5_raw_Default_MM_loads', 'T5_buffer_Default_MM_loads'}
    new_only = {'video_generations', 'VAE_decodes', 'core_CPU_reports', 'video_geometry_checks', 'source_registration_writes'}
    neutral = {'native_prefix_replays_no_decode': 1 if part == 0 else 0, 'neutral_state_byte_checks': 10 if part == 0 else 0}
    result = {}
    for key, value in old.items():
        if key in setup:
            result[key] = value
        elif key in new_only:
            result[key] = new
        elif key in neutral:
            result[key] = neutral[key]
        else:
            native.need(value % 4 == 0, 'Frozen per-source capture counter differs')
            result[key] = value//4*total
    result['row_index_upload_bytes'] = total*160*ROWS*8
    result['packet_tensor_D2H_bytes'] = total*160*PACKET_BYTES
    return result


# Change private execution geometry only. The full-M model calls and native
# OriginalGuard/NativeTap/CollectorTap implementation retain their exact source.
legacy.ROWS, legacy.ROW_SHAPES, legacy.PACKET_BYTES = ROWS, ROW_SHAPES, PACKET_BYTES
legacy.RESOURCES = RESOURCES
CURRENT_PART = None
legacy.limits = lambda: limits(CURRENT_PART)
original_packet_check = legacy.check_packet_metadata


def check_packet_metadata(metadata, rows):
    original_packet_check(metadata, rows)
    row_geometry(rows)


def byte_budget(part):
    count = 6 if part == 0 else 5
    payload = count*160*PACKET_BYTES
    # Exact tensor bodies plus conservative named format/video/metadata bounds.
    return {'source_count_including_old_neutral': count, 'new_image_clusters': 5,
            'row_tensor_payload_bytes': payload, 'state_tensor_payload_bytes': count*STATE_BYTES,
            'packet_header_max_bytes': count*160*16384, 'state_header_max_bytes': count*10*16384,
            'packet_metadata_max_bytes': count*160*32768,
            'video_file_max_bytes': 5*REPAIR['maximum_single_video_file_bytes'],
            'small_JSON_journal_source_snapshot_reserve_bytes': 256 << 20,
            'total_bound_before_actual_frozen_source_snapshot_bytes': payload+count*STATE_BYTES
            +count*160*(16384+32768)+count*10*16384+5*REPAIR['maximum_single_video_file_bytes']+(256 << 20),
            'per_scope_HF_limit_bytes': RESOURCES['new_HF_bytes']}


def validate_cohort(cohort, hf, *, verify_images=False, frozen=False):
    native.need(cohort.get('schema') == COHORT_SCHEMA and isinstance(cohort.get('sources'), list), 'New complete cohort NOT_BOUND')
    rows = cohort['sources']
    native.need(len(rows) == 10 and len({r['image']['sha256'] for r in rows}) == 10
        and len({r['source_id'] for r in rows}) == len({r['native_source_ordinal'] for r in rows}) == 10
        and sorted(r['cohort_index'] for r in rows) == list(range(10))
        and all(sum(r['part'] == p for r in rows) == 5 for p in (0, 1))
        and {k: sum(r['scene_stratum'] == k for r in rows) for k in ('subject', 'background', 'camera')}
        == {'subject': 5, 'background': 2, 'camera': 3}, 'Frozen ten image-cluster allocation differs')
    for item in rows:
        native.need(item['role'] == 'NEW_DEVELOPMENT' and item['collector_role'] == 'observation'
            and item['prompt_sha256'] == hashlib.sha256(item['prompt'].encode()).hexdigest()
            and type(item['native_source_ordinal']) is int and item['native_source_ordinal'] >= 0
            and item['native_source_ordinal'] != 894 and item['image'].get('root') == 'hf'
            and item['history'] and isinstance(item['exposure_history'], dict)
            and 'consumer20_index' not in item, 'New source identity/history differs')
        if verify_images:
            native.bound(hf, item['image'])
    if frozen:
        native.need(cohort.get('status') == 'REVIEWED_READY_FOR_RAW_DENSE_ACQUISITION'
            and isinstance(cohort.get('source_audit'), dict) and isinstance(cohort.get('source_review'), dict)
            and cohort['source_review'].get('reviewer_kind') in ('ROOT_AGENT', 'HUMAN')
            and cohort['source_review'].get('status') == 'REVIEWED_NEW_DEVELOPMENT',
            'Provisional/source history/near-duplicate review not yet closed')
    return True


def validate(plan, frozen=False):
    part = plan.get('part')
    native.need(plan.get('id') == scope_id(part) and plan.get('numeric_path') == 'BF16_ORIGINAL'
        and plan.get('limits') == limits(part) and plan.get('resources') == RESOURCES and plan.get('design') == DESIGN
        and plan.get('repair') == REPAIR and plan.get('runtime_overlay') == native.recipe()
        and plan.get('output_HF_relative') == output_relative(part) and plan.get('automatic_retry') is False,
        'Fixed full21tau acquisition scope differs')
    native.need(plan['byte_budget'] == byte_budget(part)
        and plan['byte_budget']['total_bound_before_actual_frozen_source_snapshot_bytes'] < RESOURCES['new_HF_bytes'],
        'Frozen full-trajectory output budget differs')
    for path, sha in FROZEN_SOURCES.items():
        native.need(native.sha(ROOT/path) == sha, 'Frozen source changed: '+path)
    paths = [x['path'] for x in plan['repo_sources']]
    native.need(REQUIRED <= set(paths) and len(paths) == len(set(paths)), 'Full trajectory source closure incomplete')
    if plan.get('cases') is not None:
        cases = plan['cases']
        native.need(len(cases) == (6 if part == 0 else 5) and len({c['image_sha256'] for c in cases}) == len(cases)
            and len({c['native_source_ordinal'] for c in cases}) == len(cases)
            and plan['run_ids'] == [run_id(c) for c in cases]
            and all(c['rows'] == drawings() and c['row_geometry'] == row_geometry(anchor_rows()) for c in cases)
            and [c['role'] for c in cases] == (['development'] if part == 0 else [])+['observation']*5,
            'Full-trajectory source/rows/role allocation differs')
        if part == 0:
            native.need(cases[0]['native_source_ordinal'] == 894 and cases[0]['baseline_source_index'] == 0,
                'Exact old dev894 native byte neutral required')
    if frozen:
        native.need(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION'
            and isinstance(plan.get('source_cohort'), dict) and plan.get('cases') is not None
            and isinstance(plan.get('source_acquisition_proof'), dict), 'New complete cohort/source proof NOT_BOUND')
        native.need(part == 0 or isinstance(plan.get('prior_part0_neutral'), dict), 'Part1 closed new-observer neutral NOT_BOUND')


def verify_source_cohort(plan, hf, *, verify_images=False):
    """Verify only closed source provenance/images, never raw model payloads."""
    cohort = native.read(native.bound(ROOT, plan['source_cohort']))
    validate_cohort(cohort, hf, verify_images=verify_images, frozen=True)
    native.need(isinstance(cohort['source_review'].get('document'), dict), 'Actual root source review document NOT_BOUND')
    native.bound(ROOT, cohort['source_review']['document'])
    selected = sorted((s for s in cohort['sources'] if s['part'] == plan['part']), key=lambda s: s['cohort_index'])
    actual = [c for c in plan['cases'] if c['role'] == 'observation']
    for source, case in zip(selected, actual, strict=True):
        native.need(all(case[k] == source[k] for k in source if k not in ('role', 'collector_role'))
            and case['role'] == source['collector_role'] and case['evidence_role'] == source['role']
            and case['image_sha256'] == source['image']['sha256'], 'Frozen original new source differs')
    # The actual source-extraction/review proof is required independently of the
    # source bank. Its exact schema is bound once that producer actually closes.
    source_proof = plan['source_acquisition_proof']
    source_protocol = native.read(native.bound(ROOT, source_proof['protocol']))
    source_run = native.read(native.bound(hf, source_proof['RUN']))
    source_exit_binding = source_proof['launch_exit'] if 'launch_exit' in source_proof else source_proof['exit']
    source_exit = native.read(native.bound(hf, source_exit_binding))
    legacy.exited(source_exit, source_proof['RUN'], source_proof['protocol']['sha256'], source_protocol['id'])
    native.need(source_run['id'] == source_protocol['id'] == SOURCE_ID and source_run['protocol_sha256'] == source_proof['protocol']['sha256']
        and source_run['model_calls'] == source_run['GPU_calls'] == 0
        and source_run['status'] == 'COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_HUMAN_AND_HISTORY_REVIEW'
        and source_proof == cohort['source_audit'], 'Actual source-extraction CPU proof differs')
    extracted = source_run['sources']
    native.need(len(extracted) == 10 and len({s['cohort_index'] for s in extracted}) == 10, 'Original ten extraction outputs incomplete')
    protected = ('source_id', 'native_source_ordinal', 'cohort_index', 'part', 'role', 'collector_role',
                 'dimension', 'scene_stratum', 'image_type', 'original_prompt', 'original_prompt_sha256',
                 'prompt', 'prompt_sha256', 'archive_member')
    for item in cohort['sources']:
        raw = next(s for s in extracted if s['cohort_index'] == item['cohort_index'])
        native.need(all(item[k] == raw[k] for k in protected)
            and all(item['image'][k] == raw['image'][k] for k in ('path', 'bytes', 'sha256')),
            'Reviewed bank changed actual original image/prompt/source conditioning')
    return cohort


def verify_lineage(plan, hf, *, source_bytes=False):
    previous = native.read(native.bound(ROOT, plan['parent']['protocol']))
    old_run = native.read(native.bound(hf, plan['parent']['RUN']))
    old_exit = native.read(native.bound(hf, plan['parent']['exit']))
    native.validate(previous, True)
    legacy.exited(old_exit, plan['parent']['RUN'], plan['parent']['protocol']['sha256'], previous['id'])
    native.need(previous['id'] == native.ID and old_run['status'] == 'COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING'
        and old_run['protocol_sha256'] == plan['parent']['protocol']['sha256']
        and all(plan[k] == previous[k] for k in ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers', 'private_worker_config_sha256')),
        'Original BF16 checkpoint/environment lineage differs')
    execution.require_execution_master(plan, hf)
    cohort = verify_source_cohort(plan, hf, verify_images=source_bytes)
    if plan['part'] == 0:
        case = plan['cases'][0]
        reference = next(p for p in old_run['phases'] if p['source_index'] == 0)
        original = previous['cases'][0]
        native.need(case['controls'] == reference['states'] and case['reused_video'] == reference['video']
            and all(case[k] == original[k] for k in ('source_id', 'image', 'image_sha256', 'prompt', 'prompt_sha256', 'conditioning_id')),
            'Old Dense ten-state neutral conditioning differs')
    else:
        verify_part0_neutral(plan, hf)
    return cohort


def verify_part0_neutral(plan, hf):
    proof = plan['prior_part0_neutral']
    other = native.read(native.bound(ROOT, proof['protocol']))
    validate(other, True)
    report = native.read(native.bound(hf, proof['RUN']))
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], proof['protocol']['sha256'], scope_id(0))
    native.need(other['part'] == 0 and other['source_cohort'] == plan['source_cohort']
        and report['status'] == 'COMPLETE_RAW_BF16_FULL21TAU_NEW5_PART_WITH_REGISTERED_DENSE_VIDEOS'
        and report['local_development_neutral_passed'] is True, 'Part1 needs closed successful same-cohort part0')
    native.need(not report['operation_ledger']['forbidden_attempts'] and all(report['operation_ledger'][side].get(k, 0) == n
        for side in ('predebited', 'observed') for k, n in limits(0).items()), 'Closed part0 operation ledger incomplete')
    phase = next(p for p in report['phases'] if p['role'] == 'development')
    gen = native.read(native.bound(hf, proof['GENERATION']))
    native.need(phase['generation_receipt'] == proof['GENERATION'] and gen['id'] == scope_id(0)
        and gen['status'] == 'COMPLETE_RAW_BF16_FULL21TAU_TEACHER_STATE_FULL160' and gen['role'] == 'development'
        and gen['protocol_sha256'] == proof['protocol']['sha256']
        and gen['native_source_ordinal'] == 894 and gen['all_four_original_flow_and_post_bytes_passed'] is True
        and gen['completed_native_flow_steps'] == 4 and gen['native_ffn_sites'] == 160
        and gen['states'] == proof['states'] and len(gen['states']) == 10
        and gen['row_geometry'] == row_geometry(anchor_rows()), 'Prior full21tau new-observer neutral GEN differs')
    for name, value in proof['states'].items():
        expected = other['cases'][0]['controls'][name]
        native.need(all(value[k] == expected[k] for k in ('raw_tensor_sha256', 'shape', 'dtype')), 'Prior neutral state bytes differ')
        native.bound(hf, value)
    return proof


def cpu_identity(plan, sha):
    return {'id': plan['id'], 'protocol_sha256': sha, 'part': plan['part'],
            'status': 'PASS_FULL21TAU_NEW_COHORT_ORIGINAL_IMAGES_RAW_MASTER_STAT_SOURCE_ROWS_NO_NUMERIC_IMPORTS',
            'source_closure_sha256': native.canonical(plan['repo_sources']), 'source_cohort': plan['source_cohort'],
            'source_acquisition_proof': plan['source_acquisition_proof'], 'row_geometry': row_geometry(anchor_rows()),
            'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'],
            'execution_private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
            'prior_part0_neutral': plan.get('prior_part0_neutral'), 'native_cases': len(plan['cases']),
            'new_image_clusters': 5, 'all_CUDA_devices_hidden': True, 'Torch_or_NumPy_imported': False,
            'model_calls': 0, 'GPU_calls': 0, 'raw_master_payload_bytes_rehashed': 0}


def preflight(args):
    start = time.monotonic()
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA four-CPU stdlib proof required')
    plan = native.read(args.protocol)
    validate(plan, True)
    verify_lineage(plan, args.hf_root, source_bytes=True)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    if plan['part'] == 0:
        for item in plan['cases'][0]['controls'].values():
            native.bound(args.hf_root, item)
        native.bound(args.hf_root, plan['cases'][0]['image'])
    native.model_metadata(plan)
    native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    result = cpu_identity(plan, native.sha(args.protocol))
    result['wall_seconds'] = time.monotonic()-start
    native.need(result['wall_seconds'] <= 300 and not {'numpy', 'torch'} & set(sys.modules), 'CPU source/stat/import bound exceeded')
    return result


def authorize(plan, sha):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    policy = active.get('resource_scheduling_policy_20261028', {})
    native.need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None,
        'Current registered research authorization differs')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(scopes) == 1 and scopes[0].get('protocol_sha256') == sha
        and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and scopes[0].get('limits') == limits(plan['part']) and scopes[0].get('resources') == RESOURCES
        and scopes[0].get('run_ids') == plan['run_ids'] and scopes[0].get('max_gpu_cards') == 1,
        'Unique bounded full21tau scope not registered')
    return scopes[0]


def require_cpu(scope, plan, sha, hf):
    proof = scope.get('cpu_preflight')
    native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Independent CPU proof NOT_BOUND')
    value = native.read(native.bound(hf, proof['RUN']))
    native.need(all(value.get(k) == v for k, v in cpu_identity(plan, sha).items()) and 0 <= value['wall_seconds'] <= 300,
        'Independent CPU source/row/stat proof differs')
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], sha, plan['id'])
    return proof


def registration(plan, hf):
    from ffnds_cohort_preflight import audit
    sources, samples = [], []
    for case in (c for c in plan['cases'] if c['role'] == 'observation'):
        rid = run_id(case)
        exposure = case['exposure_history']
        sources.append({'source_id': case['source_id'], 'image_id': case['image_sha256'], 'prompt_id': case['prompt_sha256'],
            'prompt': case['prompt'], 'image_path': str((hf/case['image']['path']).resolve()), 'image_name': Path(case['image']['path']).name,
            'conditioning_id': case['conditioning_id'], 'role': 'DEVELOPMENT', 'exposure_history': exposure})
        samples.append({'source_id': case['source_id'], 'variant': 'dense', 'run_id': rid, 'asset_id': 'video:'+rid,
            'seed': 42, 'sample_index': 0, 'conditioning_id': case['conditioning_id'], 'initial_noise_id': plan['id']+':'+case['source_id'],
            'scheduler_id': 'Wan22StepDistillScheduler:shift5:sigma1,.9375,.8333333134651184,.625:high,high,low,low',
            **{k: native.VIDEO[k] for k in ('frames', 'fps', 'width', 'height')}, 'status': 'PLANNED'})
    result = {'schema': 'ffnds.cohort.v1', 'mode': 'CUSTOM_PAIRED', 'experiment': plan['id'],
              'phase': 'REGISTERED_REAL_GENERATION_PLAN', 'sources': sources, 'samples': samples,
              'calibration': [], 'required_variants': ['dense']}
    check = audit(result)
    native.need(not check['errors'] and check['summary']['independent_image_clusters'] == 5, 'Five Dense acquisition registrations invalid')
    return result


# Explicit fork of the frozen V1 native_phase: only geometry/source/schema seams
# differ; the actual full-M FFN, residual write and continuous runner are intact.
def native_phase(torch,pipe,guard,plan,case,out,hf,digest,report,persist,check):
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    rid=run_id(case);folder=out/rid;folder.mkdir()
    phase={'id':plan['id'],'protocol_sha256':digest,'run_id':rid,'numeric_path':'BF16_ORIGINAL','status':'STARTED_NO_RETRY',
      'source_index':case['cohort_index'],'native_source_ordinal':case['native_source_ordinal'],'source_id':case['source_id'],
      'image_sha256':case['image_sha256'],'prompt_sha256':case['prompt_sha256'],'conditioning_id':case['conditioning_id'],
      'role':case['role'],'evidence_role':case['evidence_role'],'history':case['history'],'row_geometry':row_geometry(anchor_rows()),'mode':'dense','source_state':'teacher_input','seed':42,'states':{},'packets':[],
      'quality_or_speed_acceptance':False,'observer_cost_is_included_in_capture_wall':True}
    report['phases'].append(phase);persist();ledger=guard.ledger
    def save(value,name,key):
        native.need(name not in phase['states'],'Duplicate native state output')
        expected=torch.float32 if key in ('raw_FP32_noise_files','native_FP32_flow_files') else torch.bfloat16
        def output():
            native.need(value.dtype==expected and list(value.shape)==SHAPE,'Actual original state geometry/dtype differs')
            ledger.call(rid,{'state_output_byte_reservations':1},legacy.reserve_output_bytes,out,value.numel()*value.element_size())
            cpu=ledger.call(rid,{'state_tensor_D2H':1,'state_tensor_D2H_bytes':value.numel()*value.element_size()},lambda:value.detach().cpu().contiguous())
            native.need(bool(torch.isfinite(cpu).all()),'Nonfinite raw state')
            path=folder/(name+'.safetensors');native.need(not path.exists(),'State overwrite forbidden')
            save_file({'flow' if key=='native_FP32_flow_files' else 'latents':cpu},str(path))
            item={'root':'hf',**native.binding(path,hf),'raw_tensor_sha256':tensor_sha(cpu),'shape':list(cpu.shape),'dtype':str(cpu.dtype).removeprefix('torch.')}
            native.need(item['bytes']-value.numel()*value.element_size()<=16384,'State safetensor header exceeded frozen bound')
            if case['role']=='development':
                def neutral():
                    ref=case['controls'][name]
                    native.need(item['dtype']==ref['dtype'] and item['shape']==ref['shape'] and item['raw_tensor_sha256']==ref['raw_tensor_sha256'],'Old raw flow/post/noise neutral mismatch')
                ledger.call(rid,{'neutral_state_byte_checks':1},neutral)
            phase['states'][name]=item;persist();return item
        return ledger.call(rid,{key:1},output)
    index=ledger.call(rid,{'row_index_uploads':1,'row_index_upload_bytes':4*40*ROWS*8},torch.tensor,
      [[case['rows'][str(s)][str(b)] for b in range(40)] for s in range(4)],dtype=torch.int64,device='cuda')
    def packet(p,grid):
        native.need(grid==GRID,'Actual native tau/y/x axes differ from frozen full-trajectory geometry')
        native.need(set(p['values'])==set(ROW_SHAPES)-{'rows'},'Missing actual captured operand')
        ledger.call(rid,{'packet_output_byte_reservations':1},legacy.reserve_output_bytes,out,PACKET_BYTES)
        values={**p['values'],'rows':p['rows']};cpu={}
        for k,value in values.items():
            cpu[k]=ledger.call(rid,{'packet_tensor_D2H':1,'packet_tensor_D2H_bytes':value.numel()*value.element_size()},lambda value=value:value.detach().cpu().contiguous())
        row_ids=case['rows'][str(p['step'])][str(p['block'])]
        def validate_payload():
            native.need(all(bool(torch.isfinite(v).all()) for v in cpu.values()),'Nonfinite raw row operand')
            meta={k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.'),'raw_tensor_sha256':tensor_sha(v)} for k,v in cpu.items()}
            check_packet_metadata(meta,row_ids)
            native.need(cpu['rows'].tolist()==row_ids,'Saved row payload differs from native-ordinal draw')
            return meta
        meta=ledger.call(rid,{'packet_payload_validations':1},validate_payload)
        name='step'+str(p['step'])+'_block'+f"{p['block']:02d}";path=folder/(name+'.safetensors')
        native.need(not path.exists(),'Raw row pack overwrite forbidden')
        ledger.call(rid,{'packet_safetensor_writes':1},save_file,cpu,str(path))
        native.need(native.binding(path,hf)['bytes']-PACKET_BYTES<=16384,'Packet safetensor header exceeded frozen bound')
        record={'schema':'ffnds.raw_bf16.full21tau.operands.v1','source_id':case['source_id'],'image_sha256':case['image_sha256'],
          'prompt_sha256':case['prompt_sha256'],'cohort_index':case['cohort_index'],'native_source_ordinal':case['native_source_ordinal'],
          'seed':42,'role':case['role'],'history':case['history'],'numeric_path':'BF16_ORIGINAL','source_state':'teacher_input',
          'step':p['step'],'block':p['block'],'expert':'high' if p['step']<2 else 'low','grid':grid,'population_rows':M,
          'selected_rows':row_ids,'row_draw_identity':'FIXED_8XY_COMPLETE21TAU_PRETENSOR_GEOMETRY', 'row_geometry':row_geometry(row_ids),
          'full_operand_layout':p['source_layout'],'tensor_metadata':meta,'payload':{'root':'hf',**native.binding(path,hf)},
          'tensor_boundaries':{'V1':'original modulated LN before full raw W1','h':'original tanh GELU output / raw W2 input',
            'Y':'raw full W2 output including original bias','r':'post-attention residual immediately before FFN gated write',
            'u':'original BF16 post_process output; captured after native inplace add','gate':'actual per-channel FFN gate','shift':'actual FFN AdaLN shift','scale':'actual FFN AdaLN scale'},
          'decomposition_or_fit':False,'protocol_sha256':digest,'run_id':rid}
        native.need(len((json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())<=32768,'Packet metadata exceeded frozen byte bound')
        case_path=folder/(name+'.json');ledger.call(rid,{'packet_metadata_writes':1},native.write,case_path,record)
        phase['packets'].append({'step':p['step'],'block':p['block'],'metadata':native.binding(case_path,hf)})
    pipe.runner.scheduler.generator=None
    tap=legacy.CollectorTap(pipe,guard,rid,save,check,case,index,packet);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        if case['role']=='development':
            from lightx2v.utils.input_info import init_empty_input_info,update_input_info_from_dict
            from lightx2v.utils.utils import seed_all
            def replay():
                runner=pipe.runner;info=init_empty_input_info('i2v',pipe.support_tasks)
                update_input_info_from_dict(info,{'seed':42,'prompt':case['prompt'],'negative_prompt':'',
                  'image_path':str((hf/case['image']['path']).resolve()),'save_result_path':'UNUSED_NO_NEW_DEVELOPMENT_VIDEO'})
                seed_all(42);runner.input_info=info
                with tap,torch.inference_mode():
                    runner.inputs=runner.run_input_encoder();runner.init_run()
                    native.need(runner.video_segment_num==1,'Original one-segment runner required')
                    runner.init_run_segment(0);runner.run_segment(0)
                    runner.end_run()
                torch.cuda.synchronize();tap.finish()
            ledger.call(rid,{'native_prefix_replays_no_decode':1},replay)
            phase['all_four_original_flow_and_post_bytes_passed']=True;phase['video_reused']=case['reused_video']
        else:
            def generate():
                with tap,torch.inference_mode():
                    pipe.generate(seed=42,image_path=str((hf/case['image']['path']).resolve()),prompt=case['prompt'],negative_prompt='',save_result_path=str(folder/'output.mp4'))
                torch.cuda.synchronize();tap.finish()
            ledger.call(rid,{'video_generations':1},generate)
            from run_ub1_consumer_videos import inspect_video
            phase['video']=ledger.call(rid,{'video_geometry_checks':1},inspect_video,folder/'output.mp4',native.VIDEO)
        native.need(len(phase['packets'])==160 and set(phase['states'])=={'raw_noise_FP32','initial_noise',*[f'flow_step{s}' for s in range(4)],*[f'latent_step{s}' for s in range(4)]},'Incomplete source raw states/positions')
        phase.update(status='COMPLETE_RAW_BF16_FULL21TAU_TEACHER_STATE_FULL160',completed_native_flow_steps=4,native_ffn_sites=160,
          native_counter_delta={'linear_forward_calls':len(tap.lines),'activation_qdq_calls':0,'weight_quantizations':0},
          scheduler_generator_reset=True,timetable=tap.timetable,Default_MM_actual_operand_metadata=tap.shapes,
          source_capture_wall_seconds=time.perf_counter()-started,FFN_boundary_CUDA_seconds=sum(a.elapsed_time(b) for a,b in tap.events)/1000,
          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        native.need(phase['peak_cuda_reserved_bytes']<=RESOURCES['GPU_reserved_max_bytes'],'Full-trajectory GPU peak40GiB exceeded')
        generation=folder/'GENERATION.json';native.write(generation,phase);phase['generation_receipt']=native.binding(generation,hf)
        return phase
    except BaseException as error:
        phase.update(status='FAILED_SOURCE_FULL_DEPTH_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:native.write(folder/'PHASE.json',phase);persist()


def run(args):
    global CURRENT_PART
    started = time.monotonic()
    plan = native.read(args.protocol)
    sha = native.sha(args.protocol)
    scope = authorize(plan, sha)
    cpu = require_cpu(scope, plan, sha, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh native worker before numerical imports required')
    verify_lineage(plan, args.hf_root)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    native.model_metadata(plan)
    local, package_rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Actual original package versions differ')
    native.need(execution.execution_worker_sha() == plan['execution_private_worker_config_sha256'], 'Actual private worker differs')
    native.need(len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1'
        for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'CPU/TF32 execution contract differs')
    CURRENT_PART = plan['part']
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh bounded acquisition output; no retry')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'part': plan['part'], 'numeric_path': 'BF16_ORIGINAL',
              'status': 'STARTED_NEW_RAW_BF16_FULL21TAU_DENSE_ACQUISITION_NO_RETRY', 'automatic_retry': False,
              'CPU_preflight': cpu, 'source_cohort': plan['source_cohort'], 'source_acquisition_proof': plan['source_acquisition_proof'],
              'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'], 'phases': [], 'design': DESIGN,
              'native_part_image_clusters': len(plan['cases']), 'new_development_image_clusters': 5,
              'old_development_neutral_image_clusters': 1 if plan['part'] == 0 else 0,
              'prior_part0_neutral': plan.get('prior_part0_neutral'), 'local_development_neutral_passed': plan['part'] == 1,
              'complete_ten_source_cohort_claimed': False, 'cohort_completion': 'PART_ONLY_REQUIRES_UNIQUE10_CLOSED_SCOPE_ASSET_AUDIT',
              'state_identity': 'Raw original Dense teacher, no candidate/student/FP4 state', 'motion_correspondence': 'NOT_MEASURED',
              'official_quality': 'NOT_RUN', 'training_updates': 0, 'decompositions': 0, 'quality_or_speed_acceptance': False,
              'capture_runtime_identity': 'Original full Dense model plus passive selected-row copies/serialization; not generation throughput'}
    monitor = None
    counter = execution.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    for case in plan['cases']:
        if case['role'] == 'observation':
            counter.hold(str(counter.path(out/run_id(case)/'output.mp4')), REPAIR['maximum_single_video_file_bytes'])
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = legacy.Ledger(report.setdefault('operation_ledger', {}), plan['run_ids'],
                           lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    persist()

    def check():
        authorize(plan, sha)
        if monitor:
            monitor.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'],
                    'Host/disk reserve exceeded')
        counter.check()
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Full21tau acquisition wall1800 exhausted')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU reserved40GiB exceeded')

    def timeout(*_):
        raise TimeoutError('Fixed full21tau acquisition wall1800 exhausted; no retry')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, package_rows), native.NoHiF4()
    file_guard = execution.single_file_bound()
    try:
        file_guard.__enter__()
        check()
        snapshot = out/'EXECUTED_SOURCES'
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        reservation, monitor, report['CUDA_reservation'] = reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'matmul_TF32': False, 'float32_matmul_precision': 'highest', 'bf16_reduced_precision_reduction': True}
        # Establish the actual device reservation/monitor before copying the
        # frozen source tree. Existing samples are checked without new queries.
        snapshot.mkdir()
        copies = [(ROOT/b['path'], snapshot/b['path'], b) for b in plan['repo_sources']]
        copies += [(args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)),
                   (ROOT/plan['source_cohort']['path'], snapshot/'source_cohort.json', plan['source_cohort'])]
        counter.hold('frozen_metadata', sum(b['bytes'] for _, _, b in copies))
        for source, destination, b in copies:
            monitor.check()
            destination.parent.mkdir(parents=True, exist_ok=True)
            counter.copy_bound(source, destination, b)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Source snapshot budget not exhausted exactly')
        with execution.counted_io(counter), native.OriginalGuard(torch, plan, ledger) as guard:
            begun = time.perf_counter()
            pipe = native.build_pipeline(plan)
            setup_seconds = time.perf_counter()-begun
            report.update(raw_model_setup_seconds=setup_seconds, original_loader_paths=guard.loaded,
                          BF16_matrix_residency='Original CPU masters/four CUDA block buffers, original W1/W2/attention and prefetch unchanged')
            registered = registration(plan, args.hf_root)
            native.write(snapshot/'registered_video_plan.json', registered)
            persist()
            check()
            video_index = 0
            for case in plan['cases']:
                native.need(case['role'] == 'development' or report.get('local_development_neutral_passed') is True,
                            'New source before actual full21tau observer neutral')
                phase = native_phase(torch, pipe, guard, plan, case, out, args.hf_root, sha, report, persist, check)
                if case['role'] == 'development':
                    native.need(plan['part'] == 0 and phase['all_four_original_flow_and_post_bytes_passed'] is True,
                                'New observer old Dense byte neutral did not pass')
                    report['local_development_neutral_passed'] = True
                    report['new_observer_development_neutral_GEN'] = phase['generation_receipt']
                    persist()
                    check()
                    continue
                from ffnds_video_metrics import evaluate, validate as validate_core
                import cv2
                cv2.setNumThreads(4)
                identity = {'source_id': case['image_sha256'], 'seed': 42, 'expected_frames': 81,
                            'input_prompt_digest': case['prompt_sha256'], 'model_digest': native.canonical(plan['models']),
                            'protocol_sha256': sha, 'run_sha256': phase['generation_receipt']['sha256']}
                cost = {'generation_wall_seconds': phase['source_capture_wall_seconds'], 'ffn_boundary_seconds': phase['FFN_boundary_CUDA_seconds'],
                        'setup_seconds': setup_seconds, 'peak_gpu_memory_mib': phase['peak_cuda_allocated_bytes']/(1 << 20)}
                core = ledger.call(run_id(case), {'core_CPU_reports': 1}, evaluate, Path(phase['video']['path']), None, {'identity': identity, 'cost': cost})
                native.need(not validate_core(core), 'New81frame Dense CORE invalid')
                core_path = out/run_id(case)/'CORE_METRICS.json'
                native.write(core_path, core)
                phase['core_metrics'] = native.binding(core_path, args.hf_root)

                def save_registration():
                    folder = out/f'job{video_index:06d}'
                    folder.mkdir()
                    native.write(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
                        'job': registered['samples'][video_index], 'source': registered['sources'][video_index], 'native_ffn_sites': 160,
                        'scheduler_generator_reset': True, 'native_ffn_counters': phase['native_counter_delta'],
                        'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL',
                        'row_geometry': row_geometry(anchor_rows())})
                ledger.call(run_id(case), {'source_registration_writes': 1}, save_registration)
                video_index += 1
                persist()
                check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out)
            native.write(out/'VERIFIED_GENERATIONS.json', verified)
            report.update(status='COMPLETE_RAW_BF16_FULL21TAU_NEW5_PART_WITH_REGISTERED_DENSE_VIDEOS',
                          verified_cohort=native.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root),
                          checked_transformers_imports=finder.origins, captured_native_sites=sum(len(p['packets']) for p in report['phases']),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            native.need(video_index == 5 and report['captured_native_sites'] == 160*len(plan['cases']) and not counter.holds,
                        'Incomplete unique acquisition sites/videos/output reservations')
            check()
    except BaseException as error:
        report.update(status='FAILED_FULL21TAU_ACQUISITION_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        monitor_error = None
        if monitor:
            try:
                monitor.close()
                report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources'].get('violations'), 'Final assigned GPU resource violation')
            except BaseException as error:
                monitor_error = error
                report.update(status='FAILED_FULL21TAU_ACQUISITION_PRESERVED_NO_RETRY', resource_error=str(error))
        for item in (finder, no_hif4):
            if item in sys.meta_path:
                sys.meta_path.remove(item)
        for case in plan['cases']:
            if case['role'] != 'observation':
                continue
            path = counter.path(out/run_id(case)/'output.mp4')
            if path not in counter.files and path.exists():
                counter.observe_external(path, hold=str(path))
        try:
            report['wall_seconds'] = time.monotonic()-started
            journal.close()
            inventory = {str(p.relative_to(out)): value for p, value in counter.files.items() if p != out/'RUN.json'}
            receipt = {'schema': 'ffnds.raw_bf16.output_written_bytes.v2', 'id': plan['id'], 'protocol_sha256': sha,
                       'files_excluding_top_level_RUN_and_this_manifest': inventory, 'incremental_checkpoint': counter.summary(),
                       'complete_live_bytes_formula': 'sum(inventory bytes)+actual RUN.json bytes+actual OUTPUT_ACCOUNTING.json bytes'}
            counter.write_json(out/'OUTPUT_ACCOUNTING.json', receipt)
            report['output_accounting'] = {'receipt': counter.binding(out/'OUTPUT_ACCOUNTING.json', args.hf_root),
                                          'checkpoint_before_final_RUN_write': counter.summary(), 'recursive_output_walks': 0}
            persist()
        finally:
            file_guard.__exit__(None, None, None)
        if monitor_error is not None and sys.exc_info()[0] is None:
            raise monitor_error
    return report


def bind_repo_inputs(plan, bindings):
    """Stage every newly referenced repository metadata file in the closure."""
    existing = {b['path']: b for b in plan['repo_sources']}
    for item in bindings:
        actual = native.binding(native.bound(ROOT, item), ROOT)
        native.need(item['path'] not in existing or existing[item['path']] == actual,
                    'Conflicting repository input binding')
        existing[item['path']] = actual
    plan['repo_sources'] = [existing[k] for k in sorted(existing)]


def bind_cohort(plan, hf, cohort_binding):
    cohort = native.read(native.bound(ROOT, cohort_binding))
    validate_cohort(cohort, hf, frozen=True)
    cases = []
    if plan['part'] == 0:
        original = native.read(native.bound(ROOT, plan['parent']['protocol']))
        report = native.read(native.bound(hf, plan['parent']['RUN']))
        old = copy.deepcopy(next(c for c in original['cases'] if c['index'] == 0))
        phase = next(p for p in report['phases'] if p['source_index'] == 0)
        cases.append({**old, 'part': 0, 'cohort_index': -1, 'baseline_source_index': 0, 'native_source_ordinal': 894,
                      'role': 'development', 'evidence_role': 'HISTORICALLY_EXPOSED_DEVELOPMENT_NEUTRAL',
                      'history': 'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT', 'rows': drawings(), 'row_geometry': row_geometry(anchor_rows()),
                      'controls': phase['states'], 'reused_video': phase['video']})
    for source in sorted((s for s in cohort['sources'] if s['part'] == plan['part']), key=lambda s: s['cohort_index']):
        cases.append({**copy.deepcopy(source), 'role': 'observation', 'evidence_role': source['role'],
                      'image_sha256': source['image']['sha256'], 'conditioning_id': source['source_id']+':native_crop_condition',
                      'rows': drawings(), 'row_geometry': row_geometry(anchor_rows())})
    plan.update(source_cohort=cohort_binding, source_acquisition_proof=cohort['source_audit'], cases=cases,
                run_ids=[run_id(c) for c in cases], status='DRAFT_REAL_SOURCE_COHORT_BOUND_NEW_NATIVE_ACQUISITION_NOT_RUN',
                binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING' if plan['part'] == 0 else 'BOUND_SOURCE_COHORT_PART0_NEUTRAL_NOT_BOUND',
                data_availability='REAL_SOURCE_BYTES_AUDIT_AND_ROOT_REVIEW_BOUND_NO_NEW_TRAJECTORIES_CLAIMED')
    verify_source_cohort(plan, hf)
    bind_repo_inputs(plan, [cohort_binding, cohort['source_audit']['protocol'], cohort['source_review']['document']])
    return plan


def bind_part0_neutral(plan, hf, protocol, report, launch_exit):
    """Metadata-only binding after part0 actually exits; never a live receipt."""
    native.need(plan['part'] == 1, 'Only part1 inherits the new observer neutral')
    value = native.read(native.bound(hf, report))
    phases = [p for p in value['phases'] if p['role'] == 'development']
    native.need(len(phases) == 1, 'Unique completed full21tau old-source neutral required')
    phase = phases[0]
    plan['prior_part0_neutral'] = {'protocol': protocol, 'RUN': report, 'launch_exit': launch_exit,
                                   'GENERATION': phase['generation_receipt'], 'states': phase['states']}
    verify_part0_neutral(plan, hf)
    bind_repo_inputs(plan, [protocol])
    return plan


def draft(hf, part, cohort_binding=None):
    base = native.read(ROOT/BASE_PROTOCOL)
    keys = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers', 'private_worker_config_sha256',
            'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof', 'runtime_overlay', 'parent', 'raw_master_CPU_proof')
    plan = {k: copy.deepcopy(base[k]) for k in keys}
    plan.update(id=scope_id(part), part=part, status='DRAFT_NEW_ACQUISITION_COHORT_NOT_BOUND', numeric_path='BF16_ORIGINAL',
                assets_finalized=False, binding_status='NOT_BOUND_NEW_SOURCE_HISTORY_NEAR_DUP_HUMAN_REVIEW', automatic_retry=False,
                limits=limits(part), resources=RESOURCES, design=DESIGN, repair=REPAIR, byte_budget=byte_budget(part),
                source_cohort=None, source_acquisition_proof=None, cases=None, run_ids=None, prior_part0_neutral=None,
                source_interface_schema=COHORT_SCHEMA, source_role='NEW_DEVELOPMENT_PASSIVE_DENSE_ONLY',
                software_lineage=native.binding(ROOT/BASE_PROTOCOL, ROOT), output_HF_relative=output_relative(part),
                data_availability='NOT_BOUND_NO_NEW_IMAGES_OR_TRAJECTORIES_CLAIMED',
                temporal_estimand='Complete fixed-xy trajectories and fixed spatial neighbors over all4x40 Dense teacher sites',
                competing_mechanisms=['LAYER_STEP_HETEROGENEOUS_FUNCTION_GEOMETRY', 'STATIC_XY_OR_SPATIAL_NEIGHBOR_OR_MOTION_CORRESPONDENCE_EFFECT'],
                motion_alignment='NOT_MEASURED_FIXED_XY; nearest-of-eight is not exact motion matching',
                existing_images='One old dev894 neutral only, never counted among ten new independent source images')
    paths = {b['path'] for b in base['repo_sources']} | REQUIRED | {BASE_PROTOCOL}
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted(paths)]
    if cohort_binding is not None:
        bind_cohort(plan, hf, cohort_binding)
    validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--part', type=int, choices=(0, 1))
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--source-cohort', type=Path)
    parser.add_argument('--local-root', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.hf_root = args.hf_root.resolve()
    if args.action == 'draft':
        native.need(args.part is not None and args.out is not None and not args.out.exists(), 'Fresh part draft required')
        source = native.binding(args.source_cohort, ROOT) if args.source_cohort else None
        value = draft(args.hf_root, args.part, source)
        native.write(args.out, value)
        print(json.dumps({'id': value['id'], 'binding_status': value['binding_status'], 'model_calls': 0, 'GPU_calls': 0}))
    elif args.action == 'inspect':
        value = native.read(args.protocol)
        validate(value)
        print(json.dumps({'id': value['id'], 'resources': RESOURCES, 'limits': value['limits'], 'byte_budget': value['byte_budget'],
                          'design': DESIGN, 'binding_status': value['binding_status']}, indent=2))
    elif args.action == 'preflight':
        native.need(args.out is not None and args.local_root is not None, 'Independent CPU output/local package required')
        out = args.out.resolve()
        native.need(out.is_relative_to(args.hf_root) and not out.exists(), 'Fresh confined byte proof output')
        out.mkdir(parents=True)
        value = {'id': native.read(args.protocol)['id'], 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_FULL21TAU_CPU_PREFLIGHT'}
        native.write(out/'RUN.json', value)
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_FULL21TAU_CPU_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            native.write(out/'RUN.json', value)
        print(json.dumps({'status': value['status'], 'model_calls': 0, 'GPU_calls': 0}))
    else:
        native.need(args.local_root is not None, 'Bound local Transformers package required')
        print(json.dumps({'status': run(args)['status']}))


if __name__ == '__main__':
    main()
