"""Registered CPU-only full21tau teacher-row geometry and fixed reuse observation.

Draft/bind/inspect read small provenance JSON, never operand tensors. Only a
frozen ACTIVE-registered run streams the ten new Dense source payloads once.
Old dev894 supplies the inherited byte-neutral and is excluded from source N.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import stat
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
import raw_bf16_full21tau_geometry as core
from e0_raw_bf16_temporal_pair_diagnostic import (
    need, read, digest, canonical, binding, bound, decode_packet, OutputBudget, Meter)

HF = Path('/home/xiang/.cache/huggingface')
ID = 'E0_RAW_BF16_FULL21TAU_NEW10_GEOMETRY_REUSE_OBS_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_raw_bf16_full21tau_new10_geometry_obs_cpu_20261029/attempt1/worker/results'
COLLECTOR_ID = 'E0_RAW_BF16_FULL21TAU_NEW10_DENSE_ACQUISITION_20261029_PART'
COLLECTOR_STATUS = 'COMPLETE_RAW_BF16_FULL21TAU_NEW5_PART_WITH_REGISTERED_DENSE_VIDEOS'
PHASE_STATUS = 'COMPLETE_RAW_BF16_FULL21TAU_TEACHER_STATE_FULL160'
COHORT = 'config/e0_raw_bf16_fresh10_full21tau_sources_20261029_v1.json'
NATIVE_SOURCE = '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py'
NATIVE_GUARD = 'tools/e0_original_bf16_videos.py'
COLLECTOR = 'tools/e0_raw_bf16_full_trajectory_collect.py'
STATE_NAMES = {'raw_noise_FP32', 'initial_noise', *[f'flow_step{s}' for s in range(4)],
               *[f'latent_step{s}' for s in range(4)]}
SHAPES = {'V1': [168, 5120], 'h': [168, 13824], 'Y': [168, 5120],
          'r': [168, 5120], 'u': [168, 5120], 'gate': [5120],
          'shift': [5120], 'scale': [5120], 'rows': [168]}
PACKET_BYTES = 168*(4*5120+13824)*2+3*5120*2+168*8
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 1200, 'outer_seconds': 1500,
             'packet_full_SHA_stream_bytes_max': 32 << 30, 'new_HF_bytes': 64 << 20,
             'automatic_retry': False}
DESIGN = {
    'evidence': 'SAVED_DENSE_TEACHER_STATE_COMPLETE21TAU_GEOMETRY_AND_FIXED_ARITHMETIC_REUSE_OBSERVATION',
    'sources': 'ALL_TEN_REVIEWED_NEW_DEVELOPMENT_IMAGES; OLD894_INHERITED_NEUTRAL_ONLY',
    'unit': 'ROWS_PAIRS_NESTED_IN_SOURCE_STEP_BLOCK; EQUAL_POSITIONS_WITHIN_SOURCE_THEN_EQUAL_IMAGES',
    'source_N': 10, 'grid_tau_y_x': list(core.GRID), 'anchors_y_x': [list(p) for p in core.ANCHORS],
    'row_order': 'SORTED_NATIVE_ROW_ID_TAU_MAJOR_ANCHOR_MINOR_IDENTICAL_ALL_SOURCE_STEP_BLOCK',
    'views': list(core.VIEWS), 'operands': list(core.OPERANDS),
    'time_space': 'X=GRAND+COMMON_TAU+STATIC_XY+TAU_XY_INTERACTION; EXACT_BALANCED_ORTHOGONAL_F64_SUMS',
    'temporal_mean': 'POSTHOC_PER_XY_OUTPUT_CONSTANT_ORACLE_NOT_FREE_F_MEAN_X',
    'all_temporal_pairs': 'RATIO_OF_SUMS_BY_T_TIMES_CENTERED_ENERGY_IDENTITY_NOT_MEAN_OF_PAIR_RATIOS',
    'near_pairs': [list(p) for p in core.NEAR], 'far_pairs': [list(p) for p in core.FAR],
    'pair_control': 'SAME_TAU_FOUR_NEAR_VS_FOUR_DISJOINT_DIAGONAL_FAR_PAIRS_FIXED_BEFORE_VALUES',
    'action_order': list(core.ACTIONS), 'actions': core.action_contract(), 'action_targets': list(core.TARGETS),
    'gY': 'FP64_MULTIPLY_DECODED_BF16_OUTPUT_AND_ACTUAL_TARGET_GATE',
    'native_write': 'EAGER_SAME_DTYPE_BF16_PRODUCT_TEMPORARY_THEN_BF16_ADD; NO_COMPILE_OR_FUSED_REWRITE',
    'neutral': 'EACH_PACKET_ORIGINAL_Y_GATE_R_MUST_RESTORE_ALL_SAVED_U_BYTES_BEFORE_ITS_ACTION_STATISTICS; ANY_FAILURE_STOPS',
    'CPU_emulator': 'FP32_MULTIPLY_AND_ADD_EACH_EXPLICIT_BF16_RNE; ORIGINAL_U_BYTE_NEUTRAL_IS_REQUIRED_NOT_ASSUMED',
    'gate': 'CURRENT_STEP_BLOCK_MODULATION_AVAILABLE_BEFORE_W1; APPLY_ACTUAL_TARGET_GATE_AFTER_Y_SCATTER',
    'rowLN_u': 'FP64_PER_ROW_CENTER_POPULATION_VARIANCE_EPS1E-6_NO_AFFINE_GEOMETRY',
    'temporal_innovation': 'PER_XY_TIME_CENTERED_RESIDUAL; REPORT_ACTION_ERROR_ENERGY_RETENTION_AND_SIGNED_CROSS_SEPARATELY_FROM_DC',
    'finite_scope': 'FIXED_DENSE_PREFIX_R_AND_CAPTURED_FULL_M_REPRESENTATIVE_Y; NO_DOWNSTREAM_OR_STUDENT_STATE_REPLAY',
    'execution_boundary': 'ROW_FRACTIONS_ARE_ARITHMETIC_ONLY; GATHER_SCATTER_SMALL_M_ROUNDING_AND_KERNEL_SYS_NOT_MEASURED',
    'motion_alignment': 'NOT_MEASURED_FIXED_XY_AND_NEIGHBORS_ARE_NOT_FLOW_CORRESPONDENCE',
    'quality_or_speed_acceptance': False, 'fits': 0, 'linear_algebra_decompositions': 0,
    'model_calls': 0, 'GPU_calls': 0, 'video_generations': 0, 'official_quality_calls': 0,
    'automatic_retry': False}
REQUIRED = (
    'tools/e0_raw_bf16_full21tau_analysis.py', 'tools/raw_bf16_full21tau_geometry.py',
    'tests/test_e0_raw_bf16_full21tau_analysis.py', 'tests/test_raw_bf16_full21tau_geometry.py',
    'tools/e0_raw_bf16_temporal_pair_diagnostic.py', 'tools/worker_identity.py',
    COLLECTOR, NATIVE_SOURCE, NATIVE_GUARD)
FROZEN_READER_SHA = '32eb9764cbe1309a0fb825ffeff43f0051ae31bc3a6c21faea1a56f7a6c76e2e'


def valid_binding(item):
    need(isinstance(item, dict) and isinstance(item.get('path'), str)
         and not Path(item['path']).is_absolute() and '..' not in Path(item['path']).parts
         and type(item.get('bytes')) is int and item['bytes'] > 0
         and isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
         and set(item['sha256']) <= set('0123456789abcdef'), 'Invalid relative byte/SHA binding')


class Metadata:
    """Small immutable provenance only; records each actual unique byte read."""
    def __init__(self, hf, reader=None):
        self.hf, self.reader, self.inputs, self.cache = hf, reader, {}, {}
    def raw(self, item, root='hf'):
        valid_binding(item)
        need(item['bytes'] <= 16 << 20 and Path(item['path']).suffix in ('.json', '.md', '.py'),
             'Provenance reader refuses operand/model payloads')
        key = (root, item['path'])
        if key in self.inputs:
            need(self.inputs[key] == {'root': root, **item}, 'Conflicting metadata identity')
        else:
            target = ROOT if root == 'repo' else self.hf
            raw = self.reader(target, item) if self.reader else bound(target, item)
            self.cache[key], self.inputs[key] = raw, {'root': root, **item}
        return self.cache[key]
    def json(self, item, root='hf'):
        need(Path(item['path']).suffix == '.json', 'JSON provenance expected')
        return json.loads(self.raw(item, root))
    def manifest(self):
        return [self.inputs[k] for k in sorted(self.inputs)]


def exited(receipt, proof, expected_id):
    need(receipt.get('id') == expected_id and receipt.get('protocol_sha256') == proof['protocol']['sha256']
         and receipt.get('run_sha256') == proof['RUN']['sha256'] and receipt.get('process_exited') is True
         and type(receipt.get('child_returncode')) is int and receipt['child_returncode'] == 0
         and receipt.get('timeout') is False and receipt.get('automatic_retry') is False,
         'Independent actual exit0/RUN/protocol identity required')


def expected_geometry():
    return {'grid_tau_y_x': list(core.GRID), 'anchors_y_x': [list(p) for p in core.ANCHORS],
            'horizontal_neighbor_anchor_ids': [list(p) for p in core.NEAR],
            'row_anchor_ids': [a for _ in range(21) for a in range(8)],
            'row_taus': [t for t in range(21) for _ in range(8)],
            'trajectory_shape_tau_anchor': [21, 8],
            'row_order': 'SORTED_NATIVE_ROW_ID_EQUALS_TAU_MAJOR_ANCHOR_ID_MINOR',
            'motion_correspondence': 'NOT_MEASURED_FIXED_XY_IS_NOT_MOTION_ALIGNMENT'}


def check_tensor_metadata(value):
    need(set(value) == set(SHAPES), 'Actual nine operand tensors required')
    for name, shape in SHAPES.items():
        item = value[name]
        need(item['shape'] == shape and item['dtype'] == ('int64' if name == 'rows' else 'bfloat16')
             and isinstance(item['raw_tensor_sha256'], str) and len(item['raw_tensor_sha256']) == 64
             and set(item['raw_tensor_sha256']) <= set('0123456789abcdef'), 'Captured operand shape/dtype/rawSHA differs')
    need(value['rows']['raw_tensor_sha256'] == digest(struct.pack('<168q', *core.rows())),
         'Complete21tau row payload binding differs')


def check_packet(record, phase, reference, protocol_sha):
    step, block = reference['step'], reference['block']
    need(record['schema'] == 'ffnds.raw_bf16.full21tau.operands.v1'
         and record['numeric_path'] == 'BF16_ORIGINAL' and record['source_state'] == 'teacher_input'
         and record['seed'] == 42 and record['role'] == 'observation' and record['population_rows'] == 31668
         and record['grid'] == list(core.GRID) and record['selected_rows'] == core.rows()
         and record['row_geometry'] == expected_geometry()
         and record['row_draw_identity'] == 'FIXED_8XY_COMPLETE21TAU_PRETENSOR_GEOMETRY'
         and record['protocol_sha256'] == protocol_sha and record['run_id'] == phase['run_id']
         and record['step'] == step and record['block'] == block
         and record['expert'] == ('high' if step < 2 else 'low')
         and record['decomposition_or_fit'] is False,
         'Actual full21tau operand coordinate/numeric/protocol identity differs')
    for key in ('source_id', 'image_sha256', 'prompt_sha256', 'native_source_ordinal'):
        need(record[key] == phase[key], 'Packet source/prompt correspondence differs')
    need(record['cohort_index'] == phase['source_index'], 'Packet new source index differs')
    check_tensor_metadata(record['tensor_metadata'])
    for name in ('V1', 'h', 'Y', 'r', 'u'):
        layout = record['full_operand_layout'][name]
        need(layout['shape'] == [31668, 13824 if name == 'h' else 5120]
             and layout['dtype'] == 'bfloat16' and len(layout['stride']) == 2
             and all(type(v) is int and v > 0 for v in layout['stride']), 'Captured full-M operand geometry differs')
    valid_binding(record['payload'])
    need(record['payload'].get('root') == 'hf' and PACKET_BYTES < record['payload']['bytes'] <= PACKET_BYTES+16384,
         'Actual packet file body/header bytes differ')


def check_phase(phase, case, protocol_sha, metadata, *, neutral=False):
    need(phase['status'] == PHASE_STATUS and phase['protocol_sha256'] == protocol_sha
         and phase['numeric_path'] == 'BF16_ORIGINAL' and phase['source_state'] == 'teacher_input'
         and phase['seed'] == 42 and phase['mode'] == 'dense' and phase['row_geometry'] == expected_geometry()
         and phase['completed_native_flow_steps'] == 4 and phase['native_ffn_sites'] == 160
         and phase['native_counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 0, 'weight_quantizations': 0}
         and set(phase['states']) == STATE_NAMES and len(phase['packets']) == 160
         and {(r['step'], r['block']) for r in phase['packets']} == {(s, b) for s in range(4) for b in range(40)},
         'Closed Dense full160 phase incomplete')
    for key in ('source_id', 'native_source_ordinal', 'image_sha256', 'prompt_sha256'):
        need(phase[key] == case[key], 'Closed phase differs from frozen original source')
    need(phase['source_index'] == case['cohort_index'] and phase['role'] == case['role'], 'Closed phase role/index differs')
    for name, item in phase['states'].items():
        valid_binding(item)
        need(item['shape'] == [16, 21, 58, 104]
             and item['dtype'] == ('float32' if name == 'raw_noise_FP32' or name.startswith('flow_') else 'bfloat16')
             and len(item['raw_tensor_sha256']) == 64, 'Closed native state shape/dtype identity differs')
    generation = metadata.json(phase['generation_receipt'])
    # The immutable GENERATION precedes CORE and insertion of its own binding.
    for key, value in generation.items():
        need(phase.get(key) == value, 'Closed phase changed immutable GENERATION content')
    if neutral:
        need(phase['native_source_ordinal'] == 894 and phase['role'] == 'development'
             and phase['source_index'] == -1 and phase.get('all_four_original_flow_and_post_bytes_passed') is True,
             'Inherited old894 observer byte-neutral required')
        for name, item in phase['states'].items():
            need(all(item[k] == case['controls'][name][k] for k in ('shape', 'dtype', 'raw_tensor_sha256')),
                 'Inherited old894 raw state neutral differs')
        return None
    need(phase['role'] == 'observation' and phase.get('evidence_role') == 'NEW_DEVELOPMENT',
         'Old development image cannot count as new source')
    video = phase['video']
    need(video['frames'] == 81 and video['size'] == [832, 464] and video['fps'] == 16,
         'New passive Dense video geometry incomplete')
    core_report = metadata.json(phase['core_metrics'])
    identity = core_report['identity']
    need(identity['source_id'] == phase['image_sha256'] and identity['seed'] == 42
         and identity['input_prompt_digest'] == phase['prompt_sha256']
         and identity['run_sha256'] == phase['generation_receipt']['sha256']
         and identity['protocol_sha256'] == protocol_sha, 'Actual Dense CORE/generation identity differs')
    packets = []
    for reference in sorted(phase['packets'], key=lambda p: (p['step'], p['block'])):
        record = metadata.json(reference['metadata'])
        check_packet(record, phase, reference, protocol_sha)
        packets.append({'step': record['step'], 'block': record['block'], 'metadata': reference['metadata'],
                        'payload': record['payload'], 'tensor_metadata': record['tensor_metadata']})
    return {'source_id': phase['source_id'], 'native_source_ordinal': phase['native_source_ordinal'],
            'cohort_index': phase['source_index'], 'image_sha256': phase['image_sha256'],
            'prompt_sha256': phase['prompt_sha256'], 'generation_receipt': phase['generation_receipt'],
            'core_metrics': phase['core_metrics'], 'states': phase['states'], 'packets': packets}


def collect_inputs(plan, hf, *, reader=None):
    metadata = Metadata(hf, reader)
    cohort = metadata.json(plan['source_cohort'], 'repo')
    need(cohort.get('status') == 'REVIEWED_READY_FOR_RAW_DENSE_ACQUISITION'
         and cohort['source_review']['status'] == 'REVIEWED_NEW_DEVELOPMENT'
         and cohort['source_review']['reviewer_kind'] in ('ROOT_AGENT', 'HUMAN'), 'Actual new-source review required')
    metadata.raw(cohort['source_review']['document'], 'repo')
    acquisition = cohort['source_audit']
    source_plan = metadata.json(acquisition['protocol'], 'repo')
    source_run, source_exit = metadata.json(acquisition['RUN']), metadata.json(acquisition['exit'])
    exited(source_exit, acquisition, source_plan['id'])
    need(source_run['id'] == source_plan['id'] == 'E0_RAW_BF16_FRESH10_SOURCE_ARCHIVE_CPU_V2_20261029'
         and source_run['protocol_sha256'] == acquisition['protocol']['sha256']
         and source_run['model_calls'] == source_run['GPU_calls'] == 0 and len(source_run['sources']) == 10,
         'Actual closed source extraction differs')
    sources = sorted(cohort['sources'], key=lambda s: s['cohort_index'])
    need(len(sources) == 10 and [s['cohort_index'] for s in sources] == list(range(10))
         and len({s['source_id'] for s in sources}) == len({s['image']['sha256'] for s in sources})
         == len({s['native_source_ordinal'] for s in sources}) == 10
         and all(s['native_source_ordinal'] != 894 and s['role'] == 'NEW_DEVELOPMENT'
                 and s['collector_role'] == 'observation' for s in sources), 'Ten unique new image-clusters required')
    need(isinstance(plan.get('parents'), list) and len(plan['parents']) == 2
         and [p['part'] for p in plan['parents']] == [0, 1], 'Two closed collector parts NOT_BOUND')
    cases, parent_plans, parent_runs, neutral = [], [], [], None
    for proof in plan['parents']:
        part = proof['part']
        parent = metadata.json(proof['protocol'], 'repo')
        report, exit_receipt = metadata.json(proof['RUN']), metadata.json(proof['exit'])
        exited(exit_receipt, proof, COLLECTOR_ID+str(part))
        need(parent['id'] == report['id'] == COLLECTOR_ID+str(part)
             and parent['part'] == report['part'] == part and parent['assets_finalized'] is True
             and parent['source_cohort'] == report['source_cohort'] == plan['source_cohort']
             and parent['numeric_path'] == report['numeric_path'] == 'BF16_ORIGINAL'
             and report['protocol_sha256'] == proof['protocol']['sha256']
             and report['status'] == COLLECTOR_STATUS and report['local_development_neutral_passed'] is True
             and len(report['phases']) == (6 if part == 0 else 5)
             and report['captured_native_sites'] == 160*len(parent['cases'])
             and not report['operation_ledger']['forbidden_attempts']
             and all(report['operation_ledger'][side].get(k, 0) == n for side in ('predebited', 'observed')
                     for k, n in parent['limits'].items()), 'Complete closed collector part and actual ledger required')
        recipe = parent['runtime_overlay']
        need(recipe['use_compile'] is False and recipe['dit_quantized'] is False
             and recipe['model_cls'] == 'wan2.2_moe_distill' and recipe['norm_modulate_backend'] == 'torch'
             and recipe['infer_steps'] == 4 and recipe['num_layers'] == 40
             and recipe['feature_caching'] == 'NoCaching' and recipe['lora_configs'] is None,
             'Original eager raw BF16 full-depth identity differs')
        precision = report['runtime_precision']
        need(precision == {'matmul_TF32': False, 'float32_matmul_precision': 'highest',
                           'bf16_reduced_precision_reduction': True}, 'Actual Dense precision differs')
        source_files = {s['path']: s for s in parent['repo_sources']}
        need(all(source_files[path] == next(i for i in plan['repo_sources'] if i['path'] == path)
                 for path in (COLLECTOR, NATIVE_SOURCE, NATIVE_GUARD)), 'Parent actual native/write/capture source differs')
        observations = []
        for phase, case in zip(report['phases'], parent['cases'], strict=True):
            if case['role'] == 'development':
                need(part == 0 and neutral is None, 'Only one inherited neutral permitted')
                check_phase(phase, case, proof['protocol']['sha256'], metadata, neutral=True)
                neutral = {'part': 0, 'GENERATION': phase['generation_receipt'], 'states': phase['states']}
                continue
            source = sources[case['cohort_index']]
            need(source['part'] == part and all(case[k] == source[k] for k in
                ('source_id', 'native_source_ordinal', 'cohort_index', 'prompt_sha256'))
                 and case['image_sha256'] == source['image']['sha256'], 'Captured new source allocation differs')
            observations.append(check_phase(phase, case, proof['protocol']['sha256'], metadata))
        need(len(observations) == 5, 'Exactly five new images per actual part required')
        cases.extend(observations)
        parent_plans.append(parent)
        parent_runs.append(report)
    need(neutral is not None and parent_plans[1]['prior_part0_neutral']['protocol'] == plan['parents'][0]['protocol']
         and parent_plans[1]['prior_part0_neutral']['RUN'] == plan['parents'][0]['RUN']
         and parent_plans[1]['prior_part0_neutral']['GENERATION'] == neutral['GENERATION']
         and parent_plans[1]['prior_part0_neutral']['states'] == neutral['states']
         and parent_runs[1]['prior_part0_neutral'] == parent_plans[1]['prior_part0_neutral'],
         'Part1 must reuse this exact closed successful part0 observer neutral')
    need(all(parent_plans[0][k] == parent_plans[1][k] for k in
             ('models', 'runtime_overlay', 'execution_raw_master_CPU_proof')),
         'Raw master/recipe identity differs between acquisition parts')
    cases.sort(key=lambda c: c['cohort_index'])
    need([c['cohort_index'] for c in cases] == list(range(10)), 'Whole ten-source acquisition not complete')
    index = {'schema': 'ffnds.raw_bf16.full21tau.analysis_input_index.v1', 'cases': cases,
             'inherited_neutral': neutral, 'models_sha256': canonical(parent_plans[0]['models']),
             'recipe_sha256': canonical(parent_plans[0]['runtime_overlay'])}
    return index, metadata.manifest()


def limits(index, metadata_inputs):
    packets = [p for c in index['cases'] for p in c['packets']]
    count = len(packets)
    need(count == 1600, 'All ten-source full160 packets required')
    return {'metadata_unique_reads': len(metadata_inputs), 'metadata_bytes': sum(b['bytes'] for b in metadata_inputs),
            'packet_full_SHA_reads': count, 'packet_full_SHA_bytes': sum(p['payload']['bytes'] for p in packets),
            'tensor_raw_SHA_checks': count*9, 'row_payload_checks': count,
            'original_u_byte_neutrals': count, 'BF16_product_calls': count*4, 'BF16_add_calls': count*4,
            'gY_FP64_products': count*4, 'rowLN_FP64_calls': count*4,
            'balanced_time_space_sum_evaluations': count*7*3,
            'same_tau_near_far_pair_comparisons': count*7*(21+20+1)*8,
            'fixed_reuse_action_evaluations': count*3,
            'fixed_reuse_target_view_error_evaluations': count*3*4*3,
            'source_position_summary_additions': count, 'source_equal_summary_finalizations': 1,
            'model_calls': 0, 'GPU_calls': 0, 'fits': 0, 'linear_algebra_decompositions': 0}


def make_draft(hf):
    need(digest((ROOT/'tools/e0_raw_bf16_temporal_pair_diagnostic.py').read_bytes()) == FROZEN_READER_SHA,
         'Frozen operand reader changed')
    return {'id': ID, 'status': 'DRAFT_CLOSED_TWO_PART_ACQUISITION_NOT_BOUND', 'assets_finalized': False,
            'binding_status': 'NOT_BOUND_BOTH_ACTUAL_NATIVE_PART_EXIT0_REQUIRED', 'automatic_retry': False,
            'design': copy.deepcopy(DESIGN), 'resources': RESOURCES, 'output_HF_relative': OUTPUT,
            'source_cohort': binding(ROOT/COHORT, ROOT), 'parents': None,
            'input_index': None, 'metadata_inputs': None, 'limits': None,
            'numpy_version': importlib.metadata.version('numpy'),
            'repo_sources': [binding(ROOT/p, ROOT) for p in REQUIRED],
            'research_question': 'Does FFN output retain or reshape complete temporal innovations beyond redundancy already visible in V1 and residual R, and can fixed pre-W1 representative maps exploit it?',
            'competing_mechanisms': [
                'LARGE_COMMON_OR_STATIC_COMPONENT_MASKS_IMPORTANT_TIME_SPACE_INNOVATION; SMALL_GLOBAL_ENERGY_ERROR_CAN_REMOVE_TEMPORAL_DETAIL',
                'SOURCE_STABLE_LAYER_STEP_HETEROGENEOUS_FUNCTION_SHARING_ADDS_REUSABLE_REDUNDANCY_BEYOND_INPUT_APPEARANCE',
                'RESIDUAL_ANCHOR_AND_TARGET_GATE_MASK_LARGE_FFN_UPDATE_OR_NORMALIZED_DIRECTION_CHANGES'],
            'prediction': 'Compare V1/h/Y/gY time-space fractions and fixed-near versus fixed-far effects; preserved total energy with collapsed centered-time energy supports innovation loss, not reuse safety.',
            'r960_relation': 'THIS_DENSE_ONLY_OBSERVATION_TESTS_A_POSSIBLE_DC_VS_INNOVATION_MECHANISM; R960_CANDIDATE_TIME_TENSORS_ARE_NOT_HERE_AND_NO_CAUSAL_ATTRIBUTION_IS_MADE',
            'finite_scope': DESIGN['finite_scope'], 'motion_correspondence': 'NOT_MEASURED',
            'quality_or_speed_acceptance': False, 'payload_bytes_read_during_preparation': 0}


def bind_inputs(plan, hf, parents):
    result = copy.deepcopy(plan)
    result['parents'] = copy.deepcopy(parents)
    index, metadata_inputs = collect_inputs(result, hf)
    result.update(input_index=index, metadata_inputs=metadata_inputs, limits=limits(index, metadata_inputs),
                  status='DRAFT_BOUND_CLOSED_PARTS_READY_FOR_EXACT_REVIEW', assets_finalized=True,
                  binding_status='BOUND_READY_FOR_REGISTRATION')
    need(result['limits']['packet_full_SHA_bytes'] <= RESOURCES['packet_full_SHA_stream_bytes_max'], 'Full packet stream budget exceeded')
    validate(result)
    return result


def validate(plan, frozen=False):
    need(plan.get('id') == ID and plan.get('design') == DESIGN and plan.get('resources') == RESOURCES
         and plan.get('automatic_retry') is False and plan.get('output_HF_relative') == OUTPUT,
         'Fixed full21tau CPU observation scope differs')
    need({s['path'] for s in plan['repo_sources']} == set(REQUIRED)
         and len(plan['repo_sources']) == len(REQUIRED) and plan['source_cohort']['path'] == COHORT,
         'Exact software/source cohort closure differs')
    for item in plan['repo_sources']:
        bound(ROOT, item)
    if plan.get('input_index') is not None:
        need(plan['assets_finalized'] is True and plan['binding_status'] == 'BOUND_READY_FOR_REGISTRATION'
             and plan['limits'] == limits(plan['input_index'], plan['metadata_inputs'])
             and plan['limits']['packet_full_SHA_bytes'] <= RESOURCES['packet_full_SHA_stream_bytes_max'],
             'Closed operand index/budget differs')
    if frozen:
        need(plan.get('status') == 'FROZEN_REGISTERED' and plan.get('assets_finalized') is True
             and plan.get('input_index') is not None
             and plan.get('coordinator_review', {}).get('status') == 'REVIEWED_CODE_DATA_SOFTWARE_CONTROLS',
             'Exact coordinator-reviewed closed-data scope must be frozen and registered')


def authorize(plan, sha):
    validate(plan, True)
    active = read(ROOT/'research/ACTIVE_RESEARCH.json')
    need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True,
         'Current research authorization inactive')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    need(len(scopes) == 1 and scopes[0].get('protocol_sha256') == sha
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
         and scopes[0].get('limits') == plan['limits'] and scopes[0].get('resources') == RESOURCES
         and scopes[0].get('max_gpu_cards') == 0, 'Unique exact CPU full21tau observation not registered')


def stream_payload(hf, item, report):
    valid_binding(item)
    path = (hf/item['path']).resolve()
    need(path.is_relative_to(hf.resolve()), 'Operand path escaped HF')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == item['bytes'], 'Operand path/size differs')
    sha, chunks, count = hashlib.sha256(), [], 0
    with path.open('rb', buffering=0) as stream:
        while True:
            value = stream.read(min(8 << 20, item['bytes']+1-count))
            if not value:
                break
            count += len(value)
            report['actual_packet_bytes_read'] += len(value)
            need(count <= item['bytes'] and report['actual_packet_bytes_read'] <= RESOURCES['packet_full_SHA_stream_bytes_max'],
                 'Actual operand stream exceeded frozen bytes')
            sha.update(value)
            chunks.append(value)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
         == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
         and count == item['bytes'] and sha.hexdigest() == item['sha256'], 'Full operand SHA/stat/bytes differ')
    return b''.join(chunks), {key: getattr(after, key) for key in
                             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def audited_manifest_entry(raw, case, entry, file_stat):
    """Re-use the same verified stream/header; never read tensor bytes again."""
    header_size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+header_size])
    return {'source_id': case['source_id'], 'native_source_ordinal': case['native_source_ordinal'],
            'cohort_index': case['cohort_index'], 'step': entry['step'], 'block': entry['block'],
            'metadata': entry['metadata'], 'payload': entry['payload'],
            'actual_unchanged_stat_before_and_after_stream': file_stat,
            'actual_safetensors_header': header, 'verified_tensor_raw_metadata': entry['tensor_metadata'],
            'native_row_payload_sha256': entry['tensor_metadata']['rows']['raw_tensor_sha256'],
            'row_geometry': expected_geometry(), 'actual_full_file_SHA_and_nine_tensor_SHA_passed': True,
            'original_u_byte_neutral': True}


def conditioning_census(index):
    result = []
    for step in range(4):
        for block in range(40):
            rows = [next(p for p in c['packets'] if p['step'] == step and p['block'] == block) for c in index['cases']]
            result.append({'step': step, 'block': block, 'source_N': 10,
                           'raw_SHA_unique_count': {name: len({r['tensor_metadata'][name]['raw_tensor_sha256'] for r in rows})
                                                    for name in ('gate', 'shift', 'scale')}})
    return result


def inspect(plan, hf):
    validate(plan)
    if plan.get('input_index') is None:
        return {'id': ID, 'status': plan['binding_status'], 'model_calls': 0, 'GPU_calls': 0, 'operand_bytes_read': 0}
    index, metadata_inputs = collect_inputs(plan, hf)
    need(index == plan['input_index'] and metadata_inputs == plan['metadata_inputs'], 'Frozen parent metadata/index changed')
    return {'id': ID, 'status': 'PASS_CLOSED_PARTS_AND_METADATA_ONLY_NO_OPERAND_VALUES_READ',
            'source_N': 10, 'packets': 1600, 'limits': plan['limits'], 'model_calls': 0, 'GPU_calls': 0, 'operand_bytes_read': 0}


def run(args):
    started = time.monotonic()
    plan, sha = read(args.protocol), digest(args.protocol.read_bytes())
    authorize(plan, sha)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 4
         and not {'torch', 'numpy'} & set(sys.modules), 'Fresh hidden-CUDA four-CPU worker required')
    from worker_identity import require_worker
    require_worker()
    out = (args.hf_root/OUTPUT).resolve()
    need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh output; no automatic retry')
    out.mkdir(parents=True)
    output = OutputBudget(out, RESOURCES['new_HF_bytes']-(1 << 20))
    report = {'id': ID, 'protocol_sha256': sha, 'status': 'STARTED_SAVED_TEACHER_FULL21TAU_OBS_CPU_NO_RETRY',
              'automatic_retry': False, 'model_calls': 0, 'GPU_calls': 0, 'fits': 0, 'linear_algebra_decompositions': 0,
              'design': DESIGN, 'source_cohort': plan['source_cohort'], 'parents': plan['parents'],
              'source_N': 10, 'old894_numerical_source_N': 0, 'source_state': 'teacher_input',
              'actual_packet_bytes_read': 0, 'actual_metadata_bytes_read': 0, 'completed_packets': 0,
              'passed_original_u_byte_neutrals': 0, 'all_statistics_provisional_until_1600_neutrals_pass': True,
              'quality_or_speed_acceptance': False, 'motion_alignment': 'NOT_MEASURED',
              'supervisor_external_output_reserve_bytes': 1 << 20}
    meter = Meter(report, output, plan['limits'])
    def save():
        report['wall_seconds'] = time.monotonic()-started
        output.json('RUN.json', report, emergency=True)
    def timeout(*_):
        raise TimeoutError('Fixed full21tau CPU wall1200 reached; no retry')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    save()
    try:
        def metadata_read(root, item):
            def actual():
                raw = bound(root, item)
                report['actual_metadata_bytes_read'] += len(raw)
                return raw
            return meter.work({'metadata_unique_reads': 1, 'metadata_bytes': item['bytes']}, actual)
        index, metadata_inputs = collect_inputs(plan, args.hf_root, reader=metadata_read)
        need(index == plan['input_index'] and metadata_inputs == plan['metadata_inputs'], 'Closed lineage/index changed before numerical work')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        import numpy as np
        need(np.__version__ == plan['numpy_version'] and 'torch' not in sys.modules, 'Frozen NumPy-only environment differs')
        report['numpy_version'] = np.__version__
        report['inherited_observer_neutral'] = index['inherited_neutral']
        report['model_identity_sha256'], report['recipe_identity_sha256'] = index['models_sha256'], index['recipe_sha256']
        summary = core.SourceEqualSummary([c['source_id'] for c in index['cases']], np)
        for case in index['cases']:
            for entry in case['packets']:
                raw, file_stat = meter.work({'packet_full_SHA_reads': 1, 'packet_full_SHA_bytes': entry['payload']['bytes']},
                                           stream_payload, args.hf_root, entry['payload'], report)
                packet = meter.work({'tensor_raw_SHA_checks': 9, 'row_payload_checks': 1},
                                    decode_packet, raw, entry, core.rows(), np, SHAPES,
                                    whole_SHA_already_stream_verified=True)
                update = meter.work({'original_u_byte_neutrals': 1, 'BF16_product_calls': 1, 'BF16_add_calls': 1},
                                    core.neutral_postwrite, packet, np)
                report['passed_original_u_byte_neutrals'] += 1
                audit = audited_manifest_entry(raw, case, entry, file_stat)
                output.append('MANIFEST.jsonl', (json.dumps(audit, separators=(',', ':'), ensure_ascii=False,
                                                           allow_nan=False)+'\n').encode())
                operands = meter.work({'gY_FP64_products': 1, 'rowLN_FP64_calls': 1}, core.operands, packet, np)
                geometry = meter.work({'balanced_time_space_sum_evaluations': 7*3,
                                       'same_tau_near_far_pair_comparisons': 7*(21+20+1)*8},
                                      lambda: [core.geometry(operands[k], np) for k in core.OPERANDS])
                actions = meter.work({'fixed_reuse_action_evaluations': 3, 'fixed_reuse_target_view_error_evaluations': 3*4*3,
                                      'BF16_product_calls': 3, 'BF16_add_calls': 3,
                                      'gY_FP64_products': 3, 'rowLN_FP64_calls': 3},
                                     core.action_effects, packet, operands, update, np)
                values = core.flatten(geometry, actions)
                record = {'source_id': case['source_id'], 'native_source_ordinal': case['native_source_ordinal'],
                          'cohort_index': case['cohort_index'], 'step': entry['step'], 'block': entry['block'],
                          'metadata': entry['metadata'], 'payload': entry['payload'],
                          'row_payload_sha256': entry['tensor_metadata']['rows']['raw_tensor_sha256'],
                          'original_u_byte_neutral': True, 'scalar_order': 'SUMMARY.metric_labels', 'values': values}
                output.append('POSITIONS.jsonl', (json.dumps(record, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode())
                meter.work({'source_position_summary_additions': 1}, summary.add,
                           case['source_id'], entry['step'], entry['block'], values)
                report['completed_packets'] += 1
                report['last_completed_packet'] = [case['native_source_ordinal'], entry['step'], entry['block']]
                save()
                del raw, file_stat, audit, packet, update, operands, geometry, actions, values, record
        need(report['completed_packets'] == report['passed_original_u_byte_neutrals'] == 1600,
             'All full-depth original write byte neutrals required')
        result = meter.work({'source_equal_summary_finalizations': 1}, summary.result)
        meter.finish()
        need(report['actual_packet_bytes_read'] == plan['limits']['packet_full_SHA_bytes']
             and report['actual_metadata_bytes_read'] == plan['limits']['metadata_bytes'] and 'torch' not in sys.modules,
             'Incomplete actual single-pass CPU IO identity')
        result.update(id=ID, protocol_sha256=sha, design=DESIGN,
                      source_cohort=plan['source_cohort'], parents=plan['parents'],
                      conditioning_raw_SHA_census=conditioning_census(index),
                      original_u_byte_neutrals_passed=1600,
                      source_roles='TEN_NEW_DEVELOPMENT_IMAGE_CLUSTERS_NOT_BLIND_CONFIRMATION',
                      old894_excluded_from_statistics=True)
        output.json('SUMMARY.json', result)
        report.update(status='COMPLETE_RAW_BF16_FULL21TAU_NEW10_SAVED_TEACHER_GEOMETRY_REUSE_OBS_CPU',
                      all_statistics_provisional_until_1600_neutrals_pass=False,
                      result_files={name: binding(out/name, args.hf_root) for name in
                                    ('POSITIONS.jsonl', 'SUMMARY.json', 'MANIFEST.jsonl', 'operations.jsonl')},
                      audit_manifest={'packets': 1600, 'native_row_ids': core.rows(),
                                      'single_stream_full_SHA_and_all_nine_raw_tensor_SHA': True,
                                      'original_u_byte_neutrals': 1600,
                                      'stat_identity': 'same executing worker filesystem; recheck device/inode/size/times before reuse'})
    except BaseException as error:
        report.update(status='FAILED_FULL21TAU_OBS_CPU_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        import resource
        report['max_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['Torch_imported'] = 'torch' in sys.modules
        if 'summary' in locals():
            report['completed_sources_all160_positions'] = [s for s in summary.source_ids if sum(k[0] == s for k in summary.seen) == 160]
        try:
            report['actual_output_bytes_final_flat_audit_before_RUN_replace'] = output.final_audit()
            save()
        finally:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'bind', 'inspect', 'run'))
    parser.add_argument('--hf-root', type=Path, default=HF)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--parents', type=Path, help='JSON list of two {part,protocol,RUN,exit} exact bindings')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.hf_root = args.hf_root.resolve()
    if args.action in ('draft', 'bind'):
        need(args.out is not None and not args.out.exists(), 'Fresh draft/bound output required')
        value = make_draft(args.hf_root) if args.action == 'draft' else bind_inputs(read(args.protocol), args.hf_root, read(args.parents))
        args.out.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps({'id': ID, 'status': value['binding_status'], 'operand_bytes_read': 0, 'model_calls': 0, 'GPU_calls': 0}))
    elif args.action == 'inspect':
        print(json.dumps(inspect(read(args.protocol), args.hf_root), indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()
