"""Registered CPU OBS: same-xy temporal pairs versus matched different-xy pairs.

The eight historically exposed raw-BF16 teacher trajectories are immutable.
Draft/inspect read JSON metadata only. Only a frozen, registered run reads the
1280 complete operand packets. This program imports no model or Torch code.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import stat
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HF = Path('/home/xiang/.cache/huggingface')
ID = 'E0_RAW_BF16_S2_TEMPORAL_PAIR_OBS_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_raw_bf16_s2_temporal_pair_obs_cpu_20261029/attempt1/worker/results'
CORPUS = 'research/20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.json'
REPLAY = 'research/20261029/RAW_BF16_TEMPORAL_ROW_REPLAY_20261029.json'
GRID = (21, 29, 52)
ROWS, D, H = 256, 5120, 13824
CAL = (240, 288, 304, 896, 897, 899)
DEV = (256, 894)
ORDINALS = CAL+DEV
OPERANDS = ('X', 'h', 'Y', 'gY', 'R', 'u', 'rowLN_u')
METRICS = ('temporal_energy', 'temporal_cosine', 'matched_temporal_energy',
           'matched_temporal_cosine', 'control_energy', 'control_cosine',
           'paired_energy_difference', 'paired_cosine_difference')
TIME_BUCKETS = ('all', 'tau0_participates', 'tau0_excluded')+tuple(f'delta_tau_{d:02}' for d in range(1, 21))
SCOPES = ('all160_positions',)+tuple(f'step_{s}' for s in range(4))+tuple(f'block_{b:02}' for b in range(40))
SHAPES = {'V1': [ROWS, D], 'h': [ROWS, H], 'Y': [ROWS, D], 'r': [ROWS, D],
          'u': [ROWS, D], 'gate': [D], 'shift': [D], 'scale': [D], 'rows': [ROWS]}
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 900, 'outer_seconds': 1200,
             'packet_full_SHA_stream_bytes_max': 32 << 30, 'new_HF_bytes': 64 << 20,
             'automatic_retry': False}
DESIGN = {
    'evidence': 'SAVED_TEACHER_STATE_OBSERVATIONAL_GEOMETRY',
    'sources': 'ALL8_HISTORICALLY_EXPOSED_IMAGES_6CAL_2DEV_NO_SELECTION_OR_NEW_SAMPLING',
    'unit': 'PAIRS_NESTED_IN_SOURCE_STEP_BLOCK_THEN_POSITION_EQUAL_THEN_IMAGE_SOURCE_EQUAL',
    'row': 'tau*(29*52)+y*52+x; actual original row IDs; no motion alignment',
    'temporal': 'ALL_UNORDERED_SAME_XY_CROSS_TAU_PAIRS_ORIENTED_TAU_A_LT_TAU_B',
    'control': 'SAME_PACKET_SAME_TAU_B_NEXT_ROW_ID_CYCLIC_DIFFERENT_XY_KEEP_IA_FIXED',
    'missing_control': 'NOT_AVAILABLE_NO_IMPUTATION_TEMPORAL_ALL_AND_MATCHED_REPORTED_SEPARATELY',
    'X': 'ACTUAL_RAW_BF16_V1_FIRST_FFN_MM_INPUT_NO_Q1_QUANTIZATION',
    'gY': 'FP64_PRODUCT_OF_DECODED_BF16_Y_AND_ACTUAL_POSITION_BF16_GATE_NOT_NATIVE_ROUNDED_WRITE',
    'rowLN_u': 'FP64_ROW_CENTER_AND_POPULATION_VARIANCE_EPS1E-6_GEOMETRY_NO_AFFINE',
    'energy': 'SUM_F64((a-b)^2)/(SUM_F64(a^2)+SUM_F64(b^2)+1E-30)',
    'cosine': 'SUM_F64(a*b)/SQRT(SUM_F64(a^2)*SUM_F64(b^2)); ZERO_NORM_IS_NULL',
    'cosine_missingness': 'EACH_COSINE_HAS_ITS_OWN_NONZERO_NORM_COUNT; PAIRED_COSINE_DELTA_REQUIRES_BOTH; DO_NOT_SUBTRACT_DIFFERENT_ELIGIBILITY_MEANS',
    'paired_difference': 'TEMPORAL_MINUS_CONTROL_ON_IDENTICAL_AVAILABLE_PAIR_SET',
    'strata': 'TAU0_PARTICIPATION_AND_DELTA_TAU_SEPARATE; STEP_AND_BLOCK_SEPARATE; FULL_JOINT_COORDINATES_IN_PAIR_RECORDS',
    'interpretation': 'SPARSE_OBS_NOT_COMPLETE21_TEMPORAL_VARIANCE_OR_MOTION_ALIGNMENT_OR_VIDEO_QUALITY_RANKING',
    'numerical': 'BF16_U16_TO_U32_SHIFT16_TO_F32; F64_GEOMETRY_AND_SUMS; SINGLE_PACKET_LIFETIME',
    'full_model_calls': 0, 'GPU_calls': 0, 'fitting': 0, 'decompositions': 0,
    'automatic_retry': False}
REQUIRED = ('tools/e0_raw_bf16_temporal_pair_diagnostic.py',
            'tests/test_e0_raw_bf16_temporal_pair_diagnostic.py',
            'tools/raw_bf16_temporal_row_replay.py', 'tools/worker_identity.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def binding(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path.is_relative_to(root) and path.is_file(), 'Confined regular input required')
    value = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(value), 'sha256': digest(value)}


def bound(root, item):
    need(isinstance(item, dict) and type(item.get('bytes')) is int and item['bytes'] > 0
         and isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
         and set(item['sha256']) <= set('0123456789abcdef'), 'Malformed binding')
    path = (Path(root)/item['path']).resolve()
    need(path.is_relative_to(Path(root).resolve()) and path.is_file(), 'Input path escaped root')
    raw = path.read_bytes()
    need(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Input metadata binding differs')
    return raw


def row_draw(ordinal, step, block):
    return sorted(random.Random(42+(ordinal+1)*1000003+1009*step+37*block).sample(range(math.prod(GRID)), 512)[:ROWS])


def temporal_pairs(rows, grid=GRID):
    """Indices refer to original payload order; all choices depend only on row IDs."""
    tau_count, height, width = grid
    plane = height*width
    need(len(rows) == len(set(rows)) and all(type(r) is int and 0 <= r < tau_count*plane for r in rows),
         'Invalid native row coordinates')
    ordered = sorted((row, i) for i, row in enumerate(rows))
    by_xy, by_tau = defaultdict(list), defaultdict(list)
    for row, i in ordered:
        by_xy[row % plane].append((row, i))
        by_tau[row//plane].append((row, i))
    successor = {}
    for values in by_tau.values():
        for j, (row, i) in enumerate(values):
            other_row, other_i = values[(j+1) % len(values)]
            successor[i] = other_i if other_row % plane != row % plane else None
    pairs = []
    for values in by_xy.values():
        for (ra, ia), (rb, ib) in itertools.combinations(values, 2):
            ta, tb = ra//plane, rb//plane
            need(ta < tb, 'Temporal pair orientation differs')
            ic = successor[ib]
            need(ic is None or (rows[ic]//plane == tb and rows[ic] % plane != rb % plane),
                 'Control did not preserve target time and change spatial identity')
            pairs.append({'ia': ia, 'ib': ib, 'control_ib': ic, 'tau_a': ta, 'tau_b': tb,
                          'delta_tau': tb-ta, 'tau0_participates': ta == 0})
    return sorted(pairs, key=lambda p: (rows[p['ia']], rows[p['ib']]))


def row_schedule(index, replay):
    recorded = {(r['source'], r['step'], r['block']): r for r in replay['packets_by_source_step_block']}
    need(len(recorded) == 1280, 'Original row replay incomplete')
    summary = {'packets': 0, 'temporal_pairs': 0, 'available_controls': 0, 'NOT_AVAILABLE_controls': 0,
               'tau0_pairs': 0, 'adjacent_tau_pairs': 0, 'packets_without_adjacent_tau_pair': 0,
               'complete_21_tau_xy_trajectories': 0, 'pair_schedule_records': []}
    for case in sorted(index['cases'], key=lambda c: c['native_source_ordinal']):
        for entry in sorted(case['packets'], key=lambda p: (p['step'], p['block'])):
            ordinal, step, block = case['native_source_ordinal'], entry['step'], entry['block']
            rows = row_draw(ordinal, step, block)
            pairs = temporal_pairs(rows)
            raw_rows = struct.pack('<256q', *rows)
            old = recorded[(ordinal, step, block)]
            adjacent = sum(p['delta_tau'] == 1 for p in pairs)
            complete = sum(n == 21 for n in Counter(r % (29*52) for r in rows).values())
            need(digest(raw_rows) == entry['tensor_metadata']['rows']['raw_tensor_sha256']
                 and old['role'] == case['role'] and old['same_xy_cross_tau_pairs'] == len(pairs)
                 and old['adjacent_tau_pairs'] == adjacent and old['complete_21_tau_xy'] == complete,
                 'Frozen deterministic coordinates differ from actual row replay')
            summary['packets'] += 1
            summary['temporal_pairs'] += len(pairs)
            summary['available_controls'] += sum(p['control_ib'] is not None for p in pairs)
            summary['NOT_AVAILABLE_controls'] += sum(p['control_ib'] is None for p in pairs)
            summary['tau0_pairs'] += sum(p['tau0_participates'] for p in pairs)
            summary['adjacent_tau_pairs'] += adjacent
            summary['packets_without_adjacent_tau_pair'] += adjacent == 0
            summary['complete_21_tau_xy_trajectories'] += complete
            summary['pair_schedule_records'].append([ordinal, step, block, digest(raw_rows), canonical(pairs)])
    need(summary['packets'] == 1280 and summary['temporal_pairs'] == 26389
         and summary['adjacent_tau_pairs'] == 2511 and summary['packets_without_adjacent_tau_pair'] == 178
         and summary['complete_21_tau_xy_trajectories'] == 0, 'Actual row coverage differs')
    summary['pair_schedule_sha256'] = canonical(summary.pop('pair_schedule_records'))
    return summary


def load_inputs(plan, hf):
    proof = plan['operand_audit']
    parent = json.loads(bound(ROOT, proof['protocol']))
    run = json.loads(bound(hf, proof['RUN']))
    exit_ = json.loads(bound(hf, proof['exit']))
    need(parent['id'] == run['id'] == exit_['id'] == 'E0_RAW_BF16_FULL_DEPTH_ASSET_AUDIT_20261029'
         and run['status'] == 'COMPLETE_ASSET_BYTES_AUDITED'
         and run['protocol_sha256'] == exit_['protocol_sha256'] == proof['protocol']['sha256']
         and exit_['run_sha256'] == proof['RUN']['sha256'] and exit_['process_exited'] is True
         and type(exit_['child_returncode']) is int and exit_['child_returncode'] == 0
         and exit_['timeout'] is False and exit_['automatic_retry'] is False
         and run['manifest'] == proof['manifest'], 'Actual closed operand audit exit0 required')
    index = json.loads(bound(hf, proof['manifest']))
    corpus = json.loads(bound(ROOT, plan['corpus_report']))
    replay = json.loads(bound(ROOT, plan['row_replay']))
    need(corpus['status'] == 'COMPLETE_ASSET_AUDIT_ACTUAL_CPU_EXIT0'
         and all(corpus[k] == proof[k] for k in ('protocol', 'RUN', 'exit', 'manifest'))
         and replay['status'] == 'OBS_ROW_COORDINATES_ONLY_NO_MODEL_OR_OPERAND_PAYLOAD'
         and replay['manifest'] == proof['manifest'], 'Original corpus/row provenance differs')
    need(index['schema'] == 'ffnds.raw_bf16.full_depth.operands_index.v1'
         and index['totals'] == {'sources': 8, 'calibration': 6, 'development': 2, 'packets': 1280, 'native_positions': 160}
         and len(index['cases']) == 8 and len({c['image_sha256'] for c in index['cases']}) == 8
         and sorted(c['native_source_ordinal'] for c in index['cases'] if c['role'] == 'calibration') == list(CAL)
         and sorted(c['native_source_ordinal'] for c in index['cases'] if c['role'] == 'development') == list(DEV),
         'Eight-source role/geometry differs')
    paths = []
    for case in index['cases']:
        need(case['source_state'] == 'teacher_input' and case['history'] == 'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT'
             and len(case['packets']) == 160 and {(p['step'], p['block']) for p in case['packets']}
             == {(s, b) for s in range(4) for b in range(40)}, 'Teacher/source positions differ')
        paths.extend(p['payload']['path'] for p in case['packets'])
    need(len(set(paths)) == 1280, 'Duplicate operand paths')
    return index, row_schedule(index, replay)


def limits(index, schedule):
    packets = [p for c in index['cases'] for p in c['packets']]
    packet_bytes = sum(p['payload']['bytes'] for p in packets)
    metadata_bytes = sum(p['metadata']['bytes'] for p in packets)
    need(packet_bytes <= RESOURCES['packet_full_SHA_stream_bytes_max'], 'Packet stream exceeds 32GiB')
    return {'packet_full_SHA_reads': 1280, 'packet_full_SHA_bytes': packet_bytes,
            'packet_metadata_reads': 1280, 'packet_metadata_bytes': metadata_bytes,
            'tensor_raw_SHA_checks': 1280*len(SHAPES), 'row_coordinate_checks': 1280,
            'pair_index_constructions': 1280, 'operand_pair_evaluations': 1280*len(OPERANDS),
            'temporal_pair_comparisons': schedule['temporal_pairs']*len(OPERANDS),
            'matched_control_comparisons': schedule['available_controls']*len(OPERANDS),
            'gY_FP64_products': 1280, 'rowLN_FP64_calls': 1280,
            'model_calls': 0, 'GPU_calls': 0, 'fits': 0, 'decompositions': 0, 'new_videos': 0}


def make_draft(hf):
    proof = read(ROOT/CORPUS)
    plan = {'id': ID, 'status': 'DRAFT_NOT_REGISTERED', 'assets_finalized': False,
            'binding_status': 'BOUND_INPUTS_REVIEW_AND_FREEZE_PENDING', 'resources': RESOURCES,
            'design': DESIGN, 'automatic_retry': False, 'output_HF_relative': OUTPUT,
            'operand_audit': {k: proof[k] for k in ('protocol', 'RUN', 'exit', 'manifest')},
            'corpus_report': binding(ROOT/CORPUS, ROOT), 'row_replay': binding(ROOT/REPLAY, ROOT),
            'numpy_version': importlib.metadata.version('numpy'),
            'repo_sources': [binding(ROOT/p, ROOT) for p in REQUIRED]}
    plan['research_question'] = 'Does same-xy cross-tau geometry differ from strictly time-matched different-xy controls across the full native FFN depth?'
    plan['competing_mechanisms'] = [
        'SOURCE_SPATIAL_APPEARANCE_OR_RESIDUAL_ANCHOR_EXPLAINS_SIMILARITY_ALREADY_PRESENT_IN_X_OR_R',
        'LAYER_AND_STEP_DEPENDENT_FFN_OUTPUT_GEOMETRY_ADDS_CONDITIONAL_TEMPORAL_SHARING_VISIBLE_IN_Y_GY_OR_ROWLN_U']
    plan['claim_boundary'] = 'OBSERVATIONAL_SOURCE_EQUAL_GEOMETRY_NO_PREDICTOR_NO_INLINE_ACTION_NO_MOTION_OR_VIDEO_ACCEPTANCE'
    index, schedule = load_inputs(plan, hf)
    plan.update(row_schedule=schedule, limits=limits(index, schedule),
                sources=[{k: c[k] for k in ('source_id', 'native_source_ordinal', 'image_sha256', 'prompt_sha256', 'role', 'history', 'source_state')}
                         for c in sorted(index['cases'], key=lambda c: c['native_source_ordinal'])])
    return plan


def validate(plan, frozen=False):
    need(plan.get('id') == ID and plan.get('design') == DESIGN and plan.get('resources') == RESOURCES
         and plan.get('automatic_retry') is False and plan.get('output_HF_relative') == OUTPUT,
         'Fixed saved-data OBS scope differs')
    need(sorted(x['path'] for x in plan['repo_sources']) == sorted(REQUIRED)
         and plan['corpus_report']['path'] == CORPUS and plan['row_replay']['path'] == REPLAY, 'Source/metadata closure differs')
    for item in plan['repo_sources']:
        bound(ROOT, item)
    if frozen:
        need(plan.get('status') == 'FROZEN_REGISTERED' and plan.get('assets_finalized') is True
             and isinstance(plan.get('independent_review'), dict), 'Freeze and independent exact review required')


def decode_bf16(raw, shape, np):
    need(len(raw) == 2*math.prod(shape), 'BF16 byte geometry differs')
    value = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
    need(np.isfinite(value).all(), 'Nonfinite BF16 operand')
    return value


def decode_packet(raw, entry, expected_rows, np, shapes=SHAPES, *, whole_SHA_already_stream_verified=False):
    """Accept only a complete byte-bound safetensor; no Torch/safetensors loader."""
    need(len(raw) == entry['payload']['bytes'] and (whole_SHA_already_stream_verified
         or digest(raw) == entry['payload']['sha256']), 'Whole packet SHA/bytes differ')
    need(len(raw) >= 8, 'Short safetensor prefix')
    n = struct.unpack('<Q', raw[:8])[0]
    need(0 < n <= 1 << 20 and 8+n <= len(raw), 'Invalid safetensor header size')
    header = json.loads(raw[8:8+n])
    need(set(header)-{'__metadata__'} == set(shapes) == set(entry['tensor_metadata']), 'Packet tensor names differ')
    end, decoded = 0, {}
    body = memoryview(raw)[8+n:]
    for name in sorted(shapes, key=lambda k: header[k]['data_offsets'][0]):
        info, meta = header[name], entry['tensor_metadata'][name]
        a, b = info['data_offsets']
        need(info['shape'] == meta['shape'] == shapes[name] and type(a) is int and type(b) is int
             and a == end and b-a == math.prod(shapes[name])*(8 if name == 'rows' else 2)
             and info['dtype'] == ('I64' if name == 'rows' else 'BF16')
             and meta['dtype'] == ('int64' if name == 'rows' else 'bfloat16') and b <= len(body),
             'Operand header/shape/dtype/offset differs')
        value = body[a:b]
        need(digest(value) == meta['raw_tensor_sha256'], 'Operand raw tensor SHA differs')
        if name == 'rows':
            decoded[name] = np.frombuffer(value, dtype='<i8').tolist()
        elif name not in ('scale', 'shift'):
            decoded[name] = decode_bf16(value, shapes[name], np)
        end = b
    need(end == len(body) and decoded['rows'] == expected_rows, 'Trailing bytes or actual row coordinates differ')
    return decoded


def geometric_operands(packet, np):
    u = packet['u'].astype(np.float64)
    z = u-u.mean(axis=1, keepdims=True)
    ln = z/np.sqrt(np.mean(z*z, axis=1, keepdims=True)+1e-6)
    return {'X': packet['V1'], 'h': packet['h'], 'Y': packet['Y'],
            'gY': packet['Y'].astype(np.float64)*packet['gate'].astype(np.float64),
            'R': packet['r'], 'u': packet['u'], 'rowLN_u': ln}


def pair_geometry(values, pairs, np):
    """Return arrays; cosine zero norms remain NaN internally and JSON null."""
    n = len(pairs)
    result = np.full((n, len(METRICS)), np.nan, dtype=np.float64)
    if not n:
        return result
    ia = np.array([p['ia'] for p in pairs], dtype=np.int64)
    ib = np.array([p['ib'] for p in pairs], dtype=np.int64)
    def measure(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        aa, bb = np.sum(a*a, axis=1, dtype=np.float64), np.sum(b*b, axis=1, dtype=np.float64)
        energy = np.sum((a-b)**2, axis=1, dtype=np.float64)/(aa+bb+1e-30)
        denom = np.sqrt(aa*bb)
        cosine = np.full(len(a), np.nan, dtype=np.float64)
        np.divide(np.sum(a*b, axis=1, dtype=np.float64), denom, out=cosine, where=denom > 0)
        need(np.isfinite(energy).all() and ((energy >= 0) & (energy <= 2+1e-12)).all()
             and (np.isnan(cosine) | (np.abs(cosine) <= 1+1e-12)).all(), 'Invalid pair geometry')
        return energy, cosine
    result[:, 0], result[:, 1] = measure(values[ia], values[ib])
    available = np.array([p['control_ib'] is not None for p in pairs], dtype=bool)
    if available.any():
        ic = np.array([p['control_ib'] for p in pairs if p['control_ib'] is not None], dtype=np.int64)
        result[available, 2:4] = result[available, :2]
        result[available, 4], result[available, 5] = measure(values[ia[available]], values[ic])
        result[available, 6] = result[available, 2]-result[available, 4]
        result[available, 7] = result[available, 3]-result[available, 5]
    return result


def json_values(value, np):
    if isinstance(value, np.ndarray):
        return json_values(value.tolist(), np)
    if isinstance(value, list):
        return [json_values(v, np) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


class SourceEqualSummary:
    """Average eligible pairs in each position, positions in each image, images."""
    def __init__(self, np, ordinals=ORDINALS):
        self.np, self.ordinals = np, tuple(ordinals)
        shape = (len(ordinals), len(SCOPES), len(TIME_BUCKETS), len(OPERANDS), len(METRICS))
        self.sums, self.counts = np.zeros(shape), np.zeros(shape, dtype=np.int64)
        self.positions = np.full((len(ordinals), 4, 40, len(OPERANDS), len(METRICS)), np.nan)
        self.position_pair_counts = np.zeros(self.positions.shape, dtype=np.int64)
        self.seen = set()
    def add(self, ordinal, step, block, pairs, values):
        np = self.np
        key = (ordinal, step, block)
        need(key not in self.seen and values.shape == (len(OPERANDS), len(pairs), len(METRICS)), 'Duplicate/wrong position summary')
        self.seen.add(key)
        source = self.ordinals.index(ordinal)
        scopes = [0, 1+step, 5+block]
        masks = [np.ones(len(pairs), dtype=bool), np.array([p['tau0_participates'] for p in pairs], dtype=bool),
                 np.array([not p['tau0_participates'] for p in pairs], dtype=bool)]
        masks += [np.array([p['delta_tau'] == d for p in pairs], dtype=bool) for d in range(1, 21)]
        for bucket, mask in enumerate(masks):
            chosen = values[:, mask, :]
            valid = np.isfinite(chosen)
            count = valid.sum(axis=1)
            mean = np.zeros(count.shape)
            np.divide(np.where(valid, chosen, 0).sum(axis=1), count, out=mean, where=count > 0)
            for scope in scopes:
                self.sums[source, scope, bucket] += np.where(count > 0, mean, 0)
                self.counts[source, scope, bucket] += count > 0
            if bucket == 0:
                self.positions[source, step, block] = np.where(count > 0, mean, np.nan)
                self.position_pair_counts[source, step, block] = count
    def result(self):
        np = self.np
        source = np.full(self.sums.shape, np.nan)
        np.divide(self.sums, self.counts, out=source, where=self.counts > 0)
        valid = np.isfinite(source)
        counts = valid.sum(axis=0)
        equal = np.full(counts.shape, np.nan)
        np.divide(np.where(valid, source, 0).sum(axis=0), counts, out=equal, where=counts > 0)
        pvalid = np.isfinite(self.positions)
        pcount = pvalid.sum(axis=0)
        pequal = np.full(pcount.shape, np.nan)
        np.divide(np.where(pvalid, self.positions, 0).sum(axis=0), pcount, out=pequal, where=pcount > 0)
        return {'axis_order_source_tables': ['source', 'scope', 'time_bucket', 'operand', 'metric'],
                'source_ordinals': list(self.ordinals), 'scope': list(SCOPES), 'time_bucket': list(TIME_BUCKETS),
                'operand': list(OPERANDS), 'metric': list(METRICS),
                'source_means': json_values(source, np), 'source_eligible_position_counts': self.counts.tolist(),
                'source_equal_means_axis_without_source': json_values(equal, np), 'available_source_counts': counts.tolist(),
                'position_axis_order': ['source', 'step', 'block', 'operand', 'metric'],
                'all_pairs_per_source_position_means': json_values(self.positions, np),
                'all_pairs_per_source_position_valid_pair_counts': self.position_pair_counts.tolist(),
                'all_pairs_position_source_equal_means': json_values(pequal, np),
                'all_pairs_position_available_source_counts': pcount.tolist(),
                'weighting': 'PAIR_MEAN_WITHIN_ELIGIBLE_POSITION_THEN_EQUAL_POSITION_MEAN_WITHIN_SOURCE_THEN_EQUAL_AVAILABLE_SOURCE_MEAN',
                'missing': 'JSON_NULL_IS_NOT_MEASURED_OR_ZERO_NORM_COSINE; COUNTS_DISTINGUISH_AVAILABILITY; NO_ZERO_IMPUTATION',
                'joint_strata': 'FULL_STEP_BLOCK_TAU_A_TAU_B_XY_IN_PAIRS_JSONL; SUMMARY_SEPARATES_STEP_AND_BLOCK'}


class OutputBudget:
    """Track writes incrementally; atomic replacement checks temporary peak."""
    def __init__(self, out, limit):
        self.out, self.limit, self.sizes, self.streams = out, limit, {}, {}
    def write(self, name, raw, *, emergency=False):
        path = self.out/name
        allowance = self.limit if emergency else self.limit-(1 << 20)
        need(sum(self.sizes.values())+len(raw) <= allowance, 'Output atomic peak budget exceeded')
        tmp = path.with_name(path.name+'.tmp')
        with tmp.open('wb', buffering=0) as stream:
            count = stream.write(raw)
            self.sizes[tmp.name] = count
            need(count == len(raw), 'Partial atomic output write')
        tmp.replace(path)
        self.sizes.pop(tmp.name)
        self.sizes[name] = len(raw)
    def append(self, name, raw):
        need(sum(self.sizes.values())+len(raw) <= self.limit-(1 << 20), 'Output append budget exceeded')
        if name not in self.streams:
            self.streams[name] = (self.out/name).open('xb', buffering=0)
        count = self.streams[name].write(raw)
        self.sizes[name] = self.sizes.get(name, 0)+count
        need(count == len(raw), 'Partial output append')
    def json(self, name, value, *, emergency=False):
        self.write(name, (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode(), emergency=emergency)
    def close(self):
        for stream in self.streams.values():
            stream.close()
    def final_audit(self):
        # One final, flat audit captures a partial write interrupted by a signal.
        # There are only this scope's own few output files; never a hot-loop walk.
        measured = {}
        with os.scandir(self.out) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                need(stat.S_ISREG(info.st_mode), 'Unexpected non-file in CPU output')
                measured[entry.name] = info.st_size
        self.sizes = measured
        need(sum(measured.values()) <= self.limit, 'Actual CPU output bytes exceeded budget')
        return sum(measured.values())


class Meter:
    def __init__(self, report, output, bounds):
        self.report, self.output, self.bounds = report, output, bounds
        report['operation_ledger'] = {'predebited': {k: 0 for k in bounds}, 'observed': {k: 0 for k in bounds}}
    def work(self, costs, fn, *args, **kwargs):
        ledger = self.report['operation_ledger']
        need(costs and all(k in self.bounds and type(n) is int and n > 0 and ledger['predebited'][k]+n <= self.bounds[k]
                           for k, n in costs.items()), 'Fixed CPU OBS operation budget exceeded')
        for key, n in costs.items():
            ledger['predebited'][key] += n
        self.output.append('operations.jsonl', (json.dumps({'event': 'RESERVED_BEFORE_CALL', 'costs': costs}, separators=(',', ':'))+'\n').encode())
        result = fn(*args, **kwargs)
        for key, n in costs.items():
            ledger['observed'][key] += n
        return result
    def finish(self):
        need(all(self.report['operation_ledger'][side] == self.bounds for side in ('predebited', 'observed')),
             'Incomplete exact saved-data scope')


def authorize(plan, protocol_sha):
    validate(plan, True)
    active = read(ROOT/'research/ACTIVE_RESEARCH.json')
    need(active['user_continuation_authorization_20261027']['status'] == 'ACTIVE_CURRENT_USER_DIRECTED_SESSION',
         'Current research authorization inactive')
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    need(len(matches) == 1 and matches[0].get('protocol_sha256') == protocol_sha
         and matches[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
         and matches[0].get('limits') == plan['limits'] and matches[0].get('resources') == RESOURCES
         and matches[0].get('max_gpu_cards') == 0, 'Unique exact CPU OBS scope not registered')


def read_stream(hf, item, io, category):
    path = (hf/item['path']).resolve()
    need(path.is_relative_to(hf.resolve()), 'Payload path escaped root')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == item['bytes'], 'Payload path/size differs')
    chunks, count, sha = [], 0, hashlib.sha256()
    with path.open('rb', buffering=0) as stream:
        while True:
            chunk = stream.read(min(8 << 20, item['bytes']+1-count))
            if not chunk:
                break
            count += len(chunk)
            io[category] = io.get(category, 0)+len(chunk)
            need(io.get('packet_bytes', 0) <= RESOURCES['packet_full_SHA_stream_bytes_max'], 'Packet full-SHA stream hard budget exceeded')
            chunks.append(chunk)
            sha.update(chunk)
            need(count <= item['bytes'], 'Input grew past frozen bytes')
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
         == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
         and count == item['bytes'] and sha.hexdigest() == item['sha256'], 'Streamed SHA/stat/size differs')
    return b''.join(chunks)


def inspect(plan, hf):
    validate(plan)
    index, schedule = load_inputs(plan, hf)
    need(schedule == plan['row_schedule'] and limits(index, schedule) == plan['limits'], 'Frozen schedule/budget differs')
    return {'id': ID, 'status': 'PASS_METADATA_ONLY_NO_OPERAND_VALUES_READ', 'row_schedule': schedule,
            'limits': plan['limits'], 'model_calls': 0, 'GPU_calls': 0, 'packet_operand_bytes_read': 0}


def run(args):
    started = time.monotonic()
    plan = read(args.protocol)
    protocol_sha = digest(args.protocol.read_bytes())
    authorize(plan, protocol_sha)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 4
         and not {'numpy', 'torch'} & set(sys.modules), 'Fresh CPU-only four-thread worker required')
    if str(ROOT/'tools') not in sys.path:
        sys.path.insert(0, str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker()
    out = (args.hf_root/OUTPUT).resolve()
    need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined CPU output; no automatic retry')
    out.mkdir(parents=True)
    # Leave a further MiB outside this writer for the durable supervisor receipt/logs.
    output = OutputBudget(out, RESOURCES['new_HF_bytes']-(1 << 20))
    report = {'id': ID, 'protocol_sha256': protocol_sha, 'status': 'STARTED_CPU_SAVED_TEACHER_PAIR_OBS_NO_RETRY',
              'automatic_retry': False, 'model_calls': 0, 'GPU_calls': 0, 'fits': 0, 'decompositions': 0,
              'source_state': 'teacher_input', 'historical_exposure': 'ALL8_PREVIOUSLY_EXPOSED_6CAL_2DEV',
              'operand_audit': plan['operand_audit'], 'row_schedule': plan['row_schedule'],
              'design': DESIGN, 'actual_input_IO': {}, 'completed_packets': 0,
              'output_budget_includes_external_supervisor_reserve_bytes': 1 << 20,
              'quality_or_speed_acceptance': False, 'motion_alignment': 'NOT_MEASURED',
              'complete21_trajectory_variance': 'NOT_MEASURED_NO_COMPLETE_TRAJECTORIES'}
    meter = Meter(report, output, plan['limits'])
    def save():
        report['wall_seconds'] = time.monotonic()-started
        report['tracked_output_bytes_before_RUN_replace'] = sum(output.sizes.values())
        output.json('RUN.json', report, emergency=True)
    def timeout(*_):
        raise TimeoutError('Fixed CPU OBS wall900 reached; no retry')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    save()
    try:
        inspect(plan, args.hf_root)
        index, _ = load_inputs(plan, args.hf_root)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        import numpy as np
        need(np.__version__ == plan['numpy_version'] and 'torch' not in sys.modules, 'NumPy-only environment/version differs')
        report['numpy_version'] = np.__version__
        summary = SourceEqualSummary(np)
        for case in sorted(index['cases'], key=lambda c: c['native_source_ordinal']):
            ordinal = case['native_source_ordinal']
            for entry in sorted(case['packets'], key=lambda p: (p['step'], p['block'])):
                step, block = entry['step'], entry['block']
                rows = row_draw(ordinal, step, block)
                pairs = meter.work({'pair_index_constructions': 1}, temporal_pairs, rows)
                meta = meter.work({'packet_metadata_reads': 1, 'packet_metadata_bytes': entry['metadata']['bytes']},
                    read_stream, args.hf_root, entry['metadata'], report['actual_input_IO'], 'metadata_bytes')
                meta = json.loads(meta)
                need(meta['selected_rows'] == rows and meta['grid'] == list(GRID) and meta['step'] == step
                     and meta['block'] == block and meta['native_source_ordinal'] == ordinal
                     and meta['source_id'] == case['source_id'] and meta['tensor_metadata'] == entry['tensor_metadata'],
                     'Metadata/row/source correspondence differs')
                raw = meter.work({'packet_full_SHA_reads': 1, 'packet_full_SHA_bytes': entry['payload']['bytes']},
                    read_stream, args.hf_root, entry['payload'], report['actual_input_IO'], 'packet_bytes')
                packet = meter.work({'tensor_raw_SHA_checks': len(SHAPES), 'row_coordinate_checks': 1},
                    decode_packet, raw, entry, rows, np, whole_SHA_already_stream_verified=True)
                operands = meter.work({'gY_FP64_products': 1, 'rowLN_FP64_calls': 1}, geometric_operands, packet, np)
                costs = {'operand_pair_evaluations': len(OPERANDS), 'temporal_pair_comparisons': len(pairs)*len(OPERANDS)}
                available = sum(p['control_ib'] is not None for p in pairs)
                if available:
                    costs['matched_control_comparisons'] = available*len(OPERANDS)
                values = meter.work(costs, lambda: np.stack([pair_geometry(operands[k], pairs, np) for k in OPERANDS]))
                record = {'native_source_ordinal': ordinal, 'role': case['role'], 'step': step, 'block': block,
                          'payload_sha256': entry['payload']['sha256'], 'rows_raw_sha256': entry['tensor_metadata']['rows']['raw_tensor_sha256'],
                          'rows': rows, 'pair_fields': ['ia', 'ib', 'control_ib'],
                          'pairs': [[p['ia'], p['ib'], p['control_ib']] for p in pairs],
                          'metric_axes': ['operand', 'pair', 'metric'], 'operand_order': list(OPERANDS), 'metric_order': list(METRICS),
                          'values': json_values(values, np), 'NOT_AVAILABLE_control_count': len(pairs)-available,
                          'coordinates': 'DERIVE_TAU_XY_FROM_ACTUAL_ROWS_AND_GRID21_29_52; TAU_A_LT_TAU_B'}
                output.append('PAIRS.jsonl', (json.dumps(record, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode())
                summary.add(ordinal, step, block, pairs, values)
                report['completed_packets'] += 1
                report['last_completed_packet'] = [ordinal, step, block]
                save()
                del raw, packet, operands, values, record
        meter.finish()
        need(len(summary.seen) == 1280 and report['actual_input_IO']['packet_bytes'] == plan['limits']['packet_full_SHA_bytes']
             and report['actual_input_IO']['metadata_bytes'] == plan['limits']['packet_metadata_bytes']
             and 'torch' not in sys.modules, 'Incomplete saved-data/CPU identity')
        result = summary.result()
        result.update(id=ID, protocol_sha256=protocol_sha, design=DESIGN, sources=plan['sources'], row_schedule=plan['row_schedule'])
        output.json('SUMMARY.json', result)
        report.update(status='COMPLETE_RAW_BF16_S2_SAVED_TEACHER_TEMPORAL_PAIR_OBS_CPU',
                      result_files={name: binding(out/name, args.hf_root) for name in ('PAIRS.jsonl', 'SUMMARY.json', 'operations.jsonl')})
    except BaseException as error:
        report.update(status='FAILED_S2_TEMPORAL_PAIR_OBS_CPU_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        import resource
        report['max_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['Torch_imported'] = 'torch' in sys.modules
        seen = summary.seen if 'summary' in locals() else set()
        report['sources_with_at_least_one_completed_packet'] = sorted({key[0] for key in seen})
        report['completed_sources_all160_positions'] = sorted(o for o in ORDINALS if sum(key[0] == o for key in seen) == 160)
        try:
            report['actual_output_bytes_final_flat_audit_before_RUN_replace'] = output.final_audit()
            save()
        finally:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'inspect', 'run'))
    parser.add_argument('--hf-root', type=Path, default=HF)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.hf_root = args.hf_root.resolve()
    if args.action == 'draft':
        need(args.out is not None and not args.out.exists(), 'Fresh draft output required')
        args.out.write_text(json.dumps(make_draft(args.hf_root), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    elif args.action == 'inspect':
        print(json.dumps(inspect(read(args.protocol), args.hf_root), indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()
