"""Read-only CAL24 label provenance, budget ledger and safetensors-header contract.

No model/runtime imports. Protocol supplies ``inputs`` and ``sources`` in the
format of the frozen CAL24 binding; a future producer must bind this helper and
its own output/run identities in a separately registered protocol.
"""
import hashlib
import json
import struct
from pathlib import Path

M, D, ROWS = 31668, 5120, 256
TOTAL = {'encoder_requests': 24, 'noise_preparations': 24, 'dense_flows': 96,
         'native_blocks': 3840, 'native_ffn_calls': 3840,
         'native_ffn_linears': 7680, 'native_ffn_qdq': 7680,
         'native_non_ffn_linears': 30720, 'native_non_ffn_qdq': 30720,
         'head_calls': 96, 'layer_norm_calls': 96, 'scheduler_pre': 96,
         'scheduler_post': 96, 'decoder_calls': 0, 'training_updates': 0}
SETUP = 'NOT_BOUND'
PACK_SCHEMA = {'r_rows': ('BF16', [ROWS, D]), 'u_rows': ('BF16', [ROWS, D]),
               'head_rows': ('BF16', [ROWS, 64]), 'gate': ('BF16', [1, D]),
               'head_shift': ('BF16', [1, 1, D]), 'head_scale': ('BF16', [1, 1, D]),
               'rows': ('I64', [ROWS])}
META_KEYS = ('case_id', 'input_sha256', 'source_image_sha256', 'seed', 'step',
             'expert', 'historical_post_sha256', 'operand_sha256')
EVIDENCE_SHA_KEYS = ('head_weight_raw_sha256', 'head_bias_raw_sha256', 'head_modulation_raw_sha256', 'actual_post_sha256')
EVIDENCE_BOOL_KEYS = ('original_v1_byte_equal', 'original_y_byte_equal', 'original_gate_byte_equal')
EVIDENCE_STRIDE_KEYS = ('head_weight_stride', 'head_bias_stride', 'head_modulation_stride', 'head_shift_stride', 'head_scale_stride')


def rows_sha256(rows):
    """SHA of 256 ordered signed little-endian int64 row values."""
    _check(len(rows) == ROWS and len(set(rows)) == ROWS and all(type(x) is int and 0 <= x < M for x in rows), 'invalid rows')
    return hashlib.sha256(struct.pack('<' + 'q' * ROWS, *rows)).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    _check(path.is_relative_to(root), 'path outside HF root')
    return path


def build_cases(protocol, hf_root):
    """Verify 24 original INPUT records and images; return 96 bound cases.

    ``protocol`` is a dict containing exact frozen ``inputs``, ``sources`` and
    ``source_split``. Reads INPUT JSON and image files; no operand tensor bytes
    are read. ``part`` here is source-index // 6, independent of bank parts.
    """
    root = Path(hf_root).resolve()
    inputs, sources, split = protocol['inputs'], protocol['sources'], protocol['source_split']
    _check(len(inputs) == len(sources) == 24, 'expected 24 inputs and sources')
    by_i = {v['index']: v for v in inputs}; by_s = {v['index']: v for v in sources}
    _check(set(by_i) == set(by_s) == set(range(24)), 'duplicate or missing source indices')
    _check(split['training'] == list(range(16)) and split['diagnostic_heldout'] == list(range(16, 24)), 'split differs')
    _check(len({s['source_id'] for s in sources}) == 24, 'duplicate image clusters')
    cases = []
    for i in range(24):
        binding, source = by_i[i], by_s[i]
        path = _inside(root, binding['path'])
        _check(path.stat().st_size == binding['bytes'] and _sha(path) == binding['sha256'], 'INPUT binding differs')
        record = json.loads(path.read_text())
        _check(record['status'] == 'PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP' and record['index'] == i, 'INPUT status/index differs')
        item = record['input']; provenance = item['input_provenance']
        _check(source['seed'] == item['calibration_seed'] == 42 + i, 'seed differs')
        _check(source['source_id'] == provenance['image_sha256'] and source['source_group'] == item['source_group'], 'image/source differs')
        _check(source['dimension'] == item['dimension'], 'dimension differs')
        _check(hashlib.sha256(item['prompt'].encode('utf-8')).hexdigest() == provenance['prompt_sha256'], 'prompt differs')
        _check(len(item['img_paths']) == 1 and _sha(_inside(root, item['img_paths'][0])) == source['source_id'], 'image bytes differ')
        scopes = record['scopes']
        _check(len(scopes) == 160 and {(s['step'], s['block']) for s in scopes} == {(s, b) for s in range(4) for b in range(40)}, '160 coordinates differ')
        for s in scopes:
            row = s['selected_rows']
            _check(s['grid'] == [21,29,52] and s['rows'] == ROWS and len(row) == ROWS and len(set(row)) == ROWS
                   and all(type(v) is int and 0 <= v < M for v in row), 'row coordinates differ')
            _check(s['expert'] == ('high_noise_model' if s['step'] < 2 else 'low_noise_model'), 'expert differs')
        trajectory = record['trajectories']
        _check(trajectory['plain'] == trajectory['observed'] and [t['step'] for t in trajectory['observed']] == list(range(4)), 'trajectory differs')
        for step in range(4):
            s = next(s for s in scopes if s['step'] == step and s['block'] == 39)
            op = _inside(root, s['path'])
            _check(op.is_file() and op.stat().st_size == s['bytes'], 'operand file missing/size differs')
            cases.append({'case_id': f'cal24_{i:02d}_step{step}', 'part': i // 6, 'source_index': i,
                          'source_image_sha256': source['source_id'], 'source_id': source['semantic_source_id'],
                          'source_group': source['source_group'], 'role': 'TRAIN' if i < 16 else 'HELDOUT_EXPOSED',
                          'seed': 42+i, 'step': step, 'expert': s['expert'], 'block': 39,
                          'rows': list(s['selected_rows']), 'sampling_seed': s['sampling_seed'],
                          'input_path': binding['path'], 'input_sha256': binding['sha256'],
                          'historical_post_sha256': trajectory['observed'][step]['sha256'],
                          'operand_path': str(op.relative_to(root)), 'operand_sha256': s['sha256'],
                          'operand_bytes': s['bytes'], 'prompt_sha256': provenance['prompt_sha256']})
    return cases


def budget(part=None):
    """Exact fixed work; setup and transfers require a producer-specific binding."""
    _check(part is None or type(part) is int and 0 <= part < 4, 'invalid part')
    return {'limits': {k: v if part is None else v // 4 for k,v in TOTAL.items()},
            'setup_and_transfers': SETUP}


class Ledger:
    """Predebit then observe known operations; require each source/case once.

    Call predebit/observe for actual work, mark_source and mark_case only after
    their verified completions, and finish to check the exact fixed scope.
    A failure must be recorded by the driver, not converted into completion.
    """
    def __init__(self, cases, part):
        _check(type(part) is int and 0 <= part < 4, 'invalid part')
        selected = [c for c in cases if c['part'] == part]
        _check(len(selected) == 24 and {c['source_index'] for c in selected} == set(range(part*6, part*6+6)), 'part source membership differs')
        self.cases = {c['case_id']: c for c in selected}
        _check(len(self.cases) == 24 and all(c['case_id'] == f"cal24_{c['source_index']:02d}_step{c['step']}" and c['step'] in range(4) and c['seed'] == 42+c['source_index'] for c in selected), 'invalid cases')
        for c in selected:
            _check(c['expert'] == ('high_noise_model' if c['step'] < 2 else 'low_noise_model') and c['block'] == 39, 'case expert or block differs')
            rows_sha256(c['rows'])
        _check({(c['source_index'], c['step']) for c in selected} == {(i, s) for i in range(part*6, part*6+6) for s in range(4)}, 'source/step duplication')
        self.part = part; self.limits = budget(part)['limits']; self.predebited = dict.fromkeys(self.limits, 0); self.observed = dict.fromkeys(self.limits, 0)
        self.done_sources = set(); self.done_cases = set()

    def predebit(self, key, count=1):
        _check(key in self.limits and type(count) is int and count >= 0, 'invalid debit')
        _check(self.predebited[key] + count <= self.limits[key], 'over budget')
        self.predebited[key] += count

    def observe(self, key, count=1):
        _check(key in self.limits and type(count) is int and count >= 0, 'invalid observation')
        _check(self.observed[key] + count <= self.predebited[key], 'observation exceeds predebit')
        self.observed[key] += count

    def mark_source(self, index):
        _check(type(index) is int and index in range(self.part*6, self.part*6+6) and index not in self.done_sources, 'source invalid or repeated')
        self.done_sources.add(index)

    def mark_case(self, case):
        ident = case.get('case_id'); _check(ident in self.cases and case == self.cases[ident] and ident not in self.done_cases, 'case invalid or repeated')
        _check(case['source_index'] in self.done_sources, 'source preparation not completed')
        self.done_cases.add(ident)

    def finish(self):
        _check(len(self.done_sources) == 6 and len(self.done_cases) == 24, 'incomplete cases')
        _check(self.predebited == self.observed == self.limits, 'incomplete or inconsistent work ledger')
        return {'part': self.part, 'predebited': dict(self.predebited), 'observed': dict(self.observed),
                'source_count': len(self.done_sources), 'case_count': len(self.done_cases), 'setup_and_transfers': SETUP}


def validate_label_header(path, case):
    """Validate safetensors header and case metadata; does not read payload.

    This does not prove tensor values, rows, or payload hashes; driver must
    separately bind payload and compare actual original operands and post SHA.
    """
    path = Path(path); size = path.stat().st_size
    with path.open('rb') as f:
        length_raw = f.read(8); _check(len(length_raw) == 8, 'missing header length')
        n = struct.unpack('<Q', length_raw)[0]; _check(0 < n <= 1024*1024 and 8+n <= size, 'invalid header size')
        header = json.loads(f.read(n))
    meta = header.pop('__metadata__', {})
    expected = {k: str(case[k]) for k in META_KEYS}
    expected['rows_sha256'] = rows_sha256(case['rows'])
    required = set(expected) | set(EVIDENCE_SHA_KEYS) | set(EVIDENCE_BOOL_KEYS) | set(EVIDENCE_STRIDE_KEYS)
    _check(set(meta) == required and all(meta.get(k) == v for k,v in expected.items()) and set(header) == set(PACK_SCHEMA), 'metadata or tensor names differ')
    for key in EVIDENCE_SHA_KEYS:
        value = meta[key]
        _check(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid evidence SHA')
    _check(meta['actual_post_sha256'] == case['historical_post_sha256'], 'poststate mismatch')
    _check(all(meta[key] == 'true' for key in EVIDENCE_BOOL_KEYS), 'original operand bytes differ')
    for key in EVIDENCE_STRIDE_KEYS:
        try:
            stride = json.loads(meta[key])
        except (TypeError, ValueError):
            raise ValueError('invalid runtime stride') from None
        rank = 1 if key == 'head_bias_stride' else 2 if key == 'head_weight_stride' else 3
        _check(isinstance(stride, list) and len(stride) == rank and all(type(v) is int and v >= 0 for v in stride), 'invalid runtime stride')
    intervals = []
    for name, (dtype, shape) in PACK_SCHEMA.items():
        item = header[name]; _check(item['dtype'] == dtype and item['shape'] == shape, 'tensor schema differs')
        start, end = item['data_offsets']; elements = 1
        for v in shape: elements *= v
        width = 8 if dtype == 'I64' else 2
        _check(type(start) is int and type(end) is int and start >= 0 and end-start == elements*width, 'tensor offsets differ')
        intervals.append((start,end))
    intervals.sort(); pos = 0
    for start,end in intervals:
        _check(start == pos, 'tensor offsets overlap or have gaps'); pos = end
    _check(pos == size-8-n, 'payload size differs')
    return {'header_bytes': n, 'payload_bytes': pos, 'metadata': meta}
