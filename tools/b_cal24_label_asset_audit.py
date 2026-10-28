"""Byte-audit completed CAL24 label assets and emit a mixed-source manifest.

No Torch, model or GPU imports. Run only after all four durable workers exit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import b_cal24_label_contract as contract
import e0_b_cal24_terminal_labels as producer

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'ffnds.b_cal24.terminal_label_asset_manifest.v1'
HEAD_SHAPES = {'head_weight_runtime_KN': [5120, 64], 'head_bias': [64],
               'head_modulation': [1, 2, 5120]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve(strict=True)
    if path == root or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Asset path outside its root')
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def checked_path(root, entry):
    if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
        raise ValueError('Invalid byte binding')
    p = Path(root).resolve() / entry['path']
    actual = binding(p, root)
    if any(actual[k] != entry.get(k) for k in ('path', 'bytes', 'sha256')):
        raise ValueError('Asset binding differs: ' + entry['path'])
    return p


def read_bound(root, entry):
    return json.loads(checked_path(root, entry).read_text())


def verify_head(path, asset, expert, plan):
    header, length = producer.safetensors_header(path)
    if set(header) - {'__metadata__'} != set(HEAD_SHAPES) or set(asset['tensors']) != set(HEAD_SHAPES):
        raise ValueError('Original head fields differ')
    spans, hashes, original_weight_sha = [], {}, None
    with Path(path).open('rb') as stream:
        for key, shape in HEAD_SHAPES.items():
            field, recorded = header[key], asset['tensors'][key]
            start, end = field['data_offsets']
            size = 2
            for n in shape:
                size *= n
            if (field['dtype'] != 'BF16' or field['shape'] != shape or end - start != size
                    or recorded['shape'] != shape or recorded['dtype'] != 'torch.bfloat16'):
                raise ValueError('Head tensor shape/dtype differs')
            stream.seek(8 + length + start)
            data = stream.read(end - start)
            value = hashlib.sha256(data).hexdigest()
            if len(data) != size or value != recorded['raw_tensor_sha256']:
                raise ValueError('Head payload tensor SHA differs')
            hashes[key] = value
            if key == 'head_weight_runtime_KN':
                rows, columns = shape
                original = bytearray(size)
                for row in range(rows):
                    for col in range(columns):
                        src = 2 * (row * columns + col)
                        dst = 2 * (col * rows + row)
                        original[dst:dst + 2] = data[src:src + 2]
                original_weight_sha = hashlib.sha256(original).hexdigest()
            spans.append((start, end))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise ValueError('Head tensor overlap/gap')
        position = end
    if position + 8 + length != Path(path).stat().st_size:
        raise ValueError('Head file payload size differs')
    expected = plan['head_assets'][expert]
    if (original_weight_sha != expected['weight_raw_sha256']
            or hashes['head_bias'] != expected['bias_raw_sha256']
            or hashes['head_modulation'] != expected['modulation_raw_sha256']):
        raise ValueError('Head original bias/modulation identity differs')
    return hashes


def audit_case(result, expected, part_dir, hf_root, head_hashes):
    if result.get('case') != expected or result.get('schema') != 'ffnds.b_cal24.terminal_label_case.v1' or result.get('status') != 'PASS_CAL24_POST_AND_OPERAND_BYTES_ALIGNED':
        raise ValueError('Case identity/status differs')
    case_file = part_dir / expected['case_id'] / 'CASE.json'
    if json.loads(case_file.read_text()) != result:
        raise ValueError('CASE.json differs from RUN case')
    evidence = result['evidence']
    if (evidence.get('actual_post_sha256') != expected['historical_post_sha256']
            or result.get('actual_post_raw_FP32_sha256') != expected['historical_post_sha256']
            or any(evidence.get(k) is not True for k in contract.EVIDENCE_BOOL_KEYS)
            or result.get('native_counter_delta') != {'linear_forward_calls': 400, 'activation_qdq_calls': 400, 'weight_transfers': 0, 'cache_misses': 0}
            or result.get('native_boundary_counts') != {'pre': 1, 'pre_mm': 5, 'patch': 1, 'block': 40, 'ffn': 40, 'linear': 400, 'qdq': 400, 'norm': 1, 'head': 1, 'post': 1}
            or result.get('native_noop', {}).get('CPU_BF16_gate_mul_then_add_byte_equal') is not True):
        raise ValueError('Historical bytes/native execution evidence differs')
    for key, head_key in (('head_weight_raw_sha256', 'head_weight_runtime_KN'), ('head_bias_raw_sha256', 'head_bias'), ('head_modulation_raw_sha256', 'head_modulation')):
        if evidence.get(key) != head_hashes[head_key]:
            raise ValueError('Actual head evidence differs from exported head tensor')
    label = result['label']
    path = checked_path(part_dir, label)
    hashes = {k: v['raw_tensor_sha256'] for k, v in label['tensors'].items()}
    if set(hashes) != set(contract.PACK_SCHEMA):
        raise ValueError('Label tensor set differs')
    payload, header = producer.validate_saved_payload(path, expected, hashes)
    if (header != label['header'] or payload != label.get('serialized_payload_sha256')
            or label.get('all_serialized_tensor_bytes_match_memory') is not True
            or label.get('saved_rows_payload_matches_case') is not True):
        raise ValueError('Label header or serialized payload evidence differs')
    metadata = header['metadata']
    if any(metadata.get(k) != v for k, v in producer.label_metadata(expected, evidence).items()):
        raise ValueError('Label metadata/evidence differs')
    return {'case': expected, 'case_record': binding(case_file, hf_root),
            'label': {'root': 'hf', **binding(path, hf_root)}, 'evidence': evidence,
            'serialized_payload_sha256': payload, 'tensor_sha256': hashes,
            'actual_head_operator': result['actual_head_operator'], 'actual_u_layout': result['actual_u_layout'],
            'native_noop': result['native_noop']}


def audit(protocol_path, hf_root, *, repo_root=ROOT):
    hf_root, repo_root = Path(hf_root).resolve(), Path(repo_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    parent = binding(protocol_path, repo_root)
    plan = json.loads(protocol_path.read_text())
    producer.validate_plan(plan, 0, frozen=True)
    sources = plan['repo_sources']
    if len(sources) != 105 or len({s['path'] for s in sources}) != 105:
        raise ValueError('Exactly 105 frozen source bindings required')
    for name in ('tools/e0_b_cal24_terminal_labels.py', 'tools/b_cal24_label_contract.py'):
        item = next(s for s in sources if s['path'] == name)
        checked_path(repo_root, item)
    expected_cases = {c['case_id']: c for c in contract.build_cases(plan, hf_root)}
    if len(expected_cases) != 96 or set(expected_cases) != set(producer.run_ids()):
        raise ValueError('Frozen 96 cases differ')
    out = hf_root / plan['output_HF_relative']
    parts, cases, heads = [], [], {}
    for part in range(4):
        folder = out / f'part{part}'
        run_binding = binding(folder / 'RUN.json', hf_root)
        exit_binding = binding(folder / 'EXIT_RECEIPT.json', hf_root)
        run, receipt = read_bound(hf_root, run_binding), read_bound(hf_root, exit_binding)
        if (receipt.get('id') != plan['id'] or receipt.get('protocol_sha256') != parent['sha256']
                or receipt.get('child_returncode') != 0 or receipt.get('process_exited') is not True
                or receipt.get('run_sha256') != run_binding['sha256'] or receipt.get('timeout') is not False
                or receipt.get('automatic_retry') is not False or receipt.get('error_type') is not None):
            raise ValueError('Durable actual exit receipt differs')
        counters = {'linear_forward_calls': 9600, 'activation_qdq_calls': 9600, 'weight_transfers': 800, 'cache_misses': 0}
        if (run.get('id') != plan['id'] or run.get('part') != part or run.get('protocol_sha256') != parent['sha256']
                or run.get('status') != 'COMPLETE_CAL24_TERMINAL_LABEL_PART_POST_AND_OPERAND_BYTES_MATCH'
                or run.get('predebited') != plan['limits_per_part'] or run.get('observed') != plan['limits_per_part']
                or run.get('actual_native_counters') != counters or run.get('actual_native_counters_including_failed_work') != counters):
            raise ValueError('Completed RUN counters or ledger differs')
        for item in sources:
            checked_path(folder / 'EXECUTED_SOURCES', item)
        snapshot_protocol = folder / 'EXECUTED_SOURCES' / 'protocol.json'
        if digest(snapshot_protocol) != parent['sha256'] or snapshot_protocol.stat().st_size != parent['bytes']:
            raise ValueError('Executed protocol snapshot differs')
        per_heads = {}
        head_assets = run.get('original_head_assets', [])
        if len(head_assets) != 2 or {h.get('expert') for h in head_assets} != set(producer.EXPERTS):
            raise ValueError('Two original head exports required per part')
        for asset in head_assets:
            expert = asset['expert']
            path = checked_path(folder, asset)
            hashes = verify_head(path, asset, expert, plan)
            if expert in heads and heads[expert]['tensor_sha256'] != hashes:
                raise ValueError('Head tensor identity changed across parts')
            if expert not in heads:
                heads[expert] = {'expert': expert, 'binding': {'root': 'hf', **binding(path, hf_root)},
                                 'tensors': asset['tensors'], 'tensor_sha256': hashes}
            per_heads[expert] = hashes
        source_rows = run.get('sources', [])
        if len(source_rows) != 6 or [s.get('source_index') for s in source_rows] != list(range(part * 6, part * 6 + 6)):
            raise ValueError('Exact six ordered sources required')
        for source in source_rows:
            index = source['source_index']
            rows = source.get('cases', [])
            if (source.get('status') != 'COMPLETE_FOUR_CONTINUOUS_STEPS_CAL24_BYTES_ALIGNED'
                    or source.get('seed') != 42 + index or len(rows) != 4
                    or [r.get('case', {}).get('step') for r in rows] != list(range(4))):
                raise ValueError('Source four-step chain differs')
            for row in rows:
                key = row['case']['case_id']
                if key not in expected_cases or expected_cases[key]['part'] != part:
                    raise ValueError('Case outside part')
                cases.append(audit_case(row, expected_cases[key], folder, hf_root, per_heads[row['case']['expert']]))
        parts.append({'part': part, 'RUN': run_binding, 'exit_receipt': exit_binding})
    if len(cases) != 96 or {c['case']['case_id'] for c in cases} != set(expected_cases):
        raise ValueError('96 unique complete cases required')
    return {'schema': SCHEMA, 'id': plan['id'], 'status': 'AUDITED_COMPLETE_96_CASES',
            'parent_protocol': parent, 'source_exposure': plan['source_exposure'], 'source_split': plan['source_split'],
            'parts': parts, 'cases': cases, 'head_assets': [heads[e] for e in producer.EXPERTS],
            'totals': {'parts': 4, 'sources': 24, 'cases': 96, 'head_exports_verified': 8,
                       'execution_snapshots_verified': 420, 'dense_flows': 96, 'native_ffn_calls': 3840}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.protocol, args.hf_root)
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.out.open('x') as stream:
        stream.write(data)


if __name__ == '__main__':
    main()
