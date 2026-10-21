"""Close the fixed R9A1 native screen, preserving every source and position."""
import argparse
import json
import math
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.out.exists():
        raise RuntimeError('Preserve previous analysis')
    before_path = ROOT / 'research/20261021/R9A1_before_execution.json'
    before = json.loads(before_path.read_text())
    registration = before['registration']
    protocol_path = ROOT / registration['protocol']
    if digest(protocol_path) != registration['protocol_sha256']:
        raise RuntimeError('Registered protocol changed')
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in registration['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('Frozen native implementation changed')
    hf = Path(json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())['path']).parents[1]
    reports, receipts, matrices, full_rows, compressed_rows, supports = [], [], [], [], [], []
    identity = None
    for expert in (0, 1):
        path = args.run / f'expert{expert}/RUN.json'
        report = json.loads(path.read_text())
        if report['status'] != 'COMPLETED_LEARNED_AFFINE_COMPRESSION_SHARD' or report['expert'] != expert:
            raise RuntimeError('Both fixed expert jobs must complete')
        at = report['scientific_identity']
        if (at['protocol_sha256'] != registration['protocol_sha256']
                or at['sources'] != registration['source_sha256'] or (identity is not None and at != identity)):
            raise RuntimeError('Job scientific identities differ')
        identity = at
        if report['pipeline_generations'] != 0 or report['R256_active'] is not False:
            raise RuntimeError('Unregistered generation or compensation')
        steps = [0, 1] if expert == 0 else [2, 3]
        if [c['step'] for c in report['native_controls']] != steps or any(
                c['inputs'] != 24 or c['frozen_UB1_main_and_R_bitwise'] is not True
                or c['original_full_TeacherY_bitwise'] is not True for c in report['native_controls']):
            raise RuntimeError('Missing original native controls')
        matrix = report['affine']
        if (matrix['shared_steps'] != steps or matrix['training_native_rows'] != 6144
                or matrix['training_indices'] != protocol['inputs']['training_indices']
                or digest(hf / matrix['HF_derived_relative']) != matrix['sha256']):
            raise RuntimeError('Frozen learned matrix or train exposure changed')
        matrices.append(matrix)
        full_rows.extend(report['full_support'])
        if not report['full_support_gate_all8'] or not report['full_support_controls_complete']:
            raise RuntimeError('Compression skipped numerical prerequisite')
        if [c['step'] for c in report['compression']] != steps:
            raise RuntimeError('A fixed step was omitted')
        for case in report['compression']:
            count = protocol['groups_by_step'][case['step']]
            selected = case['selection']['groups']
            if (case['groups'] != count or case['width'] != 64 * count
                    or len(selected) != count or len(set(selected)) != count
                    or any(g < 0 or g >= 216 for g in selected)
                    or case['selection']['train_inputs'] != protocol['inputs']['training_indices']
                    or case['R256_active'] is not False
                    or case['actual_main_MAC_fraction'] != (40 + count) / 216
                    or digest(hf / case['support']['HF_derived_relative']) != case['support']['sha256']):
                raise RuntimeError('Frozen support, cost or training membership changed')
            compressed_rows.extend(case['validation'])
            supports.append({k: case[k] for k in ['step', 'groups', 'width', 'actual_main_MAC_fraction', 'selection', 'support']})
        receipts.append({'expert': expert, 'record': str(path.relative_to(ROOT)), 'sha256': digest(path)})
        reports.append(report)
    membership = {(step, 20, index) for step in range(4) for index in protocol['inputs']['indices']}
    for label, rows, threshold in [('full_support', full_rows, .01), ('compressed', compressed_rows, .98)]:
        if len(rows) != 16 or {(r['step'], r['block'], r['index']) for r in rows} != membership:
            raise RuntimeError('All16 fixed cases required')
        for row in rows:
            actual, baseline = row['native_SSE'], row['frozen_UB1_SSE']
            passed = actual <= threshold * baseline
            if (not all(math.isfinite(v) and v > 0 for v in [actual, baseline])
                    or not math.isclose(row['ratio'], actual / baseline, rel_tol=1e-12)
                    or row['gate_pass'] is not passed):
                raise RuntimeError('Saved native ratio/gate does not reproduce')
            if label == 'full_support' and not passed:
                raise RuntimeError('Compressed run without global full-support gate')
    steps = []
    for step in range(4):
        rows = [r for r in compressed_rows if r['step'] == step]
        steps.append({'step': step, 'rows': rows, 'gate_pass_count': sum(r['gate_pass'] for r in rows),
            'pooled_candidate_over_UB1': sum(r['native_SSE'] for r in rows) / sum(r['frozen_UB1_SSE'] for r in rows)})
    passed = all(r['gate_pass'] for r in compressed_rows)
    result = {'id': protocol['id'], 'status': 'COMPLETED_FIXED_LEARNED_AFFINE_NATIVE_SCREEN',
        'disposition': 'FIXED_LOCAL_SCREEN_PASS_NOT_CANDIDATE_ACCEPTANCE' if passed else 'CLOSED_FIXED_LEARNED_AFFINE_COMPRESSION_REJECTED',
        'analysis_source_sha256': digest(Path(__file__)), 'before_execution_sha256': digest(before_path),
        'scientific_identity': identity, 'job_receipts': receipts, 'affine_matrices': matrices, 'supports': supports,
        'original_native_UB1_bitwise_cases': 96, 'original_full_teacher_bitwise_cases': 96,
        'full_support': full_rows, 'full_support_gate_all16': True,
        'full_support_max_error_over_UB1': max(r['ratio'] for r in full_rows), 'steps': steps,
        'native_compression_gate_all16': passed, 'native_compression_pass_count': sum(r['gate_pass'] for r in compressed_rows),
        'pipeline_generations': 0, 'R256_active': False, 'quality_or_speed_acceptance': False,
        'data_scope': 'OriginalOP2cal24 development-exposed. Train12odd; validation4even, allfourstepsblock20. No fresh-source quality.',
        'limitations': 'Rejects only learnedperchannelOLS + fixedgreedy selectedremainder with explicitBF16matrix. Does not eliminate allaffine or nativewindow recovery. Synthetictiming is not model/full160speed; no group/budget/rank/source expansion.'}
    atomic_json(args.out, result)
    print(json.dumps({'status': result['status'], 'disposition': result['disposition'],
                      'full_support_max': result['full_support_max_error_over_UB1'],
                      'compression_pass_count': result['native_compression_pass_count'],
                      'step_pooled_ratios': [s['pooled_candidate_over_UB1'] for s in steps]}), flush=True)


if __name__ == '__main__':
    main()
