"""Reconstruct completed S2 scalar summaries; no operand tensors or models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_s2_temporal_statistics_cpu as producer
import e0_raw_bf16_temporal_video20 as evidence


def binding(path, root):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def finite_mean(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.fmean(values) if values else None


def recompute(protocol, hf):
    plan = json.loads(protocol.read_text()); out = hf/producer.OUTPUT
    run = json.loads((out/'RUN.json').read_text()); exit_path = out.parent/'EXIT_RECEIPT.json'
    rb, eb, pb = binding(out/'RUN.json', hf), binding(exit_path, hf), binding(protocol, ROOT)
    evidence.vm.exited(json.loads(exit_path.read_text()), rb, pb['sha256'], plan['id'])
    assert run['status'] == 'COMPLETE_S2_TEMPORAL_STATISTICS_CPU_SOURCE_EQUAL'
    assert run['protocol_sha256'] == pb['sha256'] and run['packets'] == run['original_u_neutrals'] == 1600
    assert run['actual_packet_bytes_read'] == plan['expected_packet_bytes'] == 18494182400
    assert all(run[k] == 0 for k in ('model_calls', 'GPU_calls', 'fits', 'new_videos'))
    summary = json.loads((out/'SUMMARY.json').read_text())
    labels = [f'{op}/{view}/{key}' for op in producer.METRICS for view in producer.VIEWS for key in producer.KEYS]
    assert summary['labels'] == labels and summary['source_count'] == 10
    sources, positions = {}, set()
    with (out/'POSITIONS.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line); coordinate = r['source_id'], r['step'], r['block']
            assert coordinate not in positions; positions.add(coordinate)
            vector = [x for op in producer.METRICS for view in producer.VIEWS
                      for x in producer.extract_scalars(r['values'][op][view])]
            sources.setdefault(r['source_id'], []).append((r['step'], r['block'], vector))
    assert len(positions) == 1600 and len(sources) == 10
    check = lambda a, b: (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
    rebuilt = []
    for saved in summary['sources']:
        rows = sources[saved['source_id']]
        assert {(s, b) for s, b, _ in rows} == {(s, b) for s in range(4) for b in range(40)}
        values = [finite_mean([v[j] for _, _, v in rows]) for j in range(len(labels))]
        assert all(check(a, b) for a, b in zip(values, saved['all160']))
        by_step = [[finite_mean([v[j] for s, _, v in rows if s == step]) for j in range(len(labels))] for step in range(4)]
        assert all(check(a, b) for row, saved_row in zip(by_step, saved['by_step']) for a, b in zip(row, saved_row))
        rebuilt.append({'source_id': saved['source_id'], 'all160': values, 'by_step': by_step})
    means = [finite_mean([r['all160'][j] for r in rebuilt]) for j in range(len(labels))]
    assert all(check(a, b) for a, b in zip(means, summary['source_equal_all160']))
    def unpack(vector):
        return {op: {view: {key: vector[labels.index(f'{op}/{view}/{key}')] for key in producer.KEYS}
                     for view in producer.VIEWS} for op in producer.METRICS}
    return {'status': run['status'], 'protocol': pb, 'RUN': rb, 'exit': eb,
        'actual_exit': 0, 'actual_outer_seconds': json.loads(exit_path.read_text())['outer_seconds'],
        'positions': binding(out/'POSITIONS.jsonl', hf), 'summary': binding(out/'SUMMARY.json', hf),
        'packet_bytes': run['actual_packet_bytes_read'], 'original_u_neutrals': 1600,
        'image_clusters': 10, 'nested_positions_per_image': 160,
        'source_equal_means': unpack(means),
        'source_equal_by_step': [unpack([finite_mean([r['by_step'][s][j] for r in rebuilt]) for j in range(len(labels))]) for s in range(4)],
        'per_source': [{'source_id': r['source_id'], 'all160': unpack(r['all160'])} for r in rebuilt],
        'source_role': 'ORIGINAL_NEW10_DEVELOPMENT_CAPTURES_NOW_ALSO_USED_FOR_VALUE_DC_AC_CALIBRATION_NOT_BLIND',
        'frequency_unit': 'CYCLES_PER_21_OR20_LATENT_TIME_SLOTS_NOT_HZ_OR_DIFFUSION_STEPS',
        'candidate_frequency_comparison': 'NOT_MEASURED_THIS_PRODUCER_READS_DENSE_ONLY',
        'motion_alignment': 'NOT_MEASURED_8_FIXED_XY', 'student_prefix': 'NOT_CAPTURED',
        'quality_or_speed_accepted': False, 'recompute_new_tensor_model_GPU_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    with args.out.open('x') as stream: json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'packets': result['original_u_neutrals'], 'actual_outer_seconds': result['actual_outer_seconds']}))


if __name__ == '__main__': main()
