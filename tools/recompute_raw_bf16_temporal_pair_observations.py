"""Compact a successful saved-pair CPU observation, preserving source identity."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('temporal_energy', 'control_energy', 'paired_energy_difference',
           'temporal_cosine', 'control_cosine', 'paired_cosine_difference')


def binding(path, root):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(root, item):
    path = root/item['path']
    assert binding(path, root) == item
    return json.loads(path.read_text())


def recompute(protocol, hf):
    plan = json.loads(protocol.read_text())
    folder = hf/plan['output_HF_relative']
    run_binding = binding(folder/'RUN.json', hf)
    exit_binding = binding(folder.parent/'EXIT_RECEIPT.json', hf)
    run, receipt = load(hf, run_binding), load(hf, exit_binding)
    assert receipt['process_exited'] is True and type(receipt['child_returncode']) is int
    assert receipt['child_returncode'] == 0 and not receipt['timeout']
    assert receipt['run_sha256'] == run_binding['sha256']
    assert run['protocol_sha256'] == receipt['protocol_sha256'] == binding(protocol, ROOT)['sha256']
    assert run['status'] == 'COMPLETE_RAW_BF16_S2_SAVED_TEACHER_TEMPORAL_PAIR_OBS_CPU'
    assert all(run['operation_ledger'][k] == plan['limits'] for k in ('predebited', 'observed'))
    assert run['completed_packets'] == 1280 and not run['Torch_imported']
    assert all(run[k] == 0 for k in ('model_calls', 'GPU_calls', 'fits', 'decompositions'))
    for item in run['result_files'].values():
        assert binding(hf/item['path'], hf) == item
    summary = load(hf, run['result_files']['SUMMARY.json'])
    assert summary['protocol_sha256'] == run['protocol_sha256']

    def table(values):
        return {op: {metric: values[i][summary['metric'].index(metric)] for metric in METRICS}
                for i, op in enumerate(summary['operand'])}

    all_source = summary['source_equal_means_axis_without_source']
    return {'status': 'COMPLETE_SAVED_TEACHER_TEMPORAL_PAIR_OBS_ACTUAL_CPU_EXIT0',
            'protocol': binding(protocol, ROOT), 'RUN': run_binding, 'exit': exit_binding,
            'full_result_files': run['result_files'], 'operand_audit': plan['operand_audit'],
            'source_count': 8, 'calibration_source_count': 6, 'development_source_count': 2,
            'source_state': 'DENSE_TEACHER_INPUT', 'row_schedule': plan['row_schedule'],
            'actual_input_IO': run['actual_input_IO'], 'operation_counts': run['operation_ledger']['observed'],
            'source_equal_all160_geometry': table(all_source[0][0]),
            'by_step_source_equal': {str(s): table(all_source[1+s][0]) for s in range(4)},
            'by_block_source_equal': {str(b): table(all_source[5+b][0]) for b in range(40)},
            'by_delta_tau_source_equal': {str(d): table(all_source[0][2+d]) for d in range(1, 21)},
            'tau0_participates': table(all_source[0][1]), 'tau0_excluded': table(all_source[0][2]),
            'per_source_all160_geometry': [{**source, 'geometry': table(summary['source_means'][i][0][0])}
                                         for i, source in enumerate(summary['sources'])],
            'weighting': summary['weighting'], 'design': plan['design'],
            'claim': 'SPARSE_SAME_XY_OBSERVATIONAL_GEOMETRY_WITH_TIME_MATCHED_DIFFERENT_XY_CONTROL',
            'complete_21tau_trajectory_variance': 'NOT_MEASURED', 'motion_alignment': 'NOT_MEASURED',
            'source_history': 'ALL_EIGHT_HISTORICALLY_EXPOSED; no heldout method selection',
            'quality_or_speed_acceptance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    if args.out.exists():
        raise ValueError('Fresh recomputation output required')
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'all160_geometry': result['source_equal_all160_geometry']}))


if __name__ == '__main__':
    main()
