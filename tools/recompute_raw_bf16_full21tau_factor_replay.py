"""Publish completed same-h factor replay receipts and source-equal components."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_full21tau_factor_replay as frozen


def recompute(protocol, hf):
    plan = frozen.data.read(protocol)
    frozen.validate(plan, True)
    out = hf/plan['output_HF_relative']
    report = frozen.data.read(out/'RUN.json')
    receipt = frozen.data.read(out.parent/'EXIT_RECEIPT.json')
    rb, eb = frozen.data.binding(out/'RUN.json', hf), frozen.data.binding(out.parent/'EXIT_RECEIPT.json', hf)
    pb = frozen.data.binding(protocol, ROOT)
    frozen.video.exited(receipt, rb, pb['sha256'], frozen.ID)
    frozen.data.need(report['status'] == 'COMPLETE_NEW10_SAME_DENSE_H_R960_TEMPORAL_COMPONENT_REPLAY'
                     and report['protocol_sha256'] == pb['sha256']
                     and report['completed_packets'] == report['passed_original_u_byte_neutrals'] == 1600
                     and not report['assigned_GPU_resources']['violations'], 'Actual complete resource-valid replay required')
    frozen.data.need(all(report['operation_ledger'][k] == plan['limits'] for k in ('predebited', 'observed')),
                     'Replay actual counts differ from frozen budget')
    for item in report['result_files'].values():
        frozen.data.bound(hf, item)
    summary = json.loads(frozen.data.bound(hf, report['result_files']['SUMMARY.json']))
    frozen.data.need(summary['protocol_sha256'] == pb['sha256'] and summary['metric_labels'] == frozen.labels(),
                     'Actual component summary identity differs')
    scalar = dict(zip(summary['metric_labels'], summary['source_equal_all160'], strict=True))
    table = {arm: {target: {view: {metric: scalar[f'{arm}/{target}/{view}/{metric}']
                                 for metric in frozen.METRICS}
                           for view in frozen.data.core.VIEWS}
                   for target in frozen.TARGETS} for arm in frozen.ARMS}
    chosen = [f'{arm}/{target}/tau1_to20/{metric}' for arm in frozen.ARMS
              for target in frozen.TARGETS for metric in
              ('error_fraction', 'DC_error_fraction', 'error_temporal_residual_fraction',
               'candidate_temporal_residual_energy_ratio', 'AC_aligned_gain', 'AC_cosine')]
    inds = [summary['metric_labels'].index(k) for k in chosen]
    joint = [{'step': step, 'block': block, 'values': {k: summary['source_equal_step4_block40'][step][block][i]
                                                    for k, i in zip(chosen, inds, strict=True)}}
             for step in range(4) for block in range(40)]
    return {'status': report['status'], 'evidence': 'SAME_DENSE_H_BOUNDARY_NATIVE_BF16_GEMMS_AND_SAVED_TEACHER_GEOMETRY',
            'protocol': pb, 'RUN': rb, 'exit': eb, 'actual_child_exit': receipt['child_returncode'],
            'actual_outer_seconds': receipt['outer_seconds'], 'source_N': 10,
            'source_input_protocol': plan['source_input_protocol'], 'factor_parent': plan['factor_parent'],
            'full_result_files': report['result_files'], 'design': plan['design'],
            'operation_counts': report['operation_ledger']['observed'],
            'component_source_equal_all160': table, 'source_equal_joint_step_block_selected_metrics': joint,
            'source_order': summary['source_order'], 'source_all160_scalar_labels': summary['metric_labels'],
            'per_source_all160_scalar_means': summary['per_source_all160'],
            'per_source_valid_position_counts': summary['per_source_valid_positions'],
            'peak_GPU_reserved_bytes': report['peak_GPU_reserved_bytes'],
            'M168_original_W2_is_paid_rounding_control': True,
            'postwrite': 'CPU_BF16_EMULATOR_ALL1600_ORIGINAL_U_BYTE_NEUTRALS_PASSED',
            'student_states': 'NOT_CAPTURED', 'motion_alignment': 'NOT_MEASURED_FIXED_XY',
            'source_exposure': 'NEW_DEVELOPMENT_NOT_BLIND_CONFIRMATION',
            'quality_or_speed_accepted': False, 'publication_replay_new_tensor_model_GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'source_N': result['source_N'], 'actual_exit': result['actual_child_exit']}))


if __name__ == '__main__':
    main()
