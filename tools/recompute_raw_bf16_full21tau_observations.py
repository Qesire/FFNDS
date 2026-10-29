"""Publish closed new10 acquisition/CPU observations; never read raw tensors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_full21tau_analysis as frozen


def recompute(protocol, hf):
    plan = frozen.read(protocol)
    frozen.validate(plan, True)
    frozen.inspect(plan, hf)
    folder = hf/plan['output_HF_relative']
    run_binding = frozen.binding(folder/'RUN.json', hf)
    exit_binding = frozen.binding(folder.parent/'EXIT_RECEIPT.json', hf)
    run = frozen.read(folder/'RUN.json')
    receipt = frozen.read(folder.parent/'EXIT_RECEIPT.json')
    proof = {'protocol': frozen.binding(protocol, ROOT), 'RUN': run_binding}
    frozen.exited(receipt, proof, frozen.ID)
    frozen.need(run['status'] == 'COMPLETE_RAW_BF16_FULL21TAU_NEW10_SAVED_TEACHER_GEOMETRY_REUSE_OBS_CPU'
                and run['protocol_sha256'] == proof['protocol']['sha256']
                and run['completed_packets'] == run['passed_original_u_byte_neutrals'] == 1600
                and run['all_statistics_provisional_until_1600_neutrals_pass'] is False
                and not run['Torch_imported'], 'Complete actual CPU numeric observation required')
    frozen.need(all(run['operation_ledger'][k] == plan['limits'] for k in ('predebited', 'observed')),
                'CPU operation ledger differs from exact frozen scope')
    for item in run['result_files'].values():
        frozen.bound(hf, item)
    summary = json.loads(frozen.bound(hf, run['result_files']['SUMMARY.json']))
    frozen.need(summary['protocol_sha256'] == proof['protocol']['sha256'] and summary['source_N'] == 10,
                'Actual summary source/protocol differs')
    labels = summary['metric_labels']
    scalar = dict(zip(labels, summary['source_equal_all_positions'], strict=True))
    geometry = {op: {view: {metric: scalar[f'geometry/{op}/{view}/{metric}']
                           for metric in frozen.core.GEOMETRY_METRICS}
                     for view in frozen.core.VIEWS} for op in frozen.core.OPERANDS}
    actions = {action: {view: {target: {metric: scalar[f'action/{action}/{view}/{target}/{metric}']
                                      for metric in frozen.core.ACTION_METRICS}
                              for target in frozen.core.TARGETS}
                        for view in frozen.core.VIEWS} for action in frozen.core.ACTIONS}
    selected = [f'geometry/{op}/tau1_to20/temporal_constant_oracle_residual_fraction' for op in frozen.core.OPERANDS]
    selected += [f'action/{a}/tau1_to20/{t}/{metric}' for a in frozen.core.ACTIONS
                 for t in ('gY_FP64', 'u_BF16', 'rowLN_u')
                 for metric in ('error_fraction', 'error_temporal_residual_fraction',
                                'candidate_temporal_residual_energy_ratio', 'signed_error_reference_temporal_residual_fraction')]
    indices = [labels.index(k) for k in selected]
    joint = [{'step': step, 'block': block, 'values': {k: summary['source_equal_joint_position'][step][block][i]
                                                    for k, i in zip(selected, indices, strict=True)}}
             for step in range(4) for block in range(40)]
    native_parts = []
    for parent in plan['parents']:
        native_run = json.loads(frozen.bound(hf, parent['RUN']))
        native_exit = json.loads(frozen.bound(hf, parent['exit']))
        native_parts.append({'part': parent['part'], 'protocol': parent['protocol'], 'RUN': parent['RUN'],
                             'exit': parent['exit'], 'actual_exit': native_exit['child_returncode'],
                             'actual_outer_seconds': native_exit['outer_seconds'],
                             'new_image_clusters': 5, 'new_videos': 5,
                             'captured_sites_including_old_neutral': native_run['captured_native_sites'],
                             'operation_counts': native_run['operation_ledger']['observed'],
                             'no_GPU_resource_violations': not native_run['assigned_GPU_resources']['violations']})
    return {'status': 'COMPLETE_NEW10_FULL160_FULL21TAU_DENSE_ACQUISITION_AND_CPU_OBSERVATION_ACTUAL_EXIT0',
            'evidence': 'NATIVE_OBS_SAVED_DENSE_TEACHER_STATE_FIXED_REUSE_ARITHMETIC',
            'source_N': 10, 'new_Dense_videos': 10, 'all_native_flows_including_old_neutral': 44,
            'all_native_FFN_including_old_neutral': 1760, 'new_source_packets': 1600,
            'old894_neutral_packets_excluded_from_statistics': 160, 'source_cohort': plan['source_cohort'],
            'native_parts': native_parts, 'CPU_protocol': proof['protocol'], 'CPU_RUN': run_binding,
            'CPU_exit': exit_binding, 'CPU_actual_exit': receipt['child_returncode'],
            'CPU_outer_seconds': receipt['outer_seconds'], 'full_result_files': run['result_files'],
            'actual_packet_full_SHA_stream_bytes': run['actual_packet_bytes_read'],
            'all_original_u_byte_neutrals': run['passed_original_u_byte_neutrals'],
            'operation_counts': run['operation_ledger']['observed'], 'design': plan['design'],
            'geometry_source_equal_all160': geometry, 'fixed_reuse_source_equal_all160': actions,
            'source_equal_joint_step_block_selected_metrics': joint,
            'source_all160_scalar_labels': labels, 'source_order': summary['source_order'],
            'per_source_all160_scalar_means': summary['source_all_position_equal'],
            'per_source_all160_valid_position_counts': summary['source_all_valid_positions'],
            'conditioning_raw_SHA_census': summary['conditioning_raw_SHA_census'],
            'source_exposure': 'NEW_TO_SCANNED_HISTORY_BANKS_REVIEWED_DEVELOPMENT_NOT_BLIND_CONFIRMATION',
            'gY': 'FP64_OF_DECODED_BF16_Y_AND_GATE_NOT_NATIVE_ROUNDED_PRODUCT',
            'rowLN': 'FP64_PER_ROW_GEOMETRY_NOT_NATIVE_SUFFIX_REPLAY',
            'motion_correspondence': 'NOT_MEASURED_FIXED_XY_ONLY',
            'quality_or_speed_acceptance': False, 'publication_replay_new_tensor_model_GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'source_N': result['source_N'], 'packets': result['new_source_packets']}))


if __name__ == '__main__':
    main()
