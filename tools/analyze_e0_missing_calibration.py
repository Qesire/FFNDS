"""Verify the four missing-block passive captures and export neutral bindings."""
import argparse
import json
from pathlib import Path

from analyze_e0_dense_residency import validate_plan
from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def analyze(hf):
    import numpy as np
    relative = 'config/e0_missing_calibration_observations_20261026.json'
    plan, digest = validate_plan(relative)
    recovery_relative = 'config/e0_missing_calibration_recovery_20261026.json'
    recovery, recovery_digest = validate_plan(recovery_relative)
    accepted = json.loads((ROOT / plan['accepted_observations']).read_text())
    result = {'id': 'E0_MISSING_CALIBRATION_OBSERVATIONS_20261026', 'status': 'COMPLETE',
              'protocol': relative, 'protocol_sha256': digest, 'analysis_sha256': sha256(Path(__file__)),
              'recovery_protocol': recovery_relative, 'recovery_protocol_sha256': recovery_digest,
              'sources': [], 'raw_bindings': [], 'new_generations': 4,
              'new_payloads': 560, 'duplicate_historical_blocks': 0,
              'peak_assigned_GPUs': 4, 'user_GPU_ceiling': 4,
              'optimizer_updates': 0, 'new_weights': 0, 'official_quality_calls': 0,
              'quality_or_speed_acceptance': False, 'stage_changed': False}
    predecessor = recovery['failed_predecessor']
    failed_path = hf / predecessor['RUN_HF_relative']
    failed = json.loads(failed_path.read_text())
    if sha256(failed_path) != predecessor['sha256'] or failed['generations_started'] != 0 or failed.get('scopes'):
        raise ValueError('Preserved failed predecessor changed')
    result['preserved_failed_attempt'] = {**predecessor, 'status': 'PROCESS_EXITED_WITHOUT_FINAL_RECEIPT',
                                         'generation_count_in_receipt': 0, 'automatic_retry': False}
    for index in plan['source_indices']:
        attempt = 2 if index == 16 else 1
        active_plan, active_digest = (recovery, recovery_digest) if index == 16 else (plan, digest)
        active_relative = recovery_relative if index == 16 else relative
        path = hf / f'e0_missing_calibration_20261026/source{index:02d}_attempt{attempt}/RUN.json'
        raw = json.loads(path.read_text())
        reference = next(s for s in accepted['sources'] if s['index'] == index)
        if raw['status'] != 'PASS_NATIVE_MISSING_CALIBRATION_OPERANDS_NOOP' or raw['protocol_sha256'] != active_digest:
            raise ValueError('Incomplete capture or protocol mismatch')
        if (raw['generations_started'] != 1 or raw['source_id'] != reference['source_id'] or
                raw['seed'] != reference['seed'] or raw['trajectory'] != reference['trajectory'] or
                raw['video_sha256'] != reference['video_sha256'] or raw['native_counts'] != reference['native_counts_each_phase']):
            raise ValueError('Native no-op identity failed')
        monitor = raw['assigned_GPU_resources']
        if not monitor['sampled_isolation_pass'] or monitor['violations']:
            raise ValueError('Assigned GPU isolation failed')
        if any(s['foreign_process_count_on_assigned_GPU'] for s in monitor['samples']):
            raise ValueError('Assigned GPU shared')
        positions = [(s['step'], s['block']) for s in raw['scopes']]
        if len(positions) != 140 or len(set(positions)) != 140 or set(positions) != {(s, b) for s in plan['steps'] for b in plan['blocks']}:
            raise ValueError('Incomplete or duplicate missing block capture')
        if set(plan['blocks']) & {0, 10, 20, 30, 39}:
            raise ValueError('Duplicate historical blocks')
        for scope in raw['scopes']:
            payload = path.parent / 'observed' / scope['path']
            if sha256(payload) != scope['sha256']:
                raise ValueError('Payload identity failed')
            data = load_arrays(payload)
            if set(data) != {'q1_post', 'q2_post', 'ffn_output', 'b1', 'b2', 'gate'}:
                raise ValueError('Unexpected fields')
            if scope['row_plan']['rows'] != sorted(scope['row_plan']['uniform_rows']) or len(scope['row_plan']['rows']) != 32:
                raise ValueError('Unexpected row panel')
            for name, width in [('q1_post', 5120), ('q2_post', 13824), ('ffn_output', 5120)]:
                if data[name].shape != (32, width) or not np.isfinite(data[name]).all():
                    raise ValueError('Invalid native operand')
        snapshot = path.parent / 'EXECUTED_SOURCES'
        if sha256(snapshot / Path(active_relative).name) != active_digest or any(
                sha256(snapshot / name) != expected for name, expected in active_plan['driver_sources_sha256'].items()):
            raise ValueError('Executed source snapshot mismatch')
        size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
        if size > plan['max_output_gib_per_source'] * (1 << 30):
            raise ValueError('HF output cap exceeded')
        result['raw_bindings'].append({'source_index': index, 'HF_relative': str(path.relative_to(hf)), 'sha256': sha256(path)})
        result['sources'].append({'source_index': index, 'source_id': raw['source_id'], 'seed': raw['seed'],
                                   'payloads': 140, 'native_counts': raw['native_counts'], 'trajectory': raw['trajectory'],
                                   'video_sha256': raw['video_sha256'], 'HF_output_bytes': size,
                                   'sampled_assigned_GPU_isolation_pass': True})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve accepted manifest')
    write_json(args.out, analyze(args.hf_root))
