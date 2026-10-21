"""R4E0: current CPU-only binding of the original OP2 operands and UB1 coefficients."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import torch

from ar2_checkpoint import atomic_json, digest
from historical_source_snapshot import HistoricalSources
from native_payload_integrity import verify_payload
from onpolicy_residual import position_cases
from run_ar2_restart import ROOT, tensor_sha
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or torch.cuda.is_initialized():
        raise RuntimeError('Preserve the first CPU-only readiness report')
    torch.set_num_threads(8)
    started = time.perf_counter()
    runs = ROOT / 'runs/worker_restart_20261020'
    protocol_path = ROOT / 'config/r4_compensated_support_screen.json'
    protocol = json.loads(protocol_path.read_text())
    records, bindings, capture = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', HistoricalSources(),
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    checks = []
    for step, block in protocol['positions']:
        position = positions.position(step, block, device='cpu')
        scopes = [next(s for s in records[i]['scopes'] if (s['step'], s['block']) == (step, block)) for i in range(24)]
        jobs = [(scope, Path(records[i]['videos']['observed']['path']).parent) for i, scope in enumerate(scopes)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            sizes = list(pool.map(lambda job: verify_payload(*job), jobs))
        # position_cases also decodes all values, original gate/bias, SHA and geometry.
        cases = position_cases(records, step, block, position['expected_capture'], device='cpu')
        if (len(cases) != 24 or any(c['scope']['original_b0_sha256'] != position['raw_b0_sha256'] for c in cases)
                or any(c['scope']['native_V1_shape'] != [c['scope']['full_rows'], 5120]
                       or c['scope']['native_V1_stride'] != [5120, 1] for c in cases)):
            raise RuntimeError('Original raw bias or supported native geometry differs')
        checks.append({'step': step, 'block': block, 'inputs': 24, 'payload_bytes': sum(sizes),
            'full_payload_SHA_header_and_all_decoded_finite_BF16_values': True,
            'original_weight_cache_and_all_coefficients_bound': True,
            'support_groups': position['groups'].tolist(), 'alpha': position['alpha'],
            'main_b2_sha256': tensor_sha(position['b2']), 'RR_parameter_sha256': position['expected_residual_sha256'],
            'gate_sha256': tensor_sha(cases[0]['gate']), 'raw_b0_sha256': position['raw_b0_sha256'],
            'input_payload_sha256': {str(c['index']): c['scope']['sha256'] for c in cases}})
        del cases, position
    if torch.cuda.is_initialized():
        raise RuntimeError('R4E0 unexpectedly initialized CUDA')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, {'status': 'PASS_R4E0_ALL48_NATIVE_OPERANDS_AND_COEFFICIENTS_BOUND_CPU_ONLY',
        'protocol_sha256': digest(protocol_path), 'collection_bindings': bindings,
        'original_UB1_policy_sha256': positions.policy_sha256, 'checks': checks,
        'sources': {name: digest(ROOT / 'tools' / name) for name in ['verify_r4_operands.py',
            'native_payload_integrity.py', 'ub1_native_data.py', 'ub1_native_replay.py', 'onpolicy_residual.py']},
        'CUDA_initialized': False, 'elapsed_seconds': time.perf_counter() - started,
        'quality_or_speed_acceptance': False,
        'scope': 'current read-only access and bindings of48 saved positions, not new native output replay or whole-population independence'})
    print('PASS_R4E0_ALL48_NATIVE_OPERANDS_AND_COEFFICIENTS_BOUND_CPU_ONLY', flush=True)


if __name__ == '__main__':
    main()
