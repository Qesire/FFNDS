"""CPU-only audit of full160 frozen UB1 metadata for native row dispatch."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the original preflight')
    import torch
    from historical_source_snapshot import HistoricalSources
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import load_metadata
    from rollout_contract import ROOT, atomic, read, sha
    torch.set_num_threads(8)
    policy = ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json'
    previous = read(ROOT / 'research/20261021/R9A1_native_affine_results.json')
    expected = previous['scientific_identity']['original_policy_sha256']
    positions = UB1Positions(policy, HistoricalSources(), expected_policy_sha256=expected)
    metadata = load_metadata(positions)
    rows = []
    for (step, block), p in metadata.items():
        rows.append({'step': step, 'block': block, 'support_width': len(p['channels']),
                     'alpha': p['alpha'], 'W2_tensor_sha256': p['W2_tensor_sha256'],
                     'auxiliary_bytes': sum(t.numel() * t.element_size() for t in p['residual'].values())})
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU metadata preflight initialized CUDA')
    report = {'id': 'R16V0_METADATA_PREFLIGHT', 'status': 'PASS_FROZEN_FULL160_METADATA_CPU_ONLY',
              'positions': rows, 'policy_sha256': sha(policy), 'original_W1_W2_calls': 0,
              'new_model_generations': 0, 'CUDA_initialized': False,
              'residual_parameter_bytes': sum(r['auxiliary_bytes'] for r in rows),
              'source_sha256': {f'tools/{name}': sha(ROOT / 'tools' / name) for name in
                                ('preflight_row_dispatch.py', 'ub1_row_dispatch.py', 'row_dispatch_kernels.py')},
              'model_execution_authority': False}
    atomic(args.output, report)
    print(json.dumps({k: report[k] for k in ('status', 'residual_parameter_bytes', 'new_model_generations')}))


if __name__ == '__main__':
    main()
