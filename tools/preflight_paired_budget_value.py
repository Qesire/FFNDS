"""CPU-only D0 operand/pack/source check; never creates a model or CUDA context."""
import argparse
import json
from pathlib import Path
import shutil
import time

from r12d0_contract import ROOT, ACTIONS, sha, groups, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ar2_checkpoint import atomic_json
    from paired_budget_runtime import audited_packs
    from run_ar2_restart import tensor_sha
    if args.output.exists():
        raise RuntimeError('Preserve prior preflight')
    started = time.monotonic()
    torch.set_num_threads(8)
    cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    variants, receipts = audited_packs(None, cache)
    if torch.cuda.is_initialized():
        raise RuntimeError('Preflight must remain CPU-only')
    bytes_all = sum(r['weight_qdq'].numel() * r['weight_qdq'].element_size()
                    for p in variants.values() for r in p['records'].values())
    if bytes_all != 1274019840:
        raise RuntimeError('Unexpected pair variants storage')
    # Boundary plan: all320 native FFN main shapes, exact MAC at each action,
    # 158 genuine residual positions and a disjoint, exhaustive24-case plan.
    macs = {}
    for arm in ACTIONS:
        macs[arm] = sum(2 * 31668 * 5120 * groups(arm, s, b) * 64 for s in range(4) for b in range(40))
        if macs[arm] != 358628091494400:
            raise RuntimeError('Unequal actual main MAC')
    schedules = [jobs('controls', a) for a in ('DQ', 'UB1')]
    schedules += [jobs(p, shard=s) for p in ('baselines', 'interventions') for s in (0, 1)]
    if len({(i, a) for _, schedule in schedules for i, a in schedule}) != 24:
        raise RuntimeError('Duplicate/missing registered cases')
    free = shutil.disk_usage(cache['path']).free
    if free < (80 << 30) + (1 << 29):
        raise RuntimeError('Insufficient reserve plus fixed artifact allowance')
    source_files = ['tools/r12d0_contract.py', 'tools/paired_budget_runtime.py',
                    'tools/paired_budget_observer.py', 'tools/run_paired_budget_value.py',
                    'tools/analyze_paired_budget_value.py', 'tools/preflight_paired_budget_value.py']
    result = {'status': 'PASS_R12_D0_CPU_PREFLIGHT_NOT_NATIVE_PROOF',
              'CUDA_initialized': False, 'new_model_generations': 0,
              'operand_audit_sha256': sha(ROOT / 'research/20261022/R12_E0_operand_audit.json'),
              'draft_sha256': sha(ROOT / 'research/20261022/R12_E1_native_value_DRAFT.json'),
              'sources': {p: sha(ROOT / p) for p in source_files}, 'all_pair_weight_variant_bytes': bytes_all,
              'main_MAC_by_raw_action': macs, 'case_plan': schedules,
              'packs': receipts, 'node_free_bytes': free, 'elapsed_seconds': time.monotonic() - started,
              'scope': 'Actual original effective weight/Smooth slices and run plan; native QDQ/GELU/writeback remain to be measured.'}
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('packs', 'case_plan', 'sources')}))


if __name__ == '__main__':
    main()
