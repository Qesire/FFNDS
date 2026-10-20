"""Audit every frozen UB1 formal case, main shape and numerical source identity."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest
from compare_ffn_benchmark import validate_candidate_runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    suite = json.loads(args.suite.read_text())
    policy_sha = digest(args.policy)
    if (suite['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY'
            or suite['identity']['indices'] != list(range(20))
            or suite['identity']['packed_policy_sha256'] != policy_sha
            or len(suite['completed']) != 20 or {v['index'] for v in suite['completed']} != set(range(20))):
        raise RuntimeError('Complete frozen UB1 formal20 required')
    numerical = {name: digest(Path(__file__).with_name(name))
                 for name in ['step_budget_runtime.py', 'trained_packed_runtime.py']}
    rows = []
    sources = set()
    for entry in suite['completed']:
        if digest(entry['case_report']) != entry['case_report_sha256']:
            raise RuntimeError('Formal case changed')
        case = json.loads(Path(entry['case_report']).read_text())
        packed = case['hif4_runtime']['packed_reference']
        validate_candidate_runtime(packed, policy_sha, require_timing=False)
        if (packed['policy']['kind'] != 'STEP_BUDGET_PACKED_REFERENCE' or packed['identity_control']
                or packed['source_sha256'] != numerical['step_budget_runtime.py']
                or packed['parent_runtime_source_sha256'] != numerical['trained_packed_runtime.py']
                or [v['step'] for v in case['trajectory']] != list(range(4))
                or case['hif4_counter_delta']['cache_misses']
                or case['hif4_counter_delta']['linear_forward_calls'] != 1600):
            raise RuntimeError('Frozen UB1 numerical sources or native coverage differ')
        source = case['input']['source_group']
        if source in sources:
            raise RuntimeError('Duplicate formal source')
        sources.add(source)
        rows.append({'index': entry['index'], 'source_group': source,
                     'case_report_sha256': entry['case_report_sha256'],
                     'actual_main_calls': len(packed['coverage']),
                     'actual_auxiliary_calls': sum(v['auxiliary_linear_calls'] for v in packed['residual_correction']['coverage'].values()),
                     'actual_main_MAC_density': packed['actual_global_main_MAC_density'],
                     'groups_by_step': packed['groups_by_step'],
                     'main_rows_by_step': [packed['coverage'][f'step{s}_block0_ffn0']['rows'] for s in range(4)],
                     'four_latent_sha256': [v['raw_tensor_sha256'] for v in case['trajectory']],
                     'video_sha256': case['video']['sha256']})
    result = {'status': 'PASS_UB1_FORMAL20_ALL160_ACTUAL_MAIN_COMPUTE_AND_FROZEN_RUNTIME_SOURCES',
              'suite_sha256': digest(args.suite), 'policy_sha256': policy_sha,
              'numerical_runtime_sha256': numerical, 'source_sha256': digest(Path(__file__)),
              'independent_sources': len(sources), 'positions': len(rows) * 160, 'cases': rows,
              'quality_or_speed_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(result['status'])


if __name__ == '__main__':
    main()
