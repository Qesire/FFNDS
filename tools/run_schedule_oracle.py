"""BO1: retrospective joint three-metric feasibility of frozen UB1/UB2 schedules."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from ar2_checkpoint import atomic_json, digest
from schedule_oracle import enumerate_assignments

ROOT = Path(__file__).resolve().parents[1]
DIMS = ['aesthetic_quality', 'imaging_quality', 'motion_smoothness']


def read_comparison(path, csv_path, expected_policy):
    value = json.loads(path.read_text())
    if (value['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
            or value['unique_source_count'] != 20 or value['filename_copies_per_source'] != 5
            or value['paired_csv_sha256'] != digest(csv_path)
            or value['candidate_policy_sha256'] != expected_policy):
        raise RuntimeError('Complete fixed-policy20-source official comparison and exact paired CSV required')
    rows = list(csv.DictReader(csv_path.open()))
    sources = sorted({v['source_group'] for v in rows})
    if (len(rows) != 60 or len(sources) != 20
            or sum(v.startswith('i2v_subject:') for v in sources) != 10
            or sum(v.startswith('i2v_background:') for v in sources) != 10):
        raise RuntimeError('Same source10/background10 cohort with all60 source/metric rows required')
    dense, candidate, prompts = np.empty((20, 3)), np.empty((20, 3)), {}
    for i, source in enumerate(sources):
        for j, dim in enumerate(DIMS):
            items = [v for v in rows if v['source_group'] == source and v['dimension'] == dim]
            if len(items) != 1 or (source in prompts and prompts[source] != items[0]['prompt']):
                raise RuntimeError('Missing, duplicate or inconsistent source/metric pair')
            row = items[0]
            prompts[source] = row['prompt']
            dense[i, j], candidate[i, j] = float(row['D_Q']), float(row['candidate'])
    if not np.isfinite(dense).all() or not np.isfinite(candidate).all() or np.any(dense <= 0):
        raise RuntimeError('Finite scores and positive paired baseline required')
    for j, dim in enumerate(DIMS):
        for array, key in [(dense, 'D_Q'), (candidate, 'candidate')]:
            if not np.isclose(array[:, j].mean(), value['metrics'][dim][key], rtol=1e-6, atol=1e-8):
                raise RuntimeError('Source rows do not reproduce the official aggregate units')
    return value, sources, prompts, dense, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Preserve the first oracle outcome')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU diagnostic must not initialize CUDA')
    protocol_path = ROOT / 'config/budget_schedule_oracle_bo1.json'
    protocol = json.loads(protocol_path.read_text())
    evidence = ROOT / 'reports/worker_20261020'
    runs = ROOT / 'runs/worker_restart_20261020'
    started = time.perf_counter()
    cohorts, bindings, seen = [], [], set()
    for cohort, names in [('formal20', ['formal20_UB1_comparison.json', 'UB2_formal20_COMPARISON.json']),
                          ('disjoint20', ['disjoint20_UB1_comparison.json', 'UB2_disjoint20_COMPARISON.json'])]:
        arms = []
        for arm, name in zip(['UB1', 'UB2'], names):
            path = evidence / name
            csv_path = runs / f'{cohort}_{arm}_comparison/COMPARISON.csv'
            arms.append(read_comparison(path, csv_path, protocol['arms'][arm]))
            bindings.append({'cohort': cohort, 'arm': arm, 'comparison': str(path.relative_to(ROOT)),
                'comparison_sha256': digest(path), 'paired_csv_sha256': digest(csv_path)})
        a, b = arms
        if (a[1] != b[1] or a[2] != b[2] or not np.array_equal(a[3], b[3])
                or a[0]['dense_protocol_sha256'] != b[0]['dense_protocol_sha256']
                or seen.intersection(a[1])):
            raise RuntimeError('Exact paired identities/baselines and disjoint development cohorts required')
        seen.update(a[1])
        result = enumerate_assignments(a[3], a[4], b[4])
        result.update(cohort=cohort, source_order=a[1], metrics=DIMS,
            oracle_UB2_sources=[a[1][i] for i in result['second_arm_indices']],
            UB1_relative_drops={dim: a[0]['metrics'][dim]['relative_drop'] for dim in DIMS},
            UB2_relative_drops={dim: b[0]['metrics'][dim]['relative_drop'] for dim in DIMS})
        cohorts.append(result)
    passed = all(v['oracle_joint_gate'] for v in cohorts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, {'status': 'COMPLETED_BO1_JOINT_ORACLE_FEASIBLE' if passed else 'COMPLETED_BO1_JOINT_ORACLE_INFEASIBLE',
        'oracle_primary_gate': passed, 'cohorts': cohorts, 'bindings': bindings,
        'protocol_sha256': digest(protocol_path), 'source_sha256': digest(Path(__file__)),
        'enumerator_sha256': digest(ROOT / 'tools/schedule_oracle.py'),
        'CUDA_initialized': torch.cuda.is_initialized(), 'numpy': np.__version__,
        'elapsed_seconds': time.perf_counter() - started,
        'quality_or_speed_acceptance': False,
        'scope': 'retrospective outcome-aware oracle on both observed development cohorts, not a trained router, new official evaluation or prospective evidence',
        'restricted_router_training': 'NOT_RUN_ORACLE_REQUIRES_SEPARATE_REGISTERED_CALIBRATION' if passed else 'NOT_RUN_BO1_ORACLE_INFEASIBLE'})
    print(json.dumps({'oracle_primary_gate': passed, 'cohorts': [{k: v[k] for k in ['cohort', 'minimax_relative_drop', 'joint_feasible_assignments', 'oracle_relative_drops']} for v in cohorts]}, indent=2))


if __name__ == '__main__':
    main()
