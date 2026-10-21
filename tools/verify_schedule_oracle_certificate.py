"""Independently certify the BO1 imaging ceiling by exact sourcewise maxima."""
import argparse
import csv
from decimal import Decimal, localcontext
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = ROOT / 'runs/worker_restart_20261020'
    oracle_path = runs / 'bo1_schedule_oracle/RUN.json'
    oracle = json.loads(oracle_path.read_text())
    if oracle['status'] != 'COMPLETED_BO1_JOINT_ORACLE_INFEASIBLE':
        raise RuntimeError('Complete rejected two-cohort BO1 outcome required')
    certificates = []
    for cohort in ['formal20', 'disjoint20']:
        paths = [runs / f'{cohort}_{arm}_comparison/COMPARISON.csv' for arm in ['UB1', 'UB2']]
        tables = []
        for arm, path in zip(['UB1', 'UB2'], paths):
            expected = next(v for v in oracle['bindings'] if v['cohort'] == cohort and v['arm'] == arm)
            if digest(path) != expected['paired_csv_sha256']:
                raise RuntimeError('Frozen oracle source-score table changed')
            rows = [v for v in csv.DictReader(path.open()) if v['dimension'] == 'imaging_quality']
            table = {v['source_group']: v for v in rows}
            if len(table) != 20 or len(rows) != 20:
                raise RuntimeError('Exactly20 independent imaging sources required')
            tables.append(table)
        if set(tables[0]) != set(tables[1]):
            raise RuntimeError('Source pairing differs')
        with localcontext() as context:
            context.prec = 50
            dense, upper = [], []
            for source in sorted(tables[0]):
                first, second = tables[0][source], tables[1][source]
                baseline = Decimal(first['D_Q'])
                choices = [Decimal(v['candidate']) for v in [first, second]]
                if (baseline != Decimal(second['D_Q']) or baseline <= 0
                        or not all(v.is_finite() for v in [baseline, *choices])):
                    raise RuntimeError('Finite paired positive baseline required')
                dense.append(baseline)
                upper.append(max(choices))
            baseline, bound = sum(dense) / 20, sum(upper) / 20
            drop, threshold = 1 - bound / baseline, Decimal('.98') * baseline
            enumerated = next(v for v in oracle['cohorts'] if v['cohort'] == cohort)
            if abs(float(drop) - enumerated['best_unconstrained_imaging_relative_drop']) > 1e-12:
                raise RuntimeError('Independent imaging upper bound differs from the enumerator')
            certificates.append({'cohort': cohort,
                'method': 'Decimal precision50 exact sum of pointwise maximum imaging source scores across the two arms, ignoring all other metric constraints',
                'source_count': 20, 'CSV_sha256': [digest(p) for p in paths],
                'D_Q_mean_decimal': str(baseline), 'max_possible_imaging_mean_decimal': str(bound),
                'required_mean_decimal': str(threshold), 'minimum_possible_relative_drop_decimal': str(drop),
                'imaging_alone_impossible': bound < threshold, 'enumeration_bound_agrees_within_1e-12': True})
    atomic_json(args.output, {'status': 'PASS_BO1_INDEPENDENT_DECIMAL_IMAGING_UPPER_BOUND_CERTIFICATE',
        'source_sha256': digest(Path(__file__)), 'raw_oracle_sha256': digest(oracle_path),
        'cohorts': certificates, 'quality_or_speed_acceptance': False,
        'scope': 'independent algebraic check on already exposed source-score tables; not an official new evaluation'})


if __name__ == '__main__':
    main()
