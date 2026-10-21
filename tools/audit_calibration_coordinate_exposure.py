"""Audit whether the learned C24 state population reused its training row draw."""
import json
import math
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT / 'runs/worker_restart_20261020'
    bindings, tables = [], {}
    for label, directory in [('dense', 'calibration24_ffn_operands'), ('student', 'onpolicy_calibration24')]:
        cases = {}
        for shard in [0, 1]:
            path = runs / f'{directory}/shard{shard}/RUN.json'
            report = json.loads(path.read_text())
            expected = ('PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP' if label == 'dense'
                        else 'PASS_ONPOLICY_CALIBRATION24_SHARD_AND_NOOP')
            if report['status'] != expected:
                raise RuntimeError('Historical coordinate audit requires both complete collections')
            bindings.append({'collection': label, 'shard': shard, 'sha256': digest(path)})
            for entry in report['completed']:
                if entry['index'] in cases or digest(entry['input_report']) != entry['input_report_sha256']:
                    raise RuntimeError('Changed or duplicate historical input report')
                cases[entry['index']] = json.loads(Path(entry['input_report']).read_text())
        if set(cases) != set(range(24)):
            raise RuntimeError('All24 historical sources required')
        tables[label] = cases
    rows, matches, total = [], 0, 0
    for index in range(24):
        dense, student = tables['dense'][index], tables['student'][index]
        if dense['input']['input_provenance'] != student['input']['input_provenance']:
            raise RuntimeError('Historical image identity differs')
        a = {(s['step'], s['block']): s for s in dense['scopes']}
        b = {(s['step'], s['block']): s for s in student['scopes']}
        if set(a) != set(b) or set(a) != {(s, block) for s in range(4) for block in range(40)}:
            raise RuntimeError('Historical position coverage differs')
        same = 0
        for key in a:
            if math.prod(a[key]['grid']) != b[key]['full_rows']:
                raise RuntimeError('Historical native row populations differ')
            equal = all(a[key][field] == b[key][field]
                        for field in ['grid', 'rows', 'selected_rows', 'sampling_seed', 'inclusion_probability'])
            same += equal
            matches += equal
            total += 1
        rows.append({'index': index, 'positions': len(a), 'exact_row_draw_matches': same})
    if matches != total or total != 3840:
        raise RuntimeError('Complete row reuse was not established; preserve the discrepancy')
    value = {'status': 'VERIFIED_ALL3840_CALIBRATION_AND_STUDENT_ROW_DRAW_REUSE',
        'positions_compared': total, 'exact_coordinate_matches': matches, 'sources': rows,
        'collection_bindings': bindings, 'source_sha256': digest(Path(__file__)),
        'sampling_namespace': 'FFNDS_CALIB24_UNIFORM_V1',
        'interpretation': 'The C24 student was trained using the original dense-calibration row draw. OP1 profiles reuse identical native coordinates, so its learned state population need not be independent of that draw.',
        'TC1_limit': 'The enumeration test proves the HT formula for a fixed population independent of sampling. It does not prove design-unbiasedness for these learned states. Existing TC1 values and rejection remain unchanged; treat them as design-weighted descriptive statistics on exposed calibration states.',
        'schema_note': 'Dense metadata has no full_rows field; exact population size is derived from its recorded native grid, then checked against OP1 full_rows.',
        'scope': 'Historical metadata lineage only; no new OP2 state statistics, fitting or GPU work',
        'quality_or_speed_acceptance': False}
    atomic_json(ROOT / 'reports/worker_20261020/calibration_row_coordinate_exposure_audit.json', value)
    print(f'Verified exact row-coordinate reuse at all{matches} historical positions')


if __name__ == '__main__':
    main()
