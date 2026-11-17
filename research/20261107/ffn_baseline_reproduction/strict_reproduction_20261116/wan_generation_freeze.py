"""Freeze the five-arm Wan comparison once its prerequisite experiments finish."""
import argparse
import hashlib
import json
from pathlib import Path

DEV11 = ('vbi_1013', 'vbi_1040', 'vbi_0886', 'vbi_0947', 'vbi_1109', 'vbi_1084',
         'vbi_0950', 'vbi_0176', 'vbi_0800', 'vbi_0392', 'vbi_0056')
DEV3 = ('vbi_1013', 'vbi_0886', 'vbi_0800')
ARMS = ('Dense', 'Q1r2', 'ICC', 'Chipmunk Author Safe', 'Global-P64')
SITES = {(expert, layer) for expert in ('high', 'low') for layer in range(40)}


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def bound(spec):
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise ValueError(f'changed frozen evidence: {path}')
    return json.loads(path.read_bytes())


def completed(spec):
    stage = Path(spec['stage']).resolve()
    config, run, exit_record, payload = (bound(spec[key]) for key in ('scope', 'run', 'exit', 'payload'))
    if (Path(spec['scope']['path']).resolve() != stage / 'scope.json'
            or Path(spec['run']['path']).resolve() != stage / 'receipt/RUN.json'
            or Path(spec['exit']['path']).resolve() != stage / 'receipt/EXIT.json'
            or not Path(spec['payload']['path']).resolve().is_relative_to(stage / 'output')):
        raise ValueError('producer receipts and output must belong to the exact frozen stage')
    actual_job_id = run['slurm_job_id'] if 'slurm_job_id' in run else run['allocation']['job_id']
    if (str(actual_job_id) != str(spec['job_id']) or str(exit_record['slurm_job_id']) != str(spec['job_id'])
            or exit_record['exit_code'] != 0 or run['scope_sha256'] != spec['scope']['sha256']):
        raise ValueError('producer did not complete the exact registered scope')
    return config, payload


def make_freeze(spec):
    stats = {}
    for key in ('icc_neutral', 'comparators_neutral', 'local_grid', 'fixed_comparators',
                'selected_decomposition', 'numeric_precision', 'true_m', 'q1_fit', 'author_shape'):
        _config, stats[key] = completed(spec['prerequisites'][key])
    if (stats['icc_neutral']['status'], stats['icc_neutral']['nfe'], stats['icc_neutral']['d4']) != ('PASS', 8, 2):
        raise ValueError('ICC host-neutral prerequisite incomplete')
    if (stats['comparators_neutral']['status'], stats['comparators_neutral']['nfe'],
            stats['comparators_neutral']['extra_d4']) != ('PASS', 24, 6):
        raise ValueError('three comparator neutral controls incomplete')
    group = {row['arm'] for row in stats['comparators_neutral']['arms']}
    if group != {'Q1r2', 'Global-P64', 'Chipmunk Author Safe'}:
        raise ValueError('wrong comparator neutrality cohort')
    grid = stats['local_grid']
    if grid['cell_count'] != 792 or len(grid['rows']) != 792 or set(row['source'] for row in grid['rows']) != set(DEV11):
        raise ValueError('local selection grid incomplete')
    decomposition = stats['selected_decomposition']
    if (decomposition['status'], decomposition['factor_actions'], len(decomposition['rows'])) != ('DECOMPOSITION_COMPLETE', 132, 66):
        raise ValueError('selected-family error readout incomplete')
    if (stats['fixed_comparators']['status'], stats['fixed_comparators']['count']) != ('LOCAL_COMPARATORS_COMPLETE', 132):
        raise ValueError('fixed comparators missing P512 cells')
    if (stats['numeric_precision']['status'], stats['numeric_precision']['count']) != ('NUMERIC_CELLS_COMPLETE', 16):
        raise ValueError('fixed numeric diagnostic incomplete')
    true_m = stats['true_m']
    if (true_m['status'], true_m['prediction_count'], true_m['arithmetic_pairs']) != ('COMPLETE', 24, 4):
        raise ValueError('author/GP64 exact true-M or same-mask evidence incomplete')
    author = stats['author_shape']
    if (author['status'], author['author_calls'], len(author['rows'])) != ('SIX_AUTHOR_CALLS_COMPLETE_NUMERICS_DESCRIPTIVE', 6, 6):
        raise ValueError('original author shape and action evidence incomplete')
    icc = bound(spec['icc_parameters'])
    if (icc['status'] != 'ALL_80_FROZEN' or icc['rank'] != 1408 or icc['family'] not in ('SVD', 'CA', 'CD')
            or icc['scores_sha256'] != spec['prerequisites']['local_grid']['payload']['sha256']
            or icc['selection_sha256'] != decomposition['selection_sha256']
            or len(icc['factor_pairs']) != 80
            or {(row['identity']['expert'], row['identity']['layer']) for row in icc['factor_pairs']} != SITES):
        raise ValueError('selected ICC family lacks reconciled full-depth factors')
    q1 = stats['q1_fit']
    if (spec['q1_parameters'] != spec['prerequisites']['q1_fit']['payload']
            or q1['status'] != 'ALL_80_FINITE_FIXED_ITERATES' or len(q1['sites']) != 80
            or sum(q1['fit_status_counts'].values()) != 80
            or {(row['expert'], row['layer']) for row in q1['sites']} != SITES):
        raise ValueError('Q1 production parameters or status evidence incomplete')
    if spec['optional_cal22_enabled'] is not False:
        raise ValueError('CAL22 and E5 are disabled in this scope')
    paired = bound(spec['paired_contract'])
    if tuple(paired['sources']) != DEV11 or tuple(paired['arms']) != ARMS:
        raise ValueError('quality frame and cohort contract differs')
    e6 = bound(spec['e6_decision'])
    if (e6['true_m_summary_sha256'] != spec['prerequisites']['true_m']['payload']['sha256']
            or e6['status'] not in ('E6_DISABLED', 'E6_TAIL52_FROZEN')):
        raise ValueError('E6 selection must bind actual scoped adaptation evidence')
    tail_sha = spec['e6_decision']['sha256'] if e6['status'] == 'E6_TAIL52_FROZEN' else None
    if tail_sha:
        witnesses = e6['witnesses']
        expected = {(s, e, 19) for s in ('vbi_1013', 'vbi_0800') for e in ('high', 'low')}
        if (e6['sources'] != list(DEV3) or not e6['affects_tail_adaptation'] or len(witnesses) != 4
                or {(row['source'], row['expert'], row['layer']) for row in witnesses} != expected
                or not all(row['legal'] and row['y_rms_inside_control_envelope'] for row in witnesses)
                or not any(row['gated_change_exceeds_twice_reference_bound'] for row in witnesses)):
            raise ValueError('E6 actual legality, numerical envelope or fixed cohort differs')
    return {'main_family_selected': True, 'selected_icc_family': icc['family'], 'main_icc_rank': 1408,
            'main_icc_state': '4S', 'parameters': {'ICC': spec['icc_parameters'], 'Q1r2': spec['q1_parameters']},
            'arms': list(ARMS), 'source_order': list(DEV11), 'seed': 42,
            'expected_main_slots': [{'arm': arm, 'source': source} for arm in ARMS for source in DEV11],
            'e4_same_factor_approved': True, 'e4_sources': list(DEV3),
            'optional_cal22_enabled': False, 'e5_video_count': 0,
            'e6_tail52_decision_sha256': tail_sha, 'e6_video_count': 3 if tail_sha else 0,
            'e6_decision': spec['e6_decision'], 'paired_contract': spec['paired_contract'],
            'prerequisites': spec['prerequisites']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('input_sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = make_freeze(bound({'path': str(args.input), 'sha256': args.input_sha256}))
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')

