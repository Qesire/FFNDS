"""Reproduce the fixed2D fits and all18 native objective outcomes on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ar2_checkpoint import atomic_json, digest
from restored_assets import ROOT
from run_ar2_restart import tensor_sha
from run_residual_gain_transport import fit_objectives, TRAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    run = args.run.resolve(); reports, entries, cases, receipts = [], [], {}, {}
    fit_path = run / 'FIT.json'; fit = json.loads(fit_path.read_text())
    prior = json.loads((ROOT / 'research/20261021/R7D2_before_execution.json').read_text())['registration']
    for shard in (0, 1):
        p = run / f'shard{shard}/RUN.json'; report = json.loads(p.read_text())
        if report['status'] != 'PASS_FIXED_OBJECTIVE_SHARD' or report['fit_sha256'] != digest(fit_path):
            raise RuntimeError('Incomplete or mismatched objective job/fit')
        identity = report['scientific_identity']
        if identity != fit['scientific_identity'] or identity['sources'] != prior['source_sha256'] or identity['protocol_sha256'] != prior['protocol_sha256']:
            raise RuntimeError('Sources/assets/protocol differ from preregistration')
        receipts[f'shard{shard}'] = {'record': str(p.relative_to(ROOT)), 'sha256': digest(p), 'completed': len(report['completed'])}
        reports.append(report); entries.extend(report['completed'])
    if len(entries) != 18:
        raise RuntimeError('Exactly18 fixed native generations required')
    refs = {'DQ': {}}
    for index in [0, 5, 10, 15]:
        old = json.loads((ROOT / f'reports/worker_20261020/TF2_DQ_input{index:02d}.json').read_text())
        record = old['steps'][3]
        if digest(record['path']) != record['sha256']:
            raise RuntimeError('Original DQ target payload changed')
        target = load_file(record['path'], device='cpu')['flow']
        if tensor_sha(target) != record['flow_sha256']:
            raise RuntimeError('Original decoded DQ target changed')
        refs['DQ'][index] = {'payloads': [None, None, None, {'flow': target}]}
    for row in entries:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Native gain case receipt changed')
        case = json.loads(Path(row['path']).read_text())
        key = (case['index'], case['label'])
        if case['status'] != 'PASS_NATIVE_GAIN_RUN' or key in cases or key != (row['index'], row['label']):
            raise RuntimeError('Invalid native gain case identity')
        record = case['steps'][3]; payload = Path(record['path'])
        if digest(payload) != record['sha256']:
            raise RuntimeError('Native gain flow payload changed')
        values = load_file(str(payload), device='cpu')
        for name in ['flow', 'post']:
            if values[name].dtype != torch.float32 or list(values[name].shape) != [16, 21, 58, 104] or tensor_sha(values[name]) != record[name + '_sha256']:
                raise RuntimeError('Decoded gain output tensor changed')
        actual = float((values['flow'].double() - refs['DQ'][case['index']]['payloads'][3]['flow'].double()).square().sum())
        if abs(actual - record['flow_error']['squared_error']) > 1e-9 * max(1., actual):
            raise RuntimeError('Recorded gain flow SSE does not reproduce')
        if [s['block'] for s in case['scopes']] != [30, 31] or not all(s['native_RNG_unchanged'] for s in case['scopes']):
            raise RuntimeError('Missing fixed-pair/RNG controls')
        cases[key] = case
    expected = {(i, a) for i in [0, 5, 10, 15] for a in ['identity', 'local_fit', 'flow_fit']} | {(i, a) for i in TRAIN for a in ['direction_A', 'direction_B']}
    if set(cases) != expected:
        raise RuntimeError('Training/heldout arm membership changed')
    fit_rows = [r for r in entries if r['label'] == 'identity'] + [r for r in entries if r['label'].startswith('direction_')]
    recomputed = fit_objectives(fit_rows, refs, fit['scientific_identity'])
    if recomputed != fit:
        raise RuntimeError('Both deterministic objective fits do not reproduce exactly')
    rows = []
    for index in [0, 5, 10, 15]:
        group = {a: cases[index, a] for a in ['identity', 'local_fit', 'flow_fit']}
        for label, case in group.items():
            desired = [1., 1.] if label == 'identity' else fit[label]['gains']
            if case['gains'] != desired or [s['gain'] for s in case['scopes']] != desired:
                raise RuntimeError('Native gains differ from frozen fit')
            for step in range(4):
                if case['steps'][step]['input_sha256'] != group['identity']['steps'][step]['input_sha256']:
                    raise RuntimeError('Gain arms received different DQ step states')
                if step < 3 and any(case['steps'][step][k] != group['identity']['steps'][step][k] for k in ['flow_sha256', 'post_sha256']):
                    raise RuntimeError('Gain change affected an earlier step')
            if any(case['scopes'][0][k] != group['identity']['scopes'][0][k] for k in ['norm_input_sha256', 'residual_sha256', 'teacher_Y_sha256']):
                raise RuntimeError('A did not receive identical original inputs/residual/teacher')
        sse = {a: c['steps'][3]['flow_error']['squared_error'] for a, c in group.items()}
        payloads = {}
        for label, case in group.items():
            record = case['steps'][3]; path = Path(record['path']); at = path.parts.index('r7d2_residual_gain_20261021')
            payloads[label] = {'HF_derived_relative': str(Path(*path.parts[at:])), 'sha256': record['sha256']}
        rows.append({'index': index, 'used_for_fit': index in TRAIN, 'flow_SSE': sse,
                     'flow_SSE_reduction_fraction': {a: 1 - v / sse['identity'] for a, v in sse.items()},
                     'local_gated_SSE': {a: sum(s['local_gated_error']['ideal_gated_SSE'] for s in c['scopes']) for a, c in group.items()},
                     'scopes': {a: c['scopes'] for a, c in group.items()}, 'payloads': payloads})
    pooled = {scope: {a: sum(r['flow_SSE'][a] for r in rows if (r['used_for_fit'] if scope == 'train' else not r['used_for_fit'] if scope == 'holdout' else True))
                      for a in ['identity', 'local_fit', 'flow_fit']} for scope in ['train', 'holdout', 'all_development']}
    prediction = {}
    g, b = np.asarray(fit['flow_fit']['gram']), np.asarray(fit['flow_fit']['linear'])
    for label in ['local_fit', 'flow_fit']:
        delta = np.asarray(fit[label]['delta'])
        predicted = pooled['train']['identity'] + float(2 * b @ delta + delta @ g @ delta)
        prediction[label] = {'predicted_training_flow_SSE': predicted,
                             'actual_training_flow_SSE': pooled['train'][label],
                             'actual_minus_predicted_SSE': pooled['train'][label] - predicted}
    result = {'id': 'R7D2_RESIDUAL_GAIN_OBJECTIVE', 'status': 'PASS_COMPLETE_FIXED_OBJECTIVE_COMPARISON',
              'analysis_source_sha256': digest(Path(__file__)), 'scientific_identity': fit['scientific_identity'],
              'completed_generations': 18, 'job_receipts': receipts, 'raw_fit_sha256': digest(fit_path), 'fit': fit,
              'fits_reproduced_exactly': True, 'rows': rows, 'pooled_flow_SSE': pooled,
              'pooled_flow_SSE_reduction_fraction': {scope: {a: 1 - v / data['identity'] for a, v in data.items()} for scope, data in pooled.items()},
              'flow_linear_prediction_check': prediction,
              'flow_fit_beats_local_fit_on_development_holdout': pooled['holdout']['flow_fit'] < pooled['holdout']['local_fit'],
              'flow_fit_beats_frozen_UB1_on_development_holdout': pooled['holdout']['flow_fit'] < pooled['holdout']['identity'],
              'native_main_MAC_fraction': .5, 'rank': 256, 'quality_or_speed_acceptance': False,
              'data_scope': 'Three already-exposed fit sources and one already-exposed fit-excluded development source; no fresh quality evaluation',
              'limitations': 'Two bounded gains and secant-flow objective only. No rank/support/layer/bounds expansion. Native extra multiplication not benchmarked as completeFFN. Teacher-forced videos are not candidate quality.'}
    atomic_json(args.out, result)
    print(json.dumps({'status': result['status'], 'fits': {k: fit[k]['gains'] for k in ['local_fit', 'flow_fit']},
                      'pooled_flow_SSE': pooled, 'holdout_flow_beats_local': result['flow_fit_beats_local_fit_on_development_holdout']}), flush=True)


if __name__ == '__main__':
    main()
