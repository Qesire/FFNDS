"""Verify R7D1 native receipts and publish only portable numerical evidence."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ar2_checkpoint import atomic_json, digest
from cross_layer_transport import transport_accounting
from restored_assets import ROOT
from run_ar2_restart import tensor_sha
from run_cross_layer_transport import INDICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    run = args.run.resolve()
    cases, receipts = {}, {}
    identity = None
    for name in ['control_DQ', 'control_UB1', 'intervention_shard0', 'intervention_shard1']:
        path = run / name / 'RUN.json'
        report = json.loads(path.read_text())
        if report['status'] != 'PASS_ALL_REGISTERED_RUNS':
            raise RuntimeError('Incomplete/failed model job')
        if identity is None:
            identity = report['scientific_identity']
        elif identity != report['scientific_identity']:
            raise RuntimeError('Model job scientific identities differ')
        receipts[name] = {'record': str(path.relative_to(ROOT)), 'sha256': digest(path), 'completed': len(report['completed'])}
        for row in report['completed']:
            p = Path(row['path'])
            if digest(p) != row['sha256']:
                raise RuntimeError('Native input receipt changed')
            case = json.loads(p.read_text())
            if case['status'] != 'PASS_FIXED_NATIVE_PAIR_RUN' or case['index'] != row['index'] or case['arm'] != row['arm']:
                raise RuntimeError('Invalid case identity')
            key = (case['index'], case['arm'])
            if key in cases:
                raise RuntimeError('Duplicate native arm')
            cases[key] = case
    arms = ['DQ', 'none', 'none_repeat', 'A', 'B', 'AB']
    frozen = json.loads((ROOT / 'research/20261021/R7D1_before_execution.json').read_text())['registration']
    if (identity['protocol_sha256'] != frozen['protocol_sha256']
            or identity['sources'] != frozen['source_sha256']):
        raise RuntimeError('Jobs differ from the pre-execution source/protocol freeze')
    for name in ['intervention_shard0', 'intervention_shard1']:
        report = json.loads((run / name / 'RUN.json').read_text())
        if report['control_bindings'] != {k: receipts[k]['sha256'] for k in ['control_DQ', 'control_UB1']}:
            raise RuntimeError('Intervention control prerequisites do not match original receipts')
    if set(cases) != {(i, a) for i in INDICES for a in arms}:
        raise RuntimeError('Exactly24 registered native generations required')
    rows = []
    for index in INDICES:
        group = {a: cases[index, a] for a in arms}
        flows, payloads = {}, {}
        for arm, case in group.items():
            step = case['steps'][3]
            path = Path(step['path'])
            if digest(path) != step['sha256']:
                raise RuntimeError('Native result tensor payload changed')
            data = load_file(str(path), device='cpu')
            if set(data) != {'flow', 'post'}:
                raise RuntimeError('Unexpected native result payload')
            for k, value in data.items():
                if value.dtype != torch.float32 or list(value.shape) != [16, 21, 58, 104] or tensor_sha(value) != step[k + '_sha256']:
                    raise RuntimeError('Decoded native result tensor changed')
            flows[arm] = data['flow']
            parts = path.parts
            offset = parts.index('r7d1_cross_layer_transport_20261021')
            payloads[arm] = {'HF_derived_relative': str(Path(*parts[offset:])), 'sha256': step['sha256'],
                             'flow_sha256': step['flow_sha256'], 'post_sha256': step['post_sha256']}
            if any(s['input_sha256'] != group['DQ']['steps'][j]['input_sha256'] for j, s in enumerate(case['steps'])):
                raise RuntimeError('Cross-arm per-step native inputs differ')
            if [s['block'] for s in case['scopes']] != [30, 31] or not all(s['native_RNG_unchanged'] for s in case['scopes']):
                raise RuntimeError('Missing pair/RNG controls')
        if not torch.equal(flows['none'], flows['none_repeat']):
            raise RuntimeError('Repeated none not bitwise identical')
        scopes = {a: {s['block']: s for s in c['scopes']} for a, c in group.items()}
        for arm in ['none_repeat', 'A', 'B', 'AB']:
            if any(scopes[arm][30][k] != scopes['none'][30][k] for k in ('norm_input_sha256', 'student_Y_sha256', 'teacher_Y_sha256')):
                raise RuntimeError('A did not receive identical pre-intervention operands')
        for left, right in [('none', 'none_repeat'), ('none', 'B'), ('A', 'AB')]:
            if any(scopes[left][31][k] != scopes[right][31][k] for k in ('norm_input_sha256', 'student_Y_sha256', 'teacher_Y_sha256')):
                raise RuntimeError('B actual incoming-state pairing differs')
        for arm in arms:
            for block in [30, 31]:
                if any(scopes[arm][block][k] != scopes['none'][block][k] for k in ('b0_sha256', 'b2_sha256', 'gate_sha256', 'teacher_shard_sha256')):
                    raise RuntimeError('Original bias/gate/teacher changed across arms')
                s = scopes[arm][block]
                if s['returned_Y_sha256'] != s['teacher_Y_sha256' if s['restored'] else 'student_Y_sha256']:
                    raise RuntimeError('Wrong output returned at the native intervention boundary')
        if not all(s['dense_identity_bitwise_equal'] for s in group['DQ']['scopes']):
            raise RuntimeError('Missing full native teacher identity control')
        metrics = transport_accounting({a: flows[a] for a in arms if a != 'DQ'}, flows['DQ'])
        for arm in arms:
            actual = float((flows[arm].double() - flows['DQ'].double()).square().sum())
            saved = group[arm]['steps'][3]['flow_error']['squared_error']
            if abs(actual - saved) > 1e-9 * max(1., actual):
                raise RuntimeError('Saved flow SSE does not reproduce')
        local = {a: scopes['none'][b]['local_before_restoration']['ideal_gated_SSE'] for a, b in [('A', 30), ('B', 31)]}
        local_prefer = 'A' if local['A'] > local['B'] else 'B' if local['A'] < local['B'] else 'tie'
        gain = metrics['flow_SSE_reduction']
        downstream_prefer = 'A' if gain['A'] > gain['B'] else 'B' if gain['A'] < gain['B'] else 'tie'
        rows.append({'index': index, **metrics, 'baseline_local_gated_SSE': local,
                     'local_larger_error': local_prefer, 'downstream_larger_benefit': downstream_prefer,
                     'local_downstream_rank_disagreement': local_prefer != downstream_prefer,
                     'scopes': {a: c['scopes'] for a, c in group.items()}, 'payloads': payloads,
                     'all4_step_original_controls': {a: [{k: v for k, v in step.items() if k not in ('path',)}
                         for step in group[a]['steps']] for a in ['DQ', 'none', 'none_repeat']}})
        print(f'input{index:02d} SSE={metrics["flow_SSE"]}; interaction={metrics["interaction_advantage"]:.6f}', flush=True)
    pooled_sse = {a: sum(r['flow_SSE'][a] for r in rows) for a in arms if a != 'DQ'}
    pooled_gain = {a: pooled_sse['none'] - s for a, s in pooled_sse.items()}
    result = {'id': 'R7D1_CROSS_LAYER_TRANSPORT', 'status': 'PASS_COMPLETE_NATIVE_FIXED_PAIR_DIAGNOSTIC',
              'analysis_source_sha256': digest(Path(__file__)),
              'scientific_identity': identity, 'job_receipts': receipts, 'completed_generations': 24,
              'original_TF2_input_flow_post_controls': 'All4 steps on4 DQ,4 none,4 repeated-none generations bitwise equal',
              'selected_full_teacher_identity_controls': 8, 'RNG_controls': 48,
              'pair_input_hash_controls': True, 'repeated_none_bitwise_equal': True,
              'rows': rows, 'pooled_flow_SSE': pooled_sse, 'pooled_flow_SSE_reduction': pooled_gain,
              'pooled_flow_SSE_reduction_fraction': {a: v / pooled_sse['none'] for a, v in pooled_gain.items()},
              'pooled_interaction_advantage': sum(r['interaction_advantage'] for r in rows),
              'rank_disagreement_source_count': sum(r['local_downstream_rank_disagreement'] for r in rows),
              'quality_or_speed_acceptance': False,
              'data_scope': 'Four development-exposed original TF2 sources; fixed pair only, no train or fresh quality test',
              'diagnostic_scope': 'Full dense restoration violates candidate budget; flow recovery is not candidate video recovery'}
    atomic_json(args.out, result)


if __name__ == '__main__':
    main()
