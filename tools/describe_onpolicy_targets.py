"""Describe captured local teacher errors; no fitting, selection or quality gate."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(8)
    rows, bindings, accepted = [], [], set()
    for path in args.collections:
        run = json.loads(path.read_text())
        bindings.append({'path': str(path.resolve()), 'sha256': digest(path), 'status': run['status'],
            'accepted_inputs': [c['index'] for c in run['completed']]})
        for entry in run['completed']:
            if entry['index'] in accepted or digest(entry['input_report']) != entry['input_report_sha256']:
                raise RuntimeError('Duplicate/changed accepted OP1 input')
            accepted.add(entry['index']); case = json.loads(Path(entry['input_report']).read_text())
            if case['status'] != 'PASS_NATIVE_ONPOLICY_FULL160_AND_NOOP':
                raise RuntimeError('Require accepted full no-op input')
            for scope in case['scopes']:
                if scope['block'] != 20: continue
                if digest(scope['path']) != scope['sha256']: raise RuntimeError('OP1 operand checksum differs')
                with safe_open(scope['path'], framework='pt', device='cpu') as handle:
                    base, old, teacher, gate = [handle.get_tensor(k) for k in ['BaseY', 'StudentY', 'TeacherY', 'gate']]
                target = (teacher * gate).double(); old = (old * gate).double(); base = (base * gate).double()
                denominator = float(target.square().sum())
                old_sse = float((old - target).square().sum()); base_sse = float((base - target).square().sum())
                centered = target - target.mean(0)
                old_centered = old - old.mean(0)
                rows.append({'index': entry['index'], 'step': scope['step'], 'block': 20,
                    'native_old_SSE': old_sse, 'native_base_SSE': base_sse,
                    'old_over_base': old_sse / base_sse, 'target_energy': denominator,
                    'old_relative_RMSE': (old_sse / denominator)**.5,
                    'old_over_teacher_centered_energy': float(old_centered.square().sum() / centered.square().sum()),
                    'sample_rows': scope['rows'], 'input_report_sha256': entry['input_report_sha256'],
                    'operand_sha256': scope['sha256']})
    pooled = []
    for step in range(4):
        selected = [r for r in rows if r['step'] == step]
        if not selected: continue
        old, base, energy = [sum(r[k] for r in selected) for k in ['native_old_SSE', 'native_base_SSE', 'target_energy']]
        pooled.append({'step': step, 'inputs': len(selected), 'old_over_base': old / base,
            'old_relative_RMSE': (old / energy)**.5})
    report = {'status': 'COMPLETE_DESCRIPTIVE_CAPTURE_SNAPSHOT', 'all24_complete': accepted == set(range(24)),
        'collections': bindings, 'accepted_inputs': sorted(accepted), 'positions': rows, 'pooled_by_step': pooled,
        'scope': 'block20 local dense D-Q FFN on frozen C24 student states, same256 SRS rows; not full dense trajectory',
        'equal_source_sampling': True, 'fitting_or_selection': False, 'quality_or_speed_acceptance': False,
        'source_sha256': digest(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.out, report)
    print(json.dumps({'accepted_inputs': len(accepted), 'pooled_by_step': pooled}, indent=2))


if __name__ == '__main__': main()
