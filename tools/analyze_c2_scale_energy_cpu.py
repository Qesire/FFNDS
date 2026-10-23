"""Compare a CPU reference to R16 energy with one analytic scale control."""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from c1_contract import BLOCKS, ROOT, SOURCES, atomic, read, sha


def main():
    from historical_source_snapshot import HistoricalSources
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import load_metadata
    from run_ar2_restart import tensor_sha
    hf = Path(os.environ['HF_HOME']); root = hf / 'derived/ffnds/C1N0_native_information'
    output = hf / 'derived/ffnds/C2_scale_energy_CPU'
    output.mkdir(exist_ok=False)
    protocol = read(ROOT / 'research/20261023/C1N0_registered_protocol.json')
    positions = UB1Positions(ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json',
                             HistoricalSources(), expected_policy_sha256=protocol['UB1_policy_sha256'])
    metadata = load_metadata(positions)
    bindings = read(positions.base['high_noise_model'].root / 'identity.json')['identity']['training_collections']
    cases = {}
    for binding in bindings:
        if sha(binding['run']) != binding['sha256']: raise RuntimeError('Changed original calibration')
        for row in read(binding['run'])['completed']:
            if sha(row['input_report']) != row['input_report_sha256']: raise RuntimeError('Changed case')
            cases[row['index']] = read(row['input_report'])
    if set(cases) != set(range(24)): raise RuntimeError('Need24calibration gate controls')
    gates = {}; controls = []
    for step in range(4):
        for block in BLOCKS:
            gate = None; receipts = []
            for i in range(24):
                s = next(x for x in cases[i]['scopes'] if (x['step'], x['block']) == (step, block))
                if sha(s['path']) != s['sha256']: raise RuntimeError('Changed original gate payload')
                with safe_open(s['path'], framework='pt', device='cpu') as h: value = h.get_tensor('gate').reshape(-1)
                if value.shape != (5120,) or (gate is not None and not torch.equal(value, gate)):
                    raise RuntimeError('Gate reuse source-independence control failed')
                gate = value; receipts.append(s['sha256'])
            gates[step, block] = gate
            controls.append({'step': step, 'block': block, '24calibration_gate_bit_identity': True,
                             'gate_sha256': tensor_sha(gate), 'payload_sha256': receipts})
    rows = []
    for r in read(root / 'RUN.json')['finite_scopes']:
        p = root / Path(r['payload_HF_relative']).name
        if sha(p) != r['payload_sha256']: raise RuntimeError('Changed native C1 payload')
        with np.load(p, allow_pickle=False) as d:
            h = torch.from_numpy(d['h256'].copy()).view(torch.bfloat16)
            rr = metadata[r['step'], r['block']]['residual']
            correction = F.linear(h, rr['second'], rr['bias_second'])
            gate = gates[r['step'], r['block']]
            energy = ((correction.float() * gate.float()) ** 2).sum(1).numpy()
            scaled = energy * d['norm_rstd'] ** 2
            loss = d['nextnorm_UB1'][d['row_ids']]
        captured = {}
        for name, score in [('original_energy_CPU_reference', energy), ('scale_times_energy_CPU_reference', scaled), ('sample_oracle', loss)]:
            chosen = np.argsort(-score, kind='stable')[:84]
            captured[name] = float(loss[chosen].sum())
        rows.append({'source': r['source'], 'step': r['step'], 'block': r['block'],
                     'total_sampled_nextnorm_loss': float(loss.sum()), 'captured': captured})
    denom = sum(r['total_sampled_nextnorm_loss'] for r in rows)
    results = {name: {'fraction_of_sampled_nextnorm_loss_captured': sum(r['captured'][name] for r in rows) / denom,
                      'per_source_captured_loss': {str(s): sum(r['captured'][name] for r in rows if r['source'] == s) for s in SOURCES}}
               for name in rows[0]['captured']}
    record = {'status': 'COMPLETE_C2_SAVED_CPU_ENERGY_REFERENCE_AND_SCALE_CHECK', 'source_sha256': sha(__file__),
              'plan_sha256': sha(ROOT / 'research/20261023/C2_scale_energy_CPU_plan.json'),
              'results': results, 'rows': rows, 'original_gate_controls': controls,
              'native_C1_h256_saved_first_linear_used': True, 'second_R256_linears_CPU_reference': 72,
              'original_FFN_or_GPU_calls': 0, 'new_training_or_factorizations': 0,
              'scope': 'CPU reference to the original R16 gated-energy score, not exact GPU scoring. Calibrated gate reuse/source semantics, BF16 CPU linear and reduction differences are disclosed in the plan. Same exposed C1 data, not an independent video or policy validation.',
              'actual_candidate_video_or_policy_speed_result': False, 'automatic_model_followup': False}
    atomic(output / 'RESULT.json', record)
    print(results)


if __name__ == '__main__': main()
