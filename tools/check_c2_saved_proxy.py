"""Portable consistency of the saved-only C2 CPU reference comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    read = lambda p: json.loads((ROOT / p).read_text())
    p = ROOT / 'research/20261023/C2_scale_energy_CPU_results.json'
    r = json.loads(p.read_text()); rows = r['rows']; c = r['original_gate_controls']
    assert len(rows) == 72 and len(c) == 12
    assert all(x['24calibration_gate_bit_identity'] and len(x['payload_sha256']) == 24 for x in c)
    assert r['original_FFN_or_GPU_calls'] == r['new_training_or_factorizations'] == 0
    assert r['second_R256_linears_CPU_reference'] == 72
    assert r['actual_candidate_video_or_policy_speed_result'] is False and not r['automatic_model_followup']
    total = sum(x['total_sampled_nextnorm_loss'] for x in rows)
    for name, v in r['results'].items():
        captured = sum(x['captured'][name] for x in rows)
        assert abs(v['fraction_of_sampled_nextnorm_loss_captured'] - captured / total) < 1e-12
        assert abs(sum(v['per_source_captured_loss'].values()) - captured) < 1e-6
    for x in rows:
        assert all(0 <= value <= x['captured']['sample_oracle'] + 1e-8 for value in x['captured'].values())
    assert r['plan_sha256'] == hashlib.sha256((ROOT / 'research/20261023/C2_scale_energy_CPU_plan.json').read_bytes()).hexdigest()
    assert r['source_sha256'] == hashlib.sha256((ROOT / 'tools/analyze_c2_scale_energy_cpu.py').read_bytes()).hexdigest()
    receipt = read('research/20261023/C2_scale_energy_CPU_results.json.redaction.json')
    assert receipt['numeric_values_exact'] and receipt['public_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    print('PASS C2 saved CPU reference:72paired scopes/12x24gate controls; zero new model/GPU calls; no video claim')


if __name__ == '__main__': main()
