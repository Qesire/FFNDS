"""One frozen synthetic comparison of three complete FFN execution forms."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from ar2_checkpoint import atomic_json, digest
from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.report.exists() or torch.cuda.device_count() != 1:
        raise RuntimeError('One assigned GPU and a new report required')
    protocol = ROOT / 'config/representation_cost_preflight_R10.json'
    p = json.loads(protocol.read_text())
    protocol_hash = digest(protocol)
    source_hash = digest(Path(__file__))
    torch.set_num_threads(8)
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    from quant.hif4_backend import hif4_qdq_tensor
    random = lambda shape, scale=1.: torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale
    m, d, h = [p[k] for k in ['M', 'D', 'H']]
    x = random((m, d))
    w0 = hif4_qdq_tensor(random((h, d), 1 / math.sqrt(d)), force_fp32=True)
    w2 = hif4_qdq_tensor(random((d, h), 1 / math.sqrt(h)), force_fp32=True)
    b0, b2, gate = random((h,), .02), random((d,), .02), random((d,), .5)
    s1, s2 = torch.ones(d, device='cuda'), torch.ones(h, device='cuda')
    # Coefficients are synthetic. No fit, singular vectors, model weights, or
    # claims about approximation quality enter this hardware comparison.
    slope, intercept = torch.full((h,), .5, device='cuda'), torch.zeros(h, device='cuda')
    af = (w2.float() * (slope / s2)[None, :]) @ w0.float()
    bf = b2.float() + w2.float() @ (slope * b0.float() / s2 + intercept)
    af, bf = af.to(torch.bfloat16), bf.to(torch.bfloat16)
    affine, factors = {}, {}
    for k, r in zip(p['affine_remainder_widths'], p['factor_ranks']):
        if k not in affine:
            affine[k] = [w0[:k].contiguous(), w2[:, :k].contiguous(), b0[:k].contiguous()]
        if r not in factors:
            factors[r] = [random((r, d), 1 / math.sqrt(d)), random((h, r), 1 / math.sqrt(r)),
                          random((r, h), 1 / math.sqrt(h)), random((d, r), 1 / math.sqrt(r))]
    checking = True
    controls = []

    def quant(value):
        if value.dtype != torch.bfloat16:
            raise RuntimeError('Quantizer must receive the original BF16 boundary')
        output = hif4_qdq_tensor(value, force_fp32=False)
        if checking:
            reference = hif4_qdq_tensor(value, force_fp32=True)
            mismatches = int(torch.count_nonzero(output.view(torch.int16) != reference.view(torch.int16)))
            finite = bool(torch.isfinite(output).all())
            controls.append({'shape': list(value.shape), 'elements': value.numel(),
                             'bit_mismatches': mismatches, 'finite': finite})
            if mismatches or not finite or output.dtype != reference.dtype:
                raise RuntimeError('Shared BF16 QDQ differs from original quantizer')
        return output

    def forward(arm, step):
        v1 = quant((x.float() / s1).to(torch.bfloat16))
        k, r = p['affine_remainder_widths'][step], p['factor_ranks'][step]
        if arm == 'dense':
            z = F.linear(v1, w0, b0)
            v2 = quant((F.gelu(z, approximate='tanh').float() / s2).to(torch.bfloat16))
            y = F.linear(v2, w2, b2)
        elif arm == 'affine_remainder':
            u, v, bias = affine[k]
            z = F.linear(v1, u, bias)
            v2 = quant((F.gelu(z, approximate='tanh').float() / s2[:k]).to(torch.bfloat16))
            remainder = (v2.float() - z.float() / s2[:k] * slope[:k] - intercept[:k]).to(torch.bfloat16)
            y = F.linear(v1, af, bf) + F.linear(remainder, v)
        else:
            u0, v0, u2, v2 = factors[r]
            z = F.linear(F.linear(v1, u0), v0, b0)
            hidden = quant((F.gelu(z, approximate='tanh').float() / s2).to(torch.bfloat16))
            y = F.linear(F.linear(hidden, u2), v2, b2)
        return y * gate

    calls = [(arm, step) for step in range(4) for arm in p['arms']]
    for arm, step in calls:
        output = forward(arm, step)
        if list(output.shape) != [m, d] or not bool(torch.isfinite(output).all()):
            raise RuntimeError('Synthetic output shape/finite control failed')
    if len(controls) != 24:
        raise RuntimeError('Every arm/step Q1 and Q2 needs its own pre-timing check')
    checking = False
    for _ in range(p['warmups']):
        for arm, step in calls:
            forward(arm, step)
    torch.cuda.synchronize()
    samples = {arm: [[] for _ in range(4)] for arm in p['arms']}
    orders = []
    for repetition in range(p['repetitions']):
        offset = repetition % len(calls)
        order = calls[offset:] + calls[:offset]
        if repetition % 2:
            order = list(reversed(order))
        orders.append([[arm, step] for arm, step in order])
        for arm, step in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); output = forward(arm, step); end.record(); end.synchronize()
            samples[arm][step].append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('Timed synthetic output is nonfinite')
            del output
    medians = {a: [statistics.median(v) for v in samples[a]] for a in p['arms']}
    totals = {a: sum(v) for a, v in medians.items()}
    ratios = {a: totals['dense'] / v for a, v in totals.items()}
    mac = {'dense': [1.] * 4,
           'affine_remainder': [(d*d + 2*d*k) / (2*d*h) for k in p['affine_remainder_widths']],
           'full_H_factorized': [r*(d+h) / (d*h) for r in p['factor_ranks']]}
    if mac['affine_remainder'] != [.25, .5, .5, .75] or sum(mac['full_H_factorized']) / 4 > .5:
        raise RuntimeError('Fixed analytic main MAC budgets changed')
    if digest(protocol) != protocol_hash or digest(Path(__file__)) != source_hash:
        raise RuntimeError('Probe source changed during execution')
    result = {'status': 'PASS_FIXED_SYNTHETIC_REPRESENTATION_COST',
        'protocol_sha256': protocol_hash, 'source_sha256': source_hash,
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'shape': {'M': m, 'D': d, 'H': h}, 'QDQ_controls': controls,
        'all_controls_before_timing': True, 'same_QDQ_for_all_arms': True,
        'main_MAC_fractions': mac, 'samples_ms': samples, 'median_ms_by_step': medians,
        'sum_four_step_complete_medians_ms': totals, 'same_run_four_step_speedups': ratios,
        'timing_order': orders, 'models_loaded': 0, 'fitted_parameters': 0,
        'quality_or_speed_acceptance': False, 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': p['interpretation'], 'precision': p['precision']}
    atomic_json(args.report, result)
    print(json.dumps({'sum_four_steps_ms': totals, 'ratios': ratios}), flush=True)


if __name__ == '__main__':
    main()
