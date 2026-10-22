"""One frozen random-tensor FFN cost check; never loads a Wan model."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from ar2_checkpoint import atomic_json, digest
from probe_pooled_nonlinear_moments import projection_cost
from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.output.exists() or torch.cuda.device_count() != 1:
        raise RuntimeError('One assigned idle GPU and new output required')
    protocol = ROOT / 'research/20261022/R13_synthetic_cost_protocol.json'
    p, protocol_sha, source_sha = json.loads(protocol.read_text()), digest(protocol), digest(__file__)
    from quant.hif4_backend import hif4_qdq_tensor
    torch.set_num_threads(8)
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    m, d, h, groups, count = [p[k] for k in ('M', 'D', 'H', 'stripes', 'two_point_stripes')]
    if m != groups*4 or h != 13824 or d != 5120:
        raise RuntimeError('Frozen native-shape geometry changed')
    rand = lambda shape, scale=1.: torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale
    x = rand((m, d))
    w1 = hif4_qdq_tensor(rand((h, d), 1/math.sqrt(d)), force_fp32=True)
    w2 = hif4_qdq_tensor(rand((d, h), 1/math.sqrt(h)), force_fp32=True)
    b1, b2, gate = rand((h,), .02), rand((d,), .02), rand((d,), .5)
    modulation_scale, modulation_shift = rand((d,), .05), rand((d,), .05)
    s1, s2 = torch.ones(d, device='cuda'), torch.ones(h, device='cuda')
    affine = (.5 * w2.float() @ w1.float()).bfloat16()
    affine_bias = (b2.float() + .5*w2.float() @ b1.float()).bfloat16()
    half_w1, half_w2, half_b1 = w1[:h//2].contiguous(), w2[:, :h//2].contiguous(), b1[:h//2].contiguous()
    first, second = rand((256, d), 1/math.sqrt(d)), rand((d, 256), 1/math.sqrt(256))
    first_bias = rand((256,), .02)
    signs = torch.randint(0, 2, (groups, 4, 1), device='cuda').float()*2-1
    all_ids = torch.arange(groups, device='cuda')
    checking, controls = True, []

    def quant(value):
        if value.dtype != torch.bfloat16:
            raise RuntimeError('Original BF16 quantizer boundary required')
        out = hif4_qdq_tensor(value, force_fp32=False)
        if checking:
            reference = hif4_qdq_tensor(value, force_fp32=True)
            mismatches = int(torch.count_nonzero(out.view(torch.int16) != reference.view(torch.int16)))
            controls.append({'shape': list(value.shape), 'bit_mismatches': mismatches})
            if mismatches or not bool(torch.isfinite(out).all()):
                raise RuntimeError('Shared BF16 QDQ control failed')
        return out

    def forward(arm):
        norm = F.layer_norm(x, (d,), eps=1e-6)
        norm = (norm.float()*(1+modulation_scale.float())+modulation_shift.float()).bfloat16()
        v1 = quant((norm.float()/s1).bfloat16())
        if arm == 'dense':
            z = F.linear(v1, w1, b1)
            hidden = quant((F.gelu(z, approximate='tanh').float()/s2).bfloat16())
            y = F.linear(hidden, w2, b2)
        elif arm == 'UB1_half_R256':
            z = F.linear(v1, half_w1, half_b1)
            hidden = quant((F.gelu(z, approximate='tanh').float()/s2[:h//2]).bfloat16())
            y = F.linear(hidden, half_w2, b2) + F.linear(F.linear(v1, first, first_bias), second)
        else:
            linear = F.linear(v1, affine, affine_bias)
            grouped = v1.reshape(groups, 4, d).float()
            mean = grouped.mean(1).bfloat16()
            contrast = ((grouped-mean.float()[:, None])*signs).sum(1).mul(.5).bfloat16()
            score = contrast.float().square().sum(1)
            selected = score.topk(count, sorted=False).indices.sort().values if arm == 'pooled_dynamic' else all_ids[:count]
            mask = torch.ones(groups, device='cuda', dtype=torch.bool)
            mask[selected] = False
            others = all_ids[mask]
            points = torch.cat((mean[others], mean[selected]+contrast[selected], mean[selected]-contrast[selected]))
            z = F.linear(points, w1, b1)
            hidden = quant((F.gelu(z, approximate='tanh').float()/s2).bfloat16())
            pooled = torch.empty((groups, h), device='cuda', dtype=torch.bfloat16)
            n = groups-count
            pooled.index_copy_(0, others, hidden[:n])
            pooled.index_copy_(0, selected, ((hidden[n:n+count].float()+hidden[n+count:].float())*.5).bfloat16())
            response = F.linear(pooled, w2, b2)
            average_linear = linear.reshape(groups, 4, d).float().mean(1).bfloat16()
            remainder = response-average_linear
            y = (linear.reshape(groups, 4, d)+remainder[:, None]).reshape(m, d)
        return x+y*gate

    for arm in p['arms']:
        output = forward(arm)
        if list(output.shape) != [m, d] or not bool(torch.isfinite(output).all()):
            raise RuntimeError('Synthetic output shape/finite control failed')
        del output
    if len(controls) != 8:
        raise RuntimeError('Each arm requires two original-boundary QDQ controls')
    checking = False
    for _ in range(p['warmups']):
        for arm in p['arms']:
            forward(arm)
    torch.cuda.synchronize()
    samples, host_samples, orders = {a: [] for a in p['arms']}, {a: [] for a in p['arms']}, []
    for repetition in range(p['repetitions']):
        offset = repetition % len(p['arms'])
        order = p['arms'][offset:]+p['arms'][:offset]
        if repetition % 2: order = list(reversed(order))
        orders.append(order)
        for arm in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.monotonic()
            begin.record(); output = forward(arm); end.record(); end.synchronize()
            host_samples[arm].append(1000*(time.monotonic()-began))
            samples[arm].append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('Timed output nonfinite')
            del output
    medians = {a: statistics.median(v) for a, v in samples.items()}
    speed = {a: medians['dense']/v for a, v in medians.items()}
    if digest(protocol) != protocol_sha or digest(__file__) != source_sha:
        raise RuntimeError('Frozen synthetic study changed during execution')
    result = {'id': p['id'], 'status': 'COMPLETED_FIXED_SYNTHETIC_RANDOM_TENSOR_COST',
        'evidence_level': 'SYNTHETIC_A100_FULL_SHAPE_NOT_WAN_MODEL',
        'protocol_sha256': protocol_sha, 'source_sha256': source_sha,
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'shape': {'M': m, 'D': d, 'H': h}, 'QDQ_controls': controls,
        'all_controls_before_timing': True, 'same_QDQ_for_all_arms': True,
        'samples_ms': samples, 'host_samples_ms': host_samples, 'median_ms': medians,
        'speedup_over_same_run_dense': speed, 'timing_order': orders,
        'projection_MAC_fraction': projection_cost(m, groups, count, d, h),
        'pooled_dynamic_speed_gate': speed['pooled_dynamic'] >= 1.5,
        'direct_implementation_disposition': 'SYNTHETIC_SPEED_SCREEN_PASSED_NO_MODEL_ELIGIBILITY' if speed['pooled_dynamic'] >= 1.5 else 'DIRECT_IMPLEMENTATION_BELOW_SYNTHETIC_SPEED_GATE_DO_NOT_SCALE',
        'max_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_reserved_bytes': torch.cuda.max_memory_reserved(),
        'models_loaded': 0, 'new_model_generations': 0, 'fitted_parameters': 0, 'new_tensor_files': 0,
        'quality_or_native_model_speed_acceptance': False,
        'limits': p['interpretation']}
    atomic_json(args.output, result)
    print(json.dumps({'median_ms': medians, 'speedup': speed, 'disposition': result['direct_implementation_disposition']}), flush=True)


if __name__ == '__main__':
    main()
