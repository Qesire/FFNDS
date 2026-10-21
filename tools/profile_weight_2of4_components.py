"""One fixed synthetic decomposition of the installed full-H sparse FFN."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

from ar2_checkpoint import atomic_json, digest
from probe_weight_2of4_shape import rel_error
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
    protocol = ROOT / 'config/weight_2of4_component_profile.json'
    p = json.loads(protocol.read_text())
    torch.set_num_threads(8)
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    SparseSemiStructuredTensor._FUSE_TRANSPOSE = True
    from quant.hif4_backend import hif4_qdq_tensor
    quant = lambda x: hif4_qdq_tensor(x, force_fp32=True).to(torch.bfloat16)
    random = lambda shape, scale=1.: torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale
    m, d, h, rank = [p[k] for k in ['M', 'D', 'H', 'rank']]
    x = random((m, d)); w0 = quant(random((h, d), 1 / math.sqrt(d))); w2 = quant(random((d, h), 1 / math.sqrt(h)))
    b0, b2 = random((h,), .02), random((d,), .02)
    s1, s2 = torch.ones(d, device='cuda'), torch.ones(h, device='cuda')
    gate = random((d,), .5)
    r0, r2 = random((rank, d), 1 / math.sqrt(d)), random((d, rank), .01 / math.sqrt(rank))

    def mask(w):
        grouped = w.reshape(w.shape[0], -1, 4)
        indices = grouped.abs().topk(2, dim=-1).indices
        return torch.zeros_like(grouped).scatter_(-1, indices, grouped.gather(-1, indices)).reshape_as(w).contiguous()

    masked0, masked2 = mask(w0), mask(w2)
    sparse0, sparse2 = to_sparse_semi_structured(masked0), to_sparse_semi_structured(masked2)
    if not torch.equal(sparse0.to_dense(), masked0) or not torch.equal(sparse2.to_dense(), masked2):
        raise RuntimeError('Packed sparse weight differs')
    v1 = quant((x.float() / s1).to(torch.bfloat16))
    z = F.linear(v1, sparse0, b0).contiguous()
    hidden = F.gelu(z, approximate='tanh')
    v2 = quant((hidden.float() / s2).to(torch.bfloat16))
    y = F.linear(v2, sparse2, b2).contiguous()
    controls = [rel_error(z, F.linear(v1, masked0, b0)), rel_error(y, F.linear(v2, masked2, b2))]
    if any(e > .01 or not math.isfinite(e) for e in controls):
        raise RuntimeError('Sparse GEMM numeric control failed')

    def chain(kind):
        q1 = quant((x.float() / s1).to(torch.bfloat16))
        a, b = (w0, w2) if kind == 'dense' else (sparse0, sparse2) if kind == 'sparse' else (masked0, masked2)
        z0 = F.linear(q1, a, b0).contiguous()
        q2 = quant((F.gelu(z0, approximate='tanh').float() / s2).to(torch.bfloat16))
        return F.linear(q2, b, b2).contiguous() * gate

    chain_error = rel_error(chain('sparse'), chain('masked'))
    operations = {
        'Q1_smooth_cast_native_QDQ': lambda: quant((x.float() / s1).to(torch.bfloat16)),
        'FFN0_sparse_bias_layout': lambda: F.linear(v1, sparse0, b0).contiguous(),
        'FFN0_dense_bias_layout': lambda: F.linear(v1, w0, b0).contiguous(),
        'full_H_GELU': lambda: F.gelu(z, approximate='tanh'),
        'Q2_smooth_cast_native_QDQ': lambda: quant((hidden.float() / s2).to(torch.bfloat16)),
        'FFN2_sparse_bias_layout': lambda: F.linear(v2, sparse2, b2).contiguous(),
        'FFN2_dense_bias_layout': lambda: F.linear(v2, w2, b2).contiguous(),
        'gate_product': lambda: y * gate,
        'R256_two_GEMMs_and_add': lambda: y + F.linear(F.linear(v1, r0), r2),
        'complete_dense': lambda: chain('dense'), 'complete_sparse': lambda: chain('sparse')}
    for _ in range(p['warmups']):
        for operation in operations.values(): operation()
    torch.cuda.synchronize()
    samples = {name: [] for name in operations}
    for repetition in range(p['repetitions']):
        order = list(operations) if repetition % 2 == 0 else list(reversed(operations))
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); output = operations[name](); end.record(); end.synchronize()
            samples[name].append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('Nonfinite timed output')
            del output
    medians = {k: statistics.median(v) for k, v in samples.items()}
    result = {'status': 'PASS_FIXED_SYNTHETIC_COMPONENT_PROFILE', 'protocol_sha256': digest(protocol),
        'source_sha256': digest(Path(__file__)), 'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'shape': {k: p[k] for k in ['M', 'D', 'H', 'rank']}, 'linear_relative_RMSE': controls,
        'chain_sparse_vs_masked_relative_RMSE': chain_error, 'samples_ms': samples, 'median_ms': medians,
        'same_run_complete_speedup': medians['complete_dense'] / medians['complete_sparse'],
        'models_loaded': 0, 'fitted_parameters': 0, 'quality_or_speed_acceptance': False,
        'limitations': 'Synthetic random values and unit smooth scales; separately timed components do not add exactly. No full160/native real-weight/video speed or quality claim.'}
    atomic_json(args.report, result)
    print(json.dumps({'median_ms': medians, 'same_run_complete_speedup': result['same_run_complete_speedup']}), flush=True)


if __name__ == '__main__':
    main()
