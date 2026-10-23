"""Synthetic CPU numerical readiness for the proposed statistics observer.

No original model, FFN, weights, training, evaluator or CUDA operation.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path

from prew1_norm_statistics import layer_norm_with_statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run this CPU fixture with CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    generator = torch.Generator(device='cpu').manual_seed(20261023)
    cases = []
    regimes = [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
               (torch.float32, torch.float32)]
    for (infer, sensitive), rows, affine, scale in itertools.product(regimes, (1, 32, 256), (False, True), (1., 4.)):
        x = (scale * torch.randn(rows, 5120, generator=generator)).to(infer)
        saved = x.clone()
        operation_dtype = torch.float32 if infer != sensitive else infer
        weight = torch.randn(5120, generator=generator).to(operation_dtype) if affine else None
        bias = torch.randn(5120, generator=generator).to(operation_dtype) if affine else None
        value = x.float() if sensitive != infer else x
        expected = torch.nn.functional.layer_norm(value, (5120,), weight, bias, 1e-6)
        if sensitive != infer:
            expected = expected.to(infer)
        actual, mean, inverse_std = layer_norm_with_statistics(x, infer_dtype=infer,
            sensitive_layer_dtype=sensitive, weight=weight, bias=bias)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        assert torch.equal(x.view(torch.uint8), saved.view(torch.uint8))
        assert tuple(mean.shape) == tuple(inverse_std.shape) == (rows, 1)
        assert mean.device.type == inverse_std.device.type == 'cpu'
        assert bool(torch.isfinite(mean).all() and torch.isfinite(inverse_std).all())
        cases.append({'rows': rows, 'dimension': 5120, 'infer_dtype': str(infer),
                      'sensitive_dtype': str(sensitive), 'affine': affine, 'input_scale': scale,
                      'output_bitwise_equal': True, 'input_unchanged': True,
                      'statistics_dtype': str(mean.dtype), 'statistics_shape': list(mean.shape)})
    # The scale collision in the mathematical discriminator also survives
    # an actual CPU BF16 normalization call; the exported statistic separates it.
    u = torch.tensor([1., -1., 1., -1.], dtype=torch.bfloat16).repeat(1, 1280)
    a, ma, ra = layer_norm_with_statistics(u, infer_dtype=torch.bfloat16, sensitive_layer_dtype=torch.bfloat16)
    b, mb, rb = layer_norm_with_statistics(4 * u, infer_dtype=torch.bfloat16, sensitive_layer_dtype=torch.bfloat16)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert float(ra[0, 0]) != float(rb[0, 0])
    assert not torch.cuda.is_initialized()
    result = {'id': 'C0_PREW1_NORM_STATISTICS_CPU_READINESS',
              'status': 'PASS_36_SYNTHETIC_CPU_BITWISE_CASES_AND_SCALE_COLLISION',
              'torch_version': str(torch.__version__), 'device': 'cpu', 'cases': cases,
              'case_count': len(cases), 'original_operator_calls': 36,
              'statistics_operator_calls': 38,
              'scale_collision': {'output_BF16_bitwise_identical': True,
                                  'inverse_std_scale1': float(ra[0, 0]), 'inverse_std_scale4': float(rb[0, 0])},
              'source_sha256': {str(p.relative_to(root)): digest(p) for p in
                  [Path(__file__), root / 'tools/prew1_norm_statistics.py']},
              'native_Wan_LNWeight_source_sha256': digest(root / '.runtime/code/LightX2V/lightx2v/common/ops/norm/layer_norm_weight.py'),
              'reference_source': 'https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/layer_norm.cpp',
              'new_model_calls': 0, 'new_FFN_calls': 0, 'new_training_updates': 0,
              'CUDA_initialized': False, 'actual_GPU_or_whole_model_bitwise_result': None,
              'actual_GPU_latency_result': None,
              'scope': 'Existing generation-environment Torch, random CPU normalization only. Proposed helper is not wired into the original model; CPU bitwise equality does not replace native GPU and full-trajectory no-op controls.'}
    (root / 'research/20261023/C0_norm_statistics_CPU_readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'case_count', 'scale_collision', 'new_model_calls', 'CUDA_initialized']}))


if __name__ == '__main__':
    main()
