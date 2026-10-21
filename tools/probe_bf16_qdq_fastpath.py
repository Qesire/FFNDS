"""Qualify the existing HiF4 BF16 CUDA entrypoint with matched dense timing.

The original model runtime and quantization cache are never modified. Both
timed dense and sparse synthetic chains receive the same proposed fast path.
"""
import json
from pathlib import Path
import sys

import torch

from ar2_checkpoint import atomic_json, digest
from probe_weight_2of4_shape import main as shape_main, ROOT


if __name__ == '__main__':
    from quant import hif4_backend
    from torch.sparse import SparseSemiStructuredTensor
    original = hif4_backend.hif4_qdq_tensor
    controls = []
    path = Path(sys.argv[sys.argv.index('--report') + 1])
    failed = path.with_suffix('.failed_controls.json')
    if failed.exists():
        raise RuntimeError('Preserve the failed BF16 path attempt')

    def fast(x, *, dim=-1, force_fp32=True, kind='activation'):
        if x.dtype != torch.bfloat16 or not force_fp32 or dim != -1:
            raise RuntimeError('Only the pre-rounded native BF16 boundary is qualified')
        result = original(x, dim=dim, force_fp32=False, kind=kind)
        # Frozen probe has2 weight calls,2 masked-chain calls,2 sparse-chain
        # calls, then8 calls in the first four-arm warmup. All checks finish
        # before timing; the remaining two warmups use the proposed path alone.
        if len(controls) < 14:
            reference = original(x, dim=dim, force_fp32=True, kind=kind)
            mismatches = int(torch.count_nonzero(result.view(torch.int16) != reference.view(torch.int16)))
            finite = bool(torch.isfinite(reference).all() and torch.isfinite(result).all())
            controls.append({'shape': list(x.shape), 'elements': x.numel(), 'bit_mismatches': mismatches,
                             'finite': finite, 'same_dtype': result.dtype == reference.dtype})
            if mismatches or not finite or result.dtype != reference.dtype:
                raise RuntimeError('Proposed BF16 path differs from original force-FP32 quantization')
        return result

    hif4_backend.hif4_qdq_tensor = fast
    SparseSemiStructuredTensor._FUSE_TRANSPOSE = True
    try:
        shape_main(ROOT / 'config/bf16_qdq_fastpath_preflight.json')
    except BaseException as error:
        atomic_json(failed, {'status': 'FAILED_BF16_PATH_PRESERVED', 'controls': controls,
                            'error': str(error), 'source_sha256': digest(Path(__file__))})
        raise
    if len(controls) != 14:
        raise RuntimeError('Incomplete frozen BF16 equivalence controls')
    report = json.loads(path.read_text())
    report.update(bf16_path_controls=controls, bf16_controls_complete_before_timing=True,
                  same_fastpath_used_for_dense_and_sparse=True,
                  original_quantizer_source_sha256=digest(ROOT / '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QTensor.py'),
                  original_CUDA_inner_source_sha256=digest(ROOT / '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu'),
                  numerical_scope='Exact observed synthetic inputs, weights and GELU outputs only; no native model path changed or real source validation.')
    atomic_json(path, report)
