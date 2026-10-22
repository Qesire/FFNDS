"""Input-only gated residual magnitude; no dense W1/teacher evaluation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _energy(R, GATE, PART, D: tl.constexpr, CHUNKS: tl.constexpr, B: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    d = chunk * B + tl.arange(0, B)
    value = tl.load(R + row * D + d, d < D, 0).to(tl.float32)
    gate = tl.load(GATE + d, d < D, 0).to(tl.float32)
    product = value * gate
    tl.store(PART + row * CHUNKS + chunk, tl.sum(product * product, 0))


def gated_energy(residual, gate):
    rows, dim = residual.shape
    if not residual.is_contiguous() or gate.numel() != dim:
        raise ValueError('Contiguous residual and one native gate vector required')
    chunks = triton.cdiv(dim, 1024)
    partial = torch.empty((rows, chunks), device=residual.device, dtype=torch.float32)
    _energy[(rows, chunks)](residual, gate, partial, dim, chunks, 1024,
                           num_warps=4, enable_fp_fusion=False)
    return partial.sum(1)
