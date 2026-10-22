"""Fixed-layout auxiliary kernels for the three-direction synthetic prototype.

These kernels begin after original Q1. They do not replace any quantizer or
GEMM, and their BF16 boundaries are checked against an explicit reference.
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _prepare(X, SIGNS, H3, MU, C, SCORES, D: tl.constexpr, K: tl.constexpr,
             CHUNKS: tl.constexpr, DYNAMIC: tl.constexpr, B: tl.constexpr):
    g, chunk = tl.program_id(0), tl.program_id(1)
    d = chunk*B+tl.arange(0, B)
    mask = d < D
    a = tl.load(X+(4*g+0)*D+d, mask, 0).to(tl.float32)
    b = tl.load(X+(4*g+1)*D+d, mask, 0).to(tl.float32)
    c = tl.load(X+(4*g+2)*D+d, mask, 0).to(tl.float32)
    e = tl.load(X+(4*g+3)*D+d, mask, 0).to(tl.float32)
    mu = (((a+b)+(c+e))*.25).to(tl.bfloat16)
    tl.store(MU+g*D+d, mu, mask)
    tl.store(H3+(3*g+0)*D+d, ((a+b-c-e)*.5).to(tl.bfloat16), mask)
    tl.store(H3+(3*g+1)*D+d, ((a-b+c-e)*.5).to(tl.bfloat16), mask)
    tl.store(H3+(3*g+2)*D+d, ((a-b-c+e)*.5).to(tl.bfloat16), mask)
    if DYNAMIC or g < K:
        s0 = tl.load(SIGNS+4*g+0)
        s1 = tl.load(SIGNS+4*g+1)
        s2 = tl.load(SIGNS+4*g+2)
        s3 = tl.load(SIGNS+4*g+3)
        m = mu.to(tl.float32)
        q = ((((a-m)*s0+(b-m)*s1)+((c-m)*s2+(e-m)*s3))*.5).to(tl.bfloat16)
        tl.store(C+g*D+d, q, mask)
        if DYNAMIC:
            f = q.to(tl.float32)
            norm = tl.sum(tl.where(mask, f*f, 0.), 0)
            tl.store(SCORES+g*CHUNKS+chunk, norm)


@triton.jit
def _pack(MU, C, IDS, POINTS, D: tl.constexpr, K: tl.constexpr,
          N: tl.constexpr, DYNAMIC: tl.constexpr, B: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    d = chunk*B+tl.arange(0, B)
    if DYNAMIC:
        g = tl.load(IDS+row)
    else:
        g = tl.where(row < N, row+K, tl.where(row < N+K, row-N, row-N-K))
    mu = tl.load(MU+g*D+d, d < D, 0)
    if row < N:
        value = mu
    else:
        contrast = tl.load(C+g*D+d, d < D, 0).to(tl.float32)
        sign = tl.where(row < N+K, 1., -1.)
        value = (mu.to(tl.float32)+sign*contrast).to(tl.bfloat16)
    tl.store(POINTS+row*D+d, value, d < D)


@triton.jit
def _pool(V2, SLOT, POOLED, H: tl.constexpr, K: tl.constexpr,
          N: tl.constexpr, DYNAMIC: tl.constexpr, B: tl.constexpr):
    g, chunk = tl.program_id(0), tl.program_id(1)
    h = chunk*B+tl.arange(0, B)
    if DYNAMIC:
        slot = tl.load(SLOT+g)
    else:
        slot = tl.where(g < K, N+g, g-K)
    first = tl.load(V2+slot*H+h, h < H, 0)
    if slot >= N:
        second = tl.load(V2+(slot+K)*H+h, h < H, 0)
        value = ((first.to(tl.float32)+second.to(tl.float32))*.5).to(tl.bfloat16)
    else:
        value = first
    tl.store(POOLED+g*H+h, value, h < H)


@triton.jit
def _finish(AH, RESPONSE, X, GATE, OUT, D: tl.constexpr, B: tl.constexpr):
    g, chunk = tl.program_id(0), tl.program_id(1)
    d = chunk*B+tl.arange(0, B)
    mask = d < D
    a = tl.load(AH+(3*g+0)*D+d, mask, 0).to(tl.float32)
    b = tl.load(AH+(3*g+1)*D+d, mask, 0).to(tl.float32)
    c = tl.load(AH+(3*g+2)*D+d, mask, 0).to(tl.float32)
    response = tl.load(RESPONSE+g*D+d, mask, 0).to(tl.float32)
    gate = tl.load(GATE+d, mask, 0).to(tl.float32)
    for i in tl.static_range(4):
        if i == 0:
            contrast = ((a+b+c)*.5).to(tl.bfloat16)
        elif i == 1:
            contrast = ((a-b-c)*.5).to(tl.bfloat16)
        elif i == 2:
            contrast = ((-a+b-c)*.5).to(tl.bfloat16)
        else:
            contrast = ((-a-b+c)*.5).to(tl.bfloat16)
        y = (contrast.to(tl.float32)+response).to(tl.bfloat16)
        gated = (y.to(tl.float32)*gate).to(tl.bfloat16)
        x = tl.load(X+(4*g+i)*D+d, mask, 0).to(tl.float32)
        tl.store(OUT+(4*g+i)*D+d, (x+gated.to(tl.float32)).to(tl.bfloat16), mask)


def prepare(v1, signs, count, dynamic):
    m, d = v1.shape
    g = m//4
    h = torch.empty((3*g, d), dtype=v1.dtype, device=v1.device)
    mu = torch.empty((g, d), dtype=v1.dtype, device=v1.device)
    c = torch.empty_like(mu)
    scores = torch.empty((g, triton.cdiv(d, BLOCK)), dtype=torch.float32, device=v1.device)
    _prepare[(g, triton.cdiv(d, BLOCK))](v1, signs, h, mu, c, scores, d, count,
        triton.cdiv(d, BLOCK), dynamic, BLOCK, num_warps=4, enable_fp_fusion=False)
    return h, mu, c, scores.sum(1) if dynamic else None


def pack(mu, c, ids, count, dynamic):
    g, d = mu.shape
    points = torch.empty((g+count, d), dtype=mu.dtype, device=mu.device)
    _pack[(g+count, triton.cdiv(d, BLOCK))](mu, c, ids, points, d, count,
        g-count, dynamic, BLOCK, num_warps=4, enable_fp_fusion=False)
    return points


def pool(hidden, slot, groups, count, dynamic):
    h = hidden.shape[1]
    out = torch.empty((groups, h), dtype=hidden.dtype, device=hidden.device)
    _pool[(groups, triton.cdiv(h, BLOCK))](hidden, slot, out, h, count,
        groups-count, dynamic, BLOCK, num_warps=4, enable_fp_fusion=False)
    return out


def finish(projected, response, x, gate):
    g, d = response.shape
    out = torch.empty_like(x)
    _finish[(g, triton.cdiv(d, BLOCK))](projected, response, x, gate, out,
        d, BLOCK, num_warps=4, enable_fp_fusion=False)
    return out
