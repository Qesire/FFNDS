"""Stable BF16 top2 and the pinned Torch CUTLASS storage in one Triton pass.

Finite BF16 input only. The selected mathematical matrix is unchanged. Zero
placeholders follow Torch's nonzero-based canonical encoding, not the original
top2 support. Importing this module does not import Torch/Triton or initialize CUDA.
"""
from functools import lru_cache

CONVERTER_SHA256 = "1759d16ca5dbdcf0a5cef676eafbd4f1477af5e961b18a8fc8dd83a1c1c6728e"
BLOCK_GROUPS = 256
NUM_WARPS = 4


def padded_rows(m):
    if type(m) is not int or m < 1:
        raise ValueError("Positive native row count required")
    return (m + 31) // 32 * 32


def canonical_indices(flags):
    """The pinned converter's Boolean map; all sixteen masks are defined."""
    if len(flags) != 4:
        raise ValueError("Four flags required")
    m0, m1, _, m3 = map(bool, flags)
    e0, e1, e2 = m0 and m1, not m0 and m1, not m0 and not m1
    first = int(e1) | (int(e2) << 1)
    second = int(e0 or e2 or m3) | (int(e1 or not m1) << 1)
    return first, second


def select_pack_group_bits(bits):
    """Scalar proof oracle over four finite IEEE BF16 uint16 encodings."""
    if (len(bits) != 4 or any(type(x) is not int or not 0 <= x < 65536 or
                            x & 0x7f80 == 0x7f80 for x in bits)):
        raise ValueError("Four finite BF16 bit patterns required")
    absolute = [x & 0x7fff for x in bits]
    rank = [sum(absolute[j] > absolute[i] or (absolute[j] == absolute[i] and j < i)
                for j in range(4)) for i in range(4)]
    masked = [x if rank[i] < 2 else 0 for i, x in enumerate(bits)]
    first, second = canonical_indices([bool(x & 0x7fff) for x in masked])
    return (masked[first], masked[second]), first | (second << 2)


def metadata_offset(row, col, rows, meta_cols):
    if (type(rows) is not int or rows < 32 or rows % 32 or
            type(meta_cols) is not int or meta_cols < 2 or meta_cols % 2 or
            not 0 <= row < rows or not 0 <= col < meta_cols):
        raise ValueError("Even-column int16 CUTLASS metadata geometry required")
    shuffled = row // 32 * 32 + (row % 8) * 4 + (row % 32) // 8
    swap = int(shuffled % 2 == 0 and col % 2 == 1) - int(shuffled % 2 == 1 and col % 2 == 0)
    shuffled, column = shuffled + swap, col - swap
    return column // 2 * rows * 2 + shuffled * 2 + column % 2


def cpu_pack_proof(value):
    """Small CPU-only executable algebra; never a fallback for the GPU probe."""
    import torch
    if (value.device.type != "cpu" or value.dtype != torch.bfloat16 or value.ndim != 2 or
            not value.is_contiguous() or value.shape[1] % 64 or value.numel() > 131072 or
            not bool(torch.isfinite(value).all())):
        raise ValueError("Small contiguous finite CPU BF16 matrix with K%64=0 required")
    m, k = value.shape
    mp = padded_rows(m)
    raw = value.view(torch.uint16).tolist()
    packed, meta = [], [None] * (mp * (k // 16))
    for row in range(mp):
        codes = []
        for group in range(k // 4):
            bits = raw[row][group * 4:group * 4 + 4] if row < m else [0] * 4
            values, code = select_pack_group_bits(bits)
            packed.extend(values)
            codes.append(code)
        for col in range(k // 16):
            word = sum(codes[col * 4 + j] << (4 * j) for j in range(4))
            offset = metadata_offset(row, col, mp, k // 16)
            if meta[offset] is not None:
                raise AssertionError("Metadata scatter collision")
            meta[offset] = word if word < 32768 else word - 65536
    if any(x is None for x in meta):
        raise AssertionError("Metadata scatter hole")
    return (torch.tensor(packed, dtype=torch.uint16).view(torch.bfloat16).reshape(mp, k // 2),
            torch.tensor(meta, dtype=torch.int16).reshape(mp, k // 16))


def edge_input(rows=35, columns=64):
    """Fixed CPU BF16 zero/signed-zero/tie/subnormal/finite-extreme cases."""
    import torch
    if (rows, columns) not in ((35, 64), (33, 128)):
        raise ValueError("Only the two frozen edge geometries are defined")
    groups = [[(0x3f80 if j % 2 == 0 else 0xbf80) if mask & (1 << j) else 0
               for j in range(4)] for mask in range(16)]
    groups += [[0x8000, 0, 0x8000, 0], [0, 0x8000, 0, 0x8000],
        [0x3f80, 0x8000, 0, 0], [0x8000, 0x3f80, 0, 0],
        [0x8000, 0, 0x3f80, 0], [0x8000, 0, 0, 0x3f80],
        [0x3f80, 0xbf80, 0x3f80, 0xbf80], [0xc000, 0x4000, 0xc000, 0x4000],
        [0x0001, 0x8001, 0x0002, 0x8002], [0, 0x8000, 0x0001, 0x8001],
        [0x7f7f, 0xff7f, 0x7f00, 0xff00]]
    raw = [groups[(r * (columns // 4) + g) % len(groups)]
           for r in range(rows) for g in range(columns // 4)]
    return torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16).reshape(rows, columns)


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def select_pack(src, dst, meta, M: tl.constexpr, K: tl.constexpr,
                    MP: tl.constexpr, MASK_ONLY: tl.constexpr, B: tl.constexpr):
        g = tl.program_id(0) * B + tl.arange(0, B)
        row, col = g // (K // 4), (g % (K // 4)) * 4
        valid = g < MP * (K // 4)
        real = valid & (row < M)
        x0 = tl.load(src + row * K + col, mask=real, other=0).to(tl.uint16, bitcast=True)
        x1 = tl.load(src + row * K + col + 1, mask=real, other=0).to(tl.uint16, bitcast=True)
        x2 = tl.load(src + row * K + col + 2, mask=real, other=0).to(tl.uint16, bitcast=True)
        x3 = tl.load(src + row * K + col + 3, mask=real, other=0).to(tl.uint16, bitcast=True)
        a0, a1, a2, a3 = x0 & 0x7fff, x1 & 0x7fff, x2 & 0x7fff, x3 & 0x7fff
        r0 = (a1 > a0).to(tl.int32) + (a2 > a0).to(tl.int32) + (a3 > a0).to(tl.int32)
        r1 = (a0 >= a1).to(tl.int32) + (a2 > a1).to(tl.int32) + (a3 > a1).to(tl.int32)
        r2 = (a0 >= a2).to(tl.int32) + (a1 >= a2).to(tl.int32) + (a3 > a2).to(tl.int32)
        r3 = (a0 >= a3).to(tl.int32) + (a1 >= a3).to(tl.int32) + (a2 >= a3).to(tl.int32)
        y0 = tl.where(real & (r0 < 2), x0, 0).to(tl.uint16)
        y1 = tl.where(real & (r1 < 2), x1, 0).to(tl.uint16)
        y2 = tl.where(real & (r2 < 2), x2, 0).to(tl.uint16)
        y3 = tl.where(real & (r3 < 2), x3, 0).to(tl.uint16)
        if MASK_ONLY:
            tl.store(dst + row * K + col, y0.to(tl.bfloat16, bitcast=True), mask=valid)
            tl.store(dst + row * K + col + 1, y1.to(tl.bfloat16, bitcast=True), mask=valid)
            tl.store(dst + row * K + col + 2, y2.to(tl.bfloat16, bitcast=True), mask=valid)
            tl.store(dst + row * K + col + 3, y3.to(tl.bfloat16, bitcast=True), mask=valid)
        else:
            m0, m1, m3 = (y0 & 0x7fff) != 0, (y1 & 0x7fff) != 0, (y3 & 0x7fff) != 0
            e0, e1, e2 = m0 & m1, ~m0 & m1, ~m0 & ~m1
            i0 = e1.to(tl.int32) | (e2.to(tl.int32) << 1)
            i1 = (e0 | e2 | m3).to(tl.int32) | ((e1 | ~m1).to(tl.int32) << 1)
            p0 = tl.where(i0 == 0, y0, tl.where(i0 == 1, y1, tl.where(i0 == 2, y2, y3)))
            p1 = tl.where(i1 == 0, y0, tl.where(i1 == 1, y1, tl.where(i1 == 2, y2, y3)))
            tl.store(dst + 2 * g, p0.to(tl.uint16).to(tl.bfloat16, bitcast=True), mask=valid)
            tl.store(dst + 2 * g + 1, p1.to(tl.uint16).to(tl.bfloat16, bitcast=True), mask=valid)
            codes = tl.reshape(i0 | (i1 << 2), (B // 4, 4))
            word = tl.sum(codes << (4 * tl.arange(0, 4)[None, :]), axis=1)
            w = tl.program_id(0) * (B // 4) + tl.arange(0, B // 4)
            rr, cc = w // (K // 16), w % (K // 16)
            rr = rr // 32 * 32 + (rr % 8) * 4 + (rr % 32) // 8
            swap = ((rr % 2 == 0) & (cc % 2 == 1)).to(tl.int32) - ((rr % 2 == 1) & (cc % 2 == 0)).to(tl.int32)
            rr, cc = rr + swap, cc - swap
            target = (cc // 2) * MP * 2 + rr * 2 + cc % 2
            tl.store(meta + target, word.to(tl.int16), mask=w < MP * (K // 16))
    return select_pack


def _input_shape(value):
    import torch
    if (value.device.type != "cuda" or value.dtype != torch.bfloat16 or value.ndim != 2 or
            not value.is_contiguous() or value.shape[1] % 64 or value.requires_grad):
        raise ValueError("Contiguous inference CUDA BF16 matrix with K%64=0 required")
    m, k = value.shape
    return m, k, padded_rows(m)


def fused_pack(value):
    """Allocations and one fused launch, with no dense masked intermediate."""
    import torch
    import triton
    m, k, mp = _input_shape(value)
    packed = torch.empty((mp, k // 2), device=value.device, dtype=torch.bfloat16)
    meta = torch.empty((mp, k // 16), device=value.device, dtype=torch.int16)
    _kernel()[(triton.cdiv(mp * (k // 4), BLOCK_GROUPS),)](
        value, packed, meta, m, k, mp, False, BLOCK_GROUPS, num_warps=NUM_WARPS)
    return packed, meta


def materialize_masked(value):
    """Same fixed selector and launch configuration; only its output differs."""
    import torch
    import triton
    m, k, mp = _input_shape(value)
    masked = torch.empty((mp, k), device=value.device, dtype=torch.bfloat16)
    _kernel()[(triton.cdiv(mp * (k // 4), BLOCK_GROUPS),)](
        value, masked, value, m, k, mp, True, BLOCK_GROUPS, num_warps=NUM_WARPS)
    return masked


def wrap_cutlass(packed, meta):
    import torch
    from torch.sparse.semi_structured import SparseSemiStructuredTensorCUTLASS
    mp, half_k = packed.shape
    if (packed.device.type != "cuda" or packed.dtype != torch.bfloat16 or meta.dtype != torch.int16 or
            packed.device != meta.device or not packed.is_contiguous() or not meta.is_contiguous() or
            mp % 32 or half_k % 32 or tuple(meta.shape) != (mp, half_k // 8)):
        raise ValueError("Pinned CUTLASS BF16 packed/meta layout required")
    return SparseSemiStructuredTensorCUTLASS(torch.Size((mp, 2 * half_k)),
        packed=packed, meta=meta, packed_t=None, meta_t=None,
        compressed_swizzled_bitmask=None, requires_grad=False)
