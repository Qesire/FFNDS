"""Bit-preserving group-128 INT8 cache writer for the four-step FFNDS port."""
from functools import lru_cache

import torch


SOURCE = r"""
struct __align__(16) IndexPair { long long first, second; };

// A lane consumes four adjacent cache channels. Channel selection (z0k) is
// performed on the original W1 slab; no full-width gathered tensor is needed.
__device__ __forceinline__ unsigned long long read_four(
    const B* src, const long long* columns, int row, int quad) {
#if SELECTED
    const IndexPair* pairs = (const IndexPair*)(columns + quad * 4);
    IndexPair ab = pairs[0], cd = pairs[1];
    const B* row_src = src + row * INPUT_WIDTH;
    unsigned long long bits = row_src[(int)ab.first];
    bits |= (unsigned long long)row_src[(int)ab.second] << 16;
    bits |= (unsigned long long)row_src[(int)cd.first] << 32;
    return bits | ((unsigned long long)row_src[(int)cd.second] << 48);
#else
    return ((const unsigned long long*)src)[row * (WIDTH / 4) + quad];
#endif
}

__device__ __forceinline__ unsigned int encode_value(float x, float reciprocal) {
    // The true magnitude is <= 129. For any BF16 ratio near an integer
    // rounding boundary, the value is m/(2*n) or an integer multiple of
    // 1/n, n <= 255. Thus its distance to any half integer, unless exactly
    // equal to it, is at least 1/510. With exponent difference <= -2 it
    // is <= 255/512. RN reciprocal followed by RN multiplication has error
    // < 0.000016 here. Rounding first to multiples of 2^-12 corrects a
    // perturbed exact tie and cannot change any other round-to-even result.
    float approx = __fmul_rn(x, reciprocal);
    float on_grid = __fsub_rn(__fadd_rn(approx, 3072.0f), 3072.0f);
    int q = __float2int_rn(on_grid);
    return (unsigned int)(q < -127 ? -127 : (q > 127 ? 127 : q)) & 255u;
}

extern "C" __global__ void write_cache8(
    const B* src, const long long* columns, const long long* probes,
    unsigned int* codes, B* scales, unsigned long long* exact,
    int groups, int probe_groups) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= groups) {
        int group = warp - groups;
        if (group >= probe_groups) return;
        int probe_idx = group / (WIDTH / 128);
        int quad = (group % (WIDTH / 128)) * 32 + lane;
        exact[group * 32 + lane] = read_four(src, columns, (int)probes[probe_idx], quad);
        return;
    }

    int row = warp / (WIDTH / 128);
    int quad = (warp % (WIDTH / 128)) * 32 + lane;
    unsigned long long bits = read_four(src, columns, row, quad);
    float values[4], maximum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = f32((B)(bits >> (16 * i)));
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    float reciprocal = 0.0f;
    if (lane == 0) {
        B scale = bf16(maximum == 0.0f ? 1.0f :
            fmaxf(__fdiv_rn(maximum, 127.0f), 0x1p-126f));
        scales[warp] = scale;
        reciprocal = __frcp_rn(f32(scale));
    }
    reciprocal = __shfl_sync(0xffffffffu, reciprocal, 0);
    unsigned int packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= encode_value(values[i], reciprocal) << (8 * i);
    }
    codes[warp * 32 + lane] = packed;
}
"""


@lru_cache(None)
def _kernel(device, width, input_width, selected):
    from fused_ops import HEADER

    constants = f"#define WIDTH {width}\n#define INPUT_WIDTH {input_width}\n#define SELECTED {int(selected)}\n"
    with torch.cuda.device(device):
        return torch.cuda._compile_kernel(constants + HEADER + SOURCE, "write_cache8")


def write(field, layer, value, selector=None):
    """Encode one layer slot. Unsupported layouts retain the registered writer."""
    width = field.width
    if (field.bits != 8 or value.ndim != 2 or not value.is_cuda
            or value.dtype != torch.bfloat16 or value.device != field.code.device
            or value.shape[0] != field.code.shape[1] or not value.is_contiguous()
            or width % 128 or not field.probes.is_contiguous()):
        return type(field).write(field, layer, value, selector)
    if selector is None:
        if value.shape[1] != width or value.data_ptr() % 8:
            return type(field).write(field, layer, value, selector)
    elif (selector.shape != (width,) or selector.dtype != torch.long
          or selector.device != value.device or not selector.is_contiguous()
          or selector.data_ptr() % 16):
        return type(field).write(field, layer, value, selector)
    scale = field.scale[layer]
    groups = scale.numel()
    probe_groups = field.probes.numel() * (width // 128)
    if groups + probe_groups:
        kernel = _kernel(value.device.index, width, value.shape[1], selector is not None)
        kernel(grid=((groups + probe_groups + 7) // 8, 1, 1), block=(256, 1, 1),
               args=[value, field.probes if selector is None else selector, field.probes,
                     field.code[layer], scale, field.exact[layer], groups, probe_groups])
    field.stores += 1
    return True
