"""Author-layout NVFP4 decode and the finite real-operand FFN schedule.

Importing this file uses only the standard library. Decode is for diagnostic
references; the measured path always calls the actual packed SM120 kernels.
"""
import math

E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6.)
WARMUP, REPEATS = 2, 5
M_VALUES = (256, 31668)
D, H = 5120, 13824
COUNTS = {'dynamic_NVFP4_quant': 120, 'packed_NVFP4_GEMM': 120,
    'decoded_BF16_GEMM': 62, 'FP32_reference_GEMM': 8, 'tanh_GELU': 62,
    'CUDA_timed_samples': 120, 'repeat_real_rows_to_fullM': 2,
    'static_weight_decodes': 4, 'M256_activation_decodes': 4,
    'numerical_linear_cases': 4, 'numerical_FFN_cases': 2,
    'new_weight_quantizations': 0, 'training_updates': 0,
    'full_model_flows': 0, 'generated_videos': 0}


def need(value, message):
    if not value:
        raise ValueError(message)


def scale_offset(row, group, k):
    """cvt_quant_to_fp4_get_sf_out_offset, in bytes, not row-major SF."""
    need(type(row) is int and row >= 0 and type(group) is int and 0 <= group < k//16
        and k > 0 and k % 16 == 0, 'Scale logical coordinate differs')
    k_tiles = (k+63)//64
    return (row//128)*k_tiles*512+(group//4)*512+(row % 32)*16+((row % 128)//32)*4+group % 4


def unswizzle_scales(scales, rows, k):
    """Same reshape/permute as the author's test_bench1, cropped to groups."""
    import torch
    mp, gp = ((rows+127)//128)*128, ((k//16+3)//4)*4
    need(k % 16 == 0 and list(scales.shape) == [mp, gp]
        and scales.dtype == torch.float8_e4m3fn and scales.is_contiguous(), 'Padded swizzled E4M3 scale layout differs')
    value = scales.float().reshape(mp//128, gp//4, 32, 4, 4)
    result = value.permute(0, 3, 2, 1, 4).reshape(mp, gp)[:rows, :k//16]
    need(bool(torch.isfinite(result).all()) and bool((result >= 0).all()), 'Nonfinite or signed magnitude scale')
    return result


def decode_unscaled(packed, scales):
    """Decode low nibble first, returning E2M1 * local SF, without global SF."""
    import torch
    need(packed.dtype == torch.uint8 and packed.ndim == 2 and packed.is_contiguous()
        and scales.device == packed.device, 'Actual packed U8 matrix required')
    rows, half_k = packed.shape; k = 2*half_k
    sf = unswizzle_scales(scales, rows, k)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, k).long()
    table = torch.tensor((*E2M1, *[-x for x in E2M1]), dtype=torch.float32, device=packed.device)
    return (table[indices].reshape(rows, k//16, 16)*sf.unsqueeze(-1)).reshape(rows, k)


def check_linear(x, layer):
    import torch
    w, sf, alpha, global_scale, bias = [layer[k] for k in ('weight', 'weight_scale', 'alpha', 'input_global_scale', 'bias')]
    need(x.ndim == 2 and x.dtype == bias.dtype == torch.bfloat16 and x.is_contiguous()
        and w.dtype == torch.uint8 and w.ndim == 2 and x.shape[1] == 2*w.shape[1]
        and bias.shape == (w.shape[0],) and all(v.device == x.device for v in (w, sf, alpha, global_scale, bias))
        and alpha.dtype == global_scale.dtype == torch.float32 and alpha.numel() == global_scale.numel() == 1,
        'Native W4A4 operand shape/dtype/device differs')


def packed_linear(x, layer, api, capture=False):
    check_linear(x, layer)
    a, sa = api.quant(x, layer['input_global_scale'])
    result = api.gemm(a, layer['weight'], sa, layer['weight_scale'], layer['alpha'], layer['bias'])
    return (result, a, sa) if capture else result


def packed_ffn(x, layers, api, capture=False):
    first, a1, sa1 = packed_linear(x, layers[0], api, capture=True)
    hidden = api.gelu(first)
    final, a2, sa2 = packed_linear(hidden, layers[1], api, capture=True)
    return {'pre_GELU': first, 'h': hidden, 'Y': final, 'A1': a1, 'S1': sa1, 'A2': a2, 'S2': sa2} if capture else final


def decoded_ffn(x, layers, api, capture=False):
    first = api.dense(x, layers[0]['decoded_BF16'], layers[0]['bias'])
    hidden = api.gelu(first)
    final = api.dense(hidden, layers[1]['decoded_BF16'], layers[1]['bias'])
    return {'pre_GELU': first, 'Y': final} if capture else final


def error_metrics(actual, reference):
    """Float64 reductions only; references can themselves be float32/BF16."""
    import torch
    a, b = actual.double(), reference.double()
    need(a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite numerical probe output')
    error = a-b; denom = b.square().sum().clamp_min(1e-30)
    return {'relative_SSE': float(error.square().sum()/denom),
        'max_abs_error': float(error.abs().max()),
        'RMSE': float(error.square().mean().sqrt()),
        'reference_RMS': float(b.square().mean().sqrt()),
        'bitwise_equal': bool(torch.equal(actual, reference)) if actual.dtype == reference.dtype else False}


def timing_summary(samples, host_seconds):
    need(len(samples) == REPEATS and all(math.isfinite(x) and x > 0 for x in samples), 'Fixed five positive CUDA timing samples required')
    average = sum(samples)/len(samples)
    return {'CUDA_ms': samples, 'CUDA_ms_mean': average,
        'CUDA_ms_std_population': math.sqrt(sum((x-average)**2 for x in samples)/len(samples)),
        'host_timed_batch_seconds_including_synchronization_and_ledger': host_seconds,
        'warmup_calls': WARMUP, 'timed_calls': REPEATS}
