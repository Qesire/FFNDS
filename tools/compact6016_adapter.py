"""Ordinary narrow BF16 FFN with offline FP32 rank adapters and exact Q2 STE.

This module never imports the native CUDA quantizer. The registered runner
supplies that callable after asset verification and GPU reservation. No weight
requantization, teacher activation, residual bank or online LoRA is used by
``compact_forward``.
"""

import math

import torch
import torch.nn.functional as F


class _ExactNativeQ2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, quantizer):
        # Returning the native result directly avoids subtract/add STE rounding.
        return quantizer(value, force_fp32=True)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def native_q2_ste(value, quantizer):
    """Native forward bytes, identity gradient at the Q2 input boundary."""
    return _ExactNativeQ2.apply(value, quantizer)


def same_bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and bool(torch.equal(left.detach().contiguous().view(torch.uint8),
                                 right.detach().contiguous().view(torch.uint8))))


def compact_forward(x, w1, b1, w2, b2, smooth2, quantizer, *,
                    training=False, return_trace=False):
    """Student's own W1/GELU/Smooth2/native Q2/W2, with fixed native biases.

    ``x`` is the actual saved post-Q1 BF16 operand, so Q1 is not repeated.
    Smooth2 is the original selected FP32 subvector. Both deployed matrices
    are BF16: the FP32 LoRA merge happens outside this function.
    """
    hidden = F.gelu(F.linear(x, w1, b1), approximate="tanh")
    pre_q2 = (hidden.float() / smooth2).to(torch.bfloat16)
    q2 = (native_q2_ste(pre_q2, quantizer) if training
          else quantizer(pre_q2, force_fp32=True)).to(torch.bfloat16)
    output = F.linear(q2, w2, b2)
    if return_trace:
        return output, {"post_GELU": hidden, "pre_Q2": pre_q2, "post_Q2": q2}
    return output


class CompactAdapters(torch.nn.Module):
    """Fixed BF16 bases; two FP32 rank32 updates merged before each forward.

    Initialization uses local generators, with the same factor shapes and
    seed for the two matched support initializations. There is no LoRA alpha
    multiplier, optimizer tuning, bias update, or weight HiF4 QDQ.
    """
    def __init__(self, w1, w2, *, rank=32, seed=42, on_random=None):
        super().__init__()
        if (w1.ndim != 2 or w2.ndim != 2 or w1.shape[::-1] != w2.shape
                or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
                or w1.device != w2.device or rank <= 0):
            raise ValueError("Fixed BF16 two-projection geometry required")
        self.register_buffer("base_W1", w1.detach().contiguous())
        self.register_buffer("base_W2", w2.detach().contiguous())
        generator = torch.Generator(device=w1.device).manual_seed(seed)
        for slot, weight in ((1, w1), (2, w2)):
            if on_random is not None:
                on_random()
            a = torch.randn((rank, weight.shape[1]), device=weight.device,
                            dtype=torch.float32, generator=generator)
            a /= math.sqrt(weight.shape[1])
            b = torch.zeros((weight.shape[0], rank), device=weight.device,
                            dtype=torch.float32)
            self.register_parameter(f"lora_A{slot}", torch.nn.Parameter(a))
            self.register_parameter(f"lora_B{slot}", torch.nn.Parameter(b))

    def merged_weights(self):
        # IEEE base + zero may canonicalize negative zero. The runner records
        # that fact and requires the zero-adapter FFN output to match the
        # original selected function byte-for-byte before any update.
        return tuple((getattr(self, f"base_W{slot}").float()
                      + getattr(self, f"lora_B{slot}")
                      @ getattr(self, f"lora_A{slot}")).to(torch.bfloat16)
                     for slot in (1, 2))


def gated_loss(prediction, target, gate):
    """Source-normalized FP32 gated pre-gate error used for optimization."""
    denominator = (target.float() * gate.float()).square().mean()
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("Nonfinite or zero source target energy")
    loss = ((prediction.float() - target.float()) * gate.float()).square().mean()
    return loss / denominator


@torch.no_grad()
def error_metrics(prediction, target, gate):
    """Separate FP32 pre-gate error and actual BF16 gate multiplication."""
    if prediction.dtype != torch.bfloat16 or target.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("Native BF16 output/target/gate required")
    pre_error = (prediction.float() - target.float()) * gate.float()
    pre_target = target.float() * gate.float()
    actual_prediction = (prediction * gate).float()
    actual_target = (target * gate).float()
    pre_sse = float(pre_error.double().square().sum())
    pre_energy = float(pre_target.double().square().sum())
    actual_sse = float((actual_prediction - actual_target).double().square().sum())
    actual_energy = float(actual_target.double().square().sum())
    if not all(math.isfinite(v) for v in (pre_sse, pre_energy, actual_sse, actual_energy)) or min(pre_energy, actual_energy) <= 0:
        raise ValueError("Invalid source metric energy")
    # These are statistics over the 256 saved row coordinates, not video time.
    error64, target64 = pre_error.double(), pre_target.double()
    mean_error = error64.mean(0, keepdim=True)
    mean_sse = float(len(error64) * mean_error.square().sum())
    centered_sse = float((error64 - mean_error).square().sum())
    centered_teacher_energy = float((target64 - target64.mean(0, keepdim=True)).square().sum())
    closure = abs(pre_sse - mean_sse - centered_sse)
    if closure > 1e-10 * max(pre_sse, 1.):
        raise ValueError("Source-row mean/centered SSE identity failed")
    return {"gated_pre_gate_relative_SSE": pre_sse / pre_energy,
            "actual_BF16_gate_relative_SSE": actual_sse / actual_energy,
            "gated_pre_gate_SSE": pre_sse, "gated_pre_gate_target_energy": pre_energy,
            "actual_BF16_gate_SSE": actual_sse, "actual_BF16_gate_target_energy": actual_energy,
            "source_row_mean_error_SSE": mean_sse,
            "source_row_centered_error_SSE": centered_sse,
            "source_row_centered_teacher_energy": centered_teacher_energy,
            "source_row_centered_relative_SSE": (centered_sse / centered_teacher_energy
                                                 if centered_teacher_energy > 0 else None),
            "source_row_centered_status": ("MEASURED" if centered_teacher_energy > 0
                                           else "NOT_MEASURED_ZERO_CENTERED_TEACHER_ENERGY"),
            "source_row_mean_error_fraction": mean_sse / pre_sse if pre_sse > 0 else None,
            "source_row_decomposition_closure_abs": closure}
