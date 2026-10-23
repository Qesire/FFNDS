"""Expose statistics from one native LayerNorm call; no model hook or launcher.

The dtype branches match Wan's torch LNWeight.apply. A future registered
observer may use the returned mean/rstd without separately reducing x.
GPU identity, actual latency and complete-model no-op checks remain required.
"""


def layer_norm_with_statistics(x, *, infer_dtype, sensitive_layer_dtype,
                               weight=None, bias=None, eps=1e-6):
    import torch
    value = x.float() if sensitive_layer_dtype != infer_dtype else x
    y, mean, inverse_std = torch.native_layer_norm(value, (value.shape[-1],), weight, bias, eps)
    if sensitive_layer_dtype != infer_dtype:
        y = y.to(infer_dtype)
    # No detach/copy of y; model writeback remains the caller's operation.
    # Keeping these returned tensors alive has a resource cost even though
    # no extra variance reduction is requested by this helper.
    return y, mean.detach(), inverse_std.detach()
