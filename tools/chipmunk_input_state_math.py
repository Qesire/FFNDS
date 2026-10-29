"""Draft single-transition P64 cache variants; no authorization or model execution."""
import torch
import torch.nn.functional as F

def encode(old_h, rows, kind):
    if old_h.ndim != 2 or old_h.dtype != torch.bfloat16:
        raise ValueError('BF16 matrix required')
    shadow = old_h.index_select(0, rows).clone()
    if kind == 'fp8':
        return old_h.clamp(-448, 448).to(torch.float8_e4m3fn), shadow, None
    if kind == 'int8':
        maximum = old_h.float().abs().amax(dim=0)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny))
        return (old_h.float() / scale).round().clamp(-127, 127).to(torch.int8), shadow, scale
    raise ValueError('unknown encoding')

def readonly_action(x, w1, b1, w2, old_y, mask, old_selected, *, mm_fp32):
    """Same arithmetic as packed baseline, except dead h clone/update omitted."""
    yc = old_y.clone()
    a = w1.index_select(0, mask); b = b1.index_select(0, mask)
    c = w2.index_select(1, mask).contiguous()
    new = F.gelu(F.linear(x, a, b), approximate='tanh')
    delta = (new.float() - old_selected.float()).to(torch.bfloat16)
    return (yc.float() + mm_fp32(delta, c.transpose(0, 1))).to(torch.bfloat16)

def selected(cache, mask, scale=None):
    v = cache.index_select(1, mask).float()
    if scale is not None:
        v = v * scale.index_select(0, mask)
    return v.to(torch.bfloat16)
