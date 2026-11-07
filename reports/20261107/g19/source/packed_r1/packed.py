"""Prepack native pinned CPU W1 and resident A before generation; no model mutation."""
import time
import torch

class PackedMatrices:
    def __init__(self, base, models, device):
        if len(models) != 2: raise RuntimeError('expected two experts')
        self.base, self.gpu = base, {}
        self.closed = False
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        for expert, model in zip(('high','low'), models):
            blocks = model.transformer_weights.blocks
            if len(blocks) != 40: raise RuntimeError('block coverage')
            for layer, block in enumerate(blocks):
                state = block.state_dict()
                w = state[f'blocks.{layer}.ffn.0.weight']
                b = state[f'blocks.{layer}.ffn.0.bias']
                a = base.gpu[(expert,layer)]
                if (w.device.type != 'cpu' or b.device.type != 'cpu' or w.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
                    or tuple(w.shape) != (5120,13824) or tuple(b.shape) != (13824,) or tuple(a.shape) != (5120,5120)):
                    raise RuntimeError('unexpected native CPU W1/b1 or resident A')
                wide = torch.empty((5120,18944), device=device, dtype=torch.bfloat16)
                bias = torch.empty((18944,), device=device, dtype=torch.bfloat16)
                wide[:,:13824].copy_(w); wide[:,13824:].copy_(a)
                bias[:13824].copy_(b); bias[13824:].zero_()
                self.gpu[(expert,layer)] = (wide, bias, 13824)
        torch.cuda.synchronize(device)
        self.setup_seconds = time.monotonic()-start
        self.payload_bytes = sum((w.numel()+b.numel())*w.element_size() for w,b,_ in self.gpu.values())
        self.setup_allocated_delta = torch.cuda.memory_allocated(device)-before
        self.setup_peak_allocated = torch.cuda.max_memory_allocated(device)

    def activate(self, expert): self.base.activate(expert)
    def get(self, expert, layer, device): return self.base.get(expert,layer,device)
    def get_packed(self, expert, layer, device):
        if self.closed or self.base.expert != expert: raise RuntimeError('invalid packed expert')
        w,b,h = self.gpu[(expert,layer)]
        if w.device != device: raise RuntimeError('packed device mismatch')
        return w,b,h
    def release(self): self.base.release()
    def close(self): self.gpu.clear(); self.base.close(); self.closed=True
