"""Process-lifetime verified matrices; request release does not evict them."""
import time
from pathlib import Path
import numpy as np
import torch


class ResidentMatrices:
    def __init__(self, entries, sha256_file, device):
        if set(entries) != {f'{e}/{l}' for e in ('high','low') for l in range(40)}:
            raise RuntimeError('matrix coverage')
        self.gpu = {}; self.expert = None; self.setup_seconds = 0.0; self.closed = False
        torch.cuda.synchronize(); start = time.monotonic()
        for expert in ('high','low'):
            for layer in range(40):
                spec = entries[f'{expert}/{layer}']; path = Path(spec['path']); before = path.stat()
                if sha256_file(path) != spec['sha256']: raise RuntimeError('matrix SHA mismatch')
                a = np.load(path, allow_pickle=False)
                after = path.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise RuntimeError('matrix changed while reading')
                if a.dtype != np.uint16 or list(a.shape) != [5120,5120] or spec['shape'] != [5120,5120]:
                    raise RuntimeError('matrix format mismatch')
                self.gpu[(expert,layer)] = torch.from_numpy(a.copy()).view(torch.bfloat16).to(device)
        torch.cuda.synchronize(); self.setup_seconds = time.monotonic() - start
        self.payload_bytes = sum(t.numel() * t.element_size() for t in self.gpu.values())
        if self.payload_bytes != 80*5120*5120*2: raise RuntimeError('matrix payload mismatch')

    def activate(self, expert):
        if self.closed or expert not in ('high','low'): raise RuntimeError('invalid resident expert')
        self.expert = expert

    def get(self, expert, layer, device):
        if self.closed or self.expert != expert: raise RuntimeError('invalid matrix access')
        value = self.gpu[(expert,layer)]
        if value.device != device: raise RuntimeError('matrix device mismatch')
        return value

    def release(self):
        # Called after each request and expert switch by the unchanged E0B core.
        self.expert = None

    def close(self):
        self.gpu.clear(); self.expert = None; self.closed = True
