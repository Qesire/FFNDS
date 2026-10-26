"""Decoded non-FFN retention plus explicit single-block dense FFN lifetime.

No model import or CUDA initialization until used by an authorized native driver.
The producer must fence outstanding consumers before ending a block.
"""


class PairedWeightResidency:
    def __init__(self, runtime, *, max_non_ffn_bytes, max_block_ffn_bytes, backend=None):
        self.runtime = runtime
        self.limits = {'non_ffn': max_non_ffn_bytes, 'ffn': max_block_ffn_bytes}
        self.backend = backend
        self.non_ffn = {}
        self.ffn = {}
        self.active = None
        self.stats = dict.fromkeys(['non_ffn_bytes', 'ffn_bytes', 'peak_non_ffn_bytes',
                                   'peak_ffn_bytes', 'non_ffn_transfers', 'ffn_transfers',
                                   'non_ffn_hits', 'ffn_hits', 'blocks_completed'], 0)

    def __enter__(self):
        if self.backend is None:
            import torch
            self.backend = torch
        self.old = self.runtime._weight
        self.runtime._weight = self.weight
        return self

    def begin_block(self, expert, block):
        if self.active is not None or self.ffn:
            raise RuntimeError('Previous FFN consumers must finish before the next block')
        self.active = (expert, int(block))

    def end_block(self, fence):
        if self.active is None or not callable(fence):
            raise RuntimeError('Active block and explicit native consumer fence required')
        fence()  # Never release a buffer still referenced by another stream.
        self.ffn.clear()
        self.stats['ffn_bytes'] = 0
        self.active = None
        self.stats['blocks_completed'] += 1

    def weight(self, *, expert, block_index, module_name, record, device, dtype):
        is_ffn = module_name.endswith(('.ffn.0', '.ffn.2'))
        kind = 'ffn' if is_ffn else 'non_ffn'
        if is_ffn and self.active != (expert, int(block_index)):
            raise RuntimeError('Dense FFN fetch outside the active block lifetime')
        cache = self.ffn if is_ffn else self.non_ffn
        key = (expert, module_name, str(device), str(dtype))
        if key in cache:
            self.stats[kind + '_hits'] += 1
            return cache[key]
        source = record.get('weight_qdq')
        if not self.backend.is_tensor(source):
            raise TypeError('Decoded native weight missing')
        size = source.numel() * self.backend.empty((), dtype=dtype, device='cpu').element_size()
        if self.stats[kind + '_bytes'] + size > self.limits[kind]:
            raise RuntimeError('Bounded decoded-weight budget exceeded: ' + kind)
        value = source.to(device=device, dtype=dtype, non_blocking=False).contiguous()
        cache[key] = value
        self.stats[kind + '_bytes'] += size
        self.stats['peak_' + kind + '_bytes'] = max(self.stats['peak_' + kind + '_bytes'], self.stats[kind + '_bytes'])
        self.stats[kind + '_transfers'] += 1
        self.runtime.counters['weight_transfers'] += 1
        return value

    def snapshot(self):
        return {**self.stats, 'active_block': self.active,
                'retained_non_ffn_tensors': len(self.non_ffn), 'retained_ffn_tensors': len(self.ffn)}

    def __exit__(self, *error):
        try:
            if self.backend.cuda.is_initialized():
                self.backend.cuda.synchronize()
        finally:
            self.runtime._weight = self.old
            self.ffn.clear()
            self.non_ffn.clear()
            self.active = None
            self.stats['ffn_bytes'] = self.stats['non_ffn_bytes'] = 0
