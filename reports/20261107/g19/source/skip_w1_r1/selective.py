"""Four-step HHLL selective block prefetch for LightX2V's native manager.

Install separately for each generation. Requires eager pinned source blocks,
two GPU buffers, unfused BF16 and the native synchronize-on-swap manager.
No model execution or scheduling occurs at import time.
"""

LAYERS = 40
EXPERTS = ('high', 'high', 'low', 'low')


def ffn_keys(layer):
    if type(layer) is not int or not 0 <= layer < LAYERS:
        raise ValueError('layer')
    return {f'blocks.{layer}.ffn.{which}.{kind}' for which in (0, 2) for kind in ('weight', 'bias')}


class FilteredBlock:
    def __init__(self, block, layer, omit, on_skip=None): self.block, self.layer, self.omit, self.on_skip = block, layer, omit, on_skip
    def state_dict(self):
        state = self.block.state_dict()
        all_keys = ffn_keys(self.layer)
        keys = {key for key in all_keys if int(key.split(".")[-2]) in self.omit}
        if not all_keys.issubset(state) or any(state[key] is None for key in all_keys):
            raise RuntimeError('missing source FFN payload')
        if self.on_skip: self.on_skip(sum(state[key].numel() * state[key].element_size() for key in keys))
        return {key: value for key, value in state.items() if key not in keys}


class TargetBlocks:
    def __init__(self, blocks, target, omit, on_skip): self.blocks, self.target, self.omit, self.on_skip = blocks, target, omit, on_skip
    def __getitem__(self, index):
        if index != self.target: raise RuntimeError('unexpected block request')
        return FilteredBlock(self.blocks[index], index, self.omit, self.on_skip)


class SelectivePrefetch:
    def __init__(self, *, skip_hits=True, skip_anchor_w1=False):
        self.skip_hits = skip_hits
        self.skip_anchor_w1 = skip_anchor_w1
        self.dispatch = None
        self.bindings = []
        self.tags = {}
        self.counts = dict(init_full=0, init_w2_only=0, prefetch_full=0, prefetch_w2_only=0, prefetch_skip=0, terminal=0)
        self.trace = []
        self.skipped_payload_bytes = 0

    def install(self, models):
        if self.bindings or set(models) != {'high','low'}: raise RuntimeError('invalid install')
        if len({id(m.offload_manager) for m in models.values()}) != 2: raise RuntimeError('manager alias')
        try:
            for expert, model in models.items(): self._bind(expert, model)
        except BaseException:
            self.uninstall(); raise

    def _bind(self, expert, model):
        mgr = model.offload_manager
        if model.lazy_load or mgr.offload_granularity != 'block' or hasattr(mgr, 'cpu_buffers') or len(mgr.cuda_buffers) != 2:
            raise RuntimeError('requires eager two-buffer block offload')
        original_init, original_prefetch = mgr.init_first_buffer, mgr.prefetch_weights
        original_need = mgr.need_init_first_buffer
        def initialize(blocks, adapter_block_idx=None):
            step = int(model.scheduler.step_index)
            if adapter_block_idx is not None or step not in (0,2) or EXPERTS[step] != expert or self.dispatch != expert:
                raise RuntimeError('unexpected first buffer')
            omit = {0} if self.skip_anchor_w1 else set()
            def record_bytes(n): self.skipped_payload_bytes += n
            result = original_init(TargetBlocks(blocks, 0, omit, record_bytes) if omit else blocks)
            self.tags[id(mgr.cuda_buffers[0])] = (step, expert, 0, 0 not in omit, 2 not in omit)
            self.counts['init_w2_only' if omit else 'init_full'] += 1
            return result
        def prefetch(target, blocks, adapter_block_idx=None):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            if step not in range(4) or EXPERTS[step] != expert or self.dispatch != expert or adapter_block_idx is not None or len(blocks) != LAYERS:
                raise RuntimeError('invalid actual dispatch')
            if target != (layer + 1) % LAYERS: raise RuntimeError('unexpected native request')
            if layer < LAYERS - 1:
                next_step, next_expert, next_layer = step, expert, layer + 1
            elif step < 3 and EXPERTS[step + 1] == expert:
                next_step, next_expert, next_layer = step + 1, expert, 0
            else:
                self.counts['terminal'] += 1
                self.trace.append((step, expert, layer, None))
                return None
            omit = {0,2} if self.skip_hits and next_step in (1,3) else ({0} if self.skip_anchor_w1 and next_step in (0,2) else set())
            def record_bytes(n): self.skipped_payload_bytes += n
            result = original_prefetch(next_layer, TargetBlocks(blocks, next_layer, omit, record_bytes) if omit else blocks)
            tag = (next_step, next_expert, next_layer, 0 not in omit, 2 not in omit)
            self.tags[id(mgr.cuda_buffers[1])] = tag
            self.counts['prefetch_skip' if len(omit)==2 else 'prefetch_w2_only' if omit else 'prefetch_full'] += 1
            self.trace.append((step, expert, layer, tag))
            return result
        self.bindings.append((mgr, original_init, original_prefetch, original_need))
        mgr.need_init_first_buffer = True
        mgr.init_first_buffer, mgr.prefetch_weights = initialize, prefetch

    def verify_current(self, model, phase, step, expert, layer):
        mgr = model.offload_manager
        w1 = not ((self.skip_hits and step in (1,3)) or (self.skip_anchor_w1 and step in (0,2)))
        w2 = not (self.skip_hits and step in (1,3))
        if self.dispatch != expert or self.tags.get(id(mgr.cuda_buffers[0])) != (step, expert, layer, w1, w2):
            raise RuntimeError('stale or wrong GPU buffer')
        for mm, present in ((phase.ffn_0,w1),(phase.ffn_2,w2)):
            if (mm.weight is not None) != present or (mm.bias is not None) != present:
                raise RuntimeError('FFN tensor presence disagrees with tag')

    def finish(self):
        expected = dict(init_full=0 if self.skip_anchor_w1 else 2,
                        init_w2_only=2 if self.skip_anchor_w1 else 0,
                        prefetch_full=(0 if self.skip_anchor_w1 else 78)+(0 if self.skip_hits else 80),
                        prefetch_w2_only=78 if self.skip_anchor_w1 else 0,
                        prefetch_skip=80 if self.skip_hits else 0, terminal=2)
        if self.counts != expected or len(self.trace) != 160: raise RuntimeError('incomplete transfer schedule')
        return {'counts':dict(self.counts), 'trace':list(self.trace), 'omitted_source_ffn_bytes':self.skipped_payload_bytes,
                'physical_transfer_measurement':'NOT_MEASURED'}

    def uninstall(self):
        for mgr, init, prefetch, _ in reversed(self.bindings):
            mgr.init_first_buffer, mgr.prefetch_weights = init, prefetch
            mgr.need_init_first_buffer = True
        self.bindings.clear(); self.tags.clear()
