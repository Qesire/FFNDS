"""Fixed native packed FFN, preserving original caller modulation and writeback."""


class ReconstructedRuntime:
    def __init__(self, runtime, packs, adapter):
        self.runtime, self.packs, self.adapter = runtime, packs, adapter
        self.pending = None
        self.counts = dict.fromkeys(['up', 'down', 'Q1', 'Q2', 'affine'], 0)

    def __enter__(self):
        import torch
        import torch.nn.functional as F
        self.old = self.runtime.apply
        owner = self

        def apply(mm, x, *, expert, step_index, block_index, module_name):
            kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            if not module_name.endswith(('.ffn.0', '.ffn.2')):
                return owner.old(mm, x, **kw)
            if expert != ('high_noise_model' if step_index < 2 else 'low_noise_model'):
                raise ValueError('Wrong expert/step')
            key = (expert, block_index)
            pack = owner.packs[key]
            if module_name.endswith('.ffn.0'):
                if owner.pending is not None or x.dtype != torch.bfloat16 or x.shape[-1] != 5120:
                    raise ValueError('Unexpected FFN input/order')
                u = (x.float() / pack['s1']).to(x.dtype)
                q = owner.adapter.native_hif4_activation_qdq(u, force_fp32=True).to(x.dtype)
                owner.counts['Q1'] += 1; owner.counts['up'] += 1
                owner.pending = (key, step_index, q)
                return F.linear(q, pack['w1'], pack['b1'])
            if owner.pending is None or owner.pending[:2] != (key, step_index) or x.shape[-1] != pack['w1'].shape[0]:
                raise ValueError('Unexpected FFN consumer/order')
            q1 = owner.pending[2]; owner.pending = None
            u = (x.float() / pack['s2']).to(x.dtype)
            q2 = owner.adapter.native_hif4_activation_qdq(u, force_fp32=True).to(x.dtype)
            owner.counts['Q2'] += 1; owner.counts['down'] += 1
            y = F.linear(q2, pack['v'], pack['bias'])
            if 'matrix' in pack:
                y = y + F.linear(q1, pack['matrix'])
                owner.counts['affine'] += 1
            return y

        self.runtime.apply = apply
        return self

    def __exit__(self, *error):
        self.runtime.apply = self.old
        if error[0] is None and self.pending is not None:
            raise RuntimeError('Incomplete FFN pair')


def neutral_check(runtime, adapter, values, expert, step, block, torch):
    """Unchanged full-width weights through the new hook must be bitwise equal.

    Saved modulated input precedes Smooth1/Q1; native caller modulation and
    residual are deliberately outside this finite integration check.
    """
    import torch.nn.functional as F
    up = runtime.records[expert][f'blocks.{block}.ffn.0']
    down = runtime.records[expert][f'blocks.{block}.ffn.2']
    pack = {'w1': up['weight_qdq'].to('cuda', dtype=torch.bfloat16),
            'v': down['weight_qdq'].to('cuda', dtype=torch.bfloat16),
            'b1': values['b1'].bfloat16().reshape(-1), 'bias': values['b2'].bfloat16().reshape(-1),
            's1': up['smooth_scale'].to('cuda').float().reshape(1, -1),
            's2': down['smooth_scale'].to('cuda').float().reshape(1, -1)}
    class Bias:
        def __init__(self, value): self.value = value
        def _get_actual_bias(self): return self.value
    mm1, mm2 = Bias(pack['b1']), Bias(pack['bias'])
    kw = dict(expert=expert, step_index=step, block_index=block)
    x = values['modulated_input'].bfloat16()
    def pair():
        z = runtime.apply(mm1, x, module_name=f'blocks.{block}.ffn.0', **kw)
        return runtime.apply(mm2, F.gelu(z, approximate='tanh'), module_name=f'blocks.{block}.ffn.2', **kw)
    reference = pair()
    with ReconstructedRuntime(runtime, {(expert, block): pack}, adapter) as hook:
        candidate = pair()
    mismatch = int((reference.view(torch.int16) != candidate.view(torch.int16)).sum())
    if mismatch:
        raise RuntimeError('Neutral full-width runtime hook changed BF16 output')
    return {'step': step, 'block': block, 'rows': len(x), 'BF16_bit_mismatches': mismatch,
            'extra_native_FFN_calls': 2, 'extra_native_QDQ_calls': 4,
            'hook_counts': hook.counts}
