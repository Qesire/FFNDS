"""Observe original norm2 statistics and scale existing R16 energy before W1.

This module never registers or launches a model. The only routing change is
score(row) = original_gated_R256_energy(row) * native_current_norm2_rstd(row)^2.
The existing RowDispatch still owns both physical FFN branches and writeback.
"""
from prew1_norm_statistics import layer_norm_with_statistics


class NormScaledEnergy:
    def __init__(self, dispatch, mode='scaled'):
        if mode not in ('scaled', 'observe_UB1'):
            raise ValueError('Explicit scaled or UB1 observation mode required')
        expected = 'dynamic_residual' if mode == 'scaled' else 'UB1'
        if dispatch.arm != expected:
            raise ValueError('Native row-dispatch arm/mode mismatch')
        self.dispatch, self.mode = dispatch, mode
        self.active, self.events = None, []

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        import row_dispatch_kernels
        self.cls, self.kernel = WanTransformerInfer, row_dispatch_kernels
        self.old_ffn = self.cls.infer_ffn
        self.old_energy = self.kernel.gated_energy
        owner = self

        def energy(residual, gate):
            if owner.mode != 'scaled' or owner.active is None or 'rstd' not in owner.active:
                raise RuntimeError('Pre-W1 native norm statistics missing')
            if owner.active['score_calls'] or owner.active['rstd'].numel() != residual.shape[0]:
                raise RuntimeError('Expected one matched score per full native FFN')
            owner.active['score_calls'] += 1
            # Same R16 kernel and reductions; only two FP32 pointwise ops added.
            return owner.old_energy(residual, gate) * owner.active['rstd'].square()

        def ffn(infer, phase, x, attention, shift, scale, gate=None):
            if owner.active is not None:
                raise RuntimeError('Nested scaled-energy FFN is not supported')
            step, block = int(infer.scheduler.step_index), int(infer.block_idx)
            scope = {'step': step, 'block': block, 'norm_calls': 0, 'score_calls': 0}
            owner.active = scope; original_norm = phase.norm2.apply
            before_events = len(owner.dispatch.events)

            def norm(value):
                if scope['norm_calls']:
                    raise RuntimeError('Expected one original norm2 per native FFN')
                y, _, rstd = layer_norm_with_statistics(value, infer_dtype=infer.infer_dtype,
                    sensitive_layer_dtype=infer.sensitive_layer_dtype, weight=phase.norm2._get_actual_weight(),
                    bias=phase.norm2._get_actual_bias(), eps=phase.norm2.eps)
                scope['norm_calls'] += 1
                scope['rstd'] = rstd.reshape(-1)
                return y

            phase.norm2.apply = norm
            try:
                result = owner.old_ffn(infer, phase, x, attention, shift, scale, gate)
                expected_calls = 1 if owner.mode == 'scaled' else 0
                if scope['norm_calls'] != 1 or scope['score_calls'] != expected_calls:
                    raise RuntimeError('Incomplete original/scaled information path')
                if len(owner.dispatch.events) != before_events + 1:
                    raise RuntimeError('Original native row dispatch missing or duplicated')
                event = owner.dispatch.events[-1]
                if (event['step'], event['block']) != (step, block):
                    raise RuntimeError('Native score/physical action position mismatch')
                if owner.mode == 'scaled':
                    event['pre_W1_signal'] = 'gated_R256_energy_times_current_norm2_rstd_squared'
                owner.events.append({'step': step, 'block': block, 'norm_calls': 1,
                                     'scaled_score_calls': expected_calls, 'CPU_tensor_transfers': 0})
                return result
            finally:
                phase.norm2.apply = original_norm
                owner.active = None

        self.cls.infer_ffn = ffn
        if self.mode == 'scaled': self.kernel.gated_energy = energy
        return self

    def __exit__(self, *error):
        self.cls.infer_ffn = self.old_ffn
        if self.mode == 'scaled': self.kernel.gated_energy = self.old_energy
