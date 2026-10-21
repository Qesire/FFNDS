"""Fixed native FFN restoration, and exact four-arm transport accounting.

Restoration is a diagnostic intervention, never a budget-compliant candidate.
No native module, activation quantizer, gate or residual writeback is replaced.
"""
import contextvars

import torch
import torch.nn.functional as F

from run_ar2_restart import tensor_sha
from variable_teacher_capture import FrozenBias, equal_bits


ARMS = {'none': (), 'none_repeat': (), 'A': (30,), 'B': (31,), 'AB': (30, 31), 'DQ': ()}


def gated_error(student, teacher, gate, chunk=1024):
    """Both ideal gated FP64 error and actual rounded native gate products."""
    if student.shape != teacher.shape or student.dtype != torch.bfloat16 or teacher.dtype != student.dtype:
        raise RuntimeError('Expected paired native BF16 FFN outputs')
    x, y = student.reshape(-1, 5120), teacher.reshape(-1, 5120)
    g = gate.reshape(1, 5120)
    ideal = native = norm = 0.
    for begin in range(0, len(x), chunk):
        a, b = x[begin:begin + chunk], y[begin:begin + chunk]
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(g).all()):
            raise RuntimeError('Nonfinite local operands')
        ideal += float(((a.double() - b.double()) * g.double()).square().sum())
        native += float(((a * g).double() - (b * g).double()).square().sum())
        norm += float((b.double() * g.double()).square().sum())
    return {'ideal_gated_SSE': ideal, 'native_gated_product_SSE': native, 'teacher_gated_norm2': norm}


def transport_accounting(flows, target):
    """FP64 identity; no cancellation inference from local error norms alone."""
    required = {'none', 'A', 'B', 'AB'}
    if not required.issubset(flows) or any(v.shape != target.shape for v in flows.values()):
        raise ValueError('Matched full-flow arms required')
    f = {k: v.double() for k, v in flows.items()}
    if not all(bool(torch.isfinite(v).all()) for v in [target, *f.values()]):
        raise ValueError('Finite full flows required')
    e = f['none'] - target.double()
    a, b = f['A'] - f['none'], f['B'] - f['none']
    c = f['AB'] - f['none'] - a - b
    sse = {k: float((v - target.double()).square().sum()) for k, v in f.items()}
    gain = {k: sse['none'] - v for k, v in sse.items()}
    linear = float(-2 * (a * b).sum())
    curved = float(-2 * ((e + a + b) * c).sum() - c.square().sum())
    interaction = gain['AB'] - gain['A'] - gain['B']
    residual = interaction - linear - curved
    tolerance = 1e-10 * max(1., *sse.values())
    if abs(residual) > tolerance:
        raise RuntimeError('Four-arm algebra identity failed')
    return {'flow_SSE': sse, 'flow_SSE_reduction': gain, 'interaction_advantage': interaction,
            'linear_direction_cross_term': linear, 'nonlinear_transport_term': curved,
            'curvature_norm2': float(c.square().sum()), 'A_flow_change_norm2': float(a.square().sum()),
            'B_flow_change_norm2': float(b.square().sum()), 'identity_residual': residual,
            'identity_absolute_tolerance': tolerance,
            'none_repeat_SSE': float((f['none_repeat'] - f['none']).square().sum()) if 'none_repeat' in f else None}


class NativePairRestoration:
    """Observe only step3/block30,31; return original or same-input teacher Y."""
    def __init__(self, runtime, teacher, arm):
        if arm not in ARMS:
            raise ValueError('Unregistered intervention')
        self.runtime, self.teacher, self.arm = runtime, teacher, arm
        self.scopes = []
        self.context = contextvars.ContextVar('ffnds_fixed_pair', default=None)

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.infer = WanTransformerInfer
        self.old_ffn, self.old_apply = self.infer.infer_ffn, self.runtime.apply
        observer = self

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            step, block = int(infer.scheduler.step_index), int(infer.block_idx)
            if step != 3 or block not in (30, 31):
                return observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
            if gate is None or gate.numel() != 5120 or gate.dtype != torch.bfloat16:
                raise RuntimeError('Expected original BF16 channel gate')
            scope = {'step': step, 'block': block, 'gate': gate.detach().reshape(1, 5120).clone()}
            token = observer.context.set(scope)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            try:
                result = observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError('Selected FFN/observer mutated native RNG')
                if not scope.get('complete'):
                    raise RuntimeError('Selected FFN escaped the intervention boundary')
                return result
            finally:
                observer.context.reset(token)

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            s = observer.context.get()
            if s is None or not module_name.endswith(('.ffn.0', '.ffn.2')):
                return observer.old_apply(mm, value, **kw)
            if (step_index, block_index, expert) != (3, s['block'], 'low_noise_model'):
                raise RuntimeError('Selected context does not match the native linear')
            bias = mm._get_actual_bias()
            if bias is None:
                raise RuntimeError('Original explicit bias missing')
            slot = 0 if module_name.endswith('.ffn.0') else 2
            if slot == 0:
                if 'norm_input' in s or value.shape != (31668, 5120):
                    raise RuntimeError('Duplicate FFN0 or changed full-M input geometry')
                s.update(norm_input=value.detach(), b0=bias.detach().clone())
            elif 'norm_input' not in s or s.get('complete'):
                raise RuntimeError('Unpaired native FFN0/FFN2')
            result = observer.old_apply(mm, value, **kw)
            if slot == 2:
                s['b2'] = bias.detach().clone()
                result = observer.finish(s, result)
                s['complete'] = True
                for key in ('norm_input', 'b0', 'b2'):
                    s.pop(key)
            return result

        self.infer.infer_ffn, self.runtime.apply = ffn, apply
        return self

    def finish(self, s, student):
        if any(v['block'] == s['block'] for v in self.scopes):
            raise RuntimeError('Duplicate selected position')
        shard = self.teacher.configure('low_noise_model', s['block'])
        r = self.teacher.runtime
        before = dict(r.counters)
        try:
            common = dict(expert='low_noise_model', step_index=3, block_index=s['block'])
            z = r.apply(FrozenBias(s['b0']), s['norm_input'], module_name=f'blocks.{s["block"]}.ffn.0', **common)
            hidden = F.gelu(z, approximate='tanh')
            del z
            y = r.apply(FrozenBias(s['b2']), hidden, module_name=f'blocks.{s["block"]}.ffn.2', **common)
            del hidden
            delta = {k: r.counters[k] - before[k] for k in before}
            if delta['linear_forward_calls'] != 2 or delta['activation_qdq_calls'] != 2 or delta['cache_misses']:
                raise RuntimeError('Teacher did not execute exactly two full native QDQ linears')
            if self.arm == 'DQ' and not equal_bits(student, y):
                raise RuntimeError('Native full teacher differs from unmodified dense FFN')
            restore = s['block'] in ARMS[self.arm]
            local = gated_error(student, y, s['gate'])
            output = y if restore else student
            self.scopes.append({'step': 3, 'block': s['block'], 'arm': self.arm, 'restored': restore,
                'full_rows': 31668, 'teacher_shard_sha256': shard['sha256'],
                'norm_input_sha256': tensor_sha(s['norm_input']), 'b0_sha256': tensor_sha(s['b0']),
                'b2_sha256': tensor_sha(s['b2']), 'gate_sha256': tensor_sha(s['gate']),
                'student_Y_sha256': tensor_sha(student), 'teacher_Y_sha256': tensor_sha(y),
                'returned_Y_sha256': tensor_sha(output), 'local_before_restoration': local,
                'local_after_restoration': {k: (0. if k != 'teacher_gated_norm2' else v) for k, v in local.items()} if restore else local,
                'dense_identity_bitwise_equal': equal_bits(student, y) if self.arm == 'DQ' else None,
                'teacher_counter_delta': delta, 'native_RNG_unchanged': True})
            return output
        finally:
            self.teacher.release()

    def __exit__(self, *error):
        self.infer.infer_ffn, self.runtime.apply = self.old_ffn, self.old_apply
        self.teacher.release()
