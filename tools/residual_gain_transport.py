"""Two existing R256 gain parameters: matched local and full-flow objectives.

This is a bounded objective diagnostic, not a new correction architecture.
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from cross_layer_transport import NativePairRestoration, gated_error
from run_ar2_restart import tensor_sha
from variable_teacher_capture import FrozenBias


def box_quadratic(gram, linear, bound=.25):
    """Exact2D convex box minimizer; evaluate stationary faces, no model search.

    Objective is constant +2*b.T*delta +delta.T*G*delta.
    """
    g, b = np.asarray(gram, dtype=np.float64), np.asarray(linear, dtype=np.float64)
    if g.shape != (2, 2) or b.shape != (2,) or not np.isfinite(g).all() or not np.isfinite(b).all():
        raise ValueError('Finite2D quadratic required')
    if bound <= 0 or not np.allclose(g, g.T, atol=1e-10) or np.linalg.eigvalsh(g).min() < -1e-9 * max(1., np.abs(g).max()):
        raise ValueError('Convex symmetric bounded quadratic required')
    candidates = [np.zeros(2)]
    unconstrained = -np.linalg.pinv(g, rcond=1e-12) @ b
    if np.max(np.abs(unconstrained)) <= bound:
        candidates.append(unconstrained)
    for fixed in (0, 1):
        free = 1 - fixed
        for value in (-bound, bound):
            delta = np.zeros(2); delta[fixed] = value
            if g[free, free] > 0:
                delta[free] = np.clip(-(b[free] + g[free, fixed] * value) / g[free, free], -bound, bound)
            else:
                delta[free] = -bound if b[free] + g[free, fixed] * value > 0 else bound
            candidates.append(delta)
    candidates.extend(np.array(v) for v in itertools.product((-bound, bound), repeat=2))
    objective = lambda delta: float(2 * b @ delta + delta @ g @ delta)
    best = min(candidates, key=lambda v: (objective(v), float(v @ v)))
    return {'delta': best.tolist(), 'gains': (1 + best).tolist(), 'quadratic_change': objective(best),
            'gram': g.tolist(), 'linear': b.tolist(), 'bound': bound,
            'condition_number': float(np.linalg.cond(g)) if np.linalg.matrix_rank(g) == 2 else None}


class NativeResidualGain(NativePairRestoration):
    """Scale only the existing residual at step3/block30,31, before native add."""
    def __init__(self, runtime, teacher, gains):
        if len(gains) != 2 or any(not .75 <= value <= 1.25 for value in gains):
            raise ValueError('Two fixed gains in[.75,1.25] required')
        super().__init__(runtime, teacher, 'none')
        self.gains = dict(zip((30, 31), gains))

    def __enter__(self):
        super().__enter__()
        self.old_predict = self.runtime.predict_residual
        observer = self

        def predict(x, parameters):
            residual = observer.old_predict(x, parameters)
            s = observer.context.get()
            if s is not None:
                if 'residual' in s:
                    raise RuntimeError('Duplicate selected residual prediction')
                s['residual'] = residual.detach()
                gain = observer.gains[s['block']]
                if gain != 1.:
                    residual = residual * gain
            return residual

        self.runtime.predict_residual = predict
        return self

    def finish(self, s, student):
        if 'residual' not in s or any(v['block'] == s['block'] for v in self.scopes):
            raise RuntimeError('Unpaired selected residual')
        shard = self.teacher.configure('low_noise_model', s['block'])
        r = self.teacher.runtime
        before = dict(r.counters)
        try:
            common = dict(expert='low_noise_model', step_index=3, block_index=s['block'])
            z = r.apply(FrozenBias(s['b0']), s['norm_input'], module_name=f'blocks.{s["block"]}.ffn.0', **common)
            hidden = F.gelu(z, approximate='tanh'); del z
            y = r.apply(FrozenBias(s['b2']), hidden, module_name=f'blocks.{s["block"]}.ffn.2', **common)
            del hidden
            delta = {k: r.counters[k] - before[k] for k in before}
            if delta['linear_forward_calls'] != 2 or delta['activation_qdq_calls'] != 2 or delta['cache_misses']:
                raise RuntimeError('Incorrect local full-teacher execution')
            local = gated_error(student, y, s['gate'])
            x, target, residual = [v.reshape(-1, 5120) for v in [student, y, s['residual']]]
            dot = norm = 0.
            for begin in range(0, len(x), 1024):
                a, b, v = [v[begin:begin + 1024].double() for v in [x, target, residual]]
                direction = v * s['gate'].double()
                dot += float(((a - b) * s['gate'].double() * direction).sum())
                norm += float(direction.square().sum())
            self.scopes.append({'step': 3, 'block': s['block'], 'gain': self.gains[s['block']],
                'norm_input_sha256': tensor_sha(s['norm_input']), 'gate_sha256': tensor_sha(s['gate']),
                'residual_sha256': tensor_sha(s['residual']), 'teacher_shard_sha256': shard['sha256'],
                'student_Y_sha256': tensor_sha(student), 'teacher_Y_sha256': tensor_sha(y),
                'local_gated_error': local, 'local_error_dot_original_residual': dot,
                'local_original_residual_norm2': norm, 'native_RNG_unchanged': True})
            s.pop('residual')
            return student
        finally:
            self.teacher.release()

    def __exit__(self, *error):
        self.runtime.predict_residual = self.old_predict
        super().__exit__(*error)
