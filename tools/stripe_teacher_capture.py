"""One-position complete-stripe observation, preserving original UB1 output.

Only used after R14N0 registration. Inherits the original Q1, teacher and RNG
controls; runs no candidate operator while a full model is resident.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from run_ar2_restart import tensor_sha
from variable_teacher_capture import FrozenBias, VariableTeacherCapture, equal_bits


def writeback(residual, y, gate, infer_dtype, sensitive_dtype):
    """Exact checked source branch, preserving the residual's actual dtype."""
    if sensitive_dtype != infer_dtype:
        return residual.to(sensitive_dtype) + y.to(sensitive_dtype) * gate.reshape(1, -1)
    output = residual.clone()
    output.add_(y * gate.reshape(1, -1))
    return output


class StripeTeacherCapture(VariableTeacherCapture):
    def __init__(self, runtime, teacher, out, index, draft, old_case):
        super().__init__(runtime, teacher, out, index)
        self.draft, self.old_case = draft, old_case
        self.planned = draft['coordinates']['planned_native_row_ids']
        self.pending = None
        self.actual_x = None
        self.active_infer = None

    def __enter__(self):
        super().__enter__()
        self.parent_ffn = self.infer.infer_ffn
        self.old_post = self.infer.post_process
        observer = self

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            observer.actual_x, observer.active_infer = x, infer
            try:
                return observer.parent_ffn(infer, phase, x, attn, shift, scale, gate)
            finally:
                observer.actual_x = observer.active_infer = None

        def post(infer, x, y, gate, pre=None):
            pending = observer.pending
            if pending is None:
                return observer.old_post(infer, x, y, gate, pre)
            values, metadata, ids = pending
            sampled = x.reshape(-1, 5120).index_select(0, ids)
            if not equal_bits(sampled, values['Residual'].to(x.device)):
                raise RuntimeError('Residual changed between FFN and original post_process')
            expected = writeback(sampled, y.reshape(-1, 5120).index_select(0, ids), gate,
                                 infer.infer_dtype, infer.sensitive_layer_dtype)
            output = observer.old_post(infer, x, y, gate, pre)
            actual = output.reshape(-1, 5120).index_select(0, ids)
            if not equal_bits(actual, expected):
                raise RuntimeError('Original sensitive-dtype residual writeback control failed')
            values['NativePostStudent'] = actual.detach().cpu().contiguous()
            metadata.update(actual_native_writeback_bitwise_control=True,
                            infer_dtype=str(infer.infer_dtype),
                            sensitive_dtype=str(infer.sensitive_layer_dtype),
                            residual_dtype=str(values['Residual'].dtype),
                            original_post_student_sha256=tensor_sha(actual))
            if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                raise RuntimeError('Nonfinite complete-stripe payload')
            path = observer.out / 'step2_block20_complete_stripes.safetensors'
            if path.exists():
                raise RuntimeError('Refuse to overwrite new stripe operands')
            temporary = path.with_suffix('.tmp')
            save_file(values, str(temporary))
            temporary.replace(path)
            metadata.update(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                            tensor_sha256={key: tensor_sha(value) for key, value in values.items()})
            observer.scopes.append(metadata)
            atomic_json(observer.out / 'SCOPE.json', metadata)
            observer.pending = None
            return output

        self.infer.infer_ffn, self.infer.post_process = ffn, post
        return self

    def observe(self, s, result):
        if (s['step'], s['block']) != (2, 20):
            # The enclosing original FFN hook owns this transient context.
            for key in ('norm_input', 'actual_v1', 'actual_residual', 'base_y', 'b0', 'b2'):
                s.pop(key, None)
            return
        if self.scopes or self.pending is not None or self.actual_x is None:
            raise RuntimeError('Exactly one target observation required')
        if (s['grid'] != [21, 29, 52] or tuple(s['actual_v1'].shape) != (31668, 5120)
                or tuple(s['actual_v1'].stride()) != (5120, 1)
                or not equal_bits(s['base_y'] + s['actual_residual'], result)):
            raise RuntimeError('Original target geometry or UB1 main/residual identity changed')
        shard = self.teacher.configure(s['expert'], s['block'])
        runtime = self.teacher.runtime
        before = dict(runtime.counters)
        s['teacher_active'] = True
        try:
            s['teacher_stage'] = 0
            hidden = runtime.apply(FrozenBias(s['b0']), s['norm_input'], expert=s['expert'],
                                   step_index=2, block_index=20, module_name='blocks.20.ffn.0')
            hidden = F.gelu(hidden, approximate='tanh')
            s['teacher_stage'] = 2
            truth = runtime.apply(FrozenBias(s['b2']), hidden, expert=s['expert'],
                                  step_index=2, block_index=20, module_name='blocks.20.ffn.2')
        finally:
            s['teacher_active'] = False
        delta = {key: runtime.counters[key] - before[key] for key in before}
        if (delta['linear_forward_calls'] != 2 or delta['activation_qdq_calls'] != 2
                or delta['cache_misses'] or not s.get('Q1_teacher_equals_actual')):
            raise RuntimeError('Original full-teacher two-linears/QDQ control failed')
        old = next(scope for scope in self.old_case['scopes'] if (scope['step'], scope['block']) == (2, 20))
        if digest(old['path']) != old['sha256']:
            raise RuntimeError('Old target payload changed')
        previous = load_file(old['path'], device='cpu')
        old_ids = torch.tensor(old['selected_rows'], device=result.device)
        current = {'V1': s['actual_v1'], 'BaseY': s['base_y'], 'StudentY': result, 'TeacherY': truth}
        controls = {}
        for key, value in current.items():
            observed = value.reshape(-1, 5120).index_select(0, old_ids)
            controls[key] = equal_bits(observed, previous[key].to(observed.device))
        controls['b2'] = equal_bits(s['b2'], previous['b2'].to(result.device))
        controls['gate'] = equal_bits(s['gate'], previous['gate'].to(result.device))
        controls['raw_b0'] = tensor_sha(s['b0']) == old['original_b0_sha256']
        controls['teacher_shard'] = shard['sha256'] == old['teacher_shard_sha256']
        if not all(controls.values()):
            raise RuntimeError('New observation differs from old original target operands')
        ids = torch.tensor(self.planned, device=result.device)
        values = {key: value.reshape(-1, 5120).index_select(0, ids).detach().cpu().contiguous()
                  for key, value in current.items()}
        values.update(Residual=self.actual_x.reshape(-1, 5120).index_select(0, ids).detach().cpu().contiguous(),
                      raw_b0=s['b0'].detach().cpu().contiguous(),
                      b2=s['b2'].detach().cpu().contiguous(), gate=s['gate'].detach().cpu().contiguous())
        metadata = {'step': 2, 'block': 20, 'expert': s['expert'], 'grid': s['grid'],
                    'full_rows': 31668, 'rows': 256, 'selected_rows': self.planned,
                    'native_V1_shape': list(s['actual_v1'].shape),
                    'native_V1_stride': list(s['actual_v1'].stride()),
                    'old_original_scope_sha256': old['sha256'], 'old_operand_controls': controls,
                    'teacher_shard_sha256': shard['sha256'], 'extra_full_teacher_GEMMs': 2,
                    'Q1_teacher_equals_actual': True,
                    'original_b0_sha256': tensor_sha(s['b0']), 'original_b2_sha256': tensor_sha(s['b2']),
                    'gate_sha256': tensor_sha(s['gate']),
                    'actual_native_BaseY_plus_old_residual_equals_StudentY': True}
        self.pending = values, metadata, ids
        self.teacher.release()

    def __exit__(self, *error):
        self.infer.post_process = self.old_post
        super().__exit__(*error)
        if not any(error) and (len(self.scopes) != 1 or self.pending is not None):
            raise RuntimeError('Target stripe/post-process observation incomplete')
