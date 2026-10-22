"""Observe the actual R12 pair, with original same-input full216 identities."""
import contextvars

import torch
import torch.nn.functional as F

from native_teacher_capture import FrozenBias, equal_bits
from r12d0_contract import groups
from run_ar2_restart import tensor_sha


class PairObserver:
    def __init__(self, runtime, teacher):
        self.runtime, self.teacher = runtime, teacher
        self.scopes, self.pending_post = [], {}
        self.context = contextvars.ContextVar('r12d0_actual_pair', default=None)

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.infer = WanTransformerInfer
        self.old_ffn, self.old_apply, self.old_post = self.infer.infer_ffn, self.runtime.apply, self.infer.post_process
        observer = self

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            step, block = int(infer.scheduler.step_index), int(infer.block_idx)
            if step != 3 or block not in (30, 31):
                return observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
            if gate is None or gate.numel() != 5120 or gate.dtype != torch.bfloat16:
                raise RuntimeError('Original BF16 row-constant gate changed')
            scope = {'step': step, 'block': block, 'gate': gate.detach().reshape(1, 5120).clone()}
            token = observer.context.set(scope)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            try:
                result = observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
                if not scope.get('complete') or not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError('Selected native FFN coverage/RNG changed')
                return result
            finally:
                observer.context.reset(token)

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            kw = dict(expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            s = observer.context.get()
            if s is None or not module_name.endswith(('.ffn.0', '.ffn.2')):
                return observer.old_apply(mm, value, **kw)
            if (step_index, block_index, expert) != (3, s['block'], 'low_noise_model'):
                raise RuntimeError('Selected pair context mismatch')
            bias = mm._get_actual_bias()
            if bias is None:
                raise RuntimeError('Missing original raw bias')
            slot = 0 if module_name.endswith('.ffn.0') else 2
            if slot == 0:
                if 'norm_input' in s or value.shape != (31668, 5120):
                    raise RuntimeError('Actual full-M pair input mismatch')
                s.update(norm_input=value.detach(), b0=bias.detach().clone())
            elif 'norm_input' not in s or s.get('complete'):
                raise RuntimeError('Unpaired selected linears')
            result = observer.old_apply(mm, value, **kw)
            if slot == 2:
                s['b2'] = bias.detach().clone()
                observer.finish(s, result)
                s['complete'] = True
                for key in ('norm_input', 'b0', 'b2'):
                    s.pop(key)
            return result

        def post(infer, x, y, gate, pre_infer_out=None):
            block = int(infer.block_idx)
            selected = int(infer.scheduler.step_index) == 3 and block in (30, 31)
            if not selected:
                return observer.old_post(infer, x, y, gate, pre_infer_out)
            row = observer.pending_post.pop(block)
            if infer.sensitive_layer_dtype != infer.infer_dtype or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
                raise RuntimeError('Frozen BF16 gate/residual branch changed')
            if tensor_sha(gate.reshape(1, 5120)) != row['gate_sha256']:
                raise RuntimeError('Gate changed between FFN and writeback')
            reference = x.clone()
            reference.add_(y * gate.squeeze())
            result = observer.old_post(infer, x, y, gate, pre_infer_out)
            if not equal_bits(result, reference):
                raise RuntimeError('Original gate/residual writeback differs from scoped reconstruction')
            row.update(native_gate_residual_bitwise_equal=True, native_RNG_unchanged=True,
                       post_residual_sha256=tensor_sha(result))
            return result

        self.infer.infer_ffn, self.runtime.apply, self.infer.post_process = ffn, apply, post
        return self

    def finish(self, s, student):
        if s['block'] in self.pending_post or any(r['block'] == s['block'] for r in self.scopes):
            raise RuntimeError('Duplicate selected pair observation')
        count = groups(self.runtime.r12_arm, 3, s['block'])
        row = {'step': 3, 'block': s['block'], 'groups': count, 'full_rows': 31668,
               'norm_input_sha256': tensor_sha(s['norm_input']), 'b0_sha256': tensor_sha(s['b0']),
               'b2_sha256': tensor_sha(s['b2']), 'gate_sha256': tensor_sha(s['gate']),
               'returned_Y_sha256': tensor_sha(student),
               **self.runtime.r12_quantized[s['block']]}
        if count == 216:
            from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
            shard = self.teacher.configure('low_noise_model', s['block'])
            r, previous_qdq = self.teacher.runtime, adapter.native_hif4_activation_qdq
            before, stages = dict(r.counters), []

            def qdq(value, *, force_fp32=True):
                result = previous_qdq(value, force_fp32=force_fp32)
                stages.append(tensor_sha(result.to(value.dtype)))
                return result

            adapter.native_hif4_activation_qdq = qdq
            try:
                common = dict(expert='low_noise_model', step_index=3, block_index=s['block'])
                z = r.apply(FrozenBias(s['b0']), s['norm_input'], module_name=f'blocks.{s["block"]}.ffn.0', **common)
                hidden = F.gelu(z, approximate='tanh')
                del z
                y = r.apply(FrozenBias(s['b2']), hidden, module_name=f'blocks.{s["block"]}.ffn.2', **common)
                del hidden
                delta = {k: r.counters[k] - before[k] for k in before}
                if (not equal_bits(student, y) or stages != [row['Q1_sha256'], row['Q2_sha256']]
                        or delta['linear_forward_calls'] != 2 or delta['activation_qdq_calls'] != 2 or delta['cache_misses']):
                    raise RuntimeError('Original full216 same-input Y/Q1/Q2 identity failed')
                row.update(full216_original_teacher_bitwise_equal=True, original_Q1_Q2_bitwise_equal=True,
                           teacher_Y_sha256=tensor_sha(y), teacher_shard_sha256=shard['sha256'],
                           offline_teacher_counter_delta=delta)
            finally:
                adapter.native_hif4_activation_qdq = previous_qdq
                self.teacher.release()
        else:
            row['offline_teacher_counter_delta'] = {'linear_forward_calls': 0, 'activation_qdq_calls': 0}
        self.pending_post[s['block']] = row
        self.scopes.append(row)

    def __exit__(self, *error):
        self.infer.infer_ffn, self.runtime.apply, self.infer.post_process = self.old_ffn, self.old_apply, self.old_post
        self.teacher.release()
