"""Distilled four-step direct FFN core; caller must supply verified native state.

No network, model loading, or experiment is performed by this module.
"""
import torch
import torch.nn.functional as F


class DirectFFN:
    def __init__(self, mode, matrices, *, selective_prefetch=False, layers=40, expected_shape=None, residual_fn=None, diagnostic=False, execution='baseline', post_fn=None):
        if mode not in ('dense', 'neutral', 'g19'):
            raise ValueError('unknown arm')
        if selective_prefetch:
            raise NotImplementedError('selective transport is not implemented or verified')
        self.mode, self.matrices, self.layers = mode, matrices, layers
        self.expected_shape = expected_shape
        self.residual_fn, self.diagnostic = residual_fn, diagnostic
        if execution not in ('baseline','fused_post','parallel') or (execution != 'baseline' and (mode != 'g19' or post_fn is None or residual_fn is None or diagnostic)):
            raise ValueError('unsupported execution')
        self.execution, self.post_fn = execution, post_fn
        self.aux = torch.cuda.Stream() if execution == 'parallel' else None
        self.slot_events = [None] * layers
        self.events = []
        self.bit_checks = []
        self.slab = None
        self.expert = None
        self.step = -1
        self.next_layer = 0
        self.counts = dict(dense=0, anchor=0, hit=0, native_ffn=0, predictions=0)
        self.closed = False

    def begin_step(self, step, expert):
        if self.closed or step != self.step + 1 or step not in range(4):
            raise RuntimeError('invalid step sequence')
        if self.step >= 0 and self.next_layer != self.layers:
            raise RuntimeError('incomplete preceding step')
        expected = ('high', 'high', 'low', 'low')[step]
        if expert != expected:
            raise RuntimeError('unexpected actual expert')
        if expert != self.expert:
            if step not in (0, 2):
                raise RuntimeError('expert changed without anchor')
            self.expert = expert
            self.valid = [False] * self.layers
            self.matrices.activate(expert) if self.mode != 'dense' else None
        self.step, self.next_layer = step, 0

    def infer_ffn(self, model, phase, x, attn_out, shift, scale, gate=None):
        if self.closed or self.step < 0 or self.next_layer >= self.layers:
            raise RuntimeError('invalid FFN lifecycle')
        if model.block_idx != self.next_layer or model.clean_cuda_cache or hasattr(phase, 'smooth_norm2_weight') or model._use_mxfp8_quant_fuse():
            raise RuntimeError('unsupported native path or layer order')
        def mark():
            if not self.diagnostic: return None
            event = torch.cuda.Event(enable_timing=True); event.record(); return event
        prefix_start = mark()
        x.add_(attn_out)
        u = phase.norm2.apply(x)
        if model.sensitive_layer_dtype != model.infer_dtype:
            u = u.to(model.sensitive_layer_dtype)
        if model._can_use_mxfp8_modulate_quant(u, scale, shift):
            raise RuntimeError('unsupported fused modulation')
        u = model.modulate_func(u, scale=scale, shift=shift).squeeze()
        if model.sensitive_layer_dtype != model.infer_dtype:
            u = u.to(model.infer_dtype)
        if u.dtype != torch.bfloat16 or u.ndim != 2 or (self.expected_shape is not None and tuple(u.shape) != self.expected_shape):
            raise RuntimeError('expected 2D BF16 native input')
        layer = self.next_layer
        action = 'dense' if self.mode == 'dense' else ('anchor' if self.step in (0, 2) else 'hit')
        if self.execution != 'baseline':
            return self._optimized(model, phase, x, u, gate, layer, action)
        prefix_end = mark()
        native_start = mark()
        if action != 'hit' or self.mode == 'neutral':
            y = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u), approximate='tanh'))
            self.counts['native_ffn'] += 1
        native_end = mark()
        gemm_start = mark()
        if action != 'dense':
            a = self.matrices.get(self.expert, layer, u.device)
            if a.dtype != u.dtype or a.device != u.device or a.shape != (u.shape[1], u.shape[1]):
                raise RuntimeError('matrix mismatch')
            q = u @ a
            gemm_end = mark()
            self.counts['predictions'] += 1
            if self.slab is None:
                self.slab = torch.empty((self.layers, *u.shape), dtype=torch.bfloat16, device=u.device)
            if self.slab.shape[1:] != u.shape or self.slab.device != u.device:
                raise RuntimeError('slab shape/device changed')
            residual_start = mark()
            if action == 'anchor':
                if self.residual_fn: self.residual_fn(y, q, self.slab[layer], subtract=True)
                else: self.slab[layer].copy_((y.float() - q.float()).to(torch.bfloat16))
                self.valid[layer] = True
            else:
                if not self.valid[layer]:
                    raise RuntimeError('uninitialized cache slot')
                if self.residual_fn:
                    prediction = torch.empty_like(q)
                    self.residual_fn(q, self.slab[layer], prediction, subtract=False)
                else: prediction = (q.float() + self.slab[layer].float()).to(torch.bfloat16)
                if self.mode == 'g19':
                    y = prediction
            residual_end = mark()
            if self.diagnostic:
                self.events.append((self.step, layer, action, prefix_start, prefix_end, native_start, native_end, gemm_start, gemm_end, residual_start, residual_end))
                length = min(4096, q.numel())
                p = q.reshape(-1)[:length]
                if action == 'anchor':
                    expected = (y.reshape(-1)[:length].float() - p.float()).to(torch.bfloat16)
                    actual = self.slab[layer].reshape(-1)[:length]
                else:
                    expected = (p.float() + self.slab[layer].reshape(-1)[:length].float()).to(torch.bfloat16)
                    actual = prediction.reshape(-1)[:length]
                self.bit_checks.append((self.step, layer, torch.count_nonzero(expected.view(torch.int16) != actual.view(torch.int16))))
        self.counts[action] += 1
        self.next_layer += 1
        return y

    def _optimized(self, model, phase, x, u, gate, layer, action):
        if (gate is None or gate.dtype != torch.bfloat16 or x.dtype != torch.bfloat16
            or model.infer_dtype != torch.bfloat16 or model.sensitive_layer_dtype != torch.bfloat16):
            raise RuntimeError('fused post requires BF16 x and current gate')
        a = self.matrices.get(self.expert, layer, u.device)
        if a.dtype != u.dtype or a.device != u.device or a.shape != (u.shape[1],u.shape[1]):
            raise RuntimeError('invalid A')
        if self.slab is None: self.slab = torch.empty((self.layers,*u.shape),dtype=torch.bfloat16,device=u.device)
        if self.slab.shape[1:] != u.shape or self.slab.device != u.device: raise RuntimeError('slab mismatch')
        slot = self.slab[layer]
        current = torch.cuda.current_stream(u.device)
        if action == 'anchor':
            if self.execution == 'parallel':
                # x is updated before aux finishes; an alias would race its GEMM input.
                if u.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(): raise RuntimeError('u aliases mutable x')
                self.aux.wait_stream(current)
                u.record_stream(self.aux); a.record_stream(self.aux); self.slab.record_stream(self.aux)
                with torch.cuda.stream(self.aux): q = u @ a
                y = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u),approximate='tanh'))
                y_ready = torch.cuda.Event(); y_ready.record(current)
                y.record_stream(self.aux); q.record_stream(self.aux)
                with torch.cuda.stream(self.aux):
                    self.aux.wait_event(y_ready)
                    self.residual_fn(y,q,slot,subtract=True)
                    done = torch.cuda.Event(); done.record(self.aux)
                self.slot_events[layer] = done
                self.post_fn(x,y,None,None,gate,action='post')
            else:
                y = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u),approximate='tanh'))
                q = u @ a
                self.post_fn(x,y,q,slot,gate,action='anchor')
            self.counts['native_ffn'] += 1
            self.valid[layer] = True
        elif action == 'hit':
            if not self.valid[layer]: raise RuntimeError('uninitialized cache')
            if self.execution == 'parallel':
                if self.slot_events[layer] is None: raise RuntimeError('missing cache event')
                current.wait_event(self.slot_events[layer])
            q = u @ a
            self.post_fn(x,None,q,slot,gate,action='hit')
        else: raise RuntimeError('unsupported optimized action')
        self.counts['predictions'] += 1; self.counts[action] += 1; self.next_layer += 1
        # Native post_process(None) returns x unchanged, avoiding a second gate.
        return None

    def finish(self):
        if self.closed or self.step != 3 or self.next_layer != self.layers:
            raise RuntimeError('incomplete schedule')
        if self.aux is not None: torch.cuda.current_stream().wait_stream(self.aux)
        self.slab = None
        if self.mode != 'dense':
            self.matrices.release()
        self.closed = True
        return dict(self.counts)

    def close_stream(self):
        if self.aux is not None: self.aux.synchronize()
