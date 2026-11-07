"""Single-branch native class binding for the E0A direct FFN core."""
import time
import torch
from direct import DirectFFN


class NativeBinding:
    def __init__(self, mode, matrices, *, diagnostic=False, layers=40, expected_shape=None, verify_phase=None, on_site=None, residual_fn=None, event_diagnostic=False, execution='baseline', post_fn=None):
        self.core = DirectFFN(mode, matrices, layers=layers, expected_shape=expected_shape, residual_fn=residual_fn, diagnostic=event_diagnostic, execution=execution, post_fn=post_fn)
        self.verify_phase = verify_phase
        self.on_site = on_site
        self.diagnostic, self.layers = diagnostic, layers
        self.dispatch = None
        self.seen = set()
        self.shadow = []
        self.reference_cache = {}
        self.owner = None
        self.summary = None

    def install(self, owner):
        if self.owner is not None or 'infer_ffn' not in owner.__dict__:
            raise RuntimeError('unexpected binding')
        original = owner.infer_ffn
        binding = self
        def ffn(model, phase, x, attn_out, shift, scale, gate=None):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            expert = 'high' if binding.dispatch == 0 else 'low' if binding.dispatch == 1 else None
            if expert is None or gate is None or (step, layer) in binding.seen:
                raise RuntimeError('invalid FFN state')
            if binding.on_site: binding.on_site(model, phase, step, expert, layer)
            if binding.verify_phase: binding.verify_phase(phase)
            if layer == 0:
                binding.core.begin_step(step, expert)
            binding.seen.add((step, layer))
            # Shadow uses a separate native call at exactly the same input state;
            # it is deliberately absent from measured arms.
            if binding.diagnostic:
                shadow_x = x.clone()
                first = phase.ffn_0.apply
                present = 'apply' in phase.ffn_0.__dict__
                previous = phase.ffn_0.__dict__.get('apply')
                captured = []
                def capture(u):
                    captured.append(u.detach().clone())
                    return first(u)
                phase.ffn_0.apply = capture
                try:
                    native = original(model, phase, shadow_x, attn_out, shift, scale, gate)
                finally:
                    if present: phase.ffn_0.apply = previous
                    else: del phase.ffn_0.apply
                if len(captured) != 1: raise RuntimeError('native input capture failure')
            y = binding.core.infer_ffn(model, phase, x, attn_out, shift, scale, gate)
            if binding.diagnostic:
                u = captured[0]
                a = binding.core.matrices.get(expert, layer, u.device)
                q = u @ a
                if step in (0, 2):
                    binding.reference_cache[layer] = (native.float() - q.float()).to(torch.bfloat16)
                    reference = native
                else:
                    reference = (q.float() + binding.reference_cache[layer].float()).to(torch.bfloat16)
                if not torch.equal(reference, y):
                    raise RuntimeError('BF16 reference discrepancy')
                if step in (0, 2) and not torch.equal(native, y):
                    raise RuntimeError('native anchor discrepancy')
                binding.shadow.append({'step': step, 'layer': layer,
                    'finite': bool(torch.isfinite(y).all()),
                    'sse': float(((native.float() - y.float()) ** 2).sum()),
                    'native_energy': float((native.float() ** 2).sum())})
            if step == 3 and layer == binding.layers - 1:
                binding.summary = binding.core.finish()
                binding.reference_cache.clear()
            return y
        self.owner, self.original = owner, original
        owner.infer_ffn = ffn

    def finish(self):
        if self.seen != {(s, l) for s in range(4) for l in range(self.layers)} or self.summary is None:
            raise RuntimeError('incomplete 4-step FFN coverage')
        if self.diagnostic and len(self.shadow) != 4 * self.layers:
            raise RuntimeError('incomplete shadow coverage')
        return self.summary

    def uninstall(self):
        if self.owner is not None:
            self.owner.infer_ffn = self.original
            self.owner = None
        self.core.close_stream()
        self.core.slab = None
        if self.core.mode != 'dense':
            self.core.matrices.release()


class ExpertMatrices:
    """Copy verified CPU BF16 matrices to only the current expert's GPU cache."""
    def __init__(self, entries, sha256_file):
        self.entries, self.sha256_file = entries, sha256_file
        self.expert, self.gpu = None, {}
        self.setup_seconds = 0.0

    def activate(self, expert):
        self.release()
        self.expert = expert

    def get(self, expert, layer, device):
        if expert != self.expert:
            raise RuntimeError('expert mismatch')
        if layer not in self.gpu:
            import numpy as np
            start = time.monotonic()
            spec = self.entries[f'{expert}/{layer}']
            path = spec['path']
            if self.sha256_file(path) != spec['sha256']:
                raise RuntimeError('matrix hash mismatch')
            a = np.load(path, allow_pickle=False)
            if a.dtype != np.uint16 or list(a.shape) != spec['shape'] or spec['shape'] != [5120, 5120]:
                raise RuntimeError('matrix format mismatch')
            self.gpu[layer] = torch.from_numpy(a.copy()).view(torch.bfloat16).to(device)
            self.setup_seconds += time.monotonic() - start
        return self.gpu[layer]

    def release(self):
        self.gpu.clear()
        self.expert = None
