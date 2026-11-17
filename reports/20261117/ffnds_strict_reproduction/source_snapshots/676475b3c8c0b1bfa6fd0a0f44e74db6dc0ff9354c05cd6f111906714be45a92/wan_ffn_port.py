"""Native PRE-W1 to PRE-gate FFN hook for the frozen LightX2V BF16 path."""

import ast
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch
import torch.nn.functional as F


def source_digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dense(u, w1, w2):
    z = w1(u)
    h = F.gelu(z, approximate='tanh')
    return z, h, w2(h)


class NativeICCPort:
    """Exact same normalization and gate as native; disk-backed expert-local cache."""

    def __init__(self, scratch, *, force_dense, factor_loader=None, cache_mode='4S'):
        if cache_mode not in ('4S', '2R') or (not force_dense and factor_loader is None):
            raise ValueError('invalid ICC mode/factors')
        self.cache_mode, self.force_dense = cache_mode, force_dense
        self.scratch = Path(scratch)
        self.factor_loader = factor_loader
        self.seen = []
        if not force_dense:
            self.scratch.mkdir()

    def _path(self, expert, layer, state):
        return self.scratch / f'{expert}_{layer:02d}_{state}.pt'

    def _store(self, expert, layer, states):
        for name, value in states.items():
            with self._path(expert, layer, name).open('xb') as f:
                torch.save(value.detach().to('cpu', copy=True).contiguous(), f)
                f.flush()
                os.fsync(f.fileno())

    def _load(self, expert, layer, name, device):
        path = self._path(expert, layer, name)
        return torch.load(path, map_location='cpu', mmap=True, weights_only=True).to(device)

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if (step, layer) != divmod(len(self.seen), 40):
            raise RuntimeError('unexpected ICC FFN site ordering')
        if tuple(u.shape) != (31668, 5120) or u.dtype != torch.bfloat16:
            raise RuntimeError('live pre-W1 coordinates or dtype changed')
        expert = 'high' if step < 2 else 'low'
        w1, w2 = phase.ffn_0.apply, phase.ffn_2.apply
        if self.force_dense:
            out = dense(u, w1, w2)[2]
            mode = 'forced_dense'
        elif step in (0, 2):
            z, h, out = dense(u, w1, w2)
            if self.cache_mode == '4S':
                self._store(expert, layer, {'u': u, 'z': z, 'h': h, 'y': out})
            else:
                first, second = self.factor_loader(expert, layer, u.device)
                self._store(expert, layer, {'r1': z - first.apply(u), 'r2': out - second.apply(h)})
            mode = 'anchor'
        else:
            first, second = self.factor_loader(expert, layer, u.device)

            def state(name):
                return self._load(expert, layer, name, u.device)

            if self.cache_mode == '4S':
                z = first.apply(u - state('u')) + state('z')
                h = F.gelu(z, approximate='tanh')
                out = second.apply(h - state('h')) + state('y')
                names = ('u', 'z', 'h', 'y')
            else:
                h = F.gelu(first.apply(u) + state('r1'), approximate='tanh')
                out = second.apply(h) + state('r2')
                names = ('r1', 'r2')
            for name in names:
                self._path(expert, layer, name).unlink()
            mode = 'hit'
        if out.dtype != u.dtype or out.shape != u.shape:
            raise RuntimeError('pre-gate result geometry or dtype changed')
        self.seen.append({'step': step, 'layer': layer, 'expert': expert, 'kind': mode})
        return out

    def finish(self):
        if len(self.seen) != 160 or (not self.force_dense and any(self.scratch.iterdir())):
            raise RuntimeError('incomplete ICC trajectory or unconsumed cache')
        return self.seen


@contextmanager
def bind_native_ffn(owner, action, *, exact_source_sha):
    """Preserve the installed inference preamble through normalized input; replace its FFN body."""
    original = owner.infer_ffn
    file = inspect.getfile(original)
    if source_digest(file) != exact_source_sha:
        raise RuntimeError('native FFN hook source changed')
    program = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = program.body[0]
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        raise RuntimeError('unexpected native FFN definition')
    ffns = [i for i, node in enumerate(function.body) if ast.unparse(node) == 'y = phase.ffn_0.apply(norm2_out)']
    if len(ffns) != 1:
        raise RuntimeError('could not locate real pre-W1 boundary')
    before_w1 = ffns[0]
    tail = function.body[before_w1:]
    if tuple(ast.unparse(node) for node in (tail[2], tail[-2], tail[-1])) != (
        "y = torch.nn.functional.gelu(y, approximate='tanh')", 'y = phase.ffn_2.apply(y)', 'return y',
    ):
        raise RuntimeError('unexpected native W1→GELU→W2 tail')
    function.body[before_w1:] = ast.parse('return _ffnds_action(self, phase, norm2_out)').body
    namespace = dict(original.__globals__, _ffnds_action=action)
    exec(compile(ast.fix_missing_locations(program), file, 'exec'), namespace)
    transformed = namespace[function.name]

    def checked(self, phase, x, attn, shift, scale, gate=None):
        if (self.clean_cuda_cache or gate is None or self._use_mxfp8_quant_fuse()
                or hasattr(phase, 'smooth_norm2_weight') or hasattr(phase, 'smooth_norm2_bias')):
            raise RuntimeError('nonreviewed native FFN configuration')
        return transformed(self, phase, x, attn, shift, scale, gate)

    owner.infer_ffn = checked
    try:
        yield
    finally:
        owner.infer_ffn = original
