"""ICC four-boundary statistics collector, independent of model scheduling.

Caller supplies actual native pre-W1 U, post-W1 Z, post-GELU H, pre-gate Y
in that order. CD anchor is losslessly written per layer and consumed/deleted
only after a complete, committed matching hit. No synthetic model inputs.
The FP32 token reduction is chunked; numerical agreement with author stat_upd
must be established by the separate author oracle before calibration use.
"""
import hashlib
import json
import os
from pathlib import Path

BOUNDARIES = ('u', 'z', 'h', 'y')


def write_json_exclusive(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())


class FourBoundaryCollector:
    """One source/expert and one family. A separate instance per transaction."""
    def __init__(self, torch, root, source, expert, family, *, tokens=31668,
                 d=5120, m=13824, layers=40, chunk_rows=256, identity=None):
        if family not in ('CA', 'CD') or expert not in ('high', 'low'):
            raise ValueError('invalid family or expert')
        if not source or min(tokens, d, m, layers, chunk_rows) <= 0 or not identity:
            raise ValueError('invalid dimensions/source or missing identity')
        self.torch, self.root = torch, Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.source, self.expert, self.family, self.identity = source, expert, family, dict(identity)
        self.tokens, self.d, self.m, self.layers, self.chunk_rows = tokens, d, m, layers, chunk_rows
        self.phase, self.layer, self.next_boundary = None, None, 0
        self.seen_anchor, self.seen_final = set(), set()
        self.values = {}

    def begin_site(self, phase, layer):
        if self.phase is not None or layer not in range(self.layers): raise RuntimeError('overlapping or invalid site')
        if phase not in (('CA',) if self.family == 'CA' else ('anchor', 'hit')):
            raise ValueError('invalid phase')
        seen = self.seen_anchor if phase == 'anchor' else self.seen_final
        if layer in seen or (phase == 'hit' and layer not in self.seen_anchor):
            raise RuntimeError('duplicate site or missing CD anchor')
        self.phase, self.layer, self.next_boundary = phase, layer, 0
        self.values = {}

    def observe(self, boundary, tensor):
        if self.phase is None or self.next_boundary >= 4 or boundary != BOUNDARIES[self.next_boundary]:
            raise RuntimeError('boundary order mismatch')
        width = self.d if boundary in ('u', 'y') else self.m
        if tensor.ndim != 2 or tuple(tensor.shape) != (self.tokens, width) or tensor.dtype != self.torch.bfloat16:
            raise ValueError('native BF16 boundary shape/dtype mismatch')
        path = self.root / f'anchor_{self.layer:02d}_{boundary}.pt'
        if self.phase == 'anchor':
            # D2H copy is complete before the native operation can mutate/reuse it.
            cpu = tensor.detach().to('cpu', copy=True).contiguous()
            for start in range(0, self.tokens, self.chunk_rows):
                if not self.torch.isfinite(cpu[start:start+self.chunk_rows]).all().item():
                    raise ValueError('nonfinite anchor')
            with path.open('xb') as f:
                self.torch.save(cpu, f); f.flush(); os.fsync(f.fileno())
            del cpu
        else:
            anchor = None
            if self.phase == 'hit':
                anchor = self.torch.load(str(path), map_location='cpu', mmap=True, weights_only=True)
                if anchor.shape != tensor.shape or anchor.dtype != tensor.dtype:
                    raise RuntimeError('anchor shape/dtype mismatch')
            total = self.torch.zeros(width, dtype=self.torch.float32)
            for start in range(0, self.tokens, self.chunk_rows):
                end = min(start + self.chunk_rows, self.tokens)
                x = tensor[start:end].detach().to(device='cpu', dtype=self.torch.float32, copy=True)
                if anchor is not None: x = x - anchor[start:end].float()
                if not self.torch.isfinite(x).all().item(): raise ValueError('nonfinite statistic input')
                total += x.abs().sum(dim=0)
            mag = total / self.tokens
            lo, hi = mag.min(), mag.max()
            normalized = (mag - lo + 1e-6) / (hi - lo + 1e-6)
            self.values[boundary] = {'magnitude': mag, 'normalized': normalized}
        self.next_boundary += 1

    def finish_site(self):
        if self.phase is None or self.next_boundary != 4: raise RuntimeError('incomplete four boundaries')
        layer, phase = self.layer, self.phase
        if phase == 'anchor':
            self.seen_anchor.add(layer)
        else:
            path = self.root / f'stats_{layer:02d}.pt'
            with path.open('xb') as f:
                self.torch.save(self.values, f); f.flush(); os.fsync(f.fileno())
            with path.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
            write_json_exclusive(self.root / f'stats_{layer:02d}.json', {
                'source': self.source, 'expert': self.expert, 'family': self.family, 'layer': layer,
                'tokens': self.tokens, 'identity': self.identity, 'stats_sha256': sha,
                'arithmetic': 'bf16 input; fp32 conversion before subtraction; chunked fp32 abs/sum; per-source minmax eps=1e-6',
                'author_oracle': 'UNVERIFIED_FOR_CHUNKED_REDUCTION'})
            self.seen_final.add(layer)
            if phase == 'hit':
                for boundary in BOUNDARIES: (self.root / f'anchor_{layer:02d}_{boundary}.pt').unlink()
        self.phase, self.layer, self.next_boundary, self.values = None, None, 0, {}

    def finish(self):
        if self.phase is not None or self.seen_final != set(range(self.layers)):
            raise RuntimeError('incomplete source/expert statistics')
        if self.family == 'CD' and self.seen_anchor != set(range(self.layers)):
            raise RuntimeError('incomplete CD anchor')
        write_json_exclusive(self.root / 'COMPLETE.json', {'source': self.source, 'expert': self.expert,
            'family': self.family, 'layers': self.layers, 'tokens': self.tokens, 'identity': self.identity,
            'chunked_reduction_oracle_status': 'UNVERIFIED'})

class FourBoundaryTap:
    """Candidate native call wrapper. Caller owns scheduler/expert/weight guards.

    `phase_name` is CA, anchor, or hit. This wrapper preserves native outputs and
    performs no gate operations. A failed transaction cannot be committed; use a
    fresh collector directory and repeat the approved indivisible CD pair.
    """
    def __init__(self, collector, phase_name, expected_step):
        self.collector, self.phase_name, self.expected_step = collector, phase_name, expected_step
        self.owner = self.original = None
        self.dispatch = None

    def install(self, owner):
        if self.owner is not None: raise RuntimeError('tap already installed')
        self.owner, self.original = owner, owner.infer_ffn
        tap, original = self, self.original
        def wrapped(model, phase, x, attn, shift, scale, gate=None):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            if step != tap.expected_step or tap.dispatch != tap.collector.expert:
                raise RuntimeError('step/expert differs from collector transaction')
            if gate is None: raise RuntimeError('missing native gate')
            tap.collector.begin_site(tap.phase_name, layer)
            fc1, fc2 = phase.ffn_0.apply, phase.ffn_2.apply
            count1 = count2 = 0
            def w1(u):
                nonlocal count1
                count1 += 1
                if count1 != 1: raise RuntimeError('duplicate W1')
                tap.collector.observe('u', u)
                z = fc1(u)
                tap.collector.observe('z', z)
                return z
            def w2(h):
                nonlocal count2
                count2 += 1
                if count2 != 1 or count1 != 1: raise RuntimeError('invalid W2 order')
                tap.collector.observe('h', h)
                y = fc2(h)
                tap.collector.observe('y', y)
                return y
            phase.ffn_0.apply, phase.ffn_2.apply = w1, w2
            try:
                out = original(model, phase, x, attn, shift, scale, gate)
                if count1 != 1 or count2 != 1: raise RuntimeError('missing native W1/W2')
                tap.collector.finish_site()
                return out
            finally:
                phase.ffn_0.apply, phase.ffn_2.apply = fc1, fc2
        owner.infer_ffn = wrapped

    def uninstall(self):
        if self.owner is not None:
            self.owner.infer_ffn = self.original
            self.owner = self.original = None
