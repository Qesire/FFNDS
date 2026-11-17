"""Q1r2 and Global-P64 pre-gate actions on the shared native Wan FFN hook."""

from pathlib import Path

import torch

from wan_ffn_port import NativeICCPort, dense


class ComparatorPort(NativeICCPort):
    def __init__(self, scratch, *, method, force_dense, q1_loader=None, gp64=None):
        if method not in ('Q1r2', 'Global-P64'):
            raise ValueError('invalid comparator')
        if not force_dense and (q1_loader if method == 'Q1r2' else gp64) is None:
            raise ValueError('missing bound comparator implementation or parameters')
        super().__init__(Path(scratch), force_dense=True)
        self.force_dense = force_dense
        self.method = method
        self.q1_loader, self.gp64 = q1_loader, gp64
        if not self.force_dense:
            self.scratch.mkdir()

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if (step, layer) != divmod(len(self.seen), 40) or u.shape != (31668, 5120) or u.dtype != torch.bfloat16:
            raise RuntimeError('unexpected native comparator coordinates')
        expert = 'high' if step < 2 else 'low'
        w1, w2 = phase.ffn_0.apply, phase.ffn_2.apply
        if self.force_dense:
            out = dense(u, w1, w2)[2]
            kind, selected = 'forced_dense', None
        elif step in (0, 2):
            _, h, out = dense(u, w1, w2)
            if self.method == 'Q1r2':
                a = self.q1_loader(expert, layer, u.device)
                residual = (out.float() - (u @ a).float()).to(torch.bfloat16)
                self._store(expert, layer, {'r': residual})
            else:
                self._store(expert, layer, {'h': h, 'y': out})
            kind, selected = 'anchor', None
        elif self.method == 'Q1r2':
            a = self.q1_loader(expert, layer, u.device)
            residual = self._load(expert, layer, 'r', u.device)
            out = ((u @ a).float() + residual.float()).to(torch.bfloat16)
            self._discard(expert, layer, 'r')
            kind, selected = 'hit', None
        else:
            out, selected = self._g64_hit(phase, u, expert, layer)
            kind = 'hit'
        if out.dtype != u.dtype or out.shape != u.shape:
            raise RuntimeError('pre-gate comparator output mismatch')
        self.seen.append({'step': step, 'layer': layer, 'expert': expert, 'kind': kind, 'selected_k': selected})
        return out

    def _g64_hit(self, phase, u, expert, layer):
        fixed_rows, score_rms, select, packed_pregate = self.gp64
        if fixed_rows.shape != (64,) or fixed_rows.dtype != torch.long:
            raise RuntimeError('invalid 64 genuine raster rows')
        fixed_rows = fixed_rows.to(u.device)
        old_h = self._load(expert, layer, 'h', u.device)
        old_y = self._load(expert, layer, 'y', u.device)
        measured = phase.ffn_0.apply(u.index_select(0, fixed_rows))
        scores = score_rms(measured, old_h.index_select(0, fixed_rows))
        indices = select(scores, 5120)
        if indices.shape != (5120,) or torch.unique(indices).numel() != 5120:
            raise RuntimeError('Global-P64 did not return exact-K unique channels')
        w1 = phase.ffn_0._get_actual_weight().T
        w2 = phase.ffn_2._get_actual_weight().T
        out = packed_pregate(u, w1, phase.ffn_0._get_actual_bias(), w2, old_h, old_y, indices)
        for name in ('h', 'y'):
            self._discard(expert, layer, name)
        return out, int(indices.numel())
