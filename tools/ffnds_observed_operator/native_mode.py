"""Conditional native Wan mode hit using the frozen LOCAL6 numerical helpers."""

import torch
import torch.nn.functional as F

from .mode_local import (
    _fit_online,
    _moments,
    compile_matrix,
    full_fp32,
    mode_output,
    mode_templates,
)


class Mode:
    """One fixed expert/site, with full-BF16 anchor and pre-gate output."""

    def __init__(self, A_bf16, params, phase):
        self.phase = phase
        self.a = A_bf16.detach()
        self.w1 = phase.ffn_0._get_actual_weight().detach().T
        self.w2 = phase.ffn_2._get_actual_weight().detach().T
        d = self.a.shape[0]
        m = self.w1.shape[0]
        if (self.a.shape != (d, d) or self.w1.shape != (m, d) or self.w2.shape != (d, m)
                or any(t.dtype != torch.bfloat16 or t.device != self.a.device
                       for t in (self.a, self.w1, self.w2))):
            raise ValueError("Mode requires a BF16 row operator and resident original Wan FFN weights")
        self.modes = torch.as_tensor(params["modes"], dtype=torch.float32, device=self.a.device)
        self.prior = torch.as_tensor(params["theta_static"], dtype=torch.float64, device=self.a.device)
        self.metric = torch.as_tensor(params["lambda_cal"], dtype=torch.float64, device=self.a.device)
        self.ridge = float(params["ridge"])
        q = int(params["q_eff"])
        if (q not in (1, 2, 3, 4) or self.modes.shape != (q, m) or self.prior.shape != (q,)
                or self.metric.shape != (q, q) or self.ridge not in (0.01, 0.1, 1.0)
                or not all(bool(torch.isfinite(t).all()) for t in (self.modes, self.prior, self.metric))):
            raise ValueError("Invalid frozen mode parameters")
        if torch.linalg.cholesky_ex(self.metric).info.item() != 0:
            raise ValueError("Invalid CAL mode metric")

    def _check_panel(self, tensor):
        if tensor.ndim != 2 or tensor.shape[1] != self.a.shape[0]:
            raise ValueError("Mode requires [tokens, channels]")
        if tensor.dtype != torch.bfloat16 or tensor.device != self.a.device:
            raise ValueError("Mode panel must be BF16 and on the bound operator device")

    @torch.inference_mode()
    def anchor(self, u0, y0):
        """Cache the original pre-W1 input and pre-gate Dense output."""
        self._check_panel(u0)
        self._check_panel(y0)
        if u0.shape != y0.shape:
            raise ValueError("Mode anchor input/output rows must correspond")
        return {"u0": u0, "y0": y0}

    @torch.inference_mode()
    def hit(self, u1, cache, probe_idx, adaptation, *, gate=None, query_idx):
        """Return the BF16 pre-gate hit and evidence about its coefficient fit."""
        self._check_panel(u1)
        u0, y0 = cache["u0"], cache["y0"]
        if u1.shape != u0.shape:
            raise ValueError("Mode hit depends on the same anchor token layout")
        with full_fp32():
            db = (u1.float() - u0.float()).bfloat16()
            base = y0[query_idx].double() + (db[query_idx].float() @ self.a.float()).double()
            theta = self.prior
            info = {"adaptation": adaptation, "probe_calls": 0, "theta": theta,
                    "prior": self.prior, "lambda": self.ridge, "base": base,
                    "current_rank": None, "condition_number": None}
            if adaptation:
                theta, probe_info = self._adapt(u1, db, y0, gate, probe_idx)
                info.update(theta=theta, **probe_info)
            current = compile_matrix(self.a, self.w1, self.w2, self.modes, theta)
            return mode_output(db, y0, current), info

    def _adapt(self, u1, db, y0, gate, probe_idx):
        if gate is None or gate.numel() != u1.shape[1] or gate.device != u1.device:
            raise ValueError("Online modes require a current gate with one value per output channel")
        gate = gate.reshape(-1)
        if not bool(torch.isfinite(gate).all()):
            raise ValueError("Nonfinite current gate")
        if probe_idx is None:
            raise ValueError("Online modes require fixed probe IDs")
        probe_idx = torch.as_tensor(probe_idx, device=u1.device)
        if (probe_idx.shape != (64,) or probe_idx.dtype not in (torch.int32, torch.int64)
                or probe_idx.unique().numel() != 64 or bool(((probe_idx < 0) | (probe_idx >= len(u1))).any())):
            raise ValueError("Online modes require exactly 64 distinct in-range probe IDs")
        current_probe = u1.index_select(0, probe_idx)
        hidden = self.phase.ffn_0.apply(current_probe)
        observed = self.phase.ffn_2.apply(F.gelu(hidden, approximate="tanh"))
        if hidden.dtype != torch.bfloat16 or observed.dtype != torch.bfloat16 or observed.shape != current_probe.shape:
            raise ValueError("Original Wan probe observer must produce BF16 pre-gate output")
        dp = db.index_select(0, probe_idx)
        base = y0[probe_idx].double() + (dp.float() @ self.a.float()).double()
        residual = observed.double() - base
        templates = mode_templates(dp, self.w1, self.w2, self.modes)
        gram, rhs = _moments(templates, residual, gate)
        theta = _fit_online(gram, rhs, self.prior, self.metric, self.ridge)
        return theta, {"probe_calls": 1, "probe_gram": gram, "probe_rhs": rhs,
                       "probe_baseline": base, "probe_residual": residual,
                       "current_rank": int(torch.linalg.matrix_rank(gram, atol=0, rtol=1e-10)),
                       "condition_number": float(torch.linalg.cond(gram + self.ridge * self.metric))}
