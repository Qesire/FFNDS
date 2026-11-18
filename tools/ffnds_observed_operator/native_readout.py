"""Matched static/online Wan pre-gate readout helper."""

import math

import torch
import torch.nn.functional as F

from .observed_local import _adapter, _deployed, _phi, _support


class Readout:
    """A single site's frozen operands and Wan dense observer.

    The owner binds the active phase, provides row indices in current token order
    and releases the returned cache after use. This helper returns pre-gate output.
    """

    @torch.inference_mode()
    def __init__(self, A_bf16, params, phase):
        if A_bf16.ndim != 2 or A_bf16.shape[0] != A_bf16.shape[1] or A_bf16.dtype != torch.bfloat16:
            raise ValueError("Expected square BF16 row-operator A")
        self.a = A_bf16
        self.d = A_bf16.shape[0]
        device = A_bf16.device

        weight = phase.ffn_0._get_actual_weight()
        bias = phase.ffn_0._get_actual_bias()
        if (weight.ndim != 2 or weight.shape[0] != self.d or weight.dtype != torch.bfloat16
                or bias is None or bias.numel() != weight.shape[1] or bias.dtype != torch.bfloat16):
            raise ValueError("Unexpected actual Wan W1/bias geometry or dtype")
        indices = torch.as_tensor(params["indices"], dtype=torch.long, device=weight.device)
        if (indices.shape != (16,) or indices.unique().numel() != 16
                or int(indices.min()) < 0 or int(indices.max()) >= weight.shape[1]):
            raise ValueError("Expected 16 distinct frozen native channels")
        self.w1 = weight.T.index_select(0, indices).to(device)
        self.b1 = bias.reshape(-1).index_select(0, indices.to(bias.device)).to(device)
        self.first = phase.ffn_0.apply
        self.second = phase.ffn_2.apply

        def operand(key, dtype):
            value = torch.as_tensor(params[key], device=device, dtype=dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite readout operand {key}")
            return value

        rank = int(params["r_eff"])
        if not 1 <= rank <= 32:
            raise ValueError("Expected an effective CAL rank between 1 and 32")
        self.beta = operand("beta", torch.float32)
        self.scale = operand("scale", torch.float32)
        self.transform = operand("T_CAL", torch.float32)
        self.prior = operand("V_static", torch.bfloat16)
        gram = operand("Gamma_CAL", torch.float64)
        if (self.beta.shape != (16,) or self.scale.shape != (32,) or self.transform.shape != (32, rank)
                or self.prior.shape != (rank, self.d) or gram.shape != (rank, rank)
                or not (self.scale > 0).all() or not torch.equal(gram, gram.T)):
            raise ValueError("Invalid frozen CAL feature or readout geometry")
        if torch.linalg.cholesky_ex(gram).info.item() != 0:
            raise ValueError("Invalid CAL readout metric")
        self.gram = gram
        self.ridge = float(params["lambda"])
        self.tau = float(params["tau"])
        if self.ridge not in (0.01, 0.1, 1.0) or not math.isfinite(self.tau) or self.tau < 0.01:
            raise ValueError("Invalid frozen CAL ridge or tau")

    def _panel(self, value):
        if (value.ndim != 2 or value.shape[1] != self.d or value.dtype != torch.bfloat16
                or value.device != self.a.device):
            raise ValueError("Expected BF16 Wan pre-gate boundary panel")

    def _phi(self, u):
        z = F.linear(u, self.w1, self.b1)
        h = F.gelu(z, approximate="tanh")
        return _phi(z, h, self.beta, self.scale, self.transform)

    @torch.inference_mode()
    def anchor(self, u0, y0):
        """Build the two required caches from this arm's actual Dense anchor."""
        self._panel(u0)
        self._panel(y0)
        if u0.shape != y0.shape:
            raise ValueError("Readout anchor input and output must share their token rows")
        base0 = u0 @ self.a
        return {"C0": (y0.float() - base0.float()).bfloat16(), "Phi0": self._phi(u0)}

    @torch.inference_mode()
    def hit(self, u1, cache, probe_idx, adaptation, *, gate=None, query_idx):
        """Apply the same BF16 cache and projection path in both arms."""
        self._panel(u1)
        c0, phi0 = cache["C0"], cache["Phi0"]
        self._panel(c0)
        if (c0.shape != u1.shape or phi0.shape != (len(u1), self.prior.shape[0])
                or phi0.dtype != torch.float32 or phi0.device != u1.device):
            raise ValueError("Readout cache does not match current token panel")
        if not isinstance(adaptation, bool):
            raise TypeError("Select static or online with an explicit boolean")
        rows = torch.as_tensor(probe_idx, device=u1.device)
        if adaptation and (rows.dtype != torch.long or rows.shape != (64,) or rows.unique().numel() != 64
                or int(rows.min()) < 0 or int(rows.max()) >= len(u1)):
            raise ValueError("Expected 64 distinct fixed probe rows")

        base1 = u1 @ self.a
        base = (base1.float() + c0.float()).bfloat16()
        x_bf16 = (self._phi(u1) - phi0).bfloat16()
        value = self.prior
        info = {"adaptation": adaptation, "probe_calls": 0,
                "lambda": self.ridge, "tau": self.tau, "current_rank": None,
                "condition_number": None}
        if adaptation:
            hidden = F.gelu(self.first(u1.index_select(0, rows)), approximate="tanh")
            actual = self.second(hidden)
            self._panel(actual)
            if actual.shape != (64, self.d):
                raise ValueError("Wan observer must return exactly the 64 probe rows")
            base_probe = base.index_select(0, rows)
            residual = actual.float() - base_probe.float()
            x = x_bf16.double()
            support = _support(x, self.gram)
            value, condition = _adapter(x, rows, residual.double(), self.prior.double(),
                                        support, self.ridge, self.tau)
            rank = len(support[2])
            info.update(probe_calls=1, current_rank=rank, condition_number=condition,
                        probe_baseline=base_probe, probe_residual=residual)
        info.update(V=value.bfloat16(), prior=self.prior, base=base.index_select(0, query_idx))
        return _deployed(x_bf16, base, value), info
