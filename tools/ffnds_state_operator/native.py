"""Six-site endpoint port on the strict Wan Q1 pre-gate binding."""

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import torch

from .bf16_replay import compile_endpoint, replay_hit, residual

SITES = {(expert, layer) for expert in ("high", "low") for layer in (4, 19, 34)}
ARMS = ("L0", "L+", "Q32", "N32")
SOURCES = ("vbi_0392", "vbi_1013", "vbi_0800")


def tensor_sha(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def native_w1(phase):
    """Actual LightX2V FFN matrix is in matmul layout, [d, m]."""
    weight = phase.ffn_0._get_actual_weight()
    bias = phase.ffn_0._get_actual_bias()
    if (weight.shape != (5120, 13824) or weight.dtype != torch.bfloat16
            or bias is None or bias.numel() != 13824 or bias.dtype != torch.bfloat16):
        raise RuntimeError("native FFN W1 or bias geometry changed")
    return weight.T, bias.reshape(-1)


def make_port(scratch, arm, q1_loader, site_loader):
    from wan_ffn_port import dense
    from wan_ffn_comparator_ports import ComparatorPort

    class StatePort(ComparatorPort):
        def __init__(self):
            super().__init__(scratch, method="Q1r2", force_dense=False, q1_loader=q1_loader)
            self.endpoints = {}
            self.gate_witness = []

        def __call__(self, model, phase, u):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            expert = "high" if step < 2 else "low"
            site = expert, layer
            if site not in SITES:
                return super().__call__(model, phase, u)
            if (step, layer) != divmod(len(self.seen), 40) or u.shape != (31668, 5120):
                raise RuntimeError("candidate native site geometry or ordering changed")
            if step in (0, 2):
                if site in self.endpoints:
                    raise RuntimeError("unexpected active endpoint on anchor")
                weight, bias = native_w1(phase) if arm == "N32" else (None, None)
                params, q1_math = site_loader(*site)
                endpoint = compile_endpoint(arm, params, q1_math, u.device, w1=weight, b1=bias)
                out = dense(u, phase.ffn_0.apply, phase.ffn_2.apply)[2]
                self._store(*site, {"r": residual(out, endpoint(u))})
                self.endpoints[site] = endpoint
                kind = "anchor"
            else:
                endpoint = self.endpoints.pop(site)
                out = replay_hit(endpoint(u), self._load(*site, "r", u.device))
                self._discard(*site, "r")
                kind = "hit"
            if out.shape != u.shape or out.dtype != torch.bfloat16:
                raise RuntimeError("candidate pre-gate output geometry changed")
            self.seen.append({"step": step, "layer": layer, "expert": expert, "kind": kind, "operator": arm})
            return out

        def finish(self):
            result = super().finish()
            wanted = {(step, layer) for step in range(4) for layer in (4, 19, 34)}
            observed = {(row["step"], row["layer"]) for row in self.gate_witness}
            if self.endpoints or observed != wanted or len(self.gate_witness) != len(wanted):
                raise RuntimeError("incomplete native gate or candidate cache lifecycle")
            return {**result, "gate_witness": self.gate_witness}

    return StatePort()


@contextmanager
def bind_gate_witness(owner, port, expected_sha, *, source_digest):
    """Observe actual native gate writes at the six candidate sites."""
    hooked_ffn, original_post = owner.infer_ffn, owner.post_process
    if source_digest(inspect.getfile(original_post)) != expected_sha:
        raise RuntimeError("native post_process source changed")
    pending = {}

    def observe_ffn(model, phase, x, attn, shift, scale, gate=None):
        y = hooked_ffn(model, phase, x, attn, shift, scale, gate)
        row = port.seen[-1]
        if "operator" in row:
            if pending or gate is None:
                raise RuntimeError("candidate FFN has an unconsumed gate witness")
            pending.update(model=model, gate=gate, y=y, row=row)
        return y

    def observe_post(model, x, y, gate, pre_infer_out=None):
        if not pending:
            return original_post(model, x, y, gate, pre_infer_out)
        if pending["model"] is not model or pending["y"] is not y or pending["gate"] is not gate:
            raise RuntimeError("native post_process did not receive current gate and pre-gate result")
        dtype = model.sensitive_layer_dtype if model.sensitive_layer_dtype != model.infer_dtype else x.dtype
        if gate.squeeze().shape != (y.shape[1],):
            raise RuntimeError("native gate ceased to be a channel vector")
        expected = torch.empty_like(x, dtype=dtype)
        for start in range(0, len(x), 1024):
            rows = slice(start, start + 1024)
            expected[rows] = x[rows].to(dtype) + y[rows].to(dtype) * gate.squeeze()
        actual = original_post(model, x, y, gate, pre_infer_out)
        as_int = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[dtype]
        if actual.dtype != dtype or not torch.equal(actual.view(as_int), expected.view(as_int)):
            raise RuntimeError("native residual write differs from one current-gate multiplication")
        port.gate_witness.append({
            **pending["row"], "gate_sha256": tensor_sha(gate),
            "gate_shape": list(gate.shape), "gate_dtype": str(gate.dtype),
            "pre_gate_sample_sha256": tensor_sha(y[:16]), "written_sample_sha256": tensor_sha(actual[:16]),
            "full_write_bitwise_equal": True,
        })
        pending.clear()
        return actual

    owner.infer_ffn, owner.post_process = observe_ffn, observe_post
    try:
        yield
        if pending:
            raise RuntimeError("final native candidate FFN did not consume its gate")
    finally:
        owner.infer_ffn, owner.post_process = hooked_ffn, original_post
