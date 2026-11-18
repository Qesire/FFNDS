"""The matched full80 pre-gate Wan port and its query evidence capture."""

from contextlib import contextmanager
import math
import os
from pathlib import Path

import numpy as np
import torch

from ffnds_state_operator.native import bind_gate_witness
from wan_ffn_comparator_ports import ComparatorPort
from wan_ffn_port import dense, source_digest

from .native_mode import Mode
from .native_readout import Readout


N_TOKENS, WIDTH = 31668, 5120
DEV3 = ("vbi_0392", "vbi_1013", "vbi_0800")
SITES = {(expert, layer) for expert in ("high", "low") for layer in range(40)}
L64_SITES = {(expert, layer) for expert in ("high", "low") for layer in (4, 19, 34)}
ARMS = {f"{family}-{condition}-80": (operator, condition == "online")
        for family, operator in (("readout", Readout), ("modes", Mode))
        for condition in ("static", "online")}


def row_ids(canonical):
    canonical = np.asarray(canonical)
    if (canonical.shape != (64,) or canonical.dtype.kind not in "iu"
            or len(np.unique(canonical)) != 64 or canonical.min() < 0 or canonical.max() >= N_TOKENS):
        raise ValueError("Historical Global-P64 must bind exactly 64 valid raster IDs")
    ids = np.arange(N_TOKENS, dtype=np.int64)
    probe = np.random.default_rng(42).permutation(ids)[:64]
    eligible = np.setdiff1d(ids, np.union1d(probe, canonical))
    query = np.random.default_rng(42).permutation(eligible)[:64]
    return probe.tolist(), query.tolist()


class Full80Port(ComparatorPort):
    def __init__(self, scratch, query_dir, source, arm, site_loader, probe, query, weight_witness):
        super().__init__(scratch, method="Q1r2", force_dense=False, q1_loader=site_loader)
        self.arm = arm
        self.operator, self.online = ARMS[arm]
        self.source = source
        self.site_loader = site_loader
        self.weight_witness = weight_witness
        self.probe = torch.tensor(probe, dtype=torch.long)
        self.query = torch.tensor(query, dtype=torch.long)
        self.query_dir = Path(query_dir)
        self.query_dir.mkdir()
        self.gate_witness = []
        self.current_gate = None

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if ((step, layer) != divmod(len(self.seen), 40) or step >= 4
                or u.shape != (N_TOKENS, WIDTH) or u.dtype != torch.bfloat16):
            raise RuntimeError("Native full80 site ordering or pre-W1 geometry changed")
        gate = self.current_gate
        if (gate is None or gate.squeeze().shape != (WIDTH,) or gate.device != u.device
                or not bool(torch.isfinite(gate).all())):
            raise RuntimeError("Missing or invalid live Wan current gate")
        expert = "high" if step < 2 else "low"
        key = f"{expert}{layer:02d}"
        query = self.query.to(u.device)
        a, params, identity = self.site_loader(expert, layer, u.device)
        self.weight_witness(expert, layer, phase, identity)
        operator = self.operator(a, params, phase)
        row = {"step": step, "layer": layer, "expert": expert, "operator": self.arm, "probe_calls": 0}
        if step in (0, 2):
            out = dense(u, phase.ffn_0.apply, phase.ffn_2.apply)[2]
            cache = operator.anchor(u, out)
            if self.operator is Readout:
                cache["query_y0"] = out.index_select(0, query)
            self._store(expert, layer, cache)
            row["kind"] = "anchor"
        else:
            names = ("C0", "Phi0") if self.operator is Readout else ("u0", "y0")
            cache = {name: self._load(expert, layer, name, u.device) for name in names}
            out, info = operator.hit(u, cache, self.probe.to(u.device), self.online, gate=gate,
                                     query_idx=query)
            if info["probe_calls"] != int(self.online):
                raise RuntimeError("Unexpected native FFN observer count")
            anchor = {name: tensor.index_select(0, query) for name, tensor in cache.items()}
            if self.operator is Readout:
                anchor["y0"] = self._load(expert, layer, "query_y0", u.device)
                names = (*names, "query_y0")
            payload = {
                "schema": "FFNDS_OBSERVED_FULL80_QUERY_V1",
                "source": self.source, "arm": self.arm, "site": key, "step": step, "identity": identity,
                "query_ids": self.query, "probe_ids": self.probe, "u1": u.index_select(0, query),
                "y_pred": out.index_select(0, query), "gate": gate.squeeze(), "anchor": anchor, **info,
            }
            if (not all(bool(torch.isfinite(t).all()) for t in info.values() if isinstance(t, torch.Tensor))
                    or any(not math.isfinite(t) for t in info.values() if isinstance(t, float))):
                raise RuntimeError("Nonfinite full80 adaptation or audit state")
            path = self.query_dir / f"{key}.pt"
            with path.open("xb") as stream:
                cpu = {name: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                       for name, value in payload.items()}
                cpu["anchor"] = {name: value.detach().cpu().contiguous() for name, value in anchor.items()}
                torch.save(cpu, stream)
                stream.flush()
                os.fsync(stream.fileno())
            for name in names:
                self._discard(expert, layer, name)
            row.update(kind="hit", probe_calls=info["probe_calls"],
                       query={"path": str(path), "sha256": source_digest(path), "bytes": path.stat().st_size})
        if out.shape != u.shape or out.dtype != torch.bfloat16 or not bool(torch.isfinite(out).all()):
            raise RuntimeError("Invalid full80 pre-gate output")
        self.seen.append(row)
        return out

    def finish(self):
        result = super().finish()
        wanted = [(step, layer) for step in range(4) for layer in range(40)]
        actual = [(row["step"], row["layer"]) for row in self.gate_witness]
        hits = [row for row in self.seen if row["kind"] == "hit"]
        if (actual != wanted or len(hits) != 80 or sum(row["probe_calls"] for row in self.seen) != 80 * self.online
                or len(list(self.query_dir.glob("*.pt"))) != 80):
            raise RuntimeError("Incomplete full80 gates, probes, or query audit")
        return {**result, "gate_witness": self.gate_witness}


@contextmanager
def bind_full80_gate(owner, port, expected_sha):
    """Expose the call's gate before W1 and retain native post-process evidence."""
    with bind_gate_witness(owner, port, expected_sha, source_digest=source_digest):
        witnessed_ffn = owner.infer_ffn

        def with_gate(model, phase, x, attn, shift, scale, gate=None):
            if port.current_gate is not None:
                raise RuntimeError("Nested Wan FFN gate")
            port.current_gate = gate
            try:
                return witnessed_ffn(model, phase, x, attn, shift, scale, gate)
            finally:
                port.current_gate = None

        owner.infer_ffn = with_gate
        try:
            yield
        finally:
            owner.infer_ffn = witnessed_ffn
