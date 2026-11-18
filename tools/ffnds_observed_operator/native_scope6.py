"""Limit the readout and the historical Global-P64 action to six Wan sites."""

from contextlib import contextmanager
import os

import torch

from wan_ffn_comparator_ports import ComparatorPort
from wan_ffn_port import source_digest

from .native_full80 import Full80Port, L64_SITES


SOURCE = "vbi_0392"
ARMS = ("readout-static-6", "readout-online-6", "Global-P64-6")
GP64 = ARMS[-1]
GP64_K = 5120


class Scope6Port(Full80Port):
    """Use the existing, exact comparator/readout calls on a fixed per-site route."""

    def __init__(self, scratch, query_dir, arm, q1, load_readout, probe, query, weight_witness,
                 *, gp64=None, gp64_identity=None):
        if arm not in ARMS or (arm == GP64) != (gp64 is not None and gp64_identity is not None):
            raise ValueError("Invalid six-site arm or historical Global-P64 binding")
        inherited_arm = "readout-online-80" if arm == "readout-online-6" else "readout-static-80"
        super().__init__(scratch, query_dir, SOURCE, inherited_arm, load_readout or q1,
                         probe, query, weight_witness)
        self.arm = arm
        self.q1_loader = q1
        self.gp64 = gp64
        self.gp64_identity = gp64_identity

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        expert = "high" if step < 2 else "low"
        if (expert, layer) not in L64_SITES:
            out = ComparatorPort.__call__(self, model, phase, u)
            self.seen[-1].update(baseline="Q1r2", probe_calls=0)
            return out
        if self.arm != GP64:
            return super().__call__(model, phase, u)

        gate = self.current_gate
        if (gate is None or gate.device != u.device or gate.squeeze().shape != (u.shape[1],)
                or not bool(torch.isfinite(gate).all())):
            raise RuntimeError("Invalid current gate at a six-site Global-P64 position")
        key = f"{expert}{layer:02d}"
        identity = self.gp64_identity[key]
        self.weight_witness(expert, layer, phase, identity)
        self.method = "Global-P64"
        try:
            if step in (0, 2):
                out = ComparatorPort.__call__(self, model, phase, u)
                self.seen[-1].update(operator=self.arm, probe_calls=0)
            else:
                out = self._global_hit(model, phase, u, gate, key, identity)
        finally:
            self.method = "Q1r2"
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("Nonfinite six-site Global-P64 result")
        return out

    def _global_hit(self, model, phase, u, gate, key, identity):
        queries = self.query.to(u.device)
        with tapped_gp64(self, queries) as capture:
            # The historical handler still calls the same W1, score, selection, packing and cache helpers.
            out = ComparatorPort.__call__(self, model, phase, u)
        if set(capture) != {"probe_scores", "selected_channels", "anchor"}:
            raise RuntimeError("Incomplete historical Global-P64 call observation")
        row = self.seen[-1]
        if row["selected_k"] != GP64_K:
            raise RuntimeError("Historical Global-P64 K changed")
        path = self.query_dir / f"{key}.pt"
        with path.open("xb") as stream:
            torch.save({
                "schema": "FFNDS_OBSERVED_SCOPE6_QUERY_V1",
                "arm": self.arm, "source": self.source, "site": key, "step": row["step"], "identity": identity,
                "probe_ids": self.gp64[0].cpu(), "query_ids": self.query,
                "u1": u.index_select(0, queries).cpu(), "y_pred": out.index_select(0, queries).cpu(),
                "gate": gate.squeeze().contiguous().cpu(), "probe_calls": 1, "probe_kind": "native_w1_only",
                **capture,
            }, stream)
            stream.flush()
            os.fsync(stream.fileno())
        row.update(operator=self.arm, probe_calls=1, probe_kind="native_w1_only",
                   query={"path": str(path), "sha256": source_digest(path), "bytes": path.stat().st_size})
        return out

    def finish(self):
        result = ComparatorPort.finish(self)
        wanted = [(step, layer) for step in range(4) for layer in (4, 19, 34)]
        observed = [(row["step"], row["layer"]) for row in self.gate_witness]
        candidate = [row for row in self.seen if row.get("operator") == self.arm]
        strict = [row for row in self.seen if row.get("baseline") == "Q1r2"]
        hits = [row for row in candidate if row["kind"] == "hit"]
        expected_probes = 0 if self.arm == "readout-static-6" else 6
        if (observed != wanted or len(candidate) != 12 or len(strict) != 148 or len(hits) != 6
                or sum(row["probe_calls"] for row in candidate) != expected_probes
                or any("query" not in row for row in hits) or len(list(self.query_dir.glob("*.pt"))) != 6):
            raise RuntimeError("Incomplete six-site coverage, gate, observer, or query evidence")
        return {**result, "gate_witness": self.gate_witness}


@contextmanager
def tapped_gp64(port, queries):
    """Observe the historical score and pack calls without replacing their arithmetic."""
    original = port.gp64
    rows, score, select, packed = original
    receipt = {}

    def observe_score(current, previous):
        scores = score(current, previous)
        receipt["probe_scores"] = scores.detach().cpu().contiguous()
        return scores

    def observe_pack(u, w1, b1, w2, old_h, old_y, selected):
        value = packed(u, w1, b1, w2, old_h, old_y, selected)
        receipt["selected_channels"] = selected.detach().cpu().contiguous()
        receipt["anchor"] = {"h0": old_h.index_select(0, queries).cpu(),
                             "y0": old_y.index_select(0, queries).cpu()}
        return value

    port.gp64 = rows, observe_score, select, observe_pack
    try:
        yield receipt
    finally:
        port.gp64 = original
