"""Deploy the reviewed 512-row, per-token anchored joint FFN on native Wan."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fused_ops import HEADER, apply
from lookup import call as lookup_secant, prepare_lookup
from native_joint import JointPort, hit, prep_hit
from resident import ResidentJoint
from wan_ffn_port import bind_native_ffn


TILE = 512
SITES = {f"{expert}{layer:02d}" for expert in ("high", "low") for layer in range(40)}
MODES = ("fast512", "resident512", "original512")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def site_loader(scope):
    manifest = scope["cal_manifest"]
    if sha(manifest["path"]) != manifest["sha256"]:
        raise ValueError("Unexpected frozen CAL manifest")
    record = json.loads(Path(manifest["path"]).read_bytes())
    if record["status"] != "PASS" or set(record["sites"]) != SITES or set(scope["active_sites"]) != SITES:
        raise ValueError("Expected both native experts and all forty layers")
    packed = {}
    for name, ref in record["sites"].items():
        path = Path(ref["path"])
        if path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"]:
            raise ValueError(f"Corrupt or stale CAL parameters at {name}")
        packed[name] = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    return lambda expert, layer: packed[f"{expert}{layer:02d}"]


@lru_cache(None)
def pointwise(device):
    from fused_ops import KERNELS
    # Preserve the original order of the three FP32 base summands. addmm still
    # receives the completed base, in the reviewed GEMM shape and data type.
    cpp = HEADER + """
        extern "C" __global__ void base(const B* y, const B* linear, float* selected, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) selected[i] = __fadd_rn(__fadd_rn(f32(y[i]), f32(linear[i])), selected[i]);
        }
    """
    with torch.cuda.device(device):
        return (torch.cuda._compile_kernel(HEADER + KERNELS["difference"], "difference"),
                torch.cuda._compile_kernel(cpp, "base"))


def prepare(phase, pack, device):
    t = prep_hit(phase, pack, device, pack["A0"].shape[0], 64)
    t["vd"] = t["v"].double()
    t["vv"] = t["vd"] @ t["vd"].T
    t["lut"] = prepare_lookup(device.index)
    t["difference"], t["base"] = pointwise(device.index)
    return t


def fast_hit(u, phase, cached, t, probes, inv_metric):
    k = t["zmap"].shape[0]
    if not u.is_contiguous() or not all(a.is_contiguous() for a in cached.values()):
        raise ValueError("Expected contiguous native and anchor rows")

    def operands(now, u0, y0, z0):
        du = torch.empty_like(now, dtype=torch.float32)
        dub = torch.empty_like(now)
        apply(t["difference"], du.numel(), now, u0, dub, du, du.numel())
        linear = dub @ t["A0"]
        ju = du @ t["jmap"]
        z = F.linear(now, t["w1k"], t["b1k"])
        dh, dh32, x = lookup_secant(z, z0, t["kernel_beta"], t["kernel_scale"], *t["lut"])
        q = (dh32 @ t["zmap"]).sub_(ju)
        base = torch.mm(dh, t["w2k"], out_dtype=torch.float32)
        apply(t["base"], base.numel(), y0, linear, base, base.numel())
        return base, q, x

    def at(i):
        return operands(u[i], cached["u0"][i], cached["y0"][i], cached["z0k"][i])

    base, qp, xp = at(probes)
    observed = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u.index_select(0, probes)), approximate="tanh"))
    target, qd, v = (observed.float() - base).double(), qp.double(), t["vd"]
    system = ((xp @ xp.T) / k).double() + inv_metric
    chol, info = torch.linalg.cholesky_ex(system)
    torch._assert_async((info == 0).all(), "Nonpositive probe covariance")
    iq = torch.cholesky_solve(qd, chol)
    gain, info = torch.linalg.solve_ex((qd.T @ iq) * t["vv"], (iq * (target @ v.T)).sum(0))
    torch._assert_async((info == 0).all(), "Singular probe gains")
    coeff = torch.cholesky_solve(target - (qd * gain) @ v, chol)
    torch._assert_async(torch.isfinite(coeff).all() & torch.isfinite(gain).all(), "Nonfinite readout")
    readout = torch.cat((gain.float()[:, None] * t["v"], 0.5 * coeff.float()))
    out = torch.empty_like(u)
    for start in range(0, len(u), TILE):
        sl = slice(start, start + TILE)
        b, q, x = at(sl)
        cross = (x @ xp.T) / k
        design = torch.cat((q, cross), dim=1)
        out[sl].copy_(torch.addmm(b, design, readout))
    out.index_copy_(0, probes, observed)
    return out, None


class ResidentPort(ResidentJoint):
    @contextmanager
    def bind(self, owner, transformer_sha):
        if self.mode == "fast512":
            with super().bind(owner, transformer_sha):
                yield self
        else:
            # The original post retains its native BF16 arithmetic and gate.
            native = owner.post_process
            def post(*args, **kwargs):
                result = native(*args, **kwargs)
                self.counters["gate_native"] += 1
                return result
            with bind_native_ffn(owner, self, exact_source_sha=transformer_sha):
                owner.post_process = post
                try:
                    yield self
                finally:
                    owner.post_process = native

    @contextmanager
    def trajectory(self):
        with super().trajectory():
            yield self
            self.counters.update(tile_rows=TILE, anchor_gpu_capacity_bytes=self.slots.bytes,
                                 anchor_d2h=0, anchor_h2d=0)


class ReviewedPort:
    def __init__(self, loader, scope):
        self.loader, self.scope = loader, scope
        self.active = None
        self.counters = {}

    @contextmanager
    def trajectory(self):
        if self.active is not None:
            raise RuntimeError("Concurrent reviewed trajectory")
        self.active = JointPort(self.loader, self.scope["observer_ids"], **self.scope["residency"])
        try:
            yield self
            result = self.active.finish()
            rows = result["sites"]
            self.counters = dict(sites=len(rows), anchors=sum(r["kind"] == "anchor" for r in rows),
                                 hits=sum(r["kind"] == "hit" for r in rows), new_packed_sites=0,
                                 gate_fused=0, gate_native=len(result["gate_witness"]), tile_rows=TILE,
                                 peak_anchor_bytes=result["resident_anchor_peak_bytes"],
                                 **result["anchor_transfer_bytes"])
        finally:
            self.active = None

    @contextmanager
    def bind(self, owner, transformer_sha):
        from run_video import bind_probe_gate
        if self.active is None:
            raise RuntimeError("Open reviewed trajectory before binding")
        with bind_native_ffn(owner, self.active, exact_source_sha=transformer_sha), \
             bind_probe_gate(owner, self.active, transformer_sha):
            yield self


def make_port(scope, mode):
    if mode not in MODES or scope["residency"]["tile_rows"] != TILE:
        raise ValueError("Unexpected backend or historical GEMM tile")
    loader = site_loader(scope)
    if mode == "original512":
        return ReviewedPort(loader, scope)
    if mode == "fast512":
        port = ResidentPort(loader, scope["observer_ids"], prepare, fast_hit)
    else:
        # Read the unmodified reviewed hit, including its original selected-data
        # setup and diagnostics. Only cross-step anchor storage changes.
        port = ResidentPort(loader, scope["observer_ids"], lambda phase, pack, dev: pack,
                            hit, prepared_budget=0)
    port.mode = mode
    return port
