"""Native Wan pre-gate implementation of the point/edge GLS half readout."""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def panel_bytes(parts):
    return sum(v.numel() * v.element_size() for v in parts.values())


def load_pack(root, expert, layer):
    return torch.load(Path(root) / expert / f"{layer:02d}.pt", map_location="cpu",
                      weights_only=True, mmap=True)


def probe_metric_inverse(probe, coords, device):
    """Match the deduplicated same-frame nearest-neighbour graph in the panel replay."""
    probe = np.asarray(probe, dtype=np.int64)
    coords = np.asarray(coords)
    pairs = set()
    for i, c in enumerate(coords):
        same = np.flatnonzero((coords[:, 0] == c[0]) & (np.arange(len(coords)) != i))
        if len(same):
            dist = ((coords[same, 1:] - c[1:]) ** 2).sum(1)
            j = int(same[np.lexsort((probe[same], dist))[0]])
            pairs.add(tuple(sorted((i,j))))
    eye = torch.eye(len(probe), dtype=torch.float64, device=device)
    if not pairs:
        return eye, 0
    i,j = np.array(sorted(pairs)).T
    diff = eye[i] - eye[j]
    metric = eye + (len(probe) / len(i)) * (diff.T @ diff)
    return torch.cholesky_inverse(torch.linalg.cholesky(metric)), len(i)


def anchor(u, phase, sel=None):
    """Use the native full row shape for W1/W2. GELU consumes W1's sole owned output."""
    z = phase.ffn_0.apply(u)
    z0k = z.index_select(1,sel) if sel is not None else None
    torch.ops.aten.gelu.out(z, approximate="tanh", out=z)
    y = phase.ffn_2.apply(z)
    return y, z0k


def prep_hit(phase, pack, device, width, observers):
    """Assemble one site's constants. Large live Wan weights keep their native owners."""
    sel = pack["sel"]
    k, rank = sel.numel(), pack["v"].shape[0]
    shapes = {"A0": (width,width), "zmap": (k,rank), "jmap": (width,rank),
              "v": (rank,width), "kernel_beta": (k,), "kernel_scale": (k,)}
    w1,w2 = (layer._get_actual_weight() for layer in (phase.ffn_0, phase.ffn_2))
    if (sel.ndim != 1 or sel.dtype != torch.long or w1.shape[0] != width or w2.shape != w1.T.shape
            or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16):
        raise ValueError("Invalid native BF16 Wan projection or frozen selection")
    tensors = {}
    for key,shape in shapes.items():
        dtype = torch.bfloat16 if key == "A0" else torch.float32
        if pack[key].shape != shape or pack[key].dtype != dtype:
            raise ValueError(f"Invalid exported {key} dtype or shape")
        tensors[key] = pack[key].to(device)
    b1 = phase.ffn_0._get_actual_bias().reshape(-1)
    tensors["w1k"] = w1.T.index_select(0,sel.to(w1.device)).contiguous().to(device)
    tensors["b1k"] = b1.index_select(0,sel.to(b1.device)).to(device)
    tensors["w2k"] = w2.index_select(0,sel.to(w2.device)).contiguous().to(device)
    return tensors


def hit(u, phase, cached, pack, probes, inv_metric, tile_rows=512):
    """Full-output correction from native observations, with only tile-sized workspaces."""
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    t = prep_hit(phase, pack, u.device, u.shape[1], len(probes))
    k,rank = t["zmap"].shape
    def operands(index):
        now = u[index]
        du = now.float() - cached["u0"][index].float()
        z1 = F.linear(now,t["w1k"],t["b1k"])
        z0 = cached["z0k"][index]
        dh = (F.gelu(z1,approximate="tanh").float() -
              F.gelu(z0,approximate="tanh").float()).bfloat16()
        dh32 = dh.float()
        x = (dh32 - (z1.float() - z0.float()) * t["kernel_beta"]) / t["kernel_scale"]
        q = dh32 @ t["zmap"] - du @ t["jmap"]
        # Keep the deployed BF16 dU cast AND the BF16 dU@A0 output rounding.
        linear = cached["y0"][index].float() + (du.bfloat16() @ t["A0"]).float()
        return dh, x, q, linear

    dhp,xp,qp,linear = operands(probes)
    # dh and selected native W2 are independently representable in BF16, while
    # the reference GEMM writes FP32. Keep this large GEMM on BF16 tensor cores
    # if the installed PyTorch/CUDA exposes the FP32-output path.
    w2k = t["w2k"]
    try:
        selected_p = torch.mm(dhp,w2k,out_dtype=torch.float32)
        native_out_dtype = True
    except (TypeError,NotImplementedError):
        w2k = w2k.float()
        selected_p = dhp.float() @ w2k
        native_out_dtype = False
    base_p = linear + selected_p
    z_obs = phase.ffn_0.apply(u.index_select(0,probes))
    observed = phase.ffn_2.apply(F.gelu(z_obs,approximate="tanh"))
    yp = (observed.float()-base_p).double()
    v, qd = t["v"].double(),qp.double()
    system = ((xp @ xp.T) / k).double() + inv_metric
    chol = torch.linalg.cholesky(system)
    siq = torch.cholesky_solve(qd,chol)
    gram = (qd.T @ siq) * (v @ v.T)
    gains = torch.linalg.solve(gram,(siq * (yp @ v.T)).sum(0))
    coeff = torch.cholesky_solve(yp - (qd * gains) @ v,chol)
    if not bool(torch.isfinite(coeff).all()) or not bool(torch.isfinite(gains).all()):
        raise RuntimeError("Non-finite current joint readout")
    # The native reference takes the small solve in FP64. Pack its result
    # once as FP32, and fold eta/gains into the corresponding output rows.
    readout = torch.cat((gains.float()[:,None] * t["v"],0.5 * coeff.float()))
    out = torch.empty_like(u)
    for start in range(0,len(u),tile_rows):
        sl = slice(start,start+tile_rows)
        dh,x,q,linear = operands(sl)
        cross = (x @ xp.T)/k
        selected = torch.mm(dh,w2k,out_dtype=torch.float32) if native_out_dtype else dh.float() @ w2k
        linear.add_(selected)
        design = torch.cat((q,cross),dim=1)
        out[sl] = torch.addmm(linear,design,readout).bfloat16()
    out.index_copy_(0,probes,observed)
    return out, {"probe_calls":1,"p":len(probes),"k":k,"rank":rank,
                 "selected_gemm":"bf16_to_fp32" if native_out_dtype else "fp32",
                 "tile_rows":tile_rows,"full_readout_writes":1,
                 "tiles":math.ceil(len(u)/tile_rows),"probe_exact":torch.equal(out[probes],observed),
                 "gain_min":float(gains.min()),"gain_max":float(gains.max())}


class JointPort:
    """One trajectory. Consume each native anchor once, with a bound on accelerator residency."""
    def __init__(self, site_loader, probe, *, active_sites=None, tile_rows=512,
                 gpu_cache_bytes=0, reserve_gpu_bytes=8*1024**3):
        self.site_loader = site_loader
        self.probe = np.asarray(probe,dtype=np.int64)
        if self.probe.shape != (64,) or len(set(self.probe)) != 64 or min(self.probe)<0 or max(self.probe)>=31668:
            raise ValueError("Invalid 64 full-sequence probe IDs")
        self.active = set(active_sites) if active_sites is not None else {
            (e,l) for e in ("high","low") for l in range(40)}
        self.tile_rows = tile_rows
        self.gpu_limit = gpu_cache_bytes
        self.reserve = reserve_gpu_bytes
        self.current_gate = None
        self.cache = {}
        self.seen, self.gate_witness, self.events = [], [], []
        self.peak = dict(cpu=0,cuda=0)
        self.live = dict(cpu=0,cuda=0)
        self.traffic = dict(anchor_d2h=0,anchor_h2d=0)
        self.cache_io = []
        self.on_gpu = None

    @torch.inference_mode()
    def __call__(self, model, phase, u):
        step,layer = int(model.scheduler.step_index),int(model.block_idx)
        if ((step,layer)!=divmod(len(self.seen),40) or step>=4 or u.shape!=(31668,5120)
                or u.dtype!=torch.bfloat16 or not u.is_cuda):
            raise RuntimeError("Unexpected native step, layer or input")
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("TF32 must be disabled for the joint FP32 contractions")
        if self.current_gate is None or self.current_gate.squeeze().shape!=(5120,):
            raise RuntimeError("Missing native pre-gate binding")
        if self.on_gpu is None:
            coords = np.column_stack(np.unravel_index(self.probe,(21,29,52)))
            inv, edges = probe_metric_inverse(self.probe,coords,u.device)
            self.on_gpu = (torch.tensor(self.probe,device=u.device),inv,edges)
        expert = "high" if step<2 else "low"
        key = (expert,layer)
        row = dict(step=step,layer=layer,expert=expert,probe_calls=0)
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        if key not in self.active:
            out,_ = anchor(u,phase)
            row["kind"] = "dense"
        elif step in (0,2):
            if key in self.cache:
                raise RuntimeError("Reused a live anchor site")
            pack = self.site_loader(expert,layer)
            out,z0k = anchor(u,phase,pack["sel"].to(u.device))
            items = {"u0":u,"y0":out,"z0k":z0k}
            size = panel_bytes(items)
            free_bytes,_ = torch.cuda.mem_get_info(u.device)
            keep_gpu = self.live["cuda"]+size<=self.gpu_limit and free_bytes>=self.reserve+size
            loc = "cuda" if keep_gpu else "cpu"
            # The Wan caller may reuse input and output storage after returning.
            if keep_gpu:
                stored = {"u0":u.clone(),"y0":out.clone(),"z0k":z0k}
            else:
                stored = {n:x.to("cpu") for n,x in items.items()}
                self.traffic["anchor_d2h"] += size
            self.cache[key] = (pack,stored,loc,size)
            self.live[loc] += size
            self.peak[loc] = max(self.peak[loc],self.live[loc])
            row.update(kind="anchor",operator="joint-gls-half",anchor_bytes=size,residency=loc)
        else:
            pack,stored,loc,size = self.cache.pop(key)
            self.live[loc] -= size
            if loc=="cpu":
                stored = {n:x.to(u.device) for n,x in stored.items()}
                self.traffic["anchor_h2d"] += size
            probes,inv,edges = self.on_gpu
            out,stats = hit(u,phase,stored,pack,probes,inv,self.tile_rows)
            row.update(kind="hit",operator="joint-gls-half",edges=edges,**stats)
        end.record()
        self.events.append((start,end))
        self.seen.append(row)
        return out

    def finish(self):
        if (len(self.seen)!=160 or self.cache or any(self.live.values())
                or sum(r["probe_calls"] for r in self.seen)!=len(self.active)
                or len(self.gate_witness)!=2*len(self.active)):
            raise RuntimeError("Incomplete native trajectory or outstanding cache/gate")
        torch.cuda.synchronize()
        for row,(start,end) in zip(self.seen,self.events):
            row["cuda_ms_including_cache"] = start.elapsed_time(end)
        self.cache_io = [dict(kind="resident_anchors",peak_bytes=self.peak,traffic_bytes=self.traffic,
                              tile_rows=self.tile_rows,probe_count=len(self.probe),disk_bytes=0,
                              peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                              peak_cuda_reserved=torch.cuda.max_memory_reserved())]
        return dict(sites=self.seen,cache_io=self.cache_io,gate_witness=self.gate_witness,
                    probe_ids=self.probe.tolist(),resident_anchor_peak_bytes=self.peak,
                    anchor_transfer_bytes=self.traffic,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved=torch.cuda.max_memory_reserved())
