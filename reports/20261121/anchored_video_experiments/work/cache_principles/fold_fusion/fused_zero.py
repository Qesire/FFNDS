"""CAL-mean hit with exact BF16 GELU lookup and tiled coordinate packing."""
from functools import lru_cache

import torch
import torch.nn.functional as F

from fused_ops import HEADER, apply
from stateless_runtime import launch


SOURCE = """
extern "C" __global__ void activate(const B* z, const B* gelu, const float* beta,
    const float* scale, const B* mean_f, B* h, float* hf, float* x, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = i % k;
        B code = z[i], value = gelu[code];
        float hv = f32(value);
        h[i] = value;
        hf[i] = hv;
        float residual = __fsub_rn(hv, __fmul_rn(f32(code), beta[j]));
        x[i] = __fsub_rn(__fdiv_rn(residual, scale[j]), f32(mean_f[j]));
    }
}
extern "C" __global__ void pack_coords(const float* hq, const float* uq, const float* mean_q,
    const float* cross, float* out, int n, int rank, int paid, float reciprocal_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int width = rank + paid, row = i / width, j = i % width;
        if (j < rank) {
            int offset = row*rank+j;
            out[i] = __fsub_rn(__fsub_rn(hq[offset],uq[offset]),mean_q[j]);
        } else {
            // torch.div_(scalar) multiplies by the reciprocal at the opmath precision.
            out[i] = __fmul_rn(cross[row*paid+j-rank],reciprocal_k);
        }
    }
}
"""


@lru_cache(None)
def prepare_fusion(device_index):
    with torch.cuda.device(device_index):
        codes = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
        table = F.gelu(codes.view(torch.bfloat16), approximate="tanh")
        src = HEADER + SOURCE
        return table, {name: torch.cuda._compile_kernel(src, name)
                       for name in ("activate", "pack_coords")}


def fused_hit(u, phase, t, probes, inv_metric, tile_rows=8192):
    if (not u.is_cuda or u.dtype != torch.bfloat16 or not u.is_contiguous()
            or probes.numel() != 64 or tile_rows <= 0):
        raise ValueError("Expected contiguous BF16 CUDA input, 64 observers and positive tile size")
    n, d = u.shape
    k, rank = t["zmap"].shape
    table, kernels = prepare_fusion(u.device.index)

    def factors(cur):
        z = F.linear(cur, t["w1k"], t["b1k"])
        h = torch.empty_like(z)
        hf = torch.empty_like(z, dtype=torch.float32)
        x = torch.empty_like(hf)
        apply(kernels["activate"], z.numel(), z, table, t["kernel_beta"], t["kernel_scale"],
              t["mean_f"], h, hf, x, z.numel(), k)
        del z
        hq = hf @ t["zmap"]
        del hf
        uq = cur.float() @ t["jmap"]
        linear = cur @ t["A0"]
        base = torch.mm(h, t["w2k"], out_dtype=torch.float32)
        launch(t["stateless_kernels"]["base"], base.numel(), linear, base,
               t["mean_c"], base.numel(), d)
        return base, hq, uq, x

    paid = u.index_select(0, probes)
    bp, hp, up, xp = factors(paid)
    qp = hp.sub_(up).sub_(t["mean_q"])
    measured = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(paid), approximate="tanh"))
    yd = (measured.float()-bp).double()
    v, qd = t["vd"], qp.double()
    chol, status = torch.linalg.cholesky_ex(((xp@xp.T)/k).double()+inv_metric)
    torch._assert_async((status==0).all(), "Probe covariance is not SPD")
    iq = torch.cholesky_solve(qd, chol)
    gain, status = torch.linalg.solve_ex((qd.T@iq)*t["vv"], (iq*(yd@v.T)).sum(0))
    torch._assert_async((status==0).all(), "Probe gains are not identifiable")
    coeff = torch.cholesky_solve(yd-(qd*gain)@v, chol)
    torch._assert_async(torch.isfinite(coeff).all() & torch.isfinite(gain).all(), "Nonfinite GLS")
    readout = torch.cat((gain.float()[:, None]*t["v"], .5*coeff.float()))
    out = torch.empty_like(u)
    for start in range(0, n, tile_rows):
        sl = slice(start, min(n, start+tile_rows))
        base, hq, uq, x = factors(u[sl])
        cross = x @ xp.T
        coords = torch.empty((len(x), rank+len(paid)), device=u.device, dtype=torch.float32)
        apply(kernels["pack_coords"], coords.numel(), hq, uq, t["mean_q"], cross, coords,
              coords.numel(), rank, len(paid), 1.0/k)
        out[sl].copy_(torch.addmm(base, coords, readout))
        del base, hq, uq, x, cross, coords
    out.index_copy_(0, probes, measured)
    return out
