"""FFNDS four-step hit with a configurable token tile and an in-place FP32 readout."""
import torch
import torch.nn.functional as F


def make_hit(tile_rows=512, reuse_base=False):
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")

    def hit(u, phase, cached, t, probes, inv_metric):
        from exact_port import apply, lookup_secant
        if not u.is_contiguous() or not all(x.is_contiguous() for x in cached.values()):
            raise ValueError("FFNDS expects contiguous token rows")
        k = t["zmap"].shape[0]

        def operands(index):
            now = u[index]
            du = torch.empty_like(now, dtype=torch.float32)
            dub = torch.empty_like(now)
            apply(t["difference"], du.numel(), now, cached["u0"][index], dub, du, du.numel())
            linear = dub @ t["A0"]
            ju = du @ t["jmap"]
            z = F.linear(now, t["w1k"], t["b1k"])
            dh, dh32, x = lookup_secant(z, cached["z0k"][index], t["kernel_beta"], t["kernel_scale"], *t["lut"])
            q = (dh32 @ t["zmap"]).sub_(ju)
            base = torch.mm(dh, t["w2k"], out_dtype=torch.float32)
            apply(t["base"], base.numel(), cached["y0"][index], linear, base, base.numel())
            return base, q, x

        base, qp, xp = operands(probes)
        observed = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u.index_select(0,probes)),approximate="tanh"))
        target, qd, v = (observed.float() - base).double(), qp.double(), t["vd"]
        system = ((xp @ xp.T) / k).double() + inv_metric
        chol, info = torch.linalg.cholesky_ex(system)
        torch._assert_async((info == 0).all(), "Nonpositive probe covariance")
        iq = torch.cholesky_solve(qd, chol)
        gain, info = torch.linalg.solve_ex((qd.T @ iq) * t["vv"], (iq * (target @ v.T)).sum(0))
        torch._assert_async((info == 0).all(), "Singular probe gains")
        coeff = torch.cholesky_solve(target - (qd * gain) @ v, chol)
        torch._assert_async(torch.isfinite(coeff).all() & torch.isfinite(gain).all(), "Nonfinite readout")
        readout = torch.cat((gain.float()[:,None] * t["v"], 0.5 * coeff.float()))
        out = torch.empty_like(u)
        for start in range(0,len(u),tile_rows):
            ix = slice(start,start+tile_rows)
            b,q,x = operands(ix)
            design = torch.cat((q,(x @ xp.T) / k),dim=1)
            if reuse_base:
                # b is private temporary storage; the cache and observed fit never alias it.
                torch.addmm(b,design,readout,out=b)
                out[ix].copy_(b)
            else:
                out[ix].copy_(torch.addmm(b,design,readout))
        out.index_copy_(0,probes,observed)
        return out,None
    return hit
