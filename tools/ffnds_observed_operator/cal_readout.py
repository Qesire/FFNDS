"""Fit the frozen observed readout and check its arithmetic on CAL panels."""

import torch

from .local_data import equal_bits
from .observed_local import (
    RIDGES, _adapter, _base, _choose, _cpu, _deployed, _hidden, _metric, _phi,
    _prior, _scale, _structural, _support,
)


def _result(status, *, tuning, parameters=None):
    return {"status": status, "tuning": tuning, "parameters": parameters}


@torch.inference_mode()
def fit_cal(site, cal):
    """Fit deployable observed-readout parameters from CAL sources alone."""
    if len(cal) < 2:
        raise ValueError("Observed readout LOCO needs at least two CAL sources")
    if any(panel.get("split", "CAL") != "CAL" for panel in cal):
        raise ValueError("Observed readout producer accepts CAL panels only")
    keys = ("w1", "w2", "b1", "b2", "A")
    weights = tuple(site[key].double() for key in keys)
    cal64 = [_structural(panel, weights) for panel in cal]
    channels, pool, scores, rounds = _choose(cal, cal64)
    selection = {"candidate_pool": _cpu(pool), "scores": _cpu(scores), "rounds": rounds}
    if channels is None:
        return _result("FEATURE_POOL_LIMITED", tuning=selection)

    w1, b1 = site["w1"][channels], site["b1"][channels]
    narrow = [(_hidden(p["u0"], w1, b1), _hidden(p["u1"], w1, b1)) for p in cal]
    dz = torch.stack([z1.double()-z0.double() for (z0, _), (z1, _) in narrow])
    dh = torch.stack([h1.double()-h0.double() for (_, h0), (_, h1) in narrow])
    beta, scale, active, raw = _scale(dz, dh)
    scaled = (raw / scale).flatten(0, 1)
    eig, vec = torch.linalg.eigh(scaled.T @ scaled / len(scaled))
    transform = vec[:, eig > eig[-1] * 1e-10]
    if not transform.numel():
        return _result("FEATURE_SPACE_EMPTY", tuning=selection)
    beta32, scale32, transform32 = beta.float(), scale.float(), transform.float()
    pairs = [(_phi(*start, beta32, scale32, transform32), _phi(*end, beta32, scale32, transform32))
             for start, end in narrow]
    xs = [(p1-p0).bfloat16() for p0, p1 in pairs]
    xd = [x.double() for x in xs]
    gram = torch.stack([x.T @ x / len(x) for x in xd]).mean(0)
    lower, info = torch.linalg.cholesky_ex(gram)
    if int(info) != 0:
        return _result("FEATURE_GRAM_NOT_SPD", tuning=selection)
    math = [((torch.cat((z[:, channels], h[:, channels]-z[:, channels]*beta32.double()), -1)
              / scale32.double()) @ transform32.double()) for z, h, *_ in cal64]
    noise = torch.stack([torch.linalg.solve_triangular(lower, (x-exact).T, upper=False).square().mean()
                         for x, exact in zip(xd, math)]).mean()
    tau = max(0.01, float(noise.sqrt()))
    bases = [_base(site, panel) for panel in cal]
    residuals = [p["y1"].double() - base for p, (base, _) in zip(cal, bases)]
    gates = [p["gate"].double() for p in cal]
    supports = [_support(x, gram) for x in xd]
    ncal = len(cal)
    xx = torch.stack([g.square()[:, None, None] * (x.T @ x / len(x))
                      for x, g in zip(xd, gates)])
    xy = torch.stack([(x.T @ residual / len(x)) * g.square()
                      for x, residual, g in zip(xd, residuals, gates)])
    prior = _prior(xx, xy, gram).bfloat16()
    tuning_rows = []
    for s, panel in enumerate(cal):
        omitted = _prior(xx, xy, gram, omit=s).bfloat16().double()
        for f, fold in enumerate(panel["folds"]):
            p, q = fold["probe"], fold["query"]
            base = bases[s][0]
            rp = panel["probe_y1"][f].double() - base[p]
            for ridge in RIDGES:
                v, _ = _adapter(xd[s], p, rp, omitted, supports[s], ridge, tau)
                pred = _deployed(xs[s], base, v)[q]
                mse = _metric(pred, panel["y1"][q], panel["y0"][q], gates[s])["mse"]
                tuning_rows.append({"source": panel["source"], "fold": f+1, "lambda": ridge, "mse": mse})
    losses = {ridge: sum(row["mse"] for row in tuning_rows if row["lambda"] == ridge)
              / (ncal * 2) for ridge in RIDGES}
    ridge = min(RIDGES, key=lambda value: (losses[value], -value))
    tuning = {**selection, "loco": tuning_rows, "mean_mse": losses, "selected_lambda": ridge}
    params = {"indices": _cpu(channels), "beta": _cpu(beta32), "scale": _cpu(scale32),
              "inactive": _cpu(~active), "T_CAL": _cpu(transform32), "Gamma_CAL": _cpu(gram),
              "V_static": _cpu(prior), "tau": tau,
              "tau_energy": float(noise), "lambda": ridge, "r_eff": prior.shape[0]}
    witnesses = []
    for panel, x, xb, (base, _), geom in zip(cal, xd, xs, bases, supports, strict=True):
        disabled, _ = _adapter(x, None, None, prior.double(), geom, ridge, tau, adaptation=False)
        static = _deployed(xb, base, prior)
        zero = torch.zeros_like(xb) @ prior
        u0, u1, a = panel["u0"], panel["u1"], site["A"]
        cache = (panel["y0"].float() - torch.mm(u0, a).float()).bfloat16()
        base_ref = (torch.mm(u1, a).float() + cache.float()).bfloat16()
        features = []
        for u in (u0, u1):
            z = torch.nn.functional.linear(u, w1, b1)
            h = torch.nn.functional.gelu(z, approximate="tanh")
            features.append((torch.cat((z.float(), h.float() - beta32*z.float()), dim=1) / scale32) @ transform32)
        x_ref = (features[1] - features[0]).bfloat16()
        for fold, observed in zip(panel["folds"], panel["probe_y1"], strict=True):
            p = fold["probe"]
            online, condition = _adapter(x, p, observed.double()-base[p], prior.double(), geom, ridge, tau)
            out = _deployed(xb, base, online)
            ref = (base_ref.float() + torch.mm(x_ref, online.bfloat16()).float()).bfloat16()
            witnesses.append({
                "source": panel["source"], "fold": fold["fold"], "solve_condition": condition,
                "probe_ids": fold["probe_ids"], "support_rank": len(geom[2]),
                "disabled_matches_static_bitwise": equal_bits(static, _deployed(xb, base, disabled)),
                "static_reference_bitwise": equal_bits(
                    static, (base_ref.float() + torch.mm(x_ref, prior).float()).bfloat16()),
                "fixed_reference_bitwise": equal_bits(out, ref),
                "zero_delta_correction_is_zero": not bool(zero.count_nonzero()),
                "gate_witness_passed": site["gate_witness"](static, panel) and site["gate_witness"](out, panel),
                "outputs_finite": bool(torch.isfinite(static).all() and torch.isfinite(out).all()),
            })
    return {**_result("CAL_FIT_COMPLETE", tuning=tuning, parameters=params), "folds": witnesses}
