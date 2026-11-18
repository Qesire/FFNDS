"""CAL-only producer and held-out local replay for the observed readout."""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .core import select_probes


RIDGES = (0.01, 0.1, 1.0)
FAMILY = "observed_readout"


def _cpu(value):
    return value.detach().cpu().tolist()


def _linear(u, w, b):
    return F.linear(u, w, b)


def _hidden(u, w, b):
    z = _linear(u, w, b)
    return z, F.gelu(z, approximate="tanh")


def _structural(panel, weights):
    w1, w2, b1, b2, a = weights
    u0, u1 = panel["u0"].double(), panel["u1"].double()
    z0, h0 = _hidden(u0, w1, b1)
    z1, h1 = _hidden(u1, w1, b1)
    y0, y1 = _linear(h0, w2, b2), _linear(h1, w2, b2)
    base = y0 + (u1 - u0) @ a
    return z1-z0, h1-h0, y0, y1, base


def _scale(dz, dh):
    denominator = dz.square().mean((0, 1))
    ratio = (dz * dh).mean((0, 1)) / denominator.clamp_min(torch.finfo(dz.dtype).tiny)
    beta = torch.where(denominator > 0, ratio, 0)
    raw = torch.cat((dz, dh - dz * beta), dim=-1)
    rms = raw.square().mean((0, 1)).sqrt()
    return beta, torch.where(rms > 0, rms, 1), rms > 0, raw


def _paired_qr(z, h):
    """FP64 pivoted, reorthogonalized QR with a two-dimensional SVD rank test.

    The closed-form left rotation of R gives the leading singular direction
    when the pair is numerically rank one. Full rank uses the same two-column
    span either way. This avoids per-channel factorization calls on CUDA.
    """
    nz = torch.linalg.vector_norm(z, dim=0)
    nh = torch.linalg.vector_norm(h, dim=0)
    swap = nh > nz
    first, second = torch.where(swap, h, z), torch.where(swap, z, h)
    a = torch.maximum(nz, nh)
    q0 = first / torch.where(a > 0, a, 1)
    b = (q0 * second).sum(dim=0)
    orthogonal = second - q0 * b
    leakage = (q0 * orthogonal).sum(dim=0)
    orthogonal = orthogonal - q0 * leakage
    b = b + leakage
    c = torch.linalg.vector_norm(orthogonal, dim=0)
    q1 = orthogonal / torch.where(c > 0, c, 1)

    # Normalize R=[[a,b],[0,c]] before evaluating the eigenvalues of R R^T.
    # Compute sigma_min via the determinant to avoid cancellation near rank one.
    scale = torch.where(a > 0, a, 1)
    b_scaled, c_scaled = b / scale, c / scale
    diagonal_gap = 1 + b_scaled.square() - c_scaled.square()
    off_diagonal = 2 * b_scaled * c_scaled
    radius = torch.hypot(diagonal_gap, off_diagonal)
    sigma_max_sq = (1 + b_scaled.square() + c_scaled.square() + radius) / 2
    keep_second = (a > 0) & (c_scaled > 1e-10 * sigma_max_sq)

    angle = torch.atan2(off_diagonal, diagonal_gap) / 2
    leading = q0 * angle.cos() + q1 * angle.sin()
    basis = torch.stack((torch.where(keep_second, q0, leading), q1), dim=-1)
    keep = torch.stack((a > 0, keep_second), dim=-1)
    return basis.permute(1, 2, 0), keep

def _screen(dz, dh, residuals, active):
    scores = dz.new_zeros(dz.shape[-1])
    for s, residual in enumerate(residuals):
        for start in range(0, len(scores), 64):
            end = start + 64
            q_transposed, keep = _paired_qr(dz[s, :, start:end], dh[s, :, start:end])
            projection = (q_transposed.reshape(-1, len(residual)) @ residual).reshape(len(keep), 2, -1)
            scores[start:end] += (projection.square().sum(-1) * keep).sum(-1) / len(residual)
    scores /= len(residuals)
    eligible = torch.where(active.reshape(2, -1).any(0))[0]
    order = torch.argsort(scores[eligible], descending=True, stable=True)
    return eligible[order[:64]], scores


def _choose(cal, structural):
    dz, dh = (torch.stack(items) for items in zip(*(value[:2] for value in structural)))
    _, scale, active, raw = _scale(dz, dh)
    dz, dh = (raw / scale).chunk(2, dim=-1)
    residuals = [(value[3] - value[4]) * panel["gate"] for panel, value in zip(cal, structural)]
    pool, scores = _screen(dz, dh, residuals, active)
    pool = pool.sort().values
    if len(pool) < 16:
        return None, pool, scores, []
    features = torch.stack((dz[..., pool], dh[..., pool]), dim=-1).flatten(-2)
    folds = []
    for s, (panel, residual) in enumerate(zip(cal, residuals)):
        for fold in panel["folds"]:
            p, q = fold["probe"], fold["query"]
            rp, rq = residual[p], residual[q]
            folds.append((s, p, q, rp @ rp.T, rq @ rp.T, rq.square().sum()))
    norms = features.transpose(1, 2) @ features / features.shape[1]
    gram_cal = norms.mean(0)
    selected, rounds = [], []
    for _ in range(16):
        remaining = [i for i in range(len(pool)) if i not in selected]
        cols = torch.tensor([selected + [i] for i in remaining], device=pool.device)
        cols = torch.stack((2 * cols, 2 * cols + 1), dim=-1).flatten(-2)
        gcal = gram_cal[cols[..., None], cols[:, None, :]]
        vals, vecs = torch.linalg.eigh(gcal)
        active_cal = vals > 1e-10 * vals[..., -1:]
        transform = vecs / torch.where(active_cal, vals, 1).sqrt().unsqueeze(1)
        transform = transform * active_cal.unsqueeze(1)
        losses = torch.zeros(len(remaining), device=pool.device, dtype=torch.float64)
        for s, p, q, pp, qp, qq in folds:
            data = features[s][:, cols].transpose(0, 1) @ transform
            u, singular, _ = torch.linalg.svd(data / np.sqrt(data.shape[1]), full_matrices=False)
            keep = (singular > 1e-12) & (singular > singular[..., :1] * 1e-10)
            coords = u * (singular * keep * np.sqrt(data.shape[1])).unsqueeze(1)
            cp, cq = coords[:, p], coords[:, q]
            system = cp.mT @ cp / len(p) + 0.1 * torch.diag_embed(singular.square() * keep + 0.01**2)
            reconstruction = cq @ torch.linalg.solve(system, cp.mT / len(p))
            losses += (qq - 2 * (reconstruction * qp).sum((1, 2))
                       + ((reconstruction @ pp) * reconstruction).sum((1, 2))) / len(q)
        which = int(losses.argmin())
        selected.append(remaining[which])
        rounds.append({"channel": int(pool[remaining[which]]), "query_mse": float(losses[which] / (len(folds) * residuals[0].shape[1]))})
    return pool[selected], pool, scores, rounds


def _phi(z, h, beta, scale, effective):
    return (torch.cat((z, h-z*beta), dim=-1) / scale) @ effective


def _support(x, gram):
    lower = torch.linalg.cholesky(gram)
    whitened = torch.linalg.solve_triangular(lower, x.T, upper=False).T
    _, singular, vh = torch.linalg.svd(whitened / np.sqrt(len(x)), full_matrices=False)
    keep = (singular > 1e-12) & (singular > singular[:1] * 1e-10)
    basis = torch.linalg.solve_triangular(lower.T, vh[keep].T, upper=True)
    return basis, x @ basis, singular[keep]


def _adapter(x, p, residual, prior, support, ridge, tau, *, adaptation=True):
    basis, coords, singular = support
    if not adaptation or len(singular) == 0:
        return prior, None
    design = coords[p]
    system = design.T @ design / len(p) + ridge * torch.diag(singular.square() + tau**2)
    rhs = design.T @ (residual - x[p] @ prior) / len(p)
    return prior + basis @ torch.linalg.solve(system, rhs), float(torch.linalg.cond(system))


def _deployed(x, base, v):
    return (base.float() + (x @ v.bfloat16()).float()).bfloat16()


def _base(site, panel):
    a = site["A"]
    base0, base1 = panel["u0"] @ a, panel["u1"] @ a
    cache = (panel["y0"].float() - base0.float()).bfloat16()
    base = (base1.float() + cache.float()).bfloat16()
    return base, (base0.float() + cache.float()).bfloat16()


def _prior(xx, xy, gram, omit=None):
    keep = [i for i in range(len(xx)) if i != omit]
    system = xx[keep].mean(0) + 1e-6 * gram
    return torch.linalg.solve(system, xy[keep].mean(0).T.unsqueeze(-1)).squeeze(-1).T


def _fixed_q32(xs, residuals, gates, gram, q32):
    r, k = gram.shape[0], q32.shape[1]
    normal = gram.new_zeros((r*k, r*k))
    rhs = gram.new_zeros((r, k))
    for x, residual, gate in zip(xs, residuals, gates):
        gated_q = q32 * gate[:, None]
        normal += torch.kron((x.T @ x / len(x)).contiguous(), gated_q.T @ gated_q)
        rhs += x.T @ ((residual * gate) @ gated_q) / len(x)
    normal /= len(xs)
    rhs /= len(xs)
    # Coefficients use the existing orthonormal Q32 coordinates.
    normal += 1e-6 * torch.kron(gram.contiguous(), torch.eye(k, device=gram.device, dtype=gram.dtype))
    return torch.linalg.solve(normal, rhs.flatten()).reshape(r, k) @ q32.T


def _metric(pred, target, anchor, gate):
    error = (pred.double() - target) * gate
    real = (target.double() - anchor) * gate
    delta = (pred.double() - anchor) * gate
    real_norm, pred_norm = real.norm(), delta.norm()
    return {
        "mse": float(error.square().mean()),
        "relative_risk": float(error.square().sum() / real_norm.square()) if real_norm else None,
        "increment_cosine": float((real * delta).sum() / (real_norm * pred_norm)) if real_norm and pred_norm else None,
        "increment_norm_ratio": float(pred_norm / real_norm) if real_norm else None,
    }


def _observe(site, u):
    if "observe" in site:
        return site["observe"](u)
    _, hidden = _hidden(u, site["w1"], site["b1"])
    return _linear(hidden, site["w2"], site["b2"])


def _report(status, **kwargs):
    return {"parameters": {}, "tuning": {}, "rows": [], "structure": [],
            "acceptance": {"status": status}, **kwargs}


def process_site(site, cal, dev, *, structure=True):
    """Select and fit on CAL, then score each common fold on captured labels."""
    if not cal:
        raise ValueError("observed readout needs CAL sources")
    keys = ("w1", "w2", "b1", "b2", "A")
    weights = tuple(site[key].double() for key in keys)
    cal64 = [_structural(panel, weights) for panel in cal]
    channels, pool, scores, rounds = _choose(cal, cal64)
    selection = {"candidate_pool": _cpu(pool), "scores": _cpu(scores), "rounds": rounds}
    if channels is None:
        return _report("FEATURE_POOL_LIMITED", tuning=selection)

    w1, b1 = site["w1"][channels], site["b1"][channels]
    panels = cal + dev
    narrow = [(_hidden(p["u0"], w1, b1), _hidden(p["u1"], w1, b1)) for p in panels]
    dz = torch.stack([z1.double()-z0.double() for (z0, _), (z1, _) in narrow[:len(cal)]])
    dh = torch.stack([h1.double()-h0.double() for (_, h0), (_, h1) in narrow[:len(cal)]])
    beta, scale, active, raw = _scale(dz, dh)
    scaled = (raw / scale).flatten(0, 1)
    eig, vec = torch.linalg.eigh(scaled.T @ scaled / len(scaled))
    transform = vec[:, eig > eig[-1] * 1e-10]
    if not transform.numel():
        return _report("FEATURE_SPACE_EMPTY", tuning=selection)
    beta32, scale32, transform32 = beta.float(), scale.float(), transform.float()
    pairs = [(_phi(*start, beta32, scale32, transform32), _phi(*end, beta32, scale32, transform32))
             for start, end in narrow]
    xs = [(p1-p0).bfloat16() for p0, p1 in pairs]
    xd = [x.double() for x in xs]
    gram = torch.stack([x.T @ x / len(x) for x in xd[:len(cal)]]).mean(0)
    lower, info = torch.linalg.cholesky_ex(gram)
    if int(info) != 0:
        return _report("FEATURE_GRAM_NOT_SPD", tuning=selection)
    math = [((torch.cat((z[:, channels], h[:, channels]-z[:, channels]*beta32.double()), -1)
              / scale32.double()) @ transform32.double()) for z, h, *_ in cal64]
    noise = torch.stack([torch.linalg.solve_triangular(lower, (x-exact).T, upper=False).square().mean()
                         for x, exact in zip(xd, math)]).mean()
    tau = max(0.01, float(noise.sqrt()))
    bases = [_base(site, panel) for panel in panels]
    residuals = [p["y1"].double() - base for p, (base, _) in zip(panels, bases)]
    gates = [p["gate"].double() for p in panels]
    supports = [_support(x, gram) for x in xd]
    ncal = len(cal)
    xx = torch.stack([g.square()[:, None, None] * (x.T @ x / len(x))
                      for x, g in zip(xd[:ncal], gates)])
    xy = torch.stack([(x.T @ residual / len(x)) * g.square()
                      for x, residual, g in zip(xd[:ncal], residuals, gates)])
    prior = _prior(xx, xy, gram).bfloat16()
    q32 = site["q32"].double()
    q32_prior = _fixed_q32(xd[:ncal], residuals[:ncal], gates[:ncal], gram, q32)
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
              "V_static": _cpu(prior), "V_q32": _cpu(q32_prior.bfloat16()), "tau": tau,
              "tau_energy": float(noise), "lambda": ridge, "r_eff": prior.shape[0]}

    rows, structural_rows, evidence = [], [], []
    for s, panel in enumerate(panels):
        cohort = "CAL" if s < ncal else "DEV"
        geometry = supports[s]
        x, xb = xd[s], xs[s]
        base, at_anchor = bases[s]
        gate = gates[s]
        static = _deployed(xb, base, prior)
        witness = site.get("gate_witness")
        static_gate = bool(witness(static, panel)) if witness else None
        disabled, _ = _adapter(x, None, None, prior.double(), geometry, ridge, tau, adaptation=False)
        zero = (pairs[s][0]-pairs[s][0]).bfloat16() @ prior
        support_rank = len(geometry[2])
        # Reconstruct the feature/cache boundary independently of the scoring helpers.
        reference_features = []
        for u in (panel["u0"], panel["u1"]):
            z = torch.nn.functional.linear(u, w1, b1)
            h = torch.nn.functional.gelu(z, approximate="tanh")
            paired = torch.cat((z.float(), h.float() - beta32 * z.float()), dim=1)
            reference_features.append((paired / scale32) @ transform32)
        reference_x = (reference_features[1] - reference_features[0]).to(torch.bfloat16)
        a = site["A"]
        a0 = torch.mm(panel["u0"], a).to(torch.bfloat16)
        a1 = torch.mm(panel["u1"], a).to(torch.bfloat16)
        cache = torch.sub(panel["y0"].float(), a0.float()).to(torch.bfloat16)
        reference_base = torch.add(a1.float(), cache.float()).to(torch.bfloat16)
        xq32 = _deployed(xb, base, q32_prior)
        math_panel = (cal64[s] if s < ncal else _structural(panel, weights)) if structure else None
        _, coords, singular = geometry
        for f, fold in enumerate(panel["folds"]):
            p, q, pool_rows = fold["probe"], fold["query"], fold["pool"]
            fixed = panel["probe_y1"][f]
            rp = fixed.double() - base[p]
            online, cond = _adapter(x, p, rp, prior.double(), geometry, ridge, tau)
            no_prior, _ = _adapter(x, p, rp, torch.zeros_like(prior, dtype=torch.float64), geometry, ridge, tau)
            whitened = _cpu(coords[pool_rows] / singular)
            pool_ids = torch.as_tensor(panel["token_ids"], device=x.device)[pool_rows]
            geometry_rows = select_probes(
                np.zeros((len(pool_rows), 0)), len(p), policy="geometry", token_ids=np.asarray(_cpu(pool_ids)),
                support=SimpleNamespace(whitened=np.asarray(whitened), rank=support_rank),
            )
            gp = pool_rows[torch.as_tensor(geometry_rows, device=x.device)]
            actual_geometry = _observe(site, panel["u1"][gp])
            geom, geom_cond = _adapter(x, gp, actual_geometry.double()-base[gp], prior.double(), geometry, ridge, tau)
            oracle = torch.linalg.pinv(x[q]) @ residuals[s][q]
            oracle_continuous = base[q].double() + x[q] @ oracle
            online_output = _deployed(xb, base, online)
            online_template = base[q].double() + x[q] @ online
            reference = torch.add(reference_base.float(), torch.mm(reference_x, online.to(torch.bfloat16)).float()).to(torch.bfloat16)
            identities = {"cohort": cohort, "split": cohort, "source": panel["source"], "site": site["key"], "fold": f+1, "family": FAMILY}
            outputs = {
                "base": base[q],
                "readout_q32": xq32[q],
                "readout_static": static[q],
                "readout_fixed64": online_output[q],
                "readout_geometry64": _deployed(xb, base, geom)[q],
                "readout_zero_prior": _deployed(xb, base, no_prior)[q],
                "readout_oracle_continuous": oracle_continuous,
                "readout_oracle_quantized": _deployed(xb, base, oracle)[q],
            }
            fixed_gap = _metric(fixed, panel["y1"][p], panel["y0"][p], gate)["mse"]
            geometry_gap = _metric(actual_geometry, panel["y1"][gp], panel["y0"][gp], gate)["mse"]
            static_mse = _metric(static[q], panel["y1"][q], panel["y0"][q], gate)["mse"]
            for arm, pred in outputs.items():
                row = _metric(pred, panel["y1"][q], panel["y0"][q], gate)
                rows.append({**identities, "arm": arm, **row,
                             "relative_static_gain": (static_mse-row["mse"]) / static_mse if static_mse else None,
                             "lambda": ridge, "tau": tau, "effective_rank": support_rank,
                             "condition_number": cond if arm == "readout_fixed64" else geom_cond if arm == "readout_geometry64" else None})
            evidence.append({
                **identities, "static_disabled_equal": bool(torch.equal(static, _deployed(xb, base, disabled))),
                "zero_delta_correction": bool(torch.count_nonzero(zero) == 0),
                "static_gate_equal": static_gate,
                "fixed64_gate_equal": bool(witness(online_output, panel)) if witness else None,
                "reference_equal": bool(torch.equal(reference, online_output)),
                "finite_output": bool(torch.isfinite(online_output).all() and torch.isfinite(fixed).all()
                                      and torch.isfinite(actual_geometry).all()),
                "output_dtype": str(online_output.dtype), "observer_dtype": str(fixed.dtype),
                "geometry_observer_dtype": str(actual_geometry.dtype),
                "feature_dtype": str(xb.dtype), "prior_dtype": str(prior.dtype),
                "endpoint_feature_dtype": str(pairs[s][0].dtype),
                "observer_rows": len(fixed), "geometry_observer_rows": len(actual_geometry),
                "fixed_full_m_probe_gap_mse": fixed_gap, "geometry_full_m_probe_gap_mse": geometry_gap,
                "anchor_cache_gap_mse": float(((at_anchor.double()-panel["y0"])*gate).square().mean()),
                "template_deployed_mse": float(((online_template-online_output[q])*gate).square().mean()),
                "update_norm": float((online - prior.double()).norm()),
                "fixed_probe_ids": _cpu(torch.as_tensor(panel["token_ids"], device=x.device)[p]),
                "geometry_probe_ids": _cpu(torch.as_tensor(panel["token_ids"], device=x.device)[gp]),
            })
            if math_panel is not None:
                structural_rows.extend(_direction_rows(site, panel, math_panel, q, p,
                                                       channels, beta32, scale32, transform32, gram, prior, ridge, tau, identities))
    status = "VALID" if all(item["static_disabled_equal"] and item["zero_delta_correction"]
                            and item["reference_equal"] and item["finite_output"] and item["observer_dtype"] == "torch.bfloat16"
                            and item["geometry_observer_dtype"] == "torch.bfloat16"
                            and item["feature_dtype"] == item["prior_dtype"] == item["output_dtype"] == "torch.bfloat16"
                            and item["endpoint_feature_dtype"] == "torch.float32"
                            and item["static_gate_equal"] is not False and item["fixed64_gate_equal"] is not False
                            and item["observer_rows"] == 64 and item["geometry_observer_rows"] == 64 for item in evidence) else "INVALID"
    return _report(status, parameters=params, tuning=tuning, rows=rows, structure=structural_rows,
                   acceptance={"status": status, "folds": evidence})


def _direction_rows(site, panel, math_panel, q, p, channels,
                    beta, scale, transform, gram, prior, ridge, tau, identities):
    dz, dh, y0, y1, base = math_panel
    dz, dh = dz[:, channels], dh[:, channels]
    x = (torch.cat((dz, dh-dz*beta.double()), -1) / scale.double()) @ transform.double()
    residual = y1 - base
    support = _support(x, gram)
    v, condition = _adapter(x, p, residual[p], prior.double(), support, ridge, tau)
    oracle = torch.linalg.pinv(x[q]) @ residual[q]
    coordinates = support[1]
    delta_target = residual - x @ prior.double()
    if coordinates.shape[1]:
        ideal = torch.linalg.pinv(coordinates, rtol=1e-10) @ delta_target
        representation_error = delta_target - coordinates @ ideal
        design = coordinates[p]
        damping = ridge * torch.diag(support[2].square() + tau**2)
        system = design.T @ design / len(p) + damping
        alias = -coordinates @ torch.linalg.solve(system, design.T @ representation_error[p] / len(p))
        shrinkage = coordinates @ torch.linalg.solve(system, damping @ ideal)
    else:
        representation_error = delta_target
        alias = shrinkage = torch.zeros_like(residual)
    gate = panel["gate"].double()
    fold = panel["folds"][identities["fold"]-1]
    rows = []
    for arm, correction in {
        "readout_static": x @ prior.double(),
        "readout_fixed64": x @ v,
        "readout_oracle_continuous": x[q] @ oracle,
    }.items():
        if arm != "readout_oracle_continuous":
            correction = correction[q]
        row = {**identities, "arm": arm, "numeric_path": "FP64 structural features and mathematical probe labels",
               **_metric(base[q] + correction, y1[q], y0[q], gate)}
        if arm == "readout_fixed64":
            reconstructed = representation_error + alias + shrinkage
            row.update({
                "condition_number": condition,
                "representation_mse": float((representation_error[q]*gate).square().mean()),
                "probe_alias_mse": float((alias[q]*gate).square().mean()),
                "shrinkage_mse": float((shrinkage[q]*gate).square().mean()),
                "decomposition_gap_max": float((reconstructed[q] - (residual[q]-correction)).abs().max()),
            })
        if "direction" in site:
            site["direction"](panel, fold, arm, correction)
        rows.append(row)
    return rows







