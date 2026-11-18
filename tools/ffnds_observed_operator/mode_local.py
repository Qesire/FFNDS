"""Local CAL and DEV worker for the hidden-coefficient secant family."""

from contextlib import contextmanager
from time import perf_counter

import torch
import torch.nn.functional as F


RIDGES = (0.01, 0.1, 1.0)


@contextmanager
def full_fp32():
    """Use the production precision policy without changing the caller's policy."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def fit_hidden_response(dz, w2, residual, gate, *, max_iterations=80, tolerance=1e-5):
    """Solve one FP64 source normal equation; return the fit and a PCG receipt."""
    n = len(dz)
    gate2 = gate.square()
    diag = dz.square().mean(dim=0) * (gate2 @ w2.square())
    active = diag > 0
    rhs = (dz * ((residual * gate2) @ w2)).sum(dim=0) / n
    rhs = torch.where(active, rhs, 0)
    result = torch.zeros_like(rhs)
    norm_rhs = torch.linalg.vector_norm(rhs).item()
    if norm_rhs == 0:
        return result, {"status": "CONVERGED", "iterations": 0, "relative_residual": 0.0,
                        "active_channels": int(active.sum()), "breakdown": False}
    penalty = 0.001 * torch.where(active, diag.clamp_min(1e-12 * diag[active].mean()), 0)
    preconditioner = torch.where(active, diag + penalty, 1)

    def normal(vector):
        response = (dz * vector) @ w2.T
        adjoint = (dz * ((response * gate2) @ w2)).sum(dim=0) / n
        return adjoint + penalty * vector

    r = rhs.clone()
    p = r / preconditioner
    rz = r @ p
    iterations = 0
    breakdown = False
    for iterations in range(1, max_iterations + 1):
        ap = normal(p)
        curvature = (p @ ap).item()
        if not 0 < curvature < float("inf"):
            breakdown = True
            break
        step = rz / curvature
        result = result + step * p
        r = r - step * ap
        if (torch.linalg.vector_norm(r) / norm_rhs).item() <= tolerance:
            break
        z = r / preconditioner
        next_rz = r @ z
        p = z + (next_rz / rz) * p
        rz = next_rz
    relative = (torch.linalg.vector_norm(rhs - normal(result)) / norm_rhs).item()
    converged = relative <= tolerance and not breakdown
    return result, {"status": "CONVERGED" if converged else "FIT_NOT_CONVERGED",
                    "iterations": iterations, "relative_residual": relative if relative < float("inf") else None,
                    "active_channels": int(active.sum()), "breakdown": breakdown}


def _structural_fit(site, cal, *, hidden_fit=None):
    w1, w2, a = (site[key].double() for key in ("w1", "w2", "A"))
    b1, b2 = (site[key].double() if site.get(key) is not None else 0 for key in ("b1", "b2"))
    solutions, panels, receipts = [], [], []
    for pair in cal:
        u0, u1, gate = (pair[key].double() for key in ("u0", "u1", "gate"))
        z0 = F.gelu(u0 @ w1.T + b1, approximate="tanh")
        z1 = F.gelu(u1 @ w1.T + b1, approximate="tanh")
        dz = z1 - z0
        residual = (z1 @ w2.T + b2) - (z0 @ w2.T + b2) - (u1 - u0) @ a
        _sync(dz)
        start = perf_counter()
        if hidden_fit is None:
            b, receipt = fit_hidden_response(dz, w2, residual, gate)
        else:
            b, receipt = hidden_fit(pair["source"], dz, w2, residual, gate)
        _sync(dz)
        normal_flops = 4 * len(dz) * w2.shape[0] * w2.shape[1]
        receipt.update(source=pair["source"], elapsed_seconds=perf_counter() - start,
                       normal_flops_per_iteration=normal_flops,
                       normal_flops_at_iteration_cap=80 * normal_flops)
        solutions.append(b)
        panels.append((dz, gate))
        receipts.append(receipt)
    if any(row["status"] != "CONVERGED" for row in receipts):
        return None, receipts, None
    solutions = torch.stack(solutions)
    gram = solutions.new_zeros((len(cal), len(cal)))
    for dz, gate in panels:
        responses = torch.stack([((dz * b) @ w2.T * gate).flatten() for b in solutions])
        gram += responses @ responses.T / (len(dz) * len(cal))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    top = torch.nonzero(eigenvalues > eigenvalues[-1] * 1e-10).flatten().flip((0,))[:4]
    modes = (eigenvectors[:, top].T @ solutions) / eigenvalues[top].sqrt()[:, None]
    # Canonical signs keep independent artifacts comparable when a mode is isolated.
    signs = modes.gather(1, modes.abs().argmax(dim=1, keepdim=True)).sign()
    spectrum = {"eigenvalues_descending": eigenvalues.flip((0,)).tolist(), "q_eff": len(top),
                "tail_action_energy": float(eigenvalues.clamp_min(0).sum() - eigenvalues[top].sum())}
    return modes * signs, receipts, spectrum


def mode_templates(db, w1, w2, modes):
    """FP32 templates in the deployed BF16 displacement coordinates."""
    dz = db.float() @ w1.float().T
    return (dz[None, :, :] * modes.float()[:, None, :]) @ w2.float().T


def compile_matrix(a, w1, w2, modes, theta):
    alpha = theta.float() @ modes.float()
    return (a.float() + (w1.float().T * alpha) @ w2.float().T).bfloat16()


def mode_output(db, y0, current):
    increment = db @ current
    return (y0.float() + increment.float()).bfloat16()


def _equal_bits(actual, expected):
    return torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


def _reference_output(site, u0, u1, y0, modes, theta):
    """Rebuild the frozen arithmetic from raw operands, with no execution helpers."""
    f32, bf16 = torch.float32, torch.bfloat16
    db = torch.sub(u1.to(f32), u0.to(f32)).to(bf16)
    coeff = torch.matmul(theta.to(f32), modes.to(f32))
    weighted = torch.mul(site["w1"].to(f32).T, coeff)
    correction = torch.mm(weighted, site["w2"].to(f32).T)
    operator = torch.add(site["A"].to(f32), correction).to(bf16)
    increment = torch.mm(db, operator).to(bf16)
    return torch.add(y0.to(f32), increment.to(f32)).to(bf16)


def _deployed_hit(site, db, y0, modes, prior, *, adaptation, fit=None, frozen_matrix=None):
    """Use the same BF16 hit for online and explicitly disabled adaptation."""
    if adaptation:
        gram, rhs, metric, ridge = fit
        theta = _fit_online(gram, rhs, prior, metric, ridge)
        current = compile_matrix(site["A"], site["w1"], site["w2"], modes, theta)
    else:
        theta = prior
        current = frozen_matrix
    return mode_output(db, y0, current), theta


def _design(templates, gate):
    return (templates.double() * gate.double()).flatten(1).T.contiguous()


def _moments(templates, residual, gate):
    design = _design(templates, gate)
    return ((design.T @ design) / residual.shape[0],
            (design.T @ (residual.double() * gate.double()).flatten()) / residual.shape[0])


def _fit_online(gram, rhs, prior, metric, ridge):
    return prior + torch.linalg.solve(gram + ridge * metric, rhs - gram @ prior)


def _oracle(templates, residual, gate):
    design = _design(templates, gate)
    target = (residual.double() * gate.double()).flatten()
    q, r = torch.linalg.qr(design, mode="reduced")
    u, singular, vh = torch.linalg.svd(r, full_matrices=False)
    support = singular > singular[0] * 1e-10
    return vh[support].T @ ((u[:, support].T @ (q.T @ target)) / singular[support])


def _geometric_probe(templates, gate, metric, pool, token_ids, count):
    """Greedy grouped log-det gain over tokens, with CAL action whitening."""
    # Each candidate contributes multiple gated channel responses under one theta.
    t = templates[:, pool].double() * gate.double()
    grams = torch.einsum("jnd,knd->njk", t, t) / count
    chol = torch.linalg.cholesky(metric)
    left = torch.linalg.solve_triangular(chol, grams, upper=False)
    grams = torch.linalg.solve_triangular(chol, left.transpose(-1, -2), upper=False).transpose(-1, -2)
    info = torch.eye(len(metric), device=metric.device, dtype=metric.dtype)
    ids = torch.as_tensor(token_ids, device=pool.device)[pool]
    order = torch.argsort(ids)
    available = torch.ones(len(pool), dtype=torch.bool, device=pool.device)
    selected = []
    for _ in range(count):
        scores = torch.linalg.slogdet(info + grams).logabsdet - torch.linalg.slogdet(info).logabsdet
        scores = scores.masked_fill(~available, -float("inf"))
        tied = available & torch.isclose(scores, scores.max(), rtol=1e-12, atol=1e-14)
        index = order[torch.nonzero(tied[order]).flatten()[0]]
        selected.append(index)
        available[index] = False
        info = info + grams[index]
    return pool[torch.stack(selected)]


def _measure(output, y0, y1, gate):
    pred = (output.double() - y0.double()) * gate.double()
    true = (y1.double() - y0.double()) * gate.double()
    err = (pred - true).square().sum()
    energy = true.square().sum()
    norm = torch.linalg.vector_norm(pred)
    target_norm = energy.sqrt()
    return {"mse": float(err / true.numel()),
            "relative_risk": float(err / energy) if energy > 0 else None,
            "increment_cosine": float((pred * true).sum() / (norm * target_norm)) if norm > 0 and energy > 0 else None,
            "increment_norm_ratio": float(norm / target_norm) if energy > 0 else None}


def _prepare(pair, site, modes):
    u0, u1, y0 = (pair[key] for key in ("u0", "u1", "y0"))
    d32 = u1.float() - u0.float()
    db = d32.bfloat16()
    base_continuous = y0.double() + (db.float() @ site["A"].float()).double()
    templates = mode_templates(db, site["w1"], site["w2"], modes)
    return db, base_continuous, templates


def _probe_moments(db, anchor, observed, gate, site, modes):
    # The fit targets the affine template before compilation rounds the operator and increment.
    templates = mode_templates(db, site["w1"], site["w2"], modes)
    base = anchor.double() + (db.float() @ site["A"].float()).double()
    return _moments(templates, observed.double() - base, gate)


def _fit_prior(gram, rhs):
    # CAL prior penalty uses the same action Gram as the fitting rows.
    return torch.linalg.pinv(gram, hermitian=True, rtol=1e-10) @ rhs / (1 + 1e-6)


def _condition(gram):
    eigen = torch.linalg.eigvalsh(gram)
    support = eigen[eigen > eigen[-1] * 1e-10]
    return {"rank": len(support), "condition": float(support[-1] / support[0]) if len(support) else None}


def _query_output(panel, y0, rows, matrix):
    # Apply the same panel GEMM shape to every arm before slicing the query.
    return mode_output(panel, y0, matrix)[rows]


def _continuous(base, templates, theta):
    return base.double() + (theta @ templates.double().flatten(1)).reshape_as(base)


def _observer_output(output, reference, rows):
    if output.dtype != torch.bfloat16 or output.shape != reference[rows].shape:
        raise ValueError("mode observer must return BF16 pre-gate output for the requested rows")
    return output


@torch.inference_mode()
def process_site(site, cal, dev, *, structure=True, hidden_fit=None):
    """Fit CAL modes and evaluate deployed/continuous mode arms on disjoint DEV queries."""
    with full_fp32():
        return _process_site(site, cal, dev, structure=structure, hidden_fit=hidden_fit)


def _process_site(site, cal, dev, *, structure, hidden_fit=None):
    key, a, w1, w2 = (site[item] for item in ("key", "A", "w1", "w2"))
    if any(t.dtype != torch.bfloat16 for t in (a, w1, w2)):
        raise ValueError("mode matrix and original FFN weights must be BF16")
    if not cal:
        raise ValueError("mode fit requires CAL sources")
    modes64, receipts, spectrum = _structural_fit(site, cal, hidden_fit=hidden_fit)
    result = {"site": key, "status": "FIT_NOT_CONVERGED", "pcg": receipts,
              "spectrum": spectrum, "parameters": None, "tuning": [], "rows": [],
              "acceptance": {"valid": False}}
    if modes64 is None:
        return result
    if not len(modes64):
        result["status"] = "NO_STRUCTURAL_SUPPORT"
        return result
    modes = modes64.float()
    source_moments = []
    for pair in cal:
        _, base, templates = _prepare(pair, site, modes)
        source_moments.append(_moments(templates, pair["y1"].float() - base, pair["gate"]))
    sum_gram = torch.stack([gram for gram, _ in source_moments]).sum(dim=0)
    sum_rhs = torch.stack([rhs for _, rhs in source_moments]).sum(dim=0)
    metric = sum_gram / len(cal)
    result["acceptance"]["cal_action_support"] = _condition(metric)
    if result["acceptance"]["cal_action_support"]["rank"] != len(modes):
        result["status"] = "DEGENERATE_DEPLOYED_MODE_GRAM"
        return result
    prior = _fit_prior(sum_gram, sum_rhs)
    scores = {ridge: [] for ridge in RIDGES}
    if len(cal) > 1:
        for pair, (gram, rhs) in zip(cal, source_moments, strict=True):
            held_in_gram = sum_gram - gram
            held_in_rhs = sum_rhs - rhs
            loo_prior = _fit_prior(held_in_gram, held_in_rhs)
            y0, y1, gate = (pair[item] for item in ("y0", "y1", "gate"))
            db = (pair["u1"].float() - pair["u0"].float()).bfloat16()
            source_errors = {ridge: [] for ridge in RIDGES}
            for fold_id, (fold, obs) in enumerate(zip(pair["folds"], pair["probe_y1"], strict=True), 1):
                probe, query = fold["probe"], fold["query"]
                obs = _observer_output(obs, y1, probe)
                pg, pr = _probe_moments(db[probe], y0[probe], obs, gate, site, modes)
                for ridge in RIDGES:
                    theta = _fit_online(pg, pr, loo_prior, metric, ridge)
                    out = _query_output(db, y0, query, compile_matrix(a, w1, w2, modes, theta))
                    row = _measure(out, y0[query], y1[query], gate)
                    source_errors[ridge].append(row["mse"])
                    result["tuning"].append(dict(row, source=pair["source"], site=key, fold=fold_id,
                                                 ridge=ridge, family="native_modes", split="CAL", cohort="CAL"))
            for ridge, errors in source_errors.items():
                scores[ridge].append(sum(errors) / len(errors))
    mean_scores = {ridge: sum(errors) / len(errors) for ridge, errors in scores.items() if errors}
    ridge = min(mean_scores, key=lambda item: (mean_scores[item], -item)) if mean_scores else RIDGES[-1]
    current_static = compile_matrix(a, w1, w2, modes, prior)
    result["parameters"] = {"modes": modes.tolist(), "theta_static": prior.tolist(),
                            "lambda_cal": metric.tolist(), "ridge": ridge, "q_eff": len(modes)}
    result["tuning_mean_mse"] = {str(value): score for value, score in mean_scores.items()}
    result["matrix_compile_cost"] = {"fp32_matmul_flops_per_compile": 2 * a.numel() * w1.shape[0],
                                     "cal_loco_compile_count": len(result["tuning"])}
    result["acceptance"]["tf32_allowed"] = torch.backends.cuda.matmul.allow_tf32
    for name in ("disabled_matches_static_bitwise", "static_reference_bitwise", "fixed_reference_bitwise",
                 "zero_delta_correction_is_zero"):
        result["acceptance"][name] = True if dev else None
    result["acceptance"]["gate_witness_passed"] = True if dev and site.get("gate_witness") else None
    for pair in dev:
        _evaluate_dev(pair, site, modes, modes64, prior, metric, ridge, current_static, result, structure)
    outputs_finite = all(torch.isfinite(torch.tensor(row["mse"], dtype=torch.float64)).item() for row in result["rows"])
    witnesses = ("disabled_matches_static_bitwise", "static_reference_bitwise", "fixed_reference_bitwise",
                 "zero_delta_correction_is_zero", "gate_witness_passed")
    result["acceptance"]["valid"] = (outputs_finite and not result["acceptance"]["tf32_allowed"]
                                      and all(result["acceptance"][name] is not False for name in witnesses))
    result["status"] = "OK" if result["acceptance"]["valid"] else "INVALID_BF16_EXECUTION"
    return result


def _evaluate_dev(pair, site, modes, modes64, prior, metric, ridge, current_static, result, structure):
    a, w1, w2 = (site[item] for item in ("A", "w1", "w2"))
    u0, u1, y0, y1, gate = (pair[item] for item in ("u0", "u1", "y0", "y1", "gate"))
    db, base, templates = _prepare(pair, site, modes)
    c0 = (y0.float() - (u0 @ a).float()).bfloat16()
    common_base = ((u1 @ a).float() + c0.float()).bfloat16()
    mode_base = mode_output(db, y0, a)
    static = mode_output(db, y0, current_static)
    disabled, _ = _deployed_hit(site, db, y0, modes, prior, adaptation=False, frozen_matrix=current_static)
    disabled_equal = _equal_bits(disabled, static)
    static_equal = _equal_bits(_reference_output(site, u0, u1, y0, modes, prior), static)
    result["acceptance"]["disabled_matches_static_bitwise"] &= disabled_equal
    result["acceptance"]["static_reference_bitwise"] &= static_equal
    witness = site.get("gate_witness")
    if witness is not None:
        result["acceptance"]["gate_witness_passed"] &= bool(witness(static, pair))
    zero_db = (u0.float() - u0.float()).bfloat16()
    zero_increment = zero_db @ current_static
    zero_output, _ = _deployed_hit(site, zero_db, y0, modes, prior, adaptation=False, frozen_matrix=current_static)
    zero_correction = bool((zero_increment == 0).all())
    result["acceptance"]["zero_delta_correction_is_zero"] &= zero_correction
    zero_gap = zero_output.double() - y0.double()
    zero_rms = float(zero_gap.square().mean().sqrt())
    zero_max = float(zero_gap.abs().max())
    exact_delta = u1.double() - u0.double()
    structural_templates = None
    if structure:
        w1_64, w2_64 = w1.double(), w2.double()
        z = exact_delta @ w1_64.T
        structural_templates = (z[None] * modes64[:, None]) @ w2_64.T
        b1, b2 = (site[key].double() if site.get(key) is not None else 0 for key in ("b1", "b2"))
        anchor64 = F.gelu(u0.double() @ w1_64.T + b1, approximate="tanh") @ w2_64.T + b2
        structural_base = anchor64 + exact_delta @ a.double()
    delta_rms = float(exact_delta.square().mean().sqrt())
    delta_rounding_rms = float((db.double() - exact_delta).square().mean().sqrt())
    for fold_id, (fold, fixed_obs) in enumerate(zip(pair["folds"], pair["probe_y1"], strict=True), 1):
        query, probe = fold["query"], fold["probe"]
        base_q, templates_q = base[query], templates[:, query]
        actual, anchor = y1[query], y0[query]
        rows = result["rows"]

        def record(arm, output, theta=None, **diagnostics):
            row = dict(_measure(output, anchor, actual, gate), family="native_modes", split="DEV", cohort="DEV",
                       source=pair["source"], site=site["key"], fold=fold_id, arm=arm, **diagnostics)
            if theta is not None:
                row["theta"] = theta.tolist()
                row["update_norm"] = float(torch.linalg.vector_norm(theta - prior))
                continuous = _continuous(base_q, templates_q, theta)
                row["compile_prediction_diff_rms"] = float(((output.double() - continuous) * gate).square().mean().sqrt())
                if structural_templates is not None:
                    corr = (theta @ structural_templates[:, query].flatten(1)).reshape_as(actual)
                    math_output = structural_base[query] + corr
                    row["math_fp64_mse_vs_capture"] = float(((math_output - actual.double()) * gate).square().mean())
                    row["math_fp64_diff_rms"] = float(((math_output - output.double()) * gate).square().mean().sqrt())
                    if site.get("direction"):
                        row["direction"] = site["direction"](pair, fold, arm, corr)
            rows.append(row)

        record("base", common_base[query])
        record("no_mode", mode_base[query], delta_rounding_rms=delta_rounding_rms, delta_rms=delta_rms)
        record("static", static[query], prior, disabled_matches_static_bitwise=disabled_equal,
               reference_bitwise=static_equal, zero_delta_correction_is_zero=zero_correction,
               zero_delta_anchor_rms=zero_rms, zero_delta_anchor_max_abs=zero_max)
        probe_arms = [("fixed", probe, fixed_obs)]
        if site.get("observe") is not None:
            geometry = _geometric_probe(templates, gate, metric, fold["pool"], pair["token_ids"], len(probe))
            probe_arms.append(("geometric", geometry, site["observe"](u1[geometry])))
        for arm, indices, observed in probe_arms:
            observed = _observer_output(observed, y1, indices)
            gram, rhs = _probe_moments(db[indices], y0[indices], observed, gate, site, modes)
            full_output, theta = _deployed_hit(site, db, y0, modes, prior, adaptation=True,
                                               fit=(gram, rhs, metric, ridge))
            reference_equal = None
            if arm == "fixed":
                reference_equal = _equal_bits(_reference_output(site, u0, u1, y0, modes, theta), full_output)
                result["acceptance"]["fixed_reference_bitwise"] &= reference_equal
                if witness is not None:
                    result["acceptance"]["gate_witness_passed"] &= bool(witness(full_output, pair))
            out = full_output[query]
            probe_id = torch.as_tensor(pair["token_ids"], device=indices.device)[indices]
            observer_diff = (observed.double() - y1[indices].double()) * gate
            record(arm, out, theta, ridge=ridge, probe_token_ids=probe_id.tolist(),
                   reference_bitwise=reference_equal, probe_support=_condition(gram),
                   observer_capture_diff_rms=float(observer_diff.square().mean().sqrt()))
        theta = _oracle(templates_q, actual.float() - base_q, gate)
        record("oracle_continuous", _continuous(base_q, templates_q, theta), theta)
        out = _query_output(db, y0, query, compile_matrix(a, w1, w2, modes, theta))
        record("oracle_compiled", out, theta)
