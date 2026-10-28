"""Weight-space output projection and finite-flow diagnostics; lazy numerical API.

No activation-trained basis, QDQ, new nonlinear feature or optimizer is used.
The caller supplies Torch and a predebited work context for every numeric unit.
"""
from contextlib import nullcontext
import math

P = {"D": 5120, "H": 13824, "head_outputs": 64, "rank": 512,
     "head_columns": 128, "eps": 1e-6, "geometry_rtol": 1e-9, "PSD_rtol": 1e-10}


def require(value, message):
    if not value: raise ValueError(message)


def mm(a, b, work):
    with work("FP64_stat_matrix_products"): return a @ b


def centered_head(torch, weight, scale, work=lambda _: nullcontext()):
    require(weight.dtype == scale.dtype == torch.bfloat16, "Native BF16 head and scale required")
    with work("geometry_BF16_gain_add"): gain = 1 + scale.reshape(-1)
    with work("geometry_head_build"):
        K = weight.double()*gain.double()[:, None]
        return K-K.mean(0, keepdim=True)


def fit_weight_basis(torch, raw_A, Ks, gates, work=lambda _: nullcontext(), p=P):
    """Raw A[H,D], fixed two-step head span128 plus weight-PCA384.

    Eigh/QR are paid library decompositions; their opaque internal arithmetic
    is not claimed as additional enumerated user-level GEMMs.
    """
    require(raw_A.dtype == torch.bfloat16 and tuple(raw_A.shape) == (p["H"], p["D"])
            and set(Ks) == set(gates) == {2, 3}, "Original low W2/two-step geometry required")
    require(2*p["head_outputs"] == p["head_columns"] < p["rank"] < p["D"], "Fixed head/PCA dimensions differ")
    with work("FP64_raw_W2_cast"): A = raw_A.double()
    with work("head_gated_columns"):
        columns = torch.cat([Ks[s]*gates[s].reshape(-1).double()[:, None] for s in (2, 3)], dim=1)
    with work("head_span_QR"): Q = torch.linalg.qr(columns, mode="reduced")[0]
    with work("weight_covariance_build"): covariance = mm(A.T, A, work)/p["H"]
    with work("head_complement_build"):
        complement = torch.eye(p["D"], dtype=A.dtype, device=A.device)-mm(Q, Q.T, work)
    with work("complement_weight_covariance"):
        C = mm(mm(complement, covariance, work), complement, work)
        C = (C+C.T)/2
    with work("weight_covariance_eigh"): eigenvalues, vectors = torch.linalg.eigh(C)
    take = p["rank"]-p["head_columns"]
    require(bool(torch.isfinite(eigenvalues).all()) and float(eigenvalues[-take]) > 0
            and float(eigenvalues[0]) >= -p["PSD_rtol"]*max(float(eigenvalues[-1]), 1e-30),
            "Weight covariance PSD/rank gate failed; no padding/search/retry")
    with work("final_basis_QR"):
        Phi = torch.linalg.qr(torch.cat((Q, vectors[:, -take:].flip(1)), dim=1), mode="reduced")[0]
    with work("basis_orthogonality_check"):
        error = mm(Phi.T, Phi, work)-torch.eye(p["rank"], dtype=A.dtype, device=A.device)
        orthogonality = float(error.abs().max())
    with work("head_span_check"):
        residual = columns-mm(Phi, mm(Phi.T, columns, work), work)
        leakage = float(residual.square().sum()/columns.square().sum())
    require(orthogonality <= p["geometry_rtol"] and leakage <= p["geometry_rtol"]**2,
            "FP64 weight-space geometry failed")
    with work("coefficient_projection"): B = mm(A, Phi, work)
    with work("BF16_projection_export_cast"):
        pack = {"B": B.bfloat16().contiguous(), "Phi": Phi.bfloat16().contiguous()}
    return pack, {"identity": "RAW_WEIGHT_SPACE_HEAD128_PLUS_WEIGHT_PCA384",
        "head_columns": p["head_columns"], "rank": p["rank"],
        "FP64_orthogonality_max_abs": orthogonality, "FP64_head_span_squared_leak": leakage,
        "smallest_eigenvalue": float(eigenvalues[0]), "retained_eigenvalue_min": float(eigenvalues[-take]),
        "activation_statistics_or_teacher_Y_used_for_basis": False,
        "BF16_export_exact_head_preservation_claimed": False}


def projected_W2(torch, hidden, pack, work=lambda _: nullcontext(), p=P):
    require(tuple(hidden.shape)[-1:] == (p["H"],) and hidden.dtype == torch.bfloat16
            and tuple(pack["B"].shape) == (p["H"], p["rank"])
            and tuple(pack["Phi"].shape) == (p["D"], p["rank"])
            and tuple(pack["b2"].shape) == (p["D"],)
            and all(t.dtype == hidden.dtype and t.device == hidden.device for t in pack.values()),
            "Static BF16 projection operands differ")
    with work("candidate_coefficient_GEMMs"): coefficients = torch.mm(hidden, pack["B"])
    with work("candidate_output_GEMMs"): result = torch.addmm(pack["b2"], coefficients, pack["Phi"].T)
    return result


def q_variance(torch, u, K, work=lambda _: nullcontext()):
    with work("FP64_q_variance_cases"):
        value = u.double(); q = mm(value, K, work)
        variance = (value-value.mean(1, keepdim=True)).square().mean(1)
        require(bool(torch.isfinite(q).all()) and bool(torch.isfinite(variance).all()), "Nonfinite native state statistics")
        return {"q": q, "variance": variance}


def head_terms(torch, head, target, stats, true_stats, work=lambda _: nullcontext(), eps=P["eps"]):
    with work("native_head_decomposition_cases"):
        q, qt = stats["q"], true_stats["q"]
        sigma = (stats["variance"]+eps).sqrt()[:, None]
        sigmat = (true_stats["variance"]+eps).sqrt()[:, None]
        direct = (q-qt)/sigmat; radial = q*(1/sigma-1/sigmat)
        continuous = direct+radial; actual = head.double()-target.double()
        remainder = actual-continuous
        energy = float(target.double().square().sum())
        require(energy > 0, "Degenerate native teacher head energy")
        squared = lambda x: float(x.square().sum())/energy
        result = {"head_target_energy": energy, "head_SSE": float(actual.square().sum()),
            "head_relative_SSE": squared(actual), "direct_relative_SSE": squared(direct),
            "radial_relative_SSE": squared(radial), "signed_cross_relative_SSE": float((2*direct*radial).sum())/energy,
            "continuous_relative_SSE": squared(continuous), "native_remainder_relative_SSE": squared(remainder),
            "native_signed_remainder_cross": float((2*continuous*remainder).sum())/energy,
            "q_SSE": float((q-qt).square().sum()),
            "variance_SSE": float((stats["variance"]-true_stats["variance"]).square().sum())}
        residual = result["head_relative_SSE"]-result["continuous_relative_SSE"]-result["native_remainder_relative_SSE"]-result["native_signed_remainder_cross"]
        require(abs(residual) <= 1e-9*max(1., result["head_relative_SSE"]), "Native head signed identity does not close")
        result["closure_residual"] = residual
        return result


def gated_Y_terms(torch, y, target, gate, work=lambda _: nullcontext()):
    with work("fullM_gated_Y_diagnostic_cases"):
        numerator = ((y.float()-target.float())*gate.float()).double().square().sum()
        denominator = (target.float()*gate.float()).double().square().sum()
        require(float(denominator) > 0, "Degenerate gated Y target")
        return {"gated_Y_SSE": float(numerator), "gated_Y_target_energy": float(denominator),
                "gated_Y_relative_SSE": float(numerator/denominator)}


def scheduler_residual(torch, before, flow, after, sigma, sigma_next, work=lambda _: nullcontext()):
    with work("native_scheduler_rounding_cases"):
        pre = before.float()-float(sigma)*flow.float()
        if sigma_next is not None: pre = pre+float(sigma_next)*flow.float()
        ideal = before.double()-(float(sigma)-float(sigma_next or 0))*flow.double()
        arithmetic = pre.double()-ideal; cast = after.double()-pre.double()
        require(torch.equal(pre.bfloat16().view(torch.uint8), after.view(torch.uint8)),
                "Original FP32 scheduler/BF16 cast reconstruction differs")
        return {"arithmetic": arithmetic, "cast": cast,
                "arithmetic_squared": float(arithmetic.square().sum()), "BF16_cast_squared": float(cast.square().sum())}


def four_corner(torch, DD, SD, DS, SS, work=lambda _: nullcontext()):
    with work("flow_four_corner_decompositions"):
        a = SD.double()-DD.double(); p = DS.double()-DD.double()
        c = SS.double()-DS.double()-a; total = SS.double()-DD.double()
        energy = float(DD.double().square().sum()); require(energy > 0, "Degenerate Dense flow")
        dot = lambda x, y: float((x*y).sum())
        terms = {"teacher_state_local": dot(a,a), "Dense_state_propagation": dot(p,p),
                 "approximation_state_change": dot(c,c), "local_state_cross": 2*dot(a,p),
                 "local_change_cross": 2*dot(a,c), "state_change_cross": 2*dot(p,c)}
        residual = dot(total,total)-sum(terms.values())
        require(abs(residual) <= 1e-9*max(1., dot(total,total)), "Four-corner signed identity does not close")
        slots = []
        for t in range(total.shape[1]):
            local = {"teacher_state_local": dot(a[:,t],a[:,t]), "Dense_state_propagation": dot(p[:,t],p[:,t]),
                     "approximation_state_change": dot(c[:,t],c[:,t]), "local_state_cross": 2*dot(a[:,t],p[:,t]),
                     "local_change_cross": 2*dot(a[:,t],c[:,t]), "state_change_cross": 2*dot(p[:,t],c[:,t])}
            slot_total = dot(total[:,t],total[:,t]); slot_residual = slot_total-sum(local.values())
            require(abs(slot_residual) <= 1e-9*max(1.,slot_total), "Time-slot signed identity does not close")
            slots.append({"slot":t,"Dense_target_energy":dot(DD.double()[:,t],DD.double()[:,t]),
                          "total_squared":slot_total,"terms":local,"closure_residual":slot_residual})
        return {"Dense_target_energy": energy, "total_squared": dot(total,total), "terms": terms,
                "relative_terms": {k:v/energy for k,v in terms.items()}, "closure_residual": residual,
                "time_slots": slots}
