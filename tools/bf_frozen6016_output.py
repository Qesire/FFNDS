"""Fixed-feature factored FFN: explicit output geometry, no model imports.

The coordinator supplies Torch and the actual registered native Q2. Training
uses the immutable A0 student's own Q2 features on saved teacher inputs. This
offline feature cache does not make deployed Q1/Q2 static: every real call
still pays Q1, frozen W1, GELU, Smooth2/Q2, Bc and Phi. Only Bc's rank32 update
is learned; the two output spaces and common train mean are fixed first.
"""
from contextlib import nullcontext
import math

P = {"D": 5120, "width": 6016, "rank": 512, "head_outputs": 64,
     "rows": 256, "adapter_rank": 32, "updates": 256, "epochs": 8,
     "lr": .0005, "epsilon": 1e-6, "seed": 42,
     "geometry_rtol": 1e-9, "PSD_rtol": 1e-10}
BASIS_MODES = ("PCA512", "HEAD128_PCA384")
TRAIN = tuple(range(16))
HELD = tuple(range(16, 24))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_parameters(p=P):
    return p["adapter_rank"] * (p["width"] + p["rank"])


def operations(p=P):
    d, h, r = p["D"], p["width"], p["rank"]
    return {"A0_MAC_per_row": 2*d*h, "BF_MAC_per_row": d*h+r*(h+d),
            "BF_to_A0_MAC_ratio": (d*h+r*(h+d))/(2*d*h),
            "deployed_BF16_GEMM": 3, "deployed_Q1": 1, "deployed_Q2": 1,
            "intermediate_new_quantizer": 0, "online_adapter_GEMM": 0,
            "trainable_parameters_per_student": count_parameters(p)}


def factored_forward(torch, feature, Bc, bc, phi, mu, work=lambda *_: nullcontext()):
    with work("factored_forward_calls"):
        coefficients = torch.nn.functional.linear(feature, Bc, bc)
        return torch.nn.functional.linear(coefficients, phi, mu)


def deployed_forward(torch, post_Q1, pack, quantizer, work=lambda *_: nullcontext()):
    """Legal online graph from the already paid *current* post-Q1 operand.

    Q1 is supplied by the original caller. Own-width Q2 is recomputed after
    this invocation's frozen W1/GELU/Smooth2; teacher outputs and the offline
    training feature cache are absent from this API. All three matrices and
    both coefficient/output biases are static BF16 export tensors.
    """
    require(post_Q1.dtype == torch.bfloat16 and pack["W1"].shape[0] % 64 == 0,
            "Current BF16 post-Q1 and whole64 own hidden width required")
    require(all(pack[key].dtype == torch.bfloat16 for key in ("W1", "b1", "Bc", "bc", "Phi", "mu")),
            "Static exported BF16 weights/biases required")
    with work("deployed_W1_GEMM"):
        hidden = torch.nn.functional.linear(post_Q1, pack["W1"], pack["b1"])
    with work("deployed_GELU"):
        hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
    with work("deployed_Smooth2"):
        pre_Q2 = (hidden.float() / pack["smooth2"]).bfloat16()
    with work("deployed_dynamic_own_Q2"):
        feature = quantizer(pre_Q2, force_fp32=True).bfloat16()
    return factored_forward(torch, feature, pack["Bc"], pack["bc"], pack["Phi"], pack["mu"], work)


def native_write(torch, residual, y, gate, work=lambda *_: nullcontext()):
    require(residual.shape == y.shape and residual.dtype == y.dtype == gate.dtype == torch.bfloat16,
            "Original BF16 residual/output/gate required")
    with work("BF16_gate_multiply"):
        delta = y * gate
    with work("BF16_residual_add"):
        result = residual.clone()
        result.add_(delta)
    return result


def centered_head(torch, weight_KN, scale, work=lambda *_: nullcontext()):
    """Return K[D,64] for q=u@K, using the actual BF16 gain addition."""
    require(weight_KN.dtype == scale.dtype == torch.bfloat16, "Original head/gain BF16 required")
    with work("geometry_BF16_gain_add"):
        gain = 1 + scale.reshape(-1)
    with work("geometry_head_build"):
        matrix = weight_KN.double() * gain.double()[:, None]
        matrix = matrix - matrix.mean(0, keepdim=True)
    return matrix


def q_and_variance(torch, state, K, work=lambda *_: nullcontext()):
    value = state.double()
    with work("FP64_q_variance_cases"):
        q = value @ K
        centered = value - value.mean(1, keepdim=True)
        variance = centered.square().mean(1)
    require(bool(torch.isfinite(q).all()) and bool(torch.isfinite(variance).all())
            and bool((variance >= 0).all()), "Nonfinite q/variance")
    return {"q": q, "variance": variance}


def head_error_decomposition(torch, prediction, target, predicted_statistics, true_statistics,
                             work=lambda *_: nullcontext(), p=P):
    """Exact continuous difference plus its actual BF16-program remainder.

    e_q=(qhat-q)/sigma; e_v=qhat*(1/sigmahat-1/sigma).
    e_q+e_v=qhat/sigmahat-q/sigma. Cross terms are signed, and all
    components use the same native target-head energy denominator.
    """
    with work("head_error_decomposition_cases"):
        require(prediction.shape == target.shape and prediction.dtype == target.dtype == torch.bfloat16,
                "Same captured BF16 target/head shape required")
        q, v = true_statistics["q"], true_statistics["variance"]
        qhat, vhat = predicted_statistics["q"], predicted_statistics["variance"]
        sigma, sigmahat = (v+p["epsilon"]).sqrt(), (vhat+p["epsilon"]).sqrt()
        direct = (qhat-q)/sigma[:, None]
        radial = qhat*(sigmahat.reciprocal()-sigma.reciprocal())[:, None]
        continuous = direct+radial
        actual = prediction.double()-target.double()
        remainder = actual-continuous
        energy = float(target.double().square().sum())
        q_energy, v_energy = float(q.square().sum()), float(v.square().sum())
        require(energy > 0 and q_energy > 0 and v_energy > 0, "Degenerate head/statistic target energy")
        sse = lambda value: float(value.square().sum())
        a, b, c = sse(direct), sse(radial), float(2*(direct*radial).sum())
        cont, rem, native = sse(continuous), sse(remainder), sse(actual)
        rem_cross = float(2*(continuous*remainder).sum())
        require(abs(cont-a-b-c) <= 1e-9*max(1., cont)
                and abs(native-cont-rem-rem_cross) <= 1e-9*max(1., native),
                "Signed head error decomposition does not close")
        result = {"head_relative_SSE": native/energy,
                  "head_SSE": native, "head_target_energy": energy,
                  "direct_relative_SSE": a/energy, "radial_relative_SSE": b/energy,
                  "signed_cross_relative_SSE": c/energy,
                  "continuous_relative_SSE": cont/energy,
                  "BF16_remainder_relative_SSE": rem/energy,
                  "BF16_signed_remainder_cross": rem_cross/energy,
                  "q_relative_SSE": sse(qhat-q)/q_energy,
                  "variance_relative_SSE": sse(vhat-v)/v_energy}
        require(all(math.isfinite(x) for x in result.values()), "Nonfinite decomposition metric")
    return result


def training_cases(cases, p=P):
    selected = [c for c in cases if c["source_index"] in TRAIN]
    require(len(selected) == 32 and len({(c["source_index"], c["step"]) for c in selected}) == 32
            and {c["source_index"] for c in selected} == set(TRAIN)
            and len({c["step"] for c in selected}) == 2,
            "Exactly train16 x two expert steps required")
    require(all(c["Y"].shape == (p["rows"], p["D"]) and c["Y"].dtype == c["gate"].dtype
                and c["role"] == "TRAIN" for c in selected), "Equal original training rows/roles required")
    return sorted(selected, key=lambda c: (c["source_index"], c["step"]))


def fit_output_spaces(torch, cases, heads_by_step, p=P, work=lambda *_: nullcontext()):
    """Exact train covariance and fixed rank512 PCA/head-preserving spaces.

    The two-step head matrix has exactly128 columns. Reduced QR supplies a
    fixed128-dimensional completion even if its algebraic rank is smaller;
    this is not rank selection. Both final spaces have exactly512 columns.
    Heldout values are never read by this routine.
    """
    train = training_cases(cases, p)
    d, r, k = p["D"], p["rank"], 2*p["head_outputs"]
    require(k < r <= d, "Output rank must contain the fixed two-step head completion")
    device = train[0]["Y"].device
    with work("train_mean_fits"):
        mu = sum((case["Y"].double().mean(0) for case in train), torch.zeros(d, dtype=torch.float64, device=device))/len(train)
    covariance = torch.zeros((d,d), dtype=torch.float64, device=device)
    for case in train:
        with work("train_covariance_cases"):
            centered = case["Y"].double()-mu
            covariance.add_(centered.T @ centered, alpha=1/(len(train)*p["rows"]))
    covariance = (covariance+covariance.T)/2
    gates = {}
    with work("step_conditioning_identity_checks"):
        for step in sorted({c["step"] for c in train}):
            group = [c for c in train if c["step"] == step]
            require(all(torch.equal(group[0]["gate"], c["gate"]) for c in group),
                    "Training-source gate is not step-constant")
            gates[step] = group[0]["gate"].reshape(-1).double()
    columns = torch.cat([heads_by_step[s]*gates[s][:, None] for s in sorted(gates)], dim=1)
    with work("two_step_head_QR"):
        qh = torch.linalg.qr(columns, mode="reduced")[0]
    with work("head_projector_build"):
        complement = torch.eye(d, dtype=torch.float64, device=device)-qh@qh.T
    with work("complement_covariance_build"):
        residual = (complement@covariance)@complement
        residual = (residual+residual.T)/2
    result, evidence = {}, {}
    for mode, matrix in zip(BASIS_MODES, (covariance, residual)):
        with work("output_covariance_eigh"):
            eigenvalues, vectors = torch.linalg.eigh(matrix)
        require(bool(torch.isfinite(eigenvalues).all())
                and float(eigenvalues[0]) >= -p["PSD_rtol"]*max(float(eigenvalues[-1]), 1e-30),
                "Covariance eigenspectrum is not PSD within fixed tolerance")
        take = r if mode == BASIS_MODES[0] else r-k
        require(float(eigenvalues[-take]) > 0, "Insufficient positive covariance rank; no padding/search/retry")
        basis = vectors[:, -take:].flip(1)
        if mode == BASIS_MODES[1]:
            with work("final_head_space_QR"):
                basis = torch.linalg.qr(torch.cat((qh,basis), dim=1), mode="reduced")[0]
        with work("output_basis_orthogonality_checks"):
            gram = basis.T@basis
            orthogonality = float((gram-torch.eye(r,dtype=gram.dtype,device=device)).abs().max())
        require(orthogonality <= p["geometry_rtol"], "Output basis not orthonormal")
        leak = None
        if mode == BASIS_MODES[1]:
            with work("head_span_preservation_checks"):
                error = columns-basis@(basis.T@columns)
                leak = float(error.square().sum()/columns.square().sum())
            require(leak <= p["geometry_rtol"]**2, "Output basis lost the actual two-step head/gate span")
        result[mode] = basis
        evidence[mode] = {"rank": r, "head_completion_columns": k if mode == BASIS_MODES[1] else 0,
                          "orthogonality_max_abs": orthogonality, "head_span_relative_squared_leak": leak,
                          "covariance_smallest_eigenvalue": float(eigenvalues[0]),
                          "retained_eigenvalue_min": float(eigenvalues[-take]),
                          "source_indices_used": list(TRAIN), "rows_are_independent_N": False}
    return mu, result, evidence


class CoefficientAdapter:
    """Fixed Phi/mean/W1; rank32 Bc update merged into static BF16 export."""
    def __init__(self, torch, phi, mu, a0, p=P, work=lambda *_: nullcontext()):
        self.torch, self.p, self.work = torch, p, work
        require(phi.dtype == mu.dtype == torch.float64 and phi.shape == (p["D"],p["rank"]),
                "Fitted FP64 basis/common mean required")
        with work("coefficient_initialization"):
            affine = torch.cat((a0["W2"].double(), (a0["b2"].double()-mu)[:, None]), dim=1)
            projected = phi.T@affine
        self.phi, self.mu = phi.bfloat16().contiguous(), mu.bfloat16().contiguous()
        self.base = projected[:, :-1].bfloat16().contiguous()
        self.bias = projected[:, -1].bfloat16().contiguous()
        generator = torch.Generator(device="cpu").manual_seed(p["seed"])
        with work("adapter_random_draws"):
            a = torch.randn((p["adapter_rank"],p["width"]),generator=generator,dtype=torch.float32)/math.sqrt(p["width"])
        if a.device != self.base.device:
            with work("adapter_random_H2D_copies"), work("adapter_random_H2D_bytes", a.numel()*a.element_size()):
                a = a.to(self.base.device)
        self.weights = torch.nn.ParameterDict({"A": torch.nn.Parameter(a),
            "B": torch.nn.Parameter(torch.zeros((p["rank"],p["adapter_rank"]),dtype=torch.float32,device=self.base.device))})
        require(sum(v.numel() for v in self.weights.values()) == count_parameters(p), "Adapter parameter count differs")

    def parameters(self):
        return self.weights.parameters()

    def merged(self):
        with self.work("coefficient_merges"):
            return (self.base.float()+self.weights["B"]@self.weights["A"]).bfloat16()

    def forward(self, feature, merged):
        torch = self.torch
        require(feature.dtype == merged.dtype == torch.bfloat16
                and feature.shape == (self.p["rows"],self.p["width"]), "Actual own-Q2 feature shape/dtype differs")
        return factored_forward(torch, feature, merged, self.bias, self.phi, self.mu, self.work)


def oracle_projection(torch, y, phi, mu, gate, K, work=lambda *_: nullcontext()):
    with work("oracle_projection_cases"):
        projected = mu+((y.double()-mu)@phi)@phi.T
    with work("oracle_precast_q_preservation_cases"):
        delta_q = ((projected-y.double())*gate.double())@K
        denominator = (y.double()*gate.double()).square().sum()*K.square().sum()
        relative_q = float(delta_q.square().sum()/denominator) if float(denominator) > 0 else None
    return projected, relative_q


def gated_loss(torch, prediction, target, gate, work=lambda *_: nullcontext()):
    with work("gated_Y_loss_calls"):
        denominator = (target.float()*gate.float()).square().mean()
        require(bool(torch.isfinite(denominator)) and float(denominator) > 0, "Degenerate gated teacher Y")
        return ((prediction.float()-target.float())*gate.float()).square().mean()/denominator


def summarize(records, modes=BASIS_MODES):
    result = {}
    for phase in ("oracle", "initial", "final"):
        result[phase] = {}
        for mode in modes:
            rows = [x for x in records if x["phase"] == phase and x["basis"] == mode]
            for split, ids in (("train16",TRAIN),("heldout8_exposed",HELD)):
                per_source = {}
                for source in ids:
                    group = [x for x in rows if x["source_index"] == source]
                    require(len(group) == 2 and len({x["step"] for x in group}) == 2,
                            "Exactly two nested expert steps per source required")
                    keys = [k for k,v in group[0]["metrics"].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
                    per_source[str(source)] = {key: sum(x["metrics"][key] for x in group)/2 for key in keys}
                result[phase][mode+":"+split] = {"independent_sources":len(ids), "per_source":per_source,
                    "equal_source_mean": {key:sum(x[key] for x in per_source.values())/len(ids) for key in keys}}
    return result


def compare_source_heads(summary, parent):
    """Read-only same-source descriptive comparison to immutable A0/BH RUN."""
    result = {}
    for phase, groups in summary.items():
        result[phase] = {}
        for key, group in groups.items():
            split = key.split(":", 1)[1]
            reference = parent[split]["per_source"]
            require(set(reference) == set(group["per_source"]), "Parent/child independent source IDs differ")
            differences = {source: {arm: metrics["head_relative_SSE"]-reference[source][arm]
                for arm in ("BH_saved", "energy94", "random94")}
                for source,metrics in group["per_source"].items()}
            result[phase][key] = {"independent_sources":len(differences),
                "per_source_child_minus_parent":differences,
                "child_lower_source_count": {arm:sum(row[arm] < 0 for row in differences.values())
                    for arm in ("BH_saved", "energy94", "random94")},
                "identity":"Same original teacher-state/selected rows/head; nested expert steps source-equal; descriptive exposed sources"}
    return result
