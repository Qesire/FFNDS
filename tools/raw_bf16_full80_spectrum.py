"""Bias-anchored output spaces and actual BF16 factor/readout primitives.

No numerical package is imported until a caller supplies real tensor operands.
The sample-Gram spectrum is an ALG continuous surrogate, not a lower bound on
rounded native GEMMs. The same factor_forward is intended for later deployment.
"""
import math
import struct


def need(ok, message):
    if not ok:
        raise ValueError(message)


def invoke(meter, key, fn, *args, **kwargs):
    return fn(*args, **kwargs) if meter is None else meter.call(key, fn, *args, **kwargs)


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def metrics(gates):
    """Two source-static step gates; RMS is a surrogate, not the joint optimum."""
    import torch
    a, b = [g.to(torch.float64).abs() for g in gates]
    return {'shared': ((a.square()+b.square())/2).sqrt(), 'step0': a, 'step1': b, 'minimum': torch.minimum(a, b)}


def anchored_rows(cases, bias, metric, *, steps, source_indices):
    """Each source/step has equal relative-risk weight; b2 is never re-fitted."""
    import torch
    rows = []
    for source in source_indices:
        for step in steps:
            case = cases[source, step]
            y, gate = case['Y'].to(torch.float64), case['gate'].to(torch.float64)
            energy = (y*gate).square().sum()/y.shape[0]
            denominator = max(float(energy), 1e-30)
            scale = 1/math.sqrt(len(source_indices)*len(steps)*y.shape[0]*denominator)
            rows.append((y-bias.to(torch.float64))*metric*scale)
    return torch.cat(rows, dim=0)


def sample_gram(a, meter=None):
    return invoke(meter, 'FP64_sample_Gram_GEMM', lambda: a@a.T)


def source_subset(a, gram, source_indices, total_sources, rows_per_source, meter=None):
    """Principal Gram deletion; held source is excluded before eig/basis fit."""
    import torch
    ids = torch.tensor([i for s in source_indices for i in range(s*rows_per_source, (s+1)*rows_per_source)], dtype=torch.int64, device=a.device)
    scale = math.sqrt(total_sources/len(source_indices))
    return a.index_select(0, ids)*scale, gram.index_select(0, ids).index_select(1, ids)*(scale*scale)


def fit_basis(a, gram, rank, meter=None):
    import torch
    need(a.dtype == gram.dtype == torch.float64 and gram.shape == (a.shape[0], a.shape[0]), 'FP64 sample-Gram geometry')
    values, vectors = invoke(meter, 'FP64_eigh', torch.linalg.eigh, gram)
    order = torch.arange(values.numel()-1, -1, -1, device=a.device)
    values = values.index_select(0, order); vectors = vectors.index_select(1, order)
    floor = max(float(values[0]) if values.numel() else 0., 1e-30)*max(gram.shape)*64*torch.finfo(torch.float64).eps
    active = min(rank, int((values > floor).sum()))
    selected = vectors[:, :active]/values[:active].sqrt().unsqueeze(0)
    basis = invoke(meter, 'FP64_basis_GEMM', lambda: a.T@selected)
    if active:
        orth = invoke(meter, 'FP64_basis_orthogonality_GEMM', lambda: basis.T@basis)
        error = float((orth-torch.eye(active, dtype=orth.dtype, device=orth.device)).abs().max())
        need(error <= 1e-5, 'Right basis numerical orthogonality failed')
    else:
        # Keep the registered diagnostic invocation even for a zero-rank target.
        invoke(meter, 'FP64_basis_orthogonality_GEMM', lambda: basis.T@basis)
        error = 0.
    basis = torch.cat((basis, torch.zeros((a.shape[1], rank-active), dtype=a.dtype, device=a.device)), dim=1)
    spectrum = values.clamp_min(0)
    trace = float(spectrum.sum())
    return basis, {'trace': trace, 'continuous_tail': float(spectrum[min(rank, spectrum.numel()):].sum()),
        'active_rank': active, 'requested_rank': rank, 'orthogonality_max_abs': error,
        'spectrum': spectrum.cpu().tolist(), 'centering': 'ORIGINAL_B2_ANCHOR_NO_EMPIRICAL_MEAN',
        'evidence': 'ALG_NATIVE_Y_MINUS_B2_SURROGATE_NOT_NATIVE_RANK_BOUND'}


def minimum_spectrum(a, meter=None):
    import torch
    gram = sample_gram(a, meter)
    eigenvalues = invoke(meter, 'FP64_eigvalsh_lower_bound', torch.linalg.eigvalsh, gram)
    return eigenvalues.clamp_min(0).flip(0).cpu().tolist()


def rounding_bridge(cases, metric, *, steps, source_indices, continuous_tail):
    """Triangle-inequality bridge from captured Y-b2 to continuous hW2.

    This bounds a continuous common-rank relaxation only. Two rounded factor
    GEMMs need not have rank-r outputs, so it is not a native BF16 floor.
    """
    risk = 0.
    for source in source_indices:
        for step in steps:
            case = cases[source, step]
            energy = max(float((case['Y'].double()*case['gate'].double()).square().sum()/case['Y'].shape[0]), 1e-30)
            risk += float((case['continuous_rounding_error']*metric).square().sum())/(len(source_indices)*len(steps)*case['Y'].shape[0]*energy)
    lower = max(0., math.sqrt(max(0., continuous_tail))-math.sqrt(max(0., risk)))**2
    return {'teacher_addmm_rounding_relative_RMS_squared': risk,
        'continuous_hW_rank_lower_relaxation_after_rounding_triangle': lower,
        'identity': 'CONTINUOUS_ONLY_MAX_SQRT_TAIL_MINUS_ROUNDING_RMS_NOT_BF16_RANK_BOUND'}


def hidden_moment_commutator(a, b, meter=None):
    """Exact full-hidden-space identity, using only sample-size Gram matrices.

    C0=A'A, C1=B'B. This measures uncentered moments, not empirical centered
    covariances, and does not materialize either13824x13824 matrix.
    """
    multiply = lambda x, y: invoke(meter, 'FP64_covariance_commutator_GEMM', lambda: x@y)
    k0, k1, cross = multiply(a, a.T), multiply(b, b.T), multiply(a, b.T)
    p, q, square = multiply(k0, cross), multiply(cross, k1), multiply(cross.T, cross)
    first, second = float((p*q).sum()), float(square.square().sum())
    denominator = max(float(k0.square().sum())*float(k1.square().sum()), 1e-300)
    signed = 2*(first-second)/denominator
    need(signed >= -1e-9, 'Moment commutator cancellation exceeds FP64 tolerance')
    return {'normalized_Frobenius_squared': max(0., signed), 'unclamped_signed': signed,
        'identity': 'EXACT_SAMPLE_TRACE_FOR_FULL_UNCENTERED_HIDDEN_SECOND_MOMENTS',
        'centering': 'NONE', 'source_role': 'CALIBRATION6_DESCRIPTIVE_NOT_HELDOUT'}


def compile_factors(scaled_weight, bias, metric, basis, meter=None):
    """W2 M U and U' M+; original b2 remains the second GEMM bias."""
    import torch
    need(scaled_weight.dtype == basis.dtype == metric.dtype == torch.float64, 'FP64 factor compilation')
    left = invoke(meter, 'FP64_factor_compile_GEMM', lambda: scaled_weight@basis)
    inverse = torch.where(metric > 0, metric.reciprocal(), torch.zeros_like(metric))
    right = basis.T*inverse.unsqueeze(0)
    # Store checkpoint-style [out,in], runtime transpose has native (1,K) stride.
    result = {'left.weight': left.T.to(torch.bfloat16).contiguous(),
        'right.weight': right.T.to(torch.bfloat16).contiguous(),
        'right.bias': bias.to(torch.bfloat16).contiguous()}
    need(all(bool(torch.isfinite(v).all()) for v in result.values()), 'Nonfinite BF16 export')
    return result


def factor_forward(h, factors, meter=None):
    import torch
    left, right, bias = factors['left.weight'], factors['right.weight'], factors['right.bias']
    need(all(v.dtype == torch.bfloat16 and v.device == h.device for v in (h, left, right, bias))
        and h.ndim == 2 and h.shape[1] == left.shape[1] and right.shape[1] == left.shape[0]
        and bias.shape == (right.shape[0],), 'BF16 factor native geometry/dtype/device')
    middle = invoke(meter, 'BF16_factor_GEMM', torch.mm, h, left.T)
    return invoke(meter, 'BF16_factor_GEMM', torch.addmm, bias, middle, right.T)


def postwrite(r, y, gate, beta=1., meter=None, *, return_gate=False):
    import torch
    beta = fp32(beta)
    def gain():
        return (gate.to(torch.float32)*beta).to(torch.bfloat16)
    effective = invoke(meter, 'fixed_gate_gain_calls', gain)
    def write():
        product = y*effective
        result = r.clone(); result.add_(product)
        return result
    result = invoke(meter, 'BF16_postwrite_calls', write)
    return (result, effective) if return_gate else result


def centered(value):
    return value-value.mean(dim=-1, keepdim=True)


def row_statistics(r, y, u, gate, predicted_y, predicted_u, epsilon=1e-6, *, effective_gate=None):
    import torch
    r, y, u, gate, yh, uh = [x.to(torch.float64) for x in (r, y, u, gate, predicted_y, predicted_u)]
    effective = gate if effective_gate is None else effective_gate.to(torch.float64)
    def relative(error, reference):
        return float(error.square().sum()/reference.square().sum().clamp_min(1e-30))
    zd, zs = centered(u), centered(uh); delta = zs-zd
    vd, vs = zd.square().mean(-1, keepdim=True), zs.square().mean(-1, keepdim=True)
    sd, ss = (vd+epsilon).sqrt(), (vs+epsilon).sqrt()
    direct, radial = delta/sd, zs*(ss.reciprocal()-sd.reciprocal())
    denom = (zd/sd).square().sum().clamp_min(1e-30)
    direct_e = float(direct.square().sum()/denom); radial_e = float(radial.square().sum()/denom)
    cross = float(2*(direct*radial).sum()/denom)
    normalized = relative(zs/ss-zd/sd, zd/sd)
    ratio = ss/sd
    return {'gated_Y_relative_SSE': relative(yh*effective-y*gate, y*gate),
        'unscaled_factor_gated_Y_relative_SSE': relative((yh-y)*gate, y*gate),
        'gated_Y_metric_identity': 'FP64_OF_BF16_Y_AND_ACTUAL_TYPED_EFFECTIVE_GATE_BEFORE_PRODUCT_ROUNDING',
        'post_u_relative_SSE': relative(uh-u, u), 'post_centered_relative_SSE': relative(delta, zd),
        'post_mean_squared_error': float((uh.mean(-1)-u.mean(-1)).square().mean()),
        'std_ratio_mean': float(ratio.mean()), 'std_ratio_std': float(ratio.std(unbiased=False)),
        'std_ratio_MSE': float((ratio-1).square().mean()),
        'variance_change_cross_signed_mean': float(2*(zd*delta).mean()),
        'variance_change_square_mean': float(delta.square().mean()),
        'normalized_u_relative_SSE': normalized, 'normalized_direct': direct_e,
        'normalized_radial': radial_e, 'normalized_cross_signed': cross,
        'normalized_closure_error': normalized-direct_e-radial_e-cross,
        'identity': 'FP64_CONTINUOUS_NORMALIZED_U_PROXY_NOT_NATIVE_HEAD_OR_VIDEO'}


def _quadratic(a, b, c):
    scale = max(abs(a), abs(b), abs(c), 1e-300)
    if abs(a) <= scale*1e-14:
        return [] if abs(b) <= scale*1e-14 else [-c/b]
    disc = b*b-4*a*c
    if disc < -scale*scale*1e-14:
        return []
    root = math.sqrt(max(0., disc)); q = -.5*(b+math.copysign(root, b))
    return [-b/(2*a)] if q == 0 else [q/a, c/q]


def bounded_quartic_minimum(coefficients, low=0., high=2.):
    """All derivative roots on a fixed interval, no eig/optimizer/grid search."""
    q0, q1, q2, q3, q4 = map(float, coefficients)
    def derivative(x):
        return ((4*q4*x+3*q3)*x+2*q2)*x+q1
    def loss(x):
        return (((q4*x+q3)*x+q2)*x+q1)*x+q0
    critical = sorted({low, high, *[x for x in _quadratic(12*q4, 6*q3, 2*q2) if low < x < high]})
    candidates = [low, high, min(high, max(low, 1.))]
    tolerance = max(sum(abs(x) for x in coefficients), 1e-300)*1e-12
    for point in critical:
        if abs(derivative(point)) <= tolerance:
            candidates.append(point)
    for a, b in zip(critical, critical[1:]):
        fa, fb = derivative(a), derivative(b)
        if fa*fb < 0:
            for _ in range(80):
                midpoint = (a+b)/2; fm = derivative(midpoint)
                if (fa > 0) == (fm > 0):
                    a, fa = midpoint, fm
                else:
                    b = midpoint
            candidates.append((a+b)/2)
    best = min(loss(x) for x in candidates)
    tied = [x for x in candidates if loss(x) <= best+tolerance]
    beta = min(tied, key=lambda x: (abs(x-1), x))
    return {'beta_fit_FP64': beta, 'beta_deployed_FP32': fp32(beta), 'continuous_loss': loss(beta),
        'bounds': [low, high], 'method': 'FIXED_CUBIC_ROOT_BRACKETS_80_BISECTIONS_NO_EIG_NO_GRID'}


def fit_gains(training_cases, predictions, meter=None):
    """Each source contributes equally; each step/block gets one fixed scalar."""
    import torch
    numerator = denominator = 0.; coefficients = [0.]*5
    for case, predicted in zip(training_cases, predictions):
        r, y, u, gate, yh = [x.to(torch.float64) for x in (case['r'], case['Y'], case['u'], case['gate'], predicted)]
        true, approx = y*gate, yh*gate
        energy = max(float(true.square().sum()), 1e-30)
        numerator += float((true*approx).sum())/energy
        denominator += float(approx.square().sum())/energy
        rc, dc, uc = centered(r), centered(approx), centered(u)
        a = dc.square().mean(-1); b = (rc*dc).mean(-1)
        target = uc.square().mean(-1); c = rc.square().mean(-1)-target
        weight = (target+1e-6).square().reciprocal()/target.numel()/len(training_cases)
        terms = (c*c, 4*b*c, 4*b*b+2*a*c, 4*a*b, a*a)
        for i, term in enumerate(terms):
            coefficients[i] += float((weight*term).sum())
    def ls():
        beta = min(2., max(0., numerator/denominator)) if denominator > 1e-30 else 1.
        return {'beta_fit_FP64': beta, 'beta_deployed_FP32': fp32(beta), 'bounds': [0., 2.], 'objective': 'SOURCE_EQUAL_GATED_Y_CONTINUOUS_LS'}
    def variance():
        return {**bounded_quartic_minimum(coefficients), 'objective': 'SOURCE_EQUAL_POST_VARIANCE_CONTINUOUS_QUARTIC'}
    return {'gated_LS': invoke(meter, 'fixed_LS_gain_fits', ls),
        'variance': invoke(meter, 'fixed_variance_gain_fits', variance)}
