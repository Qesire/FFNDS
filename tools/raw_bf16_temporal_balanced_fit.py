"""Offline same-row DC/AC balancing; no numerical import or execution at import.

The target is still Y-b2 and the deployed representation is the unchanged
rank-r two-GEMM factor. Bias participates in reference risk normalization only.
"""
import math


def need(ok, message):
    if not ok:
        raise ValueError(message)


def temporal_transform(target, reference, gate, *, balance):
    """Keep tau0 weight; give rest-time DC and AC equal relative-risk mass.

    target/reference are [21,8,D], already in the caller's exact FP64 geometry.
    Weights use reference Y*g including original b2, never candidate errors.
    Transforming in row space retains exactly168 rows instead of stacking DC
    and AC as additional observations. No temporal labels are needed online.
    """
    import torch
    need(target.shape == reference.shape and len(target.shape) == 3
         and target.shape[:2] == (21, 8) and gate.shape == (target.shape[-1],)
         and target.dtype == reference.dtype == gate.dtype == torch.float64,
         'Complete FP64 temporal geometry required')
    rest, truth = target[1:], reference[1:]*gate
    mean = rest.mean(dim=0, keepdim=True)
    mean_truth = truth.mean(dim=0, keepdim=True)
    dc = float(20*mean_truth.square().sum())
    ac = float((truth-mean_truth).square().sum())
    total = dc+ac
    active = int(dc > 0)+int(ac > 0)
    if balance and active:
        alpha = total/(active*dc) if dc > 0 else 0.
        beta = total/(active*ac) if ac > 0 else 0.
    else:
        alpha = beta = 1.
    need(all(math.isfinite(v) and v >= 0 for v in (alpha, beta)), 'Invalid temporal calibration weights')
    output = torch.cat((target[:1], math.sqrt(alpha)*mean+math.sqrt(beta)*(rest-mean)), dim=0)
    need(bool(torch.isfinite(output).all()), 'Nonfinite temporal target')
    return output, {'reference_DC_energy_tau1_to20': dc, 'reference_AC_energy_tau1_to20': ac,
                    'DC_weight': alpha, 'AC_weight': beta, 'active_components': active,
                    'tau0_weight_changed': False, 'bias': 'REFERENCE_INCLUDES_B2_TARGET_IS_Y_MINUS_B2',
                    'zero_component_policy': 'AVAILABLE_COMPONENTS_SHARE_MASS; BOTH_ZERO_USES_IDENTITY'}


def anchored_rows(cases, bias, metric, *, steps, source_indices, balance):
    """Same source/step relative weights as the frozen ordinary output fit."""
    import torch
    rows, records = [], []
    for source in source_indices:
        for step in steps:
            case = cases[source, step]
            y, gate = case['Y'].double(), case['gate'].double()
            need(y.shape == (168, metric.numel()) and bias.shape == gate.shape == metric.shape,
                 'Native168 rows and actual bias/gate geometry required')
            energy = max(float((y*gate).square().sum()/168), 1e-30)
            scale = 1/math.sqrt(len(source_indices)*len(steps)*168*energy)
            transformed, record = temporal_transform((y-bias.double()).reshape(21, 8, -1),
                                                     y.reshape(21, 8, -1), gate, balance=balance)
            rows.append(transformed.reshape(168, -1)*metric*scale)
            records.append({'source': source, 'local_step': step, 'scale': scale, **record})
    return torch.cat(rows, dim=0), records


def temporal_metrics(y, prediction, gate):
    """Native BF16 outputs, continuous gated geometry, no suffix execution."""
    a = (prediction.double()*gate.double()).reshape(21, 8, -1)
    b = (y.double()*gate.double()).reshape(21, 8, -1)
    rest_a, rest_b = a[1:], b[1:]
    am, bm = rest_a.mean(0, keepdim=True), rest_b.mean(0, keepdim=True)
    ac_a, ac_b = rest_a-am, rest_b-bm
    dc_e = float(20*(am-bm).square().sum())
    dc_b = float(20*bm.square().sum())
    ac_e = float((ac_a-ac_b).square().sum())
    ac_ref = float(ac_b.square().sum())
    ac_energy = float(ac_a.square().sum())
    dot = float((ac_a*ac_b).sum())
    t0_ref = float(b[0].square().sum())
    return {'tau1_to20_DC_relative_SSE': dc_e/dc_b if dc_b > 0 else None,
            'tau1_to20_AC_relative_SSE': ac_e/ac_ref if ac_ref > 0 else None,
            'tau1_to20_AC_energy_ratio': ac_energy/ac_ref if ac_ref > 0 else None,
            'tau1_to20_AC_aligned_gain': dot/ac_ref if ac_ref > 0 else None,
            'tau1_to20_AC_cosine': dot/math.sqrt(ac_energy*ac_ref) if ac_energy*ac_ref > 0 else None,
            'tau0_relative_SSE': float((a[0]-b[0]).square().sum())/t0_ref if t0_ref > 0 else None,
            'time_error_closure_signed': float((rest_a-rest_b).square().sum())-dc_e-ac_e}
