"""Synthetic CPU test of nonlinear pooling and coupled HiF4 group scales.

No Wan weights, model forward, calibration fit or CUDA extension is loaded.
Small synthetic arrays discriminate marginal moment matching from joint
two-point cubature; none is a native FFN quality/speed result.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/20261022/R13_pooled_moment_probe_protocol.json'


def gaussian_gelu_mean(mean, variance):
    if variance < 0:
        raise ValueError('Variance must be nonnegative')
    scale = math.sqrt(1 + variance)
    arg = mean / scale
    return mean * .5 * (1 + math.erf(arg / math.sqrt(2))) + variance / scale * math.exp(-arg * arg / 2) / math.sqrt(2 * math.pi)


def projection_cost(rows=31668, stripes=7917, two_point=3958, d=5120, h=13824):
    if rows <= 0 or d <= 0 or h <= 0 or not 0 <= two_point <= stripes <= rows:
        raise ValueError('Invalid pooling geometry')
    # First projection: stripes+two_point rows. Second: stripes rows.
    # Averaging the two post-Q2 points before W2 avoids a second W2.
    return {'full_affine': d / (2 * h), 'pooled_W1': (stripes + two_point) / (2 * rows),
            'pooled_W2': stripes / (2 * rows),
            'total': d / (2 * h) + (2 * stripes + two_point) / (2 * rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first pooled-moment probe')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ar2_checkpoint import digest
    from audit_hif4_group_separability import QROOT, reference
    from run_ar2_restart import tensor_sha
    torch.set_num_threads(8)
    protocol_sha = digest(PROTOCOL)
    original_binding = json.loads((ROOT / 'research/20261022/R12_HiF4_group_separability.json').read_text())
    for name in ('QType.py', 'QFuncs/hifx.py'):
        if digest(ROOT / QROOT / name) != original_binding['source_sha256'][QROOT + name]:
            raise RuntimeError('Original quantizer source differs from previous CPU audit')
    qtype, original = reference()

    def post_q2(z):
        gelu = F.gelu(z.bfloat16(), approximate='tanh')
        return original(gelu.float(), qtype, -1).bfloat16().double()

    nodes, weights = np.polynomial.hermite.hermgauss(64)
    exact_checks = []
    for mean in (-2., -.5, 0., .5, 2.):
        for variance in (0., .25, 1., 4.):
            values = [mean + math.sqrt(2 * variance) * float(n) for n in nodes]
            exact = math.fsum(float(w) * z * .5 * (1 + math.erf(z / math.sqrt(2)))
                              for z, w in zip(values, weights)) / math.sqrt(math.pi)
            native_tanh = math.fsum(float(w) * .5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + .044715 * z ** 3)))
                                    for z, w in zip(values, weights)) / math.sqrt(math.pi)
            exact_checks.append({'mean': mean, 'variance': variance, 'closed_form': gaussian_gelu_mean(mean, variance),
                                 'quadrature': exact, 'absolute_error': abs(exact - gaussian_gelu_mean(mean, variance)),
                                 'tanh_GELU_expectation_difference': native_tanh - exact})
    if max(r['absolute_error'] for r in exact_checks) > 1e-7:
        raise RuntimeError('Gaussian exact-GELU expectation check failed')
    latent = torch.tensor([-2., -.5, .5, 2.], dtype=torch.float64)
    amplitude = torch.pow(2., torch.linspace(-2, 2, 64, dtype=torch.float64))
    offset = .2 * torch.sin(torch.arange(64, dtype=torch.float64))
    a = (latent[:, None] * amplitude + offset).bfloat16().double()
    b = torch.stack([a[:, k].roll(k % 4) for k in range(64)], dim=1)
    marginal_equal = torch.equal(a.sort(0).values, b.sort(0).values)
    if not marginal_equal:
        raise RuntimeError('Fixed synthetic channel marginals do not match')
    true_a, true_b = post_q2(a).mean(0), post_q2(b).mean(0)
    mean_gap = true_a - true_b
    rows = []
    signs = torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64)
    for label, z in (('shared_channel_order', a), ('rotated_channel_order', b)):
        mean, centered = z.mean(0), z - z.mean(0)
        # Input-space construction commutes with any real affine W1. Here z
        # is synthetic preactivation, so no original/model GEMM is invoked.
        contrasts = signs @ centered / math.sqrt(len(z))
        covariance = centered.T @ centered / len(z)
        sketch_covariance = contrasts.T @ contrasts / len(signs)
        covariance_error = float((covariance - sketch_covariance).abs().max())
        quadratic_prediction = (mean.square() + contrasts.square()).mean(0)
        quadratic_error = float((quadratic_prediction - z.square().mean(0)).abs().max())
        if covariance_error > 1e-10 or quadratic_error > 1e-10:
            raise RuntimeError('Exact sign covariance or quadratic identity failed')
        true_mean = post_q2(z).mean(0)
        one = post_q2(mean[None])[0]
        cubatures = .5 * (post_q2(mean[None] + contrasts) + post_q2(mean[None] - contrasts))
        prediction = cubatures.mean(0)
        bias_sse = float((prediction - true_mean).square().sum())
        variance_sse = float((cubatures - prediction).square().sum(1).mean())
        actual_mse = float((cubatures - true_mean).square().sum(1).mean())
        if not math.isclose(actual_mse, bias_sse + variance_sse, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError('Cubature bias/variance decomposition failed')
        rows.append({'case': label, 'preactivation_sha256': tensor_sha(z),
                     'mean_post_Q2_sha256': tensor_sha(true_mean),
                     'one_point_mean_response_SSE': float((one - true_mean).square().sum()),
                     'two_point_ensemble_bias_SSE': bias_sse,
                     'single_sketch_variance_SSE': variance_sse,
                     'single_sketch_expected_SSE': actual_mse,
                     'random_original_row_anchor_expected_SSE': float((post_q2(z) - true_mean).square().sum(1).mean()),
                     'sign_covariance_max_error': covariance_error,
                     'quadratic_moment_max_error': quadratic_error})
    if torch.cuda.is_initialized():
        raise RuntimeError('Synthetic CPU probe initialized CUDA')
    if digest(PROTOCOL) != protocol_sha:
        raise RuntimeError('Synthetic scope changed during execution')
    result = {'id': 'R13S3_POOLED_NONLINEAR_MOMENT_DISCRIMINATOR',
        'evidence_level': 'SYNTHETIC_CPU_QUANTIZER_REFERENCE_AND_EXACT_ALGEBRA_ONLY',
        'protocol_sha256': protocol_sha, 'source_sha256': digest(__file__),
        'original_reference_source_sha256': {p: digest(ROOT / p) for p in (QROOT + 'QType.py', QROOT + 'QFuncs/hifx.py')},
        'gaussian_exact_GELU_checks': exact_checks,
        'marginal_sufficiency_counterexample': {'identical_per_channel_empirical_marginals': marginal_equal,
            'mean_post_Q2_channel_mismatches': int((mean_gap != 0).sum()),
            'mean_post_Q2_L2_difference': float(mean_gap.norm()),
            'mean_post_Q2_max_absolute_difference': float(mean_gap.abs().max()),
            'conclusion': 'Per-channel marginals alone are insufficient for this original grouped quantizer.' if bool((mean_gap != 0).any()) else 'No grouped-quantizer counterexample in this fixed synthetic case.'},
        'joint_cubature_cases': rows,
        'full_projection_MAC_fraction': projection_cost(),
        'native_arithmetic_caveat': 'Averaging original post-Q2 activations before W2 is a different rounded computation from averaging two native Y outputs. BF16 pooling, anchor GEMM shapes, mixed-affine reconstruction and complete residual writeback need their own native controls.',
        'remaining_costs': 'All-row originalQ1, up to11875 pooledQ2 rows, pooling/sketch/ranking, averaging two post-Q2 vectors, R9 full-BF16 affine storage/traffic, centered remainder construction and per-row output writes must be charged. No timing claim.',
        'new_model_generations': 0, 'models_loaded': 0, 'fitted_parameters': 0, 'CUDA_initialized': False,
        'selection_or_native_success': False,
        'limits': 'Synthetic activations with no Wan weights or W2/gate. Correct moments for a quadratic and a Gaussian do not imply accurate GELU+HiF4 pooling on Wan or input-conditioned benefit. All16 signs are an analytic ensemble, not a free deployment ensemble.'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counterexample': result['marginal_sufficiency_counterexample'], 'cubature': rows,
                      'main_MAC': result['full_projection_MAC_fraction']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
