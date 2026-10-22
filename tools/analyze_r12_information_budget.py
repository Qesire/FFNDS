"""Small algebraic discriminators for selector information and cold starts.

Synthetic math and exact budget arithmetic only; no Wan performance claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def gelu(value):
    # Same tanh approximation used by the frozen FFN; Python scalar arithmetic
    # demonstrates the information obstruction, not BF16/CUDA equivalence.
    return .5 * value * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + .044715 * value ** 3)))


def analyze():
    tiles = [[[2., 0.], [-2., 0.]], [[0., 2.], [0., -2.]]]
    means = [[sum(row[k] for row in tile) / len(tile) for k in range(2)] for tile in tiles]
    energies = [[sum(gelu(row[k]) ** 2 for row in tile) for k in range(2)] for tile in tiles]
    assert means[0] == means[1] == [0., 0.]
    assert [max(range(2), key=row.__getitem__) for row in energies] == [0, 1]
    variances = [[sum(row[k] ** 2 for row in tile) / len(tile) for k in range(2)] for tile in tiles]
    return {'id': 'R12_INFORMATION_AND_BUDGET_DISCRIMINATORS',
        'status': 'COMPLETED_STDLIB_SYNTHETIC_AND_EXACT_ARITHMETIC',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_model_generations': 0, 'CUDA_initialized': False,
        'mean_only_selector_counterexample': {'operator': 'Synthetic2D W1=W2=identity, zero biases, unit gate, scalar tanh-GELU; no native QDQ/BF16 equivalence claimed.',
            'inputs': tiles, 'identical_means': means,
            'tanh_GELU_hidden_energy': energies, 'best_hidden_coordinate': [0, 1],
            'second_moments': variances,
            'finding': 'Equal tile means can hide opposite useful nonlinear directions. A mean-only signal has no universal sufficient-information guarantee; this is not a Wan empirical failure.'},
        'dense_anchor_delta_budget': {'experts': ['high', 'high', 'low', 'low'],
            'assumption': 'One full dense initial cache per expert, followed by sparse hidden updates at fractions rho1/rho3; equalM and widths.',
            'global_main_MAC_fraction': '(2 + rho1 + rho3)/4',
            'examples': [{'rho1': rho, 'rho3': rho, 'fraction': (2 + 2 * rho) / 4} for rho in (0., .125, .25, .5)],
            'finding': 'Any positive sparse update after both dense cold starts exceeds50% mainMAC. This reconfirms docs/55; partial anchors or a different representation lie outside the assumption.'},
        'two_GEMM_GELU_proxy_budget': {'assumption': 'One approximate W1 selector retains input fraction k; exact selected W1/W2 use hidden fraction rho. Equal unit MAC costs; quantized kernel speed factors must be measured separately.',
            'physical_MAC_fraction_including_proxy': 'rho + k/2',
            'examples': [{'rho': .5, 'k': k, 'all_projection_MAC_fraction': .5 + k / 2} for k in (.0, .1, .3)],
            'finding': 'A retained50% main path leaves nonzero selector work. Prox SwiGLU three-projection/INT4 GEMV cost constants cannot be copied into this two-GEMM, fullM HiF4 baseline.'},
        'scope': 'Mathematical counterexample and source-aligned cost accounting. No rejection of every mean-based empirical router, sparse-delta algorithm or proxy family, and no new model eligibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous discriminator record')
    result = analyze()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Equal means/opposite hidden directions: reproduced; dense-anchor and proxy costs: accounted;0 model runs.')


if __name__ == '__main__':
    main()
