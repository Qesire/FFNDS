"""Exact finite checks of a shared-input functional support construction.

Integer linear cases are exhaustively enumerated. Tanh-GELU examples use
stdlib scalar arithmetic and describe a continuous mathematical function,
not the native BF16/HiF4 implementation or real Wan tensors.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_SHARED_PERMUTATION_FUNCTIONAL_20261027'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def collapse(w, c):
    return [dot(column, c) for column in zip(*w, strict=True)]


def support(x, v, groups):
    """Exact extrema over independent within-group input permutations."""
    lower = upper = 0
    for start, end in groups:
        xs, vs = sorted(x[start:end]), sorted(v[start:end])
        lower += dot(xs, vs[::-1])
        upper += dot(xs, vs)
    return lower, upper


def histogram_prefix_support(histogram, sorted_v):
    """Known finite-alphabet counts and a precomputed sorted prefix sum."""
    prefix = [0]
    for v in sorted_v:
        prefix.append(prefix[-1] + v)
    position = result = 0
    for code, count in sorted(histogram):
        if count < 0 or int(count) != count:
            raise ValueError('Nonnegative integer counts required')
        result += code * (prefix[position + count] - prefix[position])
        position += count
    if position != len(sorted_v):
        raise ValueError('Counts do not cover the group')
    return result


def orbit(x, groups):
    choices = [sorted(set(itertools.permutations(x[start:end])))
               for start, end in groups]
    for blocks in itertools.product(*choices):
        yield tuple(itertools.chain.from_iterable(blocks))


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


CASES = [
    {'name': 'duplicate_rows_cancel', 'x': [-1, 1],
     'w': [[1, -1], [1, -1]], 'c': [1, -1], 'groups': [[0, 2]]},
    {'name': 'opposite_rows_linear_cancel', 'x': [-1, 1],
     'w': [[1, -1], [-1, 1]], 'c': [1, 1], 'groups': [[0, 2]]},
    {'name': 'signed_three_row', 'x': [-2, -1, 0, 2],
     'w': [[1, 0, -1, 2], [0, 2, 1, -1], [-1, 1, 0, 1]],
     'c': [2, -1, 3], 'groups': [[0, 4]]},
    {'name': 'two_independent_scale_intersections', 'x': [-2, -1, 0, 2, -1, 0, 1, 2],
     'w': [[1, 0, -1, 2, 2, -1, 0, 1], [0, 2, 1, -1, 1, 0, -2, 2],
           [-1, 1, 0, 1, 0, 2, 1, -1]],
     'c': [2, -1, 3], 'groups': [[0, 4], [4, 8]]},
]


def linear_check(case):
    x, w, c, groups = (case[k] for k in ('x', 'w', 'c', 'groups'))
    v = collapse(w, c)
    row_bounds = [support(x, row, groups) for row in w]
    box = [sum(coef * bounds[0 if coef >= 0 else 1]
               for coef, bounds in zip(c, row_bounds, strict=True)),
           sum(coef * bounds[1 if coef >= 0 else 0]
               for coef, bounds in zip(c, row_bounds, strict=True))]
    ys, witnesses, states = [], {}, set()
    for xp in orbit(x, groups):
        z = tuple(dot(row, xp) for row in w)
        y = dot(c, z)
        if y != dot(v, xp):
            raise ArithmeticError('Collapsed exact projection identity failed')
        ys.append(y); witnesses.setdefault(y, list(xp)); states.add(z)
    actual = [min(ys), max(ys)]
    predicted = list(support(x, v, groups))
    if actual != predicted:
        raise ArithmeticError('Shared support differs from exhaustive orbit')
    if box[0] > actual[0] or box[1] < actual[1]:
        raise ArithmeticError('Marginal outer bound is unsound')
    return {'name': case['name'], 'input_permutations': len(ys),
            'distinct_hidden_states': len(states), 'collapsed_weights': v,
            'marginal_row_bounds': row_bounds, 'marginal_output_box': box,
            'joint_linear_exact_range': actual, 'sorted_support_range': predicted,
            'extreme_input_witnesses': [witnesses[actual[0]], witnesses[actual[1]]],
            'canonical_ordered_input_exact_value': dot(v, x)}


def nonlinear_checks():
    # Identical row marginals do not identify their joint coordinate alignment.
    positive = CASES[0]['w']
    negative = CASES[1]['w']
    assert all(sorted(a) == sorted(b) for a, b in zip(positive, negative, strict=True))
    ranges = [support([-1, 1], collapse(w, [1, 1]), [[0, 2]]) for w in (positive, negative)]
    if ranges != [(-4, 4), (0, 0)]:
        raise ArithmeticError('Weight marginal nonidentifiability example changed')
    values = [gelu(z) + gelu(-z) for z in (-2, 2)]
    if not all(y > 0 for y in values):
        raise ArithmeticError('Nonlinear signed remainder example failed')
    choices = [-2, -1, 1]
    ordered = max(choices, key=lambda z: -z)
    nonlinear = max(choices, key=lambda z: -gelu(z))
    if (ordered, nonlinear) != (-2, -1):
        raise ArithmeticError('Tanh-GELU negative-tail example failed')
    return {
        'row_histogram_nonidentifiability': {'same_row_histograms': True,
            'output_ranges_with_c_ones': ranges},
        'antipodal_gelu': {'linear_function_range': [0, 0],
            'discrete_nonlinear_values': values,
            'convexified_input_midpoint_output': gelu(0) + gelu(0),
            'discrete_output_never_attains_midpoint_zero': True},
        'negative_tail_extreme_reversal': {'possible_preactivations': choices,
            'linear_maximizer': ordered, 'nonlinear_maximizer': nonlinear,
            'signed_gelu_values': [-gelu(z) for z in choices]},
        'numerical_scope': 'Continuous tanh-GELU scalar examples; no native QDQ/BF16 or Wan data.',
    }


def run(args):
    plan = json.loads(args.protocol.read_text())
    digest = sha(args.protocol)
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    if plan['id'] != ID or plan['cases'] != CASES or plan['limits'] != {
            'cpu_threads': 1, 'wall_seconds': 30, 'input_permutations': 604,
            'output_bytes': 1048576, 'GPU': 0, 'model_calls': 0}:
        raise ValueError('Fixed discriminator identity/budget changed')
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen source changed')
    if not any(s.get('id') == ID and s.get('protocol_sha256') == digest
               and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
               for s in active['experiment_authority']['current_user_authorized_scopes']):
        raise ValueError('Exact study scope not registered')
    hf = args.hf_root.resolve(); out = (hf / plan['output']).resolve()
    if not out.is_relative_to(hf) or out == hf:
        raise ValueError('HF output required')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'ATTEMPT.json').write_text(json.dumps({'id': ID, 'attempt': 1, 'protocol_sha256': digest}))
    started = time.monotonic()
    result = {'id': ID, 'protocol_sha256': digest, 'source_sha256': plan['source_sha256'],
              'status': 'STARTED', 'GPU_assignments': 0, 'model_calls': 0,
              'new_videos': 0, 'actual_Wan_observations': 0, 'quality_accepted': False}
    try:
        result['linear_cases'] = [linear_check(c) for c in CASES]
        count = sum(c['input_permutations'] for c in result['linear_cases'])
        if count != plan['limits']['input_permutations']:
            raise ArithmeticError('Permutation accounting mismatch')
        result['nonlinear_examples'] = nonlinear_checks()
        result.update(status='PASS_EXACT_FINITE_LINEAR_CHECKS_AND_SCALAR_COUNTEREXAMPLES',
                      input_permutations=count)
    except BaseException as error:
        result.update(status='FAILED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if len(payload.encode()) > plan['limits']['output_bytes'] or result['elapsed_seconds'] > 30:
            raise RuntimeError('Fixed resource bound exceeded')
        (out / 'RUN.json').write_text(payload)
    print(json.dumps({'status': result['status'], 'input_permutations': count,
                      'RUN_sha256': sha(out / 'RUN.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, required=True)
    run(parser.parse_args())
