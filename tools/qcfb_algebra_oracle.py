#!/usr/bin/env python3
"""Synthetic QCFB algebra oracle: no real Wan model, HiF4 kernel or GPU."""
import itertools
import json
import math


def gelu(x):
    return .5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm2(v):
    return dot(v, v)


def boundary_checks():
    """Deterministic finite examples; no native tensors or timing claims."""
    checks = []
    # Antiparallel is insufficient if magnitudes do not match.
    assert (100. - 1.)**2 == 9801.
    checks.append("negative_cross_not_low_risk")
    # Three groups can cancel without a near-zero pair.
    tri = ((1., 0.), (-.5, math.sqrt(3)/2), (-.5, -math.sqrt(3)/2))
    assert norm2(tuple(sum(v[k] for v in tri) for k in range(2))) < 1e-24
    for a, b in itertools.combinations(tri, 2):
        assert math.isclose(norm2(tuple(x+y for x,y in zip(a,b))), 1.)
    checks.append("three_group_bundle_pair_search_blindspot")
    # Z is constant; a posterior action sees D and is not Z-measurable.
    states = (-1., 1.)
    actual = sum((1-d*int(d==1))**2 for d in states)/2
    invalid = sum(1+int(d==1)**2 for d in states)/2
    assert actual == .5 and invalid == 1.5
    checks.append("nonmeasurable_oracle_counterexample")
    samples = ((1., -1.), (-1., 1.))
    mu = [sum(row[j] for row in samples)/2 for j in range(2)]
    gamma = [[sum(row[i]*row[j] for row in samples)/2 for j in range(2)] for i in range(2)]
    assert mu == [0., 0.] and gamma == [[1.,-1.],[-1.,1.]]
    checks.append("conditional_covariance_required")
    # Sample outer products construct the augmented moment matrix.
    obs = ((1., 2., -1.), (-2., -1., 3.), (.5, 0., 2.))
    H = [[sum(o[i]*o[j] for o in obs)/len(obs) for j in range(3)] for i in range(3)]
    for m in itertools.product((0,1), repeat=2):
        w = (1., -m[0], -m[1])
        pred = sum(w[i]*H[i][j]*w[j] for i in range(3) for j in range(3))
        direct = sum((o[0]-m[0]*o[1]-m[1]*o[2])**2 for o in obs)/len(obs)
        assert pred >= -1e-12 and math.isclose(pred,direct,abs_tol=1e-12)
    assert 1 - 2*2 + 1 == -2  # C=Gamma=1, b=2 is impossible.
    checks.append("augmented_PSD_and_inconsistent_b")
    # eQ=d=1: relative-reference improvement, but drift from dense.
    assert (1.-1.)**2 == 0 and 1.**2 == 1
    checks.append("reference_improvement_not_dense_equivalence")
    # Physical row metric vs averaging; cross-token P=[1,1].
    row_errors = (1.,-1.)
    assert sum(e*e for e in row_errors) == 2 and sum(row_errors)**2 == 0
    assert sum(e*e for e in (1.,1.)) == 2 and sum((1.,1.))**2 == 4
    checks.append("row_separable_vs_mixing_readout")
    # Projection misses the second coordinate.
    v = ((1.,100.),(-1.,100.))
    assert sum(x[0] for x in v) == 0 and sum(x[1] for x in v) == 200
    checks.append("readout_nullspace_not_full_output")
    assert 0.*1. == 0 and 1.**2 - 0.**2 == 1
    checks.append("zero_J_not_zero_finite_response")
    # No information in Z2: min E L = .5, E min L = 0.
    risks = [sum(int(a!=s) for s in (0,1))/2 for a in (0,1)]
    oracle = sum(min(int(a!=s) for a in (0,1)) for s in (0,1))/2
    assert min(risks) == .5 and oracle == 0
    checks.append("staged_conditional_expectation_required")
    # Per-state frozen linear algebra exactness vs a nonlinear input-panel action.
    dense = gelu(1.+1.)
    changed = gelu(1.)
    assert not math.isclose(changed,dense)
    checks.append("input_panel_changes_later_activation")
    # All masks obey the derived per-action error bound, for a fixed example.
    true_g = ((2.,-.5),(-.5,1.))
    estimated_g = ((2.1,-.5),(-.5,1.1))
    b = (.2,-.1)
    bhat = (.23,-.14)
    eps_g, eps_b = .1, .05
    for m in itertools.product((0,1), repeat=2):
        k = sum(m)
        error = sum(m[i]*(estimated_g[i][j]-true_g[i][j])*m[j]
                    for i in range(2) for j in range(2))
        error -= 2*sum((bhat[i]-b[i])*m[i] for i in range(2))
        assert abs(error) <= k*eps_g+2*math.sqrt(k)*eps_b+1e-12
    checks.append("finite_action_error_bound")
    # Adding the second group reduces error; a single-score ordering is insufficient.
    assert (1.-1.)**2 < 1.**2
    checks.append("nonmonotone_set_risk")
    return checks


def verify():
    # Equal unsigned scale / code magnitude does not fix a signed action.
    assert math.isclose(gelu(1) - gelu(-1), 1.0, abs_tol=1e-12)
    assert math.isclose(gelu(1) - gelu(1), 0, abs_tol=1e-12)

    # Conditional Gram is PSD despite negative signed cross-terms.
    d = ((2., 1.), (-2., -1.), (0., 3.))
    gram = [[dot(a, b) for b in d] for a in d]
    assert gram[0][1] < 0
    for m in itertools.product((0, 1), repeat=3):
        summed = tuple(sum(m[j] * d[j][k] for j in range(3)) for k in range(2))
        quad = sum(m[i] * gram[i][j] * m[j] for i in range(3) for j in range(3))
        assert quad >= -1e-12
        assert math.isclose(norm2(summed), quad, abs_tol=1e-12)

        # Joint quantization residual and omission cross term.
        q = (1., -2.)
        diff = tuple(q[k] - summed[k] for k in range(2))
        rhs = norm2(q) + norm2(summed) - 2 * dot(q, summed)
        assert math.isclose(norm2(diff), rhs, abs_tol=1e-12)

    # Per-row cancellation may disappear under a shared physical tile mask.
    rows = ((1., -1.), (1., 1.))
    assert sum(rows[0]) == 0
    tile_pair = sum(a * b for a, b in rows)
    assert tile_pair == 0
    assert sum(sum(r) ** 2 for r in rows) == 4

    return {
        "status": "ALG_SYNTHETIC_PASS",
        "scope": "ordinary algebra and tanh-GELU; NOT Wan BF16/HiF4/native",
        "checks": ["signed_scale_collision", "gram_PSD", "quant_cross_term",
                   "rowtile_cross_cancellation"] + boundary_checks(),
        "negative_offdiagonal": gram[0][1],
        "new_model_runs": 0, "new_gpu_runs": 0
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
