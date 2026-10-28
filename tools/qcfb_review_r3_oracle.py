#!/usr/bin/env python3
"""Finite algebra checks for the QCFB R3 author response; no model/native work."""
import json
import math
from fractions import Fraction as F


def verify():
    # Equal two-group omission budgets, different state-optimal actions.
    ds = ((1, -1, 3), (1, 3, -1))
    masks = ((1, 1, 0), (1, 0, 1))
    loss = [[sum(d*m for d, m in zip(row, mask))**2 for mask in masks] for row in ds]
    assert loss == [[0, 16], [16, 0]]
    assert min(sum(row[a] for row in loss)/2 for a in range(2)) == 8
    assert sum(min(row) for row in loss)/2 == 0
    # Exact Z=S has zero gap; a constant bin loses the bit.
    collision = [[0, 1], [1, 0]]
    risk_full_z = sum(min(row) for row in collision)/2
    risk_bin = min(sum(row[a] for row in collision)/2 for a in range(2))
    assert (risk_full_z, risk_bin) == (0, .5)

    # Dictionary, information and policy losses telescope exactly.
    prob = (F(1, 5), F(3, 10), F(1, 2))
    all_loss = ((0, 4, 4), (4, 0, 4), (4, 4, 0))
    oracle_all = sum(p*min(row) for p, row in zip(prob, all_loss))
    oracle_dict = sum(p*min(row[:2]) for p, row in zip(prob, all_loss))
    bayes_z = min(sum(p*row[a] for p, row in zip(prob, all_loss)) for a in (0, 1))
    policy = sum(p*row[0] for p, row in zip(prob, all_loss))
    gaps = (oracle_dict-oracle_all, bayes_z-oracle_dict, policy-bayes_z)
    assert gaps == (F(2), F(4, 5), F(2, 5))
    assert sum(gaps) == policy-oracle_all == F(16, 5)

    # Large relative numerical error can still leave a certified wide margin.
    r_a, r_b, bound_a, bound_b = 1, 100, 3, 0
    assert r_b-r_a > bound_a+bound_b and (-1-1)**2 < 100
    assert (-1-1)**2 > 1.1  # Narrow nominal ordering need not survive.

    # Two injections in a frozen scalar linear state: B0=2, A1=3, B1=-6.
    delta_1 = -2
    delta_2 = 3*delta_1 - (-6)
    transported = (3*2, -6)
    quad = sum(x*y for x in transported for y in transported)
    assert delta_2 == 0 and quad == 0
    assert sum(x*x for x in transported) == 72  # Offdiagonals matter.
    assert max(-1, 0)+1 == 1 and -1+1 == 0  # Nonlinear path differs.

    zero_failure_probability = .99**20
    upper95 = 1-.05**(1/20)
    assert .81 < zero_failure_probability < .82
    assert .13 < upper95 < .14
    return {
        "status": "SYNTHETIC_ALGEBRA_ONLY_PASS",
        "checks": ["conditional_bundle_vs_static", "coarsening_not_original_lower_bound",
                   "dictionary_information_policy_decomposition", "native_ordering_margin",
                   "cross_call_linear_gram", "nonlinear_cancellation_failure",
                   "ideal_independent_binomial_example"],
        "loss_table": loss,
        "gap_components": [str(x) for x in gaps],
        "zero_failure_probability_if_p001_n20": zero_failure_probability,
        "zero_failure_one_sided95_upper_n20": upper95,
        "model_runs": 0,
        "scope": "Constructed examples and ideal sampling calculation; no Wan/source-level empirical estimate."
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
