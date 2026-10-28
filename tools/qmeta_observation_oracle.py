#!/usr/bin/env python3
"""Finite mathematical/size checks for Q-Meta timing; no model, CUDA or timing."""
import hashlib
import itertools
import json
import math
from pathlib import Path


def magnitude_tree64(x):
    """Only the abs/max tree; not a HiF4 encoding implementation."""
    assert len(x) == 64
    g4 = tuple(max(abs(v) for v in x[i:i+4]) for i in range(0, 64, 4))
    g8 = tuple(max(g4[i:i+2]) for i in range(0, 16, 2))
    return max(g8), g8, g4


def gelu_tanh(x):
    return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))


def verify():
    x = tuple([1., -2., .5, -1.] * 16)
    assert magnitude_tree64(x) == magnitude_tree64(tuple(-v for v in x))
    p, n = gelu_tanh(1)**2, gelu_tanh(-1)**2
    losses = ((p, n), (n, p))
    static = min(sum(row[a] for row in losses)/2 for a in range(2))
    signed = sum(min(row) for row in losses)/2
    assert static > signed and p > n > 0

    xp, yp, w = (1, -1, 0, 0), (-1, 1, 0, 0), (1, -1, 0, 0)
    assert sorted(xp) == sorted(yp)
    assert magnitude_tree64(xp+(0,)*60) == magnitude_tree64(yp+(0,)*60)
    assert sum(a*b for a,b in zip(xp,w)) == 2
    assert sum(a*b for a,b in zip(yp,w)) == -2

    # Finite actions: subtracting a common reference preserves the argmin.
    risks = (3., 1., 4., 2.)
    contrasts = tuple(v-risks[0] for v in risks)
    assert min(range(4), key=risks.__getitem__) == min(range(4), key=contrasts.__getitem__)
    # Uniform score errors give <= 2 epsilon conditional regret.
    eps = .1
    worst_regret = 0
    for truth in itertools.product((0., .1, .2, .3), repeat=3):
        for err in itertools.product((-eps, 0., eps), repeat=3):
            pred = [a+b for a,b in zip(truth,err)]
            chosen = min(range(3), key=pred.__getitem__)
            regret = truth[chosen]-min(truth)
            assert regret <= 2*eps+1e-12
            worst_regret = max(worst_regret, regret)

    # XOR: individually useless observations can be jointly valuable.
    states = tuple(itertools.product((0,1),repeat=2))
    def bayes(features):
        buckets = {}
        for s in states:
            buckets.setdefault(tuple(s[j] for j in features), []).append(s[0]^s[1])
        return sum(min(v.count(0),v.count(1)) for v in buckets.values())/len(states)
    assert (bayes(()),bayes((0,)),bayes((1,)),bayes((0,1))) == (.5,.5,.5,0)
    assert .1+.1 < .5  # Acquiring the pair can have positive net value.

    row_contrib = (1,-1)
    assert sum(v*v for v in row_contrib) == 2
    assert sum(row_contrib)**2 == 0

    M,D,H,B,CTA,tile = 31668,5120,13824,64,4096,16
    assert D//B == 80 and CTA//B == 64
    def touched(row):
        return set(range(row*D//CTA, ((row+1)*D-1)//CTA+1))
    assert all(len(touched(row)) == 2 for row in range(4))
    assert len(set.union(*(touched(row) for row in range(4)))) == 5
    assert len(set.union(*(touched(row) for row in range(tile)))) == 20
    byte_counts = {
        "Q1_groups": M*D//B,
        "Q1_BF16_extra_read": 2*M*D,
        "Q2_groups": M*H//B,
        "Q2_BF16_extra_read": 2*M*H,
        "Q1_ideal_packed_metadata": M*D//B*4,
        "Q2_ideal_packed_metadata": M*H//B*4,
        "Q1_naive_25fp32_metadata": M*D//B*25*4,
        "eight_fp32_perrow": M*8*4,
        "eight_fp32_pertile16": math.ceil(M/tile)*8*4,
        "uint8_class_pertile16": math.ceil(M/tile),
        "naive_eight_fp32_pergroup_partials": M*(D//B)*8*4,
    }
    assert byte_counts["Q1_BF16_extra_read"] == 324280320
    assert byte_counts["Q2_BF16_extra_read"] == 875556864
    assert byte_counts["Q1_ideal_packed_metadata"]*32 == byte_counts["Q1_BF16_extra_read"]

    root = Path(__file__).resolve().parents[1]
    src = root/"upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu"
    source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    assert source_hash == "07120a5d2519e3ed3c8782acb8897a44e31103f93dd84d353ae7bb86b043e00c"
    return {
        "status": "SYNTHETIC_ALGEBRA_SIZE_AND_SOURCE_HASH_PASS",
        "checks": [
            "magnitude_tree_sign_invariance", "gelu_action_collision",
            "signed_histogram_position_collision", "finite_action_contrasts",
            "uniform_error_regret_bound", "observation_complementarity",
            "row_metric_aggregation", "CTA_row_alignment", "byte_budget", "pinned_source_hash"
        ],
        "gelu_losses": {"positive": p, "negative": n, "static": static, "sign_visible": signed},
        "worst_finite_regret": worst_regret,
        "ideal_payload_bytes": byte_counts,
        "source_sha256": source_hash,
        "model_runs": 0,
        "scope": "Finite examples, ideal payload counts and pinned source; no HiF4 native equality, Wan distribution, hardware traffic or latency measurement."
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

