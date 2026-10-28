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
                   "rowtile_cross_cancellation"],
        "negative_offdiagonal": gram[0][1]
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
