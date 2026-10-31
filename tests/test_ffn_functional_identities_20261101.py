"""Tiny algebra fixtures for G18. Synthetic float64 only; no Wan, GPU or video evidence."""
import math
import numpy as np


def gelu_exact(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def gelu_tanh(x):
    t = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(t))


def gelu_tanh_prime(x):
    z = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    t = np.tanh(z)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x**2)


def check():
    rng = np.random.default_rng(20261101)
    u = rng.normal(size=(17, 6))
    w1 = rng.normal(size=(6, 11))
    b1 = rng.normal(size=11)
    w2 = rng.normal(size=(11, 6))
    b2 = rng.normal(size=6)
    a = u @ w1 + b1
    dense = gelu_tanh(a) @ w2 + b2

    # Both definitions obey the odd/even identity; they are not equal to each other.
    for phi in (gelu_exact, gelu_tanh):
        np.testing.assert_allclose(phi(a) - phi(-a), a, atol=2e-14)
    psi = lambda t: gelu_tanh(t) - 0.5 * t
    folded = u @ (0.5 * (w1 @ w2)) + 0.5 * b1 @ w2 + b2 + psi(a) @ w2
    np.testing.assert_allclose(dense, folded, atol=2e-13)

    # Antipodal preactivations admit an exact linear-plus-even representation.
    z = rng.normal(size=17)
    vp, vm = rng.normal(size=6), rng.normal(size=6)
    pair = gelu_tanh(z)[:, None] * vp + gelu_tanh(-z)[:, None] * vm
    rewrite = 0.5 * z[:, None] * (vp - vm) + psi(z)[:, None] * (vp + vm)
    np.testing.assert_allclose(pair, rewrite, atol=2e-14)

    # Exact merging for identical preactivation functions.
    center_w = rng.normal(size=6)
    center_b = 0.3
    v = rng.normal(size=(4, 6))
    exact = gelu_tanh(u @ np.repeat(center_w[:, None], 4, axis=1) + center_b) @ v
    merged = gelu_tanh(u @ center_w + center_b)[:, None] * v.sum(axis=0)
    np.testing.assert_allclose(exact, merged, atol=2e-14)

    # First-order group aggregate equals individually Taylor-expanded neurons.
    eps = rng.normal(size=(6, 4))
    eta = rng.normal(size=4)

    def errors(scale):
        center = u @ center_w + center_b
        delta = scale * (u @ eps + eta)
        true = gelu_tanh(center[:, None] + delta) @ v
        m = (scale * eps) @ v
        c = (scale * eta) @ v
        approx = gelu_tanh(center)[:, None] * v.sum(axis=0) + gelu_tanh_prime(center)[:, None] * (u @ m + c)
        return float(np.max(np.abs(true - approx)))

    err_large, err_small = errors(0.04), errors(0.02)
    assert err_large > err_small and err_small / err_large < 0.4, (err_large, err_small)

    # A generic hidden basis rotation is not an equivalence across GELU.
    q = np.array([[1.0, 0.7], [0.0, 1.0]])
    short_u = rng.normal(size=(8, 2))
    short_w1 = rng.normal(size=(2, 2))
    short_w2 = rng.normal(size=(2, 2))
    original = gelu_tanh(short_u @ short_w1) @ short_w2
    rotated = gelu_tanh(short_u @ short_w1 @ q) @ np.linalg.inv(q) @ short_w2
    assert np.max(np.abs(original - rotated)) > 1e-4

    # Ideal leading MAC only; these inequalities do not establish hardware speedup.
    d, m = 5120, 13824
    assert (d * d + 2 * d * (400 + 250)) < 2 * d * m
    print("G18 algebra fixtures PASS; synthetic float64 only")
    print(f"Taylor max error 0.04: {err_large:.6g}; 0.02: {err_small:.6g}")


if __name__ == "__main__":
    check()
