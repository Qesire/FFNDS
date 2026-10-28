"""FP64 terminal-head geometry and finite identities; no model or GPU calls."""
import numpy as np


def _array(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite input')
    return value


def head_geometry(weight, scale, *, rank_rtol=1e-10):
    """One reduced QR for span{1, rows(W diag(1+scale))}; reject rank loss."""
    w, scale = _array(weight), _array(scale)
    if w.ndim != 2 or scale.shape != (w.shape[1],) or w.shape[0] + 1 > w.shape[1]:
        raise ValueError('head geometry mismatch')
    t = w * (1.0 + scale)[None, :]
    basis = np.column_stack((np.ones(w.shape[1]), t.T))
    q, r = np.linalg.qr(basis, mode='reduced')
    diag = np.abs(np.diag(r))
    if diag.min() <= rank_rtol * diag.max():
        raise ValueError('rank-deficient head geometry')
    return {'T': t, 'Q': q, 'rank': q.shape[1],
            'qr_diagonal': diag, 'orthogonality_error': float(np.linalg.norm(q.T @ q - np.eye(q.shape[1])))}


def directions(u, geometry, *, seed=42, degeneracy_rtol=1e-6):
    """Three unit-L2 directions, or None for a degenerate direction; one row."""
    u = _array(u)
    t, q = geometry['T'], geometry['Q']
    if u.shape != (t.shape[1],) or np.linalg.norm(u) == 0:
        raise ValueError('invalid state')
    z = u - u.mean()
    visible = t[0] - t[0].mean()
    def unit(v, ref):
        n = np.linalg.norm(v)
        return None if n <= degeneracy_rtol * max(ref, np.finfo(np.float64).tiny) else v / n
    visible = unit(visible, np.linalg.norm(t[0]))
    v = z - q @ (q.T @ z)
    norm = unit(v, np.linalg.norm(z))
    random = np.random.default_rng(seed).standard_normal(u.size)
    nvec = random - q @ (q.T @ random)
    if norm is not None:
        nvec = nvec - norm * np.dot(norm, nvec)
    null = unit(nvec, np.linalg.norm(random)) if norm is not None else None
    return {'visible': visible, 'norm': norm, 'null': null,
            'constant': np.ones(u.size, dtype=np.float64) / np.sqrt(u.size)}


def nominal_delta(u, direction, amplitude):
    u, direction = _array(u), _array(direction)
    if u.ndim != 1 or direction.shape != u.shape or amplitude not in (1/16, 1/4):
        raise ValueError('invalid perturbation')
    if not np.isclose(np.linalg.norm(direction), 1, rtol=1e-10, atol=1e-12):
        raise ValueError('direction must be unit L2')
    return amplitude * np.linalg.norm(u) * direction


def finite_and_linear(u, delta, t, *, epsilon=1e-6):
    """Return exact real head difference, Jacobian prediction, and components."""
    u, delta, t = _array(u), _array(delta), _array(t)
    if u.ndim != 1 or delta.shape != u.shape or t.ndim != 2 or t.shape[1] != u.size or epsilon <= 0:
        raise ValueError('incompatible state/head')
    z = u - u.mean()
    centered_delta = delta - delta.mean()
    sigma2 = np.dot(z, z) / u.size + epsilon
    sigma_new2 = sigma2 + (2 * np.dot(z, delta) + np.dot(centered_delta, centered_delta)) / u.size
    if sigma_new2 <= 0:
        raise ValueError('invalid variance')
    q = t @ z
    direct = t @ centered_delta
    finite = (q + direct) / np.sqrt(sigma_new2) - q / np.sqrt(sigma2)
    linear = (direct - q * np.dot(z, delta) / (u.size * sigma2)) / np.sqrt(sigma2)
    return {'finite': finite, 'linear': linear, 'linear_head': direct / np.sqrt(sigma2),
            'linear_norm': -q * np.dot(z, delta) / (u.size * sigma2 ** 1.5),
            'sigma2': float(sigma2), 'sigma_new2': float(sigma_new2)}
