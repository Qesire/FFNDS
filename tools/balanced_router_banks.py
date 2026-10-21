"""Fixed equal-size assignment variant; the original RB1 algorithm is immutable."""
import numpy as np

from router_bank_feasibility import DegenerateBanks, squared_distances


def balanced_assignments(distances):
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 4 or len(distances) % 4 or not np.isfinite(distances).all():
        raise ValueError('Finite N-by4 distances with equal integer capacities required')
    nearest = np.sort(distances, axis=1)[:, :2]
    margin = nearest[:, 1] - nearest[:, 0]
    order = np.argsort(-margin, kind='stable')
    remaining = np.full(4, len(distances) // 4, dtype=np.int64)
    labels = np.empty(len(distances), dtype=np.int64)
    for row in order:
        costs = np.where(remaining > 0, distances[row], np.inf)
        bank = int(costs.argmin())
        labels[row] = bank
        remaining[bank] -= 1
    if remaining.any():
        raise RuntimeError('Equal-size assignment did not fill all capacities')
    return labels


def fit_balanced_banks(profiles, keep, seed, *, min_bank=128):
    x = np.asarray(profiles, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 216 or len(x) % 4 or len(x) < 4 * min_bank
            or not 0 < keep < 216 or not np.isfinite(x).all()
            or not np.allclose(x.sum(1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Valid normalized216-group equal-size training profiles required')
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))].copy()]
    for _ in range(3):
        distance = squared_distances(x, np.asarray(centers)).min(1)
        if distance.sum() <= 0:
            raise DegenerateBanks('Degenerate kmeans++ initialization; do not reseed')
        centers.append(x[int(rng.choice(len(x), p=distance / distance.sum()))].copy())
    centers = np.asarray(centers)
    for _ in range(20):
        assignments = balanced_assignments(squared_distances(x, centers))
        centers = np.stack([x[assignments == i].mean(0) for i in range(4)])
    assignments = balanced_assignments(squared_distances(x, centers))
    counts = np.bincount(assignments, minlength=4)
    if np.any(counts != len(x) // 4) or np.any(counts < min_bank):
        raise DegenerateBanks('Balanced bank does not meet unchanged sample minimum')
    scores = np.stack([x[assignments == i].mean(0) for i in range(4)])
    masks = np.sort(np.argsort(-scores, axis=1, kind='stable')[:, :keep], axis=1)
    static = np.sort(np.argsort(-x.mean(0), kind='stable')[:keep])
    return {'centers': centers, 'assignments': assignments, 'counts': counts,
            'masks': masks, 'static': static}
