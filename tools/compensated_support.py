"""Fixed-feature real OLS and a finite equal-budget support-swap proxy."""
import itertools

import numpy as np
import torch


class FixedDesign:
    def __init__(self, hidden):
        if hidden.ndim != 2 or len(hidden) < 2 or not torch.isfinite(hidden).all():
            raise ValueError('Finite training hidden rows required')
        self.phi = self.features(hidden)
        u, s, vh = torch.linalg.svd(self.phi, full_matrices=False,
                                  driver='gesvd' if hidden.device.type == 'cuda' else None)
        self.rtol = max(self.phi.shape) * torch.finfo(torch.float64).eps
        selected = s > self.rtol * s[0]
        self.u, self.s, self.vh = u[:, selected], s[selected], vh[selected]
        self.rank = len(self.s)
        self.condition = float(self.s[0] / self.s[-1])

    @staticmethod
    def features(hidden):
        return torch.cat([hidden.double(), torch.ones((len(hidden), 1), dtype=torch.float64,
                                                     device=hidden.device)], dim=1)

    def fit(self, target):
        if target.ndim != 2 or len(target) != len(self.phi) or not torch.isfinite(target).all():
            raise ValueError('Finite targets paired with training design required')
        return self.vh.T @ ((self.u.T @ target.double()) / self.s[:, None])

    def residual(self, values):
        if values.ndim != 2 or len(values) != len(self.phi):
            raise ValueError('Training rows required for the implicit projection')
        values = values.double()
        return values - self.u @ (self.u.T @ values)


def proposal_pool(kept, linear, *, width=8):
    kept = sorted(map(int, kept))
    linear = np.asarray(linear, dtype=np.float64)
    if (linear.ndim != 1 or not np.isfinite(linear).all() or len(set(kept)) != len(kept)
            or not kept or not set(kept).issubset(range(len(linear))) or width < 1):
        raise ValueError('Valid unique kept groups and finite group scores required')
    omitted = sorted(set(range(len(linear))) - set(kept))
    removed = sorted(kept, key=lambda i: (linear[i], i))[:width]
    added = sorted(omitted, key=lambda i: (-linear[i], i))[:width]
    return sorted(removed), sorted(added)


def linear_upper_bound(kept, linear, q=2):
    removed, added = proposal_pool(kept, linear, width=len(linear))
    incoming = sorted((linear[i] for i in added), reverse=True)
    outgoing = sorted((-linear[i] for i in removed), reverse=True)
    return float(max([0.] + [sum(incoming[:k]) + sum(outgoing[:k])
                            for k in range(1, min(q, len(added), len(removed)) + 1)]))


def finite_swaps(removed, added, linear, gram, q=2):
    pool = list(removed) + list(added)
    if (len(pool) != len(set(pool)) or not removed or not added or q < 1
            or not np.isfinite(gram).all() or np.shape(gram) != (len(pool), len(pool))):
        raise ValueError('Disjoint proposal groups and a finite matching Gram required')
    local = {group: i for i, group in enumerate(pool)}
    linear = np.asarray(linear, dtype=np.float64)
    scores = linear[pool]
    actions = []
    for count in range(min(q, len(removed), len(added)) + 1):
        for outgoing in itertools.combinations(sorted(removed), count):
            for incoming in itertools.combinations(sorted(added), count):
                s = np.zeros(len(pool))
                for i in outgoing:
                    s[local[i]] = -1
                for i in incoming:
                    s[local[i]] = 1
                gain = float(scores @ s - s @ gram @ s)
                actions.append({'removed': list(outgoing), 'added': list(incoming),
                                'proxy_advantage': gain, 'swap_count': count})
    actions.sort(key=lambda v: (-v['proxy_advantage'], v['removed'], v['added']))
    best_nonzero = next(v for v in actions if v['swap_count'])
    return {'candidate_count_including_identity': len(actions), 'best': actions[0],
            'best_nonzero': best_nonzero, 'all_actions': actions}


def discrepancy_certificate(error, proxy_delta, native_delta):
    error, proxy_delta, native_delta = [v.double() for v in [error, proxy_delta, native_delta]]
    eta = native_delta - proxy_delta
    native = float(-2 * (error * native_delta).sum() - native_delta.square().sum())
    proxy = float(-2 * (error * proxy_delta).sum() - proxy_delta.square().sum())
    bound = float(2 * torch.linalg.vector_norm(error + proxy_delta) * torch.linalg.vector_norm(eta)
                  + eta.square().sum())
    tolerance = 1e-10 * max(1., abs(native), abs(proxy), bound)
    if abs(native - proxy) > bound + tolerance:
        raise RuntimeError('Selected-action native discrepancy inequality failed')
    return {'native_advantage': native, 'anchored_proxy_advantage': proxy,
        'absolute_advantage_discrepancy': abs(native - proxy), 'bound': bound,
        'eta_L2': float(torch.linalg.vector_norm(eta)),
        'proxy_delta_L2': float(torch.linalg.vector_norm(proxy_delta)),
        'inequality_verified': True, 'scope': 'this selected action only; no uniform neighborhood bound'}
