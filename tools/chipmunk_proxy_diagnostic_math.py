"""CPU-only numerical primitives for a *proposed* saved-h diagnostic.

All selections other than the frozen proxy are retrospective oracles.
"""
import numpy as np


def stable_topk(values, k):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= k <= len(values):
        raise ValueError('invalid scores or K')
    return np.sort(np.lexsort((np.arange(len(values)), -values))[:k])


def diagnose(delta_h, weight, gate, proxy):
    """Return diagonal energies and signed cross terms with fixed masks.

    delta_h [rows,H], weight [D,H], gate [D]; float64 accumulation. This is
    an oracle analysis of already-produced h, not an implementable selector.
    """
    h, w, g = (np.asarray(x, dtype=np.float64) for x in (delta_h, weight, gate))
    if h.ndim != 2 or w.ndim != 2 or w.shape[1] != h.shape[1] or g.shape != (w.shape[0],):
        raise ValueError('operand shape mismatch')
    if not all(np.isfinite(x).all() for x in (h,w,g)):
        raise ValueError('nonfinite operand')
    p = np.asarray(proxy, dtype=np.int64) if len(proxy) == 0 else np.asarray(proxy)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer) or len(np.unique(p)) != len(p) or np.any(p < 0) or np.any(p >= h.shape[1]):
        raise ValueError('invalid proxy indices')
    energy = np.sum(h*h,axis=0); consumer = np.sum((w*g[:,None])**2,axis=0); weighted=energy*consumer
    masks={'proxy':np.sort(p),'activation_oracle':stable_topk(energy,len(p)), 'consumer_oracle':stable_topk(weighted,len(p))}
    full = h @ (w*g[:,None]).T
    full_energy = float(np.sum(full*full))
    arms={}
    for name, selected in masks.items():
        omitted=np.ones(h.shape[1],dtype=bool); omitted[selected]=False
        y=h[:,omitted] @ (w[:,omitted]*g[:,None]).T
        total=float(np.sum(y*y)); diag=float(np.sum(weighted[omitted]));
        arms[name]={'selected':selected.tolist(),'omitted_diagonal_energy':diag,'omitted_projected_energy':total,'signed_cross_energy':total-diag, 'relative_to_full_delta_energy': total/full_energy if full_energy else None}
    return {'full_projected_energy':full_energy,'activation_energy':energy.tolist(),'consumer_column_norm_squared':consumer.tolist(),'arms':arms}
