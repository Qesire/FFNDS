"""Stage-3 local P512 ICC scoring primitives; no model or job launch.

Canonical dense endpoints call the actual BWS ridge_fit_core.native_ffn. ICC
factor orientation and arithmetic follow the fixed author-four-state and BWS
2R functions; this module retains hidden predictions for the selected 4S grid.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

FAMILIES = ('SVD', 'CA', 'CD')
RANKS = (512, 1408)
MODES = ('4S', '2R')
SITES = (('high',4),('high',19),('high',34),('low',4),('low',19),('low',34))
DEV11 = ('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084',
         'vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')

@dataclass(frozen=True)
class FactorPair:
    left: torch.Tensor
    right: torch.Tensor
    def prefix(self, rank):
        if self.left.ndim != 2 or self.right.ndim != 2 or self.left.shape[1] != self.right.shape[0] or rank > self.left.shape[1]:
            raise ValueError('factor shape/rank mismatch')
        if self.left.dtype != torch.bfloat16 or self.right.dtype != torch.bfloat16:
            raise ValueError('execution factors must already be BF16')
        return FactorPair(self.left[:, :rank], self.right[:rank, :])
    def apply(self, x):
        return F.linear(F.linear(x, self.right), self.left)


def canonical_endpoints(native_ffn, u0, u1, w1, b1, w2, b2, *, rows=512, d=5120, m=13824):
    """Build both endpoints once using the pinned native_ffn call, including biases."""
    operands = (u0,u1,w1,b1,w2,b2)
    if any(x.dtype != torch.bfloat16 for x in operands): raise ValueError('native BF16 operands required')
    if u0.shape != (rows,d) or u1.shape != (rows,d) or w1.shape != (m,d) or b1.shape != (m,) or w2.shape != (d,m) or b2.shape != (d,):
        raise ValueError('native target geometry mismatch')
    z0,h0,y0 = native_ffn(u0,w1,b1,w2,b2)
    z1,h1,y1 = native_ffn(u1,w1,b1,w2,b2)
    return {'u0':u0,'u1':u1,'z0':z0,'h0':h0,'y0':y0,'z1':z1,'h1':h1,'y1':y1}


def icc_prediction(c, f1, f2, rank, mode):
    if mode not in MODES: raise ValueError('invalid cache mode')
    a, b = f1.prefix(rank), f2.prefix(rank)
    if mode == '4S':
        z = a.apply(c['u1'] - c['u0']) + c['z0']
        h = F.gelu(z, approximate='tanh')
        y = b.apply(h - c['h0']) + c['y0']
    else:
        residual_z = c['z0'] - a.apply(c['u0'])
        residual_y = c['y0'] - b.apply(c['h0'])
        z = a.apply(c['u1']) + residual_z
        h = F.gelu(z, approximate='tanh')
        y = b.apply(h) + residual_y
    if not torch.isfinite(h).all().item() or not torch.isfinite(y).all().item():
        raise ValueError('nonfinite ICC prediction')
    return h, y


def gated_risk(pred, target, gate):
    """F64 lift before subtraction and gate, normalized by P*d."""
    if pred.shape != target.shape or pred.ndim != 2 or gate.numel() != pred.shape[1]:
        raise ValueError('prediction/target/gate mismatch')
    error = (pred.double() - target.double()) * gate.reshape(1,-1).double()
    val = error.square().mean()
    if not torch.isfinite(val).item(): raise ValueError('nonfinite risk')
    return float(val)


def delta_alignment(y0, y1, prediction, gate, norm_floor):
    """Compare the true and predicted increments from the same Dense anchor."""
    g = gate.reshape(1, -1).double()
    true = (y1.double() - y0.double()) * g
    predicted = (prediction.double() - y0.double()) * g
    true_norm = float(torch.linalg.vector_norm(true))
    predicted_norm = float(torch.linalg.vector_norm(predicted))
    squared_error = (predicted - true).square()
    result = {'true_norm': true_norm, 'predicted_norm': predicted_norm,
              'absolute_mse': float(squared_error.mean()),
              'norm_ratio': None, 'cosine': None, 'relative_error': None}
    if true_norm > norm_floor:
        result['norm_ratio'] = predicted_norm / true_norm
        result['relative_error'] = float(squared_error.sum()) / true_norm ** 2
        if predicted_norm > norm_floor:
            result['cosine'] = float((true * predicted).sum()) / (true_norm * predicted_norm)
    return result


def error_decomposition(c, h_pred, y_pred, f2, gate):
    """Two F64 B2 factor actions; native BF16 values lifted before subtraction.

    E_read contains native Dense numerical-path error. E_num is residual-defined;
    closure alone is not an independent scientific validation.
    """
    left, right = f2.left.double(), f2.right.double()
    h0 = c['h0'].double()
    def action(x): return F.linear(F.linear(x, right), left)
    A = action(c['h1'].double() - h0)
    B = action(h_pred.double() - h0)
    e_read = (c['y1'].double() - c['y0'].double()) - A
    e_first = A - B
    total = c['y1'].double() - y_pred.double()
    e_num = total - e_read - e_first
    g = gate.reshape(1,-1).double()
    v = [x*g for x in (e_read,e_first,e_num)]
    energy = [float(x.square().mean()) for x in v]
    cross = {'read_first':float(2*(v[0]*v[1]).mean()),
             'read_num':float(2*(v[0]*v[2]).mean()),
             'first_num':float(2*(v[1]*v[2]).mean())}
    gated_total = total*g
    closure = gated_total - sum(v)
    return {'energy':dict(zip(('read','first','num'),energy)), 'cross':cross,
            'total_energy':float(gated_total.square().mean()),
            'closure_max_abs':float(closure.abs().max()),
            'read_semantics':'includes native Dense numerical path',
            'num_semantics':'residual-defined; closure is not independent evidence'}


def validate_grid(rows, sources=DEV11, sites=SITES, ranks=RANKS, modes=MODES, families=FAMILIES):
    expected = {(s,e,l,f,r,c) for s in sources for e,l in sites for f in families for r in ranks for c in modes}
    got = {}
    for row in rows:
        key = (row['source'],row['expert'],row['layer'],row['family'],row['rank'],row['mode'])
        if key not in expected or key in got: raise ValueError('unexpected or duplicated local cell')
        value = row['risk']
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid local risk')
        got[key] = float(value)
    if set(got) != expected: raise ValueError(f'incomplete grid: {len(got)}/{len(expected)}')
    return got


def select_family(rows, *, sources=DEV11, sites=SITES, ranks=RANKS, modes=MODES):
    grid = validate_grid(rows,sources=sources,sites=sites,ranks=ranks,modes=modes)
    if 1408 not in ranks or '4S' not in modes: raise ValueError('main selection subset missing')
    experts = ('high','low')
    if any(len([l for e,l in sites if e == expert]) != 3 for expert in experts):
        raise ValueError('expected three equal-weight layers per expert')
    scores = {}
    for family in FAMILIES:
        per_source = []
        for source in sources:
            per_expert = []
            for expert in experts:
                vals = [grid[(source,e,l,family,1408,'4S')] for e,l in sites if e == expert]
                per_expert.append(math.fsum(vals)/len(vals))
            per_source.append(math.fsum(per_expert)/len(per_expert))
        scores[family] = math.fsum(per_source)/len(per_source)
    return {'family':min(FAMILIES,key=lambda f: (scores[f], FAMILIES.index(f))),
            'risk_by_family':scores,'rule':'source/expert/layer equal-weight, r1408 4S; exact ties SVD then CA then CD'}
