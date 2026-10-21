"""Explicit matched-input functions for the separately scoped original-O audit."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def raw_native(u,parameters,second_stride):
    w0 = parameters['0.weight'].T.contiguous()
    source2 = parameters['2.weight'].T
    w2 = torch.empty_strided(tuple(source2.shape),tuple(second_stride),device=u.device,dtype=u.dtype)
    w2.copy_(source2)
    z = torch.empty((len(u),w0.shape[1]),device=u.device,dtype=u.dtype)
    torch.addmm(parameters['0.bias'],u,w0,out=z)
    hidden = F.gelu(z,approximate='tanh')
    y = torch.empty((len(u),w2.shape[1]),device=u.device,dtype=u.dtype)
    torch.addmm(parameters['2.bias'],hidden,w2,out=y)
    return z,hidden,y


@torch.no_grad()
def quantized_native(u,records,bias0,bias2,quantizer):
    r0,r2 = records
    q1 = quantizer((u.float()/r0['smooth_scale']).to(u.dtype),force_fp32=True)
    z = F.linear(q1,r0['weight_qdq'],bias0)
    hidden = F.gelu(z,approximate='tanh')
    q2 = quantizer((hidden.float()/r2['smooth_scale']).to(u.dtype),force_fp32=True)
    y = F.linear(q2,r2['weight_qdq'],bias2)
    return q2,y


def paired_error(candidate,reference):
    if (candidate.shape != reference.shape or not torch.isfinite(candidate).all()
            or not torch.isfinite(reference).all()):
        raise ValueError('Paired finite same-shape native outputs required')
    x,y = candidate.double(),reference.double()
    error = float((x-y).square().sum())
    norm = float(y.square().sum())
    if norm <= 0:
        raise ValueError('Positive reference norm required')
    return {'squared_error':error,'reference_squared_norm':norm,'relative_RMSE':math.sqrt(error/norm)}


def error_alignment(distilled,original,quantized):
    first,second = (distilled.double()-original.double()).reshape(-1),(quantized.double()-distilled.double()).reshape(-1)
    denominator = float(first.norm()*second.norm())
    return float(first@second)/denominator if denominator > 0 else None


@torch.no_grad()
def concentration(hidden,weight,gate):
    if hidden.shape[1] != 13824 or weight.shape != (5120,13824):
        raise ValueError('Original216 whole64-channel groups required')
    scores = torch.empty((len(hidden),216),device=hidden.device,dtype=torch.float64)
    h,w,g = hidden.float(),weight.float(),gate.float()
    for group in range(216):
        sl = slice(group*64,(group+1)*64)
        contribution = F.linear(h[:,sl].contiguous(),w[:,sl].contiguous())*g
        scores[:,group] = contribution.double().square().sum(1)
    return normalized_concentration(scores)


def normalized_concentration(scores):
    if scores.ndim != 2 or scores.shape[1] != 216:
        raise ValueError('Exactly216 original group scores required')
    scores = scores.double()
    totals = scores.sum(1,keepdim=True)
    if not torch.isfinite(scores).all() or torch.any(scores < 0) or torch.any(totals <= 0):
        raise ValueError('Finite positive contribution energies required')
    p = scores/totals
    entropy = -torch.where(p>0,p*torch.log(p),0.).sum(1)/math.log(216)
    return {'mean_row_oracle_top108_retained_C':float(p.topk(108,dim=1).values.sum(1).mean()),
            'mean_row_normalized_C_entropy':float(entropy.mean()),
            'scope':'same-D-gated descriptive original-coordinate contribution energies; offline oracle, no selector cost or intrinsic sparsity claim'}
