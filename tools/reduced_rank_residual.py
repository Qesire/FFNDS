"""Whitened reduced-rank ridge and its explicit native BF16 predictor."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_rrr(x,y,regularization,max_rank,*,seed=20261021):
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or regularization<=0:
        raise ValueError('Invalid residual training geometry/regularization')
    x,y=x.double(),y.double()
    mean_x,mean_y=x.mean(0),y.mean(0)
    scale=(x-mean_x).square().mean().sqrt().clamp_min(1e-8)
    centered=(x-mean_x)/scale
    covariance=centered.T@centered/len(x)
    covariance.diagonal().add_(regularization)
    chol=torch.linalg.cholesky(covariance)
    cross=centered.T@(y-mean_y)/len(x)
    whitened=torch.linalg.solve_triangular(chol,cross,upper=False)
    q=min(max_rank+32,min(whitened.shape))
    devices=[x.device.index] if x.device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        u,s,v=torch.svd_lowrank(whitened,q=q,niter=3)
    rank=min(max_rank,len(s))
    left=torch.linalg.solve_triangular(chol.T,u[:,:rank]*s[:rank],upper=True)/scale
    return {'left':left,'right':v[:,:rank],'mean_x':mean_x,'mean_y':mean_y,
            'whitened_singular_values':s,'scale_x':scale}


def native_parameters(model,gate,rank):
    if not 0<rank<=model['left'].shape[1]:
        raise ValueError('Rank outside fitted factors')
    gate=gate.reshape(-1).double()
    inverse=torch.where(gate!=0,gate.reciprocal(),0.)
    first=model['left'][:,:rank].T
    second=model['right'][:,:rank]*inverse[:,None]
    bias_first=-(model['mean_x']@model['left'][:,:rank])
    bias_second=model['mean_y']*inverse
    parameters={n:t.to(torch.bfloat16).contiguous() for n,t in
                [('first',first),('second',second),('bias_first',bias_first),('bias_second',bias_second)]}
    if any(not bool(torch.isfinite(t).all()) for t in parameters.values()):
        raise FloatingPointError('Nonfinite native residual predictor')
    return parameters


def predict_y(x,parameters,*,fp32=False):
    dtype=torch.float32 if fp32 else torch.bfloat16
    hidden=F.linear(x.to(dtype),parameters['first'].to(dtype),parameters['bias_first'].to(dtype))
    return F.linear(hidden,parameters['second'].to(dtype),parameters['bias_second'].to(dtype))
