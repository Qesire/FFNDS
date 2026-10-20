"""Full-rank retained-feature ridge, folded into one native packed FFN2."""
import torch


@torch.no_grad()
def fit_reconstruction(x,dense_y,base_y,gate,prior_weight,prior_bias,regularization):
    if (x.ndim!=2 or dense_y.shape!=base_y.shape or len(x)!=len(dense_y)
            or regularization<=0 or prior_weight.shape!=(dense_y.shape[1],x.shape[1])):
        raise ValueError('Invalid anchored reconstruction geometry/regularization')
    x=x.double();n=len(x)
    mean_x=x.mean(0)
    scale=(x-mean_x).square().mean().sqrt().clamp_min(1e-8)
    centered=(x-mean_x)/scale
    g=gate.reshape(-1).double()
    residual=((dense_y.float()-base_y.float())*gate.reshape(1,-1).float()).double()
    mean_r=residual.mean(0);target=residual-mean_r
    if n<=x.shape[1]:
        covariance=centered@centered.T/n
        covariance.diagonal().add_(regularization)
        dual=torch.cholesky_solve(target/n,torch.linalg.cholesky(covariance))
        correction=centered.T@dual/scale
    else:
        covariance=centered.T@centered/n
        covariance.diagonal().add_(regularization)
        correction=torch.cholesky_solve(centered.T@target/n,torch.linalg.cholesky(covariance))/scale
    inverse=torch.where(g!=0,g.reciprocal(),0.)
    delta_weight=correction.T*inverse[:,None]
    delta_bias=(mean_r-mean_x@correction)*inverse
    weight=(prior_weight.double()+delta_weight).to(torch.bfloat16).contiguous()
    bias=(prior_bias.double()+delta_bias).to(torch.bfloat16).contiguous()
    if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
        raise FloatingPointError('Nonfinite reconstructed native parameters')
    return weight,bias


def bf16_bits(tensor):
    if tensor.dtype!=torch.bfloat16:raise ValueError('Exact BF16 encoding required')
    return tensor.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def from_bf16_bits(array,device='cpu'):
    tensor=torch.from_numpy(array)
    if tensor.dtype!=torch.uint16:raise ValueError('Exact unsigned16 BF16 payload required')
    return tensor.view(torch.bfloat16).to(device)
