"""Positive scalar energy matching with explicit original-bias accounting."""
import numpy as np
import torch


def affine_reference(value,beta,shift,original_bias,*,fp32=False):
    centered=value.float()-original_bias.float()
    scaled=centered*float(np.float32(beta))
    result=(scaled+shift.float())+original_bias.float()
    return result if fp32 else result.to(torch.bfloat16)


def fit_output_moments(dense,candidate,gate,original_bias):
    if (dense.shape!=candidate.shape or dense.ndim!=2 or dense.dtype!=torch.bfloat16
            or candidate.dtype!=torch.bfloat16 or original_bias.numel()!=dense.shape[1]
            or gate.numel()!=dense.shape[1]):
        raise ValueError('Need matching native BF16 outputs and one gate/bias vector')
    d=dense.double()-original_bias.reshape(1,-1).double()
    s=candidate.double()-original_bias.reshape(1,-1).double()
    md,ms=d.mean(0),s.mean(0);g=gate.reshape(1,-1).double()
    ed=float(((d-md)*g).square().sum()/len(d))
    es=float(((s-ms)*g).square().sum()/len(s))
    if not np.isfinite(ed+es) or min(ed,es)<=dense.shape[1]*1e-24:
        raise ValueError('Degenerate or nonfinite centered gated energy')
    beta_fit=float(np.sqrt(ed/es));beta=float(np.float32(beta_fit))
    if not .5<=beta<=2:
        raise ValueError('Variance coefficient outside the registered range; no clipping')
    shift=(md-beta*ms).float()
    if not bool(torch.isfinite(shift).all()):raise ValueError('Nonfinite affine mean shift')
    return {'beta':beta,'beta_fit_FP64':beta_fit,'shift':shift,'dense_centered_energy':ed,
        'candidate_centered_energy':es,'original_centered_energy_ratio':es/ed}
