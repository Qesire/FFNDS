"""One anchored nonlinear-feature update followed by a replacement V1 residual."""
import torch
import torch.nn.functional as F

from anchored_reconstruction import fit_reconstruction
from reduced_rank_residual import fit_rrr,native_parameters,predict_y


def project_rows(features,weight,bias,*,rows=512):
    return torch.cat([F.linear(part,weight,bias) for part in features.split(rows)])


@torch.no_grad()
def joint_update(features,x,dense,base,gate,prior_weight,prior_bias,initial_residual,quantizer,
                 *,regularization_W=1.,regularization_R=1.,rank=256,basis_rank=512):
    # The existing BF16 sum is the actual teacher-target residual here. This
    # does not fold alpha into W2: features already contain its native cast.
    initial=base+predict_y(x,initial_residual)
    w,b=fit_reconstruction(features,dense,initial,gate,prior_weight,prior_bias,regularization_W)
    qw=quantizer(w,force_fp32=True).to(torch.bfloat16).contiguous()
    if qw.shape!=prior_weight.shape or not bool(torch.isfinite(qw).all()):
        raise FloatingPointError('Invalid reconstructed HiF4 effective W2')
    updated_base=project_rows(features,qw,b)
    target=(dense.float()-updated_base.float())*gate.reshape(1,-1).float()
    fit=fit_rrr(x,target,regularization_R,basis_rank)
    residual=native_parameters(fit,gate,rank)
    return {'weight':qw,'pre_quant_weight':w,'bias':b,'residual':residual}


def output(features,x,model,*,fp32=False):
    dtype=torch.float32 if fp32 else torch.bfloat16
    main=project_rows(features.to(dtype),model['weight'].to(dtype),model['bias'].to(dtype))
    return main+predict_y(x,model['residual'],fp32=fp32)
