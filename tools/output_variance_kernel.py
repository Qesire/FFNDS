"""Fused FP32 signal affine with native BF16 input/output and explicit bias."""
import torch
import triton
import triton.language as tl


@triton.jit
def _affine(y,beta,shift,bias,out,N:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    columns=offsets%D
    value=tl.load(y+offsets,offsets<N,other=0).to(tl.float32)
    original=tl.load(bias+columns).to(tl.float32)
    coefficient=tl.load(beta).to(tl.float32)
    displacement=tl.load(shift+columns).to(tl.float32)
    centered=value-original
    scaled=centered*coefficient
    corrected=scaled+displacement
    corrected=corrected+original
    tl.store(out+offsets,corrected,offsets<N)


def affine_cuda(value,parameters,*,identity=False):
    if (value.device.type!='cuda' or value.dtype!=torch.bfloat16 or not value.is_contiguous()
            or value.ndim!=2 or value.shape[1]!=5120):
        raise ValueError('Expected contiguous native CUDA BF16 FFN2 output')
    if identity:return value
    if (parameters['beta'].dtype!=torch.float32 or parameters['shift'].dtype!=torch.float32
            or parameters['original_b2'].dtype!=torch.bfloat16
            or any(v.device!=value.device for v in parameters.values())):
        raise ValueError('Affine coefficient device/dtype differs')
    output=torch.empty_like(value)
    _affine[(triton.cdiv(value.numel(),1024),)](value,parameters['beta'],parameters['shift'],parameters['original_b2'],
        output,value.numel(),5120,1024,num_warps=4,enable_fp_fusion=False)
    return output
