"""The frozen dense D-Q FFN function, starting at actual native postQ1 V1."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def from_v1(v1,w0,b0,w2,b2,smooth2,qdq,*,linear=F.linear):
    if (v1.dtype!=torch.bfloat16 or w0.dtype!=v1.dtype or w2.dtype!=v1.dtype
            or v1.shape[-1]!=5120 or w0.shape!=(13824,5120) or w2.shape!=(5120,13824)):
        raise ValueError('Need original full-width native BF16 D-Q FFN operands')
    z=linear(v1,w0,b0)
    hidden=F.gelu(z,approximate='tanh')
    v2=qdq((hidden.float()/smooth2).to(torch.bfloat16),force_fp32=True)
    y=linear(v2,w2,b2)
    return y,v2
