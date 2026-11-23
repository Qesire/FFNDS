"""Single-pass native BF16 GELU lookup and normalized secant. The table is 128 KiB."""
from functools import lru_cache

import torch
import torch.nn.functional as F

from fused_ops import HEADER,apply


SRC="""
extern "C" __global__ void lookup_secant(const B* z,const B* a,const B* table,
    const float* beta,const float* scale,B* db,float* df,float* x,int n,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int j=i%k;B z1=z[i],z0=a[i];
        B b=bf16(f32(table[z1])-f32(table[z0]));float d=f32(b);
        db[i]=b;df[i]=d;x[i]=__fsub_rn(d,__fmul_rn(f32(z1)-f32(z0),beta[j]))/scale[j];}
}"""


@lru_cache(None)
def prepare_lookup(device):
    with torch.cuda.device(device):
        bit_patterns=torch.arange(65536,dtype=torch.int32,device="cuda").to(torch.int16)
        table=F.gelu(bit_patterns.view(torch.bfloat16),approximate="tanh")
        kernel=torch.cuda._compile_kernel('extern "C" {\n'+HEADER+SRC+'\n}',"lookup_secant")
    return table,kernel


def call(z1,z0,beta,scale,table,kernel):
    db=torch.empty_like(z1)
    df=torch.empty_like(z1,dtype=torch.float32)
    x=torch.empty_like(df)
    apply(kernel,z1.numel(),z1,z0,table,beta,scale,db,df,x,z1.numel(),z1.shape[-1])
    return db,df,x
