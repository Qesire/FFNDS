"""CUDA pointwise passes for BF16 joint secants. All launches use the calling stream."""
from functools import lru_cache

import torch


HEADER = """
typedef unsigned short B;
__device__ __forceinline__ float f32(B x) { return __uint_as_float(((unsigned int)x)<<16); }
__device__ __forceinline__ B bf16(float x) {
    B out;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
    return out;
}
"""
KERNELS = {
    "difference": """
extern "C" __global__ void difference(const B* u,const B* a,B* db,float* df,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float d=f32(u[i])-f32(a[i]);db[i]=bf16(d);df[i]=d;}
}""",
    "secant": """
extern "C" __global__ void secant(const B* z,const B* za,const B* h,const B* ha,
    const float* beta,const float* scale,B* db,float* df,float* x,int n,int k){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int j=i%k; B b=bf16(f32(h[i])-f32(ha[i]));float d=f32(b);
        db[i]=b;df[i]=d;x[i]=__fsub_rn(d,__fmul_rn(f32(z[i])-f32(za[i]),beta[j]))/scale[j];}
}""",
    "writeback": """
extern "C" __global__ void writeback(const B* a,const B* linear,const float* selected,
    const float* correction,B* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){out[i]=bf16(((f32(a[i])+f32(linear[i]))+selected[i])+correction[i]);}
}"""
}


@lru_cache(None)
def compile_for(device_index):
    with torch.cuda.device(device_index):
        # A single translation unit works with the 2.8 and 2.10 NVRTC APIs.
        return {k:torch.cuda._compile_kernel('extern "C" {\n'+HEADER+src+'\n}',k)
                for k,src in KERNELS.items()}


def apply(kernel,n,*args):
    kernel(grid=((n+255)//256,1,1),block=(256,1,1),args=list(args))
