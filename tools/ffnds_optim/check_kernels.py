"""Reproduce offline sm80 compilation or explicitly validate CUDA kernels.

Compilation does not initialize a CUDA device. CUDA mode runs synthetic
operator checks only; its tolerances are numerical test criteria, not model
quality bounds or speed measurements. BF16 Triton interpreter is not used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def compile_sm80():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from .partition_triton import _partition_kernel,_selected_kernel
    common=dict(N=31668,H=40,D=128,QSN=5120,QSH=128,KSN=5120,KSH=128,
                VSN=5120,VSH=128,RSH=248*31,RSQ=31,TOP=31,BLOCK_SIZE=128,
                BLOCK_M=32,BLOCK_K=64,SCALE=128**-.5)
    descriptions=[
        (_partition_kernel,dict(ROUTE_PAD=32,WRITE_CENTERED=True),
         dict(Q='*bf16',K='*bf16',V='*bf16',ROUTE='*i32',MEAN='*fp32',INVERSE='*fp32',
              FULL='*bf16',SELECTED='*bf16',RHO='*fp32',FULL_LSE='*fp32',SELECTED_LSE='*fp32',CENTERED='*bf16')),
        (_selected_kernel,dict(NQ=64),
         dict(Q='*bf16',K='*bf16',V='*bf16',ROUTE='*i32',ROW_MAP='*i32',ACTIVE_TILES='*i32',OUT='*bf16',LSE='*fp32')),
    ]
    rows=[]
    for function,extra,signature in descriptions:
        kernel=triton.compile(ASTSource(function,signature=signature,constexprs=dict(common,**extra)),
                              target=GPUTarget('cuda',80,32),
                              options=dict(num_warps=4,num_stages=2,enable_fp_fusion=True))
        rows.append(dict(kernel=function.__name__,target='sm80',
                         ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),
                         ptx_bytes=len(kernel.asm['ptx']),cubin_bytes=len(kernel.asm['cubin']),
                         static_shared_memory_bytes=kernel.metadata.shared,
                         signature=signature,constants=dict(common,**extra),
                         options=dict(num_warps=4,num_stages=2,enable_fp_fusion=True)))
    return dict(status='OFFLINE_SM80_COMPILED_NO_GPU_EXECUTION',torch_version=torch.__version__,
                triton_version=triton.__version__,new_gpu_executions=0,
                kernel_source_sha256=hashlib.sha256(Path(__file__).with_name('partition_triton.py').read_bytes()).hexdigest(),
                kernels=rows)


def _metrics(actual,reference):
    a,b=actual.double(),reference.double()
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def verify_cuda():
    import triton
    import torch.nn.functional as F
    from .attention import partition_attention,selected_attention,value_moments
    if not torch.cuda.is_available():
        return dict(status='PENDING_CUDA',reason='No CUDA device available',new_gpu_executions=0)
    torch.backends.cuda.matmul.allow_tf32=False
    gen=torch.Generator(device='cuda').manual_seed(20261128)
    n,heads,dim,block=259,2,128,128
    q,k,v=[torch.randn(n,heads,dim,generator=gen,device='cuda').bfloat16() for _ in range(3)]
    route=torch.tensor([[[2,0],[1,2],[0,2]],[[1,0],[2,0],[2,1]]],device='cuda',dtype=torch.int32)
    ids=torch.arange(n,device='cuda')
    logits=(q.transpose(0,1).double()@k.permute(1,2,0).double())*dim**-.5
    keyblocks=torch.arange(n,device='cuda')//block
    mask=(route[:,ids//block,:,None]==keyblocks[None,None,None,:]).any(2)
    weights=logits.softmax(-1)
    native=weights@v.transpose(0,1).double()
    small=logits.masked_fill(~mask,-torch.inf).softmax(-1)@v.transpose(0,1).double()
    rho=(weights*(~mask)).sum(-1,keepdim=True)
    mean,_,inverse=value_moments(v)
    centered=(native-(1-rho)*small-rho*mean.double())*inverse.double()
    actual=partition_attention(q,k,v,route,backend='triton',mean=mean,inverse=inverse)
    # The frozen operators write BF16; compare both the BF16 numerical target
    # and full FP64 oracle. Bounds below are solely these seeded operator tests.
    tolerance=dict(value_atol=.015625,value_rtol=.03,mass_atol=3e-5,mass_rtol=3e-4)
    checks={}
    for name,reference in [('full',native),('selected',small),('centered',centered)]:
        value=getattr(actual,name)
        torch.testing.assert_close(value.float(),reference.bfloat16().float(),
                                   atol=tolerance['value_atol'],rtol=tolerance['value_rtol'])
        checks[name]=_metrics(value,reference)
    torch.testing.assert_close(actual.rho.double(),rho,
                               atol=tolerance['mass_atol'],rtol=tolerance['mass_rtol'])
    checks['rho']=_metrics(actual.rho,rho)
    # Original token IDs span all route blocks, are intentionally unsorted,
    # and include the final partial block.
    subset=torch.tensor([258,0,129,127,256,33,128,4,255],device='cuda')
    selected,_=selected_attention(q[subset],k,v,route,subset,backend='triton')
    torch.testing.assert_close(selected.float(),small[:,subset].bfloat16().float(),
                               atol=tolerance['value_atol'],rtol=tolerance['value_rtol'])
    checks['selected_subset']=_metrics(selected,small[:,subset])
    head=lambda x:x.transpose(0,1).unsqueeze(0)
    sdpa=F.scaled_dot_product_attention(head(q),head(k),head(v),dropout_p=0.,is_causal=False,scale=dim**-.5)[0]
    checks['full_vs_native_sdpa']=_metrics(actual.full,sdpa)
    torch.cuda.synchronize()
    return dict(status='SYNTHETIC_CUDA_OPERATOR_CHECKS_PASS',torch_version=torch.__version__,
                triton_version=triton.__version__,device=torch.cuda.get_device_name(),
                shape=[n,heads,dim],input_dtype='bfloat16',seed=20261128,
                new_gpu_executions=2,model_forwards=0,latency_benchmarks=0,
                scope='synthetic operator correctness only; no model-quality or latency claim',
                test_tolerances=tolerance,metrics=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('compile','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compile_sm80() if args.mode=='compile' else verify_cuda()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status']=='PENDING_CUDA':
        raise SystemExit(2)


if __name__=='__main__':
    main()
