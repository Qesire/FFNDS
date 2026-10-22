"""Frozen random-tensor cost study; no model weights, fitting or videos."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from ar2_checkpoint import atomic_json,digest
import mean_contrast_kernels as fused
import spatial_contrast_kernels as previous
from probe_three_direction_backend import reference_prepare,reference_finish
from probe_mean_contrast_representation import mac_ledger
from worker_identity import require_worker

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/20261023/R15_mean_contrast_backend_protocol.json'


def reconstruct_reference(mean,contrasts):
    g,h=mean.shape
    a,b,c=contrasts.reshape(g,3,h).float().unbind(1)
    m=mean.float()
    return torch.stack((m+(a+b+c)*.5,m+(a-b-c)*.5,
                        m+(-a+b-c)*.5,m+(-a-b+c)*.5),1).bfloat16().reshape(4*g,h)


def pool_reference(v2):
    g,h=len(v2)//4,v2.shape[1]
    a,b,c,e=v2.reshape(g,4,h).float().unbind(1)
    return (((a+b)+(c+e))*.25).bfloat16()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require_worker()
    if args.output.exists() or torch.cuda.device_count()!=1:
        raise RuntimeError('One assigned idle GPU and new report required')
    p=json.loads(PROTOCOL.read_text());protocol_sha=digest(PROTOCOL)
    for name,expected in p['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise RuntimeError('Frozen source changed: '+name)
    torch.set_num_threads(8)
    torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    from quant.hif4_backend import hif4_qdq_tensor
    m,d,h,g,r,rf,k=[p[n] for n in ('M','D','H','stripes','contrast_rank','full_input_factor_rank','two_point_stripes')]
    if (m,d,h,g,r,rf,k)!=(31668,5120,13824,7917,256,1536,3958):
        raise RuntimeError('Fixed synthetic comparison dimensions changed')
    rand=lambda shape,scale=1.:torch.randn(shape,device='cuda',dtype=torch.bfloat16)*scale
    x=rand((m,d))
    w1=hif4_qdq_tensor(rand((h,d),1/math.sqrt(d)),force_fp32=True)
    w2=hif4_qdq_tensor(rand((d,h),1/math.sqrt(h)),force_fp32=True)
    b1,b2,gate=rand((h,),.02),rand((d,),.02),rand((d,),.5)
    scale,shift=rand((d,),.05),rand((d,),.05)
    smooth1,smooth2=torch.ones(d,device='cuda'),torch.ones(h,device='cuda')
    affine=(.5*w2.float()@w1.float()).bfloat16()
    half1,half2,halfb=w1[:h//2].contiguous(),w2[:,:h//2].contiguous(),b1[:h//2].contiguous()
    c1,c2,cb=rand((256,d),1/math.sqrt(d)),rand((d,256),1/math.sqrt(256)),rand((256,),.02)
    v,u=rand((r,d),1/math.sqrt(d)),rand((h,r),1/math.sqrt(r))
    fd=[rand((rf,d),1/math.sqrt(d)),rand((h,rf),1/math.sqrt(rf)),
        rand((rf,h),1/math.sqrt(h)),rand((d,rf),1/math.sqrt(rf))]
    signs=torch.randint(0,2,(g,4),device='cuda').float()*2-1
    ids=torch.arange(g,device='cuda')
    n=g-k
    checks,quant_checks=[],[]
    checking=False

    def compare(name,a,b):
        mismatch=int((a.view(torch.int16)!=b.view(torch.int16)).sum())
        finite=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        checks.append({'name':name,'shape':list(a.shape),'BF16_bit_mismatches':mismatch,'finite':finite})
        if mismatch or not finite:
            raise RuntimeError('Fixed fused arithmetic mismatch: '+name)

    def norm_input():
        norm=F.layer_norm(x,(d,),eps=1e-6)
        return ((norm.float()*(1+scale.float())+shift.float())/smooth1).bfloat16()

    def quant(value):
        if value.dtype!=torch.bfloat16:
            raise RuntimeError('Original BF16 quantizer input required')
        out=hif4_qdq_tensor(value,force_fp32=False)
        if checking:
            ref=hif4_qdq_tensor(value,force_fp32=True)
            mismatch=int((out.view(torch.int16)!=ref.view(torch.int16)).sum())
            quant_checks.append({'shape':list(value.shape),'BF16_bit_mismatches':mismatch})
            if mismatch:raise RuntimeError('Original quantizer control failed')
        return out

    def hidden(z):
        return quant((F.gelu(z,approximate='tanh').float()/smooth2).bfloat16())

    def forward(arm):
        v1=quant(norm_input())
        if arm=='dense':
            return x+F.linear(hidden(F.linear(v1,w1,b1)),w2,b2)*gate
        if arm=='UB1_half_R256_proxy':
            z=F.linear(v1,half1,halfb)
            v2=quant((F.gelu(z,approximate='tanh').float()/smooth2[:h//2]).bfloat16())
            y=F.linear(v2,half2,b2)+F.linear(F.linear(v1,c1,cb),c2)
            return x+y*gate
        if arm=='full_input_factors_r1536':
            v0,u0,v2,u2=fd
            z=F.linear(F.linear(v1,v0),u0,b1)
            return x+F.linear(F.linear(hidden(z),v2),u2,b2)*gate
        if arm=='R14_two_point_fused_fixed':
            hh,mu,c,_=previous.prepare(v1,signs,k,False)
            points=previous.pack(mu,c,ids,k,False)
            response=F.linear(previous.pool(hidden(F.linear(points,w1,b1)),ids,g,k,False),w2,b2)
            return previous.finish(F.linear(hh,affine),response,x,gate)
        use_fused=arm=='mean_contrast_r256_fused'
        hh,mu=(fused.prepare(v1) if use_fused else reference_prepare(v1,signs,0,False)[:2])
        zmean=F.linear(mu,w1,b1)
        contrasts=F.linear(F.linear(hh,v),u)
        z=(fused.reconstruct if use_fused else reconstruct_reference)(zmean,contrasts)
        pooled=(fused.pool if use_fused else pool_reference)(hidden(z))
        response=F.linear(pooled,w2,b2)
        projected=F.linear(hh,affine)
        return (previous.finish if use_fused else reference_finish)(projected,response,x,gate)

    v1=hif4_qdq_tensor(norm_input(),force_fp32=False)
    refh,refmu,_,_=reference_prepare(v1,signs,0,False)
    hh,mu=fused.prepare(v1)
    compare('prepared_three_contrasts',hh,refh);compare('prepared_mean',mu,refmu)
    zmean=F.linear(mu,w1,b1);ch3=F.linear(F.linear(hh,v),u)
    z=fused.reconstruct(zmean,ch3)
    compare('four_preactivation_rows',z,reconstruct_reference(zmean,ch3))
    v2=hidden(z)
    pooled=fused.pool(v2)
    compare('pooled_four_post_Q2_rows',pooled,pool_reference(v2))
    response=F.linear(pooled,w2,b2);projected=F.linear(hh,affine)
    compare('inverse_gate_residual',previous.finish(projected,response,x,gate),reference_finish(projected,response,x,gate))
    compare('full_pipeline',forward('mean_contrast_r256_fused'),forward('mean_contrast_r256_unfused'))
    del v1,refh,refmu,hh,mu,zmean,ch3,z,v2,pooled,response,projected
    checking=True
    for arm in p['arms']:
        out=forward(arm)
        if list(out.shape)!=[m,d] or not bool(torch.isfinite(out).all()):
            raise RuntimeError('Synthetic output finite/shape control failed')
        del out
    if len(quant_checks)!=12:raise RuntimeError('Every arm needs both original QDQ controls')
    checking=False
    for _ in range(p['warmups']):
        for arm in p['arms']:
            out=forward(arm);del out
    torch.cuda.synchronize()
    samples={a:[] for a in p['arms']};host={a:[] for a in p['arms']};orders=[]
    for repeat in range(p['repetitions']):
        order=p['arms'][repeat%len(p['arms']):]+p['arms'][:repeat%len(p['arms'])]
        if repeat%2:order=list(reversed(order))
        orders.append(order)
        for arm in order:
            torch.cuda.synchronize();start=time.monotonic()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();out=forward(arm);end.record();end.synchronize()
            host[arm].append((time.monotonic()-start)*1000)
            samples[arm].append(begin.elapsed_time(end))
            if not bool(torch.isfinite(out).all()):raise RuntimeError('Nonfinite timed output')
            del out
    medians={a:statistics.median(s) for a,s in samples.items()}
    mad={a:statistics.median(abs(v-medians[a]) for v in s) for a,s in samples.items()}
    for name,expected in p['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Source changed during timing')
    if digest(PROTOCOL)!=protocol_sha or torch.cuda.max_memory_allocated()>32*2**30:
        raise RuntimeError('Frozen scope changed or allocated memory cap exceeded')
    saving=medians['mean_contrast_r256_unfused']-medians['mean_contrast_r256_fused']
    margin=max(.5,3*max(mad.values()))
    result={'id':p['id'],'status':'COMPLETED_FIXED_SYNTHETIC_CONTRAST_REPRESENTATION_COST',
            'protocol_sha256':protocol_sha,'source_sha256':p['source_sha256'],
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'fusion_controls':checks,'original_QDQ_controls':quant_checks,'all_controls_before_timing':True,
            'samples_ms':samples,'host_samples_ms':host,'timing_order':orders,'median_ms':medians,'MAD_ms':mad,
            'speedup_over_same_run_dense':{a:medians['dense']/v for a,v in medians.items()},
            'mean_contrast_MAC_ledger':mac_ledger(),'full_input_factor_MAC_fraction':rf*(d+h)/(d*h),
            'fused_saving_ms':saving,'noise_margin_ms':margin,'backend_fusion_effect':saving>=margin,
            'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),
            'models_loaded':0,'new_model_generations':0,'fitted_parameters':0,'tensor_files_written':0,
            'native_model_or_quality_acceptance':False,'model_execution_eligibility_change':False,
            'limitations':p['limitations']}
    atomic_json(args.output,result)
    print(json.dumps({'median_ms':medians,'same_run_dense_speedups':result['speedup_over_same_run_dense'],
                      'backend_fusion_effect':result['backend_fusion_effect']}),flush=True)


if __name__=='__main__':
    main()
