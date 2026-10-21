"""Synthetic A100 prerequisite for full-width 2:4 weights in both FFN linears."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

from ar2_checkpoint import atomic_json, digest
from worker_identity import require_worker

ROOT=Path(__file__).resolve().parents[1]


def rel_error(actual,reference):
    a,b=actual.float(),reference.float()
    return float(((a-b).square().sum()/b.square().sum()).sqrt())


@torch.inference_mode()
def main(protocol_path=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    require_worker()
    if args.report.exists():
        raise RuntimeError('Preserve first shape-preflight report')
    protocol_path=protocol_path or ROOT/'config/weight_2of4_shape_preflight.json'
    p=json.loads(protocol_path.read_text())
    if not torch.backends.cusparselt.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Require existing cuSPARSELt and exactly one assigned device')
    torch.set_num_threads(8)
    torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    m,d,h,r=(p[k] for k in ['M','D','H','rank'])
    from quant.hif4_backend import hif4_qdq_tensor
    quant=lambda x:hif4_qdq_tensor(x,force_fp32=True).to(torch.bfloat16)
    random=lambda shape,scale=1.:torch.randn(shape,device='cuda',dtype=torch.bfloat16)*scale
    start=time.monotonic()
    x=random((m,d));w0=quant(random((h,d),1/math.sqrt(d)));w2=quant(random((d,h),1/math.sqrt(h)))
    b0,b2=random((h,),.02),random((d,),.02)
    s1,s2=torch.ones(d,device='cuda'),torch.ones(h,device='cuda')
    gate=random((d,),.5);r0,r2=random((r,d),1/math.sqrt(d)),random((d,r),.01/math.sqrt(r))

    def mask(w):
        groups=w.reshape(w.shape[0],-1,4)
        take=groups.abs().topk(2,dim=-1).indices
        chosen=torch.zeros_like(groups).scatter_(-1,take,groups.gather(-1,take))
        return chosen.reshape_as(w).contiguous()

    masked0,masked2=mask(w0),mask(w2)
    packed0,packed2=to_sparse_semi_structured(masked0),to_sparse_semi_structured(masked2)
    if not torch.equal(packed0.to_dense(),masked0) or not torch.equal(packed2.to_dense(),masked2):
        raise RuntimeError('Sparse weight roundtrip differs numerically')
    controls=[]
    for left,dense,sparse,label in [(x,masked0,packed0,'FFN0'),(random((m,h)),masked2,packed2,'FFN2')]:
        reference=F.linear(left,dense)
        raw=F.linear(left,sparse)
        observed=raw.contiguous()
        error=rel_error(observed,reference)
        controls.append({'linear':label,'relative_RMSE':error,'all_values_finite':bool(torch.isfinite(observed).all()),
                         'packed_weight_numeric_roundtrip':True,
                         'sparse_raw_output_stride':list(raw.stride()),
                         'sparse_raw_output_is_contiguous':raw.is_contiguous()})
        if not controls[-1]['all_values_finite'] or error>.01:
            raise RuntimeError('Sparse linear did not pass registered numerical control')
        del reference,observed,raw
    # Release last full-H diagnostic input before peak-memory/timing measurement.
    del left

    def forward(arm):
        v1=quant((x.float()/s1).to(torch.bfloat16))
        if arm=='dense_full':a,b=w0,w2
        elif arm=='dense_masked_control':a,b=masked0,masked2
        else:a,b=packed0,packed2
        z=F.linear(v1,a,b0).contiguous()
        y=F.gelu(z,approximate='tanh')
        v2=quant((y.float()/s2).to(torch.bfloat16))
        result=F.linear(v2,b,b2).contiguous()
        if arm=='sparse_2of4_plus_R256_workload':
            result=result+F.linear(F.linear(v1,r0),r2)
        return result*gate

    native_masked=forward('dense_masked_control')
    native_sparse=forward('sparse_2of4')
    chain_error=rel_error(native_sparse,native_masked)
    if not bool(torch.isfinite(native_sparse).all()):
        raise RuntimeError('Nonfinite complete synthetic sparse output')
    del native_masked,native_sparse
    preparation_seconds=time.monotonic()-start
    samples={arm:[] for arm in p['arms']}
    for _ in range(p['warmups']):
        for arm in p['arms']:forward(arm)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for repetition in range(p['repetitions']):
        order=p['arms'] if repetition%2==0 else list(reversed(p['arms']))
        for arm in order:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();output=forward(arm);end.record();end.synchronize()
            samples[arm].append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):raise RuntimeError('Nonfinite timed output')
            del output
    medians={k:statistics.median(v) for k,v in samples.items()}
    result={'status':'COMPLETE_SYNTHETIC_2OF4_SHAPE_PREFLIGHT',
            'protocol_sha256':digest(protocol_path),'source_sha256':digest(Path(__file__)),
            'entrypoint_sha256':digest(Path(sys.argv[0])),
            'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
            'torch':torch.__version__,'cuda_build':torch.version.cuda,'backend':type(packed0).__name__,
            'shape':{'M':m,'D':d,'H':h,'rank':r},'linear_controls':controls,
            'complete_chain_sparse_vs_dense_masked_relative_RMSE':chain_error,
            'milliseconds_samples':samples,'median_milliseconds':medians,
            'dense_full_over_sparse_median_ratio':medians['dense_full']/medians['sparse_2of4'],
            'dense_full_over_sparse_plus_R256_workload_median_ratio':medians['dense_full']/medians['sparse_2of4_plus_R256_workload'],
            'maximum_allocated_GPU_bytes':torch.cuda.max_memory_allocated(),
            'preparation_seconds_excluded_from_timing':preparation_seconds,
            'external_sources':{'.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py':digest(ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py'),
                                'installed_torch/sparse/semi_structured.py':digest(Path(torch.__file__).parent/'sparse/semi_structured.py')},
            'models_loaded':0,'fitted_parameters':0,'quality_or_speed_acceptance':False,
            'limitations':['Synthetic random weights/inputs; not real Wan accuracy or a deployed candidate.',
                           'One shape and installed default sparse algorithm; no full160 benchmark or video speed.',
                           'Full-H native HiF4 activation QDQ, GELU, bias, output layout copies and gate included.',
                           'Smooth scales are ones. R256 arm adds two GEMMs/addition only, not original residual preprocessing or coefficients.',
                           'Resident tensors; compression, transfers and setup excluded. All timing samples retained.',
                           'Sparse BF16 accumulation can differ; full QDQ chain discrepancy is reported, not assumed bitwise equal.']}
    args.report.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.report,result)
    print(json.dumps({'status':result['status'],'median_milliseconds':medians,
                      'dense_full_over_sparse_median_ratio':result['dense_full_over_sparse_median_ratio'],
                      'linear_controls':controls,'complete_chain_relative_RMSE':chain_error}),flush=True)


if __name__=='__main__':main()
