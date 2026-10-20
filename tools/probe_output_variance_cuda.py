"""Check native fused affine bits on real calibration values and full token shapes."""
import argparse
import json
from pathlib import Path
import time

import torch

from ar2_checkpoint import CheckpointStore,atomic_json,digest
from output_variance import affine_reference
from output_variance_kernel import affine_cuda
from run_ar2_restart import tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8)
    fit=json.loads(args.fit_report.read_text());root=Path(fit['checkpoint_root'])
    store=CheckpointStore(root,json.loads((root/'identity.json').read_text())['identity'])
    if store.fingerprint!=fit['fingerprint']:raise RuntimeError('Moment fingerprint mismatch')
    cases=[];started=time.perf_counter()
    for step in [0,2]:
        name=f'step{step}_block20';part=store.load('positions',name);data=store.load('probe_data',name)
        if part is None or data is None:raise RuntimeError('Missing real pilot operator probe data')
        arrays=part['arrays']
        parameters={'beta':torch.from_numpy(arrays['beta']).cuda(),'shift':torch.from_numpy(arrays['shift']).cuda(),
            'original_b2':torch.from_numpy(arrays['original_b2']).to(device='cuda',dtype=torch.bfloat16)}
        original=torch.from_numpy(data['arrays']['candidate']).to(torch.bfloat16)
        for rows in [1,127,128,31668,31941]:
            value=original.repeat((rows+len(original)-1)//len(original),1)[:rows].contiguous().cuda()
            actual=affine_cuda(value,parameters)
            expected=affine_reference(value,float(parameters['beta'][0]),parameters['shift'],parameters['original_b2'])
            if not torch.equal(actual.view(torch.int16),expected.view(torch.int16)):
                raise RuntimeError(f'CUDA affine differs from explicit GPU reference: {name}, rows{rows}')
            if rows<=128:
                cpu=affine_reference(value.cpu(),float(parameters['beta'][0]),parameters['shift'].cpu(),parameters['original_b2'].cpu())
                if not torch.equal(actual.cpu().view(torch.int16),cpu.view(torch.int16)):
                    raise RuntimeError('Fused CUDA affine differs from explicit CPU scalar reference')
            identity={'beta':torch.ones_like(parameters['beta']),'shift':torch.zeros_like(parameters['shift']),
                'original_b2':parameters['original_b2']}
            unchanged=affine_cuda(value,identity,identity=True)
            if unchanged.data_ptr()!=value.data_ptr():raise RuntimeError('Identity control touched the input')
            cases.append({'step':step,'block':20,'rows':rows,'shape':list(value.shape),
                'input_sha256':tensor_sha(value),'output_sha256':tensor_sha(actual),
                'bitwise_GPU_FP32_reference':True,'bitwise_CPU_reference':rows<=128,
                'identity_returns_same_storage':True})
            del value,actual,expected,unchanged
    args.out.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS_VM1_CUDA_AFFINE_BITWISE_AND_IDENTITY','fit_fingerprint':store.fingerprint,
        'fit_report_sha256':digest(args.fit_report),'kernel_source_sha256':digest(Path(__file__).with_name('output_variance_kernel.py')),
        'source_sha256':digest(Path(__file__)),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(),
        'cases':cases,'elapsed_seconds':time.perf_counter()-started,'quality_or_speed_acceptance':False}
    atomic_json(args.out/'RUN.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
