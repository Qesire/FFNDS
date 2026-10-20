"""Apply frozen output moments after the unchanged native RR1 BF16 sum."""
import json
from pathlib import Path
import types

import numpy as np
import torch

from ar2_checkpoint import CheckpointStore,digest
from output_variance_kernel import affine_cuda
from run_ar2_restart import tensor_sha


def create_runtime(cache_path,policy_path):
    from trained_packed_runtime import create_runtime as create_base
    policy=json.loads(Path(policy_path).read_text())
    if policy['kind']!='OUTPUT_VARIANCE_CALIBRATED_REFERENCE':
        raise RuntimeError('Unsupported output affine policy')
    for name in ['base_policy','moment_fit','kernel_probe']:
        binding=policy[name]
        if digest(binding['path'])!=binding['sha256']:
            raise RuntimeError(f'Frozen output affine {name} changed')
    kernel=Path(__file__).with_name('output_variance_kernel.py')
    if (digest(kernel)!=policy['kernel_source_sha256']
            or digest(Path(__file__))!=policy['runtime_source_sha256']):
        raise RuntimeError('Affine GPU implementation differs from verified policy')
    fit=json.loads(Path(policy['moment_fit']['path']).read_text())
    probe=json.loads(Path(policy['kernel_probe']['path']).read_text())
    if (fit['status']!='COMPLETED_VM1_FULL160_MOMENT_FIT' or len(fit['positions'])!=160
            or probe['status']!='PASS_VM1_CUDA_AFFINE_BITWISE_AND_IDENTITY'
            or probe['fit_fingerprint']!=fit['fingerprint']
            or probe['kernel_source_sha256']!=digest(kernel)):
        raise RuntimeError('Incomplete moment fit or native CUDA operator proof')
    root=Path(fit['checkpoint_root'])
    identity=json.loads((root/'identity.json').read_text())['identity']
    store=CheckpointStore(root,identity)
    if store.fingerprint!=fit['fingerprint'] or identity['base_policy_sha256']!=policy['base_policy']['sha256']:
        raise RuntimeError('Output moments were trained for a different base')
    runtime=create_base(cache_path,policy['base_policy']['path'])
    runtime.output_affine_parameters={}
    coverage={}
    identity_control=bool(policy.get('identity_control',False))
    for step in range(4):
        for block in range(40):
            part=store.load('positions',f'step{step}_block{block:02d}')
            if part is None:raise RuntimeError('Missing output affine position')
            a,m=part['arrays'],part['metadata']
            beta=torch.from_numpy(a['beta']);shift=torch.from_numpy(a['shift'])
            b0=torch.from_numpy(a['original_b2']).to(torch.bfloat16)
            if (beta.shape!=(1,) or shift.shape!=(5120,) or b0.shape!=(5120,)
                    or beta.dtype!=torch.float32 or shift.dtype!=torch.float32
                    or not np.isfinite(float(beta[0])) or not .5<=float(beta[0])<=2
                    or not bool(torch.isfinite(shift).all())
                    or tensor_sha(shift)!=m['shift_FP32_sha256']
                    or tensor_sha(b0)!=m['original_b2_sha256']
                    or tensor_sha(b0)!=runtime.positions[step,block]['original_b2_sha256']):
                raise RuntimeError('Invalid or inconsistent frozen affine coefficients')
            runtime.output_affine_parameters[step,block]={'beta':torch.ones_like(beta) if identity_control else beta,
                'shift':torch.zeros_like(shift) if identity_control else shift,'original_b2':b0}
    original_apply,original_report,original_begin=runtime.apply,runtime.report,runtime.begin_request

    def begin(self):
        original_begin();coverage.clear()

    def apply(self,mm_weight,input_tensor,*,expert,step_index,block_index,module_name):
        result=original_apply(mm_weight,input_tensor,expert=expert,step_index=step_index,
            block_index=block_index,module_name=module_name)
        if module_name.endswith('.ffn.2'):
            key=(step_index,block_index)
            if key in coverage:raise RuntimeError('Output affine executed twice')
            cpu_parameters=self.output_affine_parameters[key]
            transferred=sum(v.numel()*v.element_size() for v in cpu_parameters.values() if v.device!=result.device)
            parameters={k:v.to(result.device) for k,v in cpu_parameters.items()}
            result=affine_cuda(result,parameters,identity=identity_control)
            coverage[key]={'step':step_index,'block':block_index,'rows':result.shape[0],
                'coefficient_H2D_bytes':transferred,'pointwise_kernel_calls':0 if identity_control else 1}
            if key==(3,39) and len(coverage)!=160:
                raise RuntimeError('Output affine did not cover every160 position')
        return result

    def report(self):
        result=original_report();packed=result['packed_reference']
        packed['base_policy_sha256']=packed['policy_sha256']
        packed.update(policy=policy,policy_sha256=digest(policy_path))
        packed['output_affine']={'fit_fingerprint':store.fingerprint,'actual_positions':len(coverage),
            'expected_positions':160,'identity_control':identity_control,'coverage':list(coverage.values()),
            'kernel_source_sha256':digest(kernel),'runtime_source_sha256':digest(Path(__file__)),
            'timing_includes_affine_cost':bool(getattr(self,'full_ffn_profiler_active',False)),
            'precision':'four sequential FP32 operations, FMA disabled, one BF16 cast before original gate',
            'quality_or_speed_acceptance':False}
        return result

    runtime.apply=types.MethodType(apply,runtime)
    runtime.report=types.MethodType(report,runtime)
    runtime.begin_request=types.MethodType(begin,runtime)
    return runtime
