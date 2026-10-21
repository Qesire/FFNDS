"""UB2: frozen quarter/quarter/half/full main, with no terminal residual path."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import canonical_json,digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import tensor_sha


def effective_groups(policy):
    if (policy['kind'] != 'TERMINAL_DENSE_STEP_REFERENCE' or type(policy['identity_control']) is not bool
            or policy['groups_by_step'] != [54,54,108,216] or policy['residual_rank'] != 256
            or policy['terminal_step'] != 3):
        raise RuntimeError('Only frozen UB2 or its original-UB1 identity mode is supported')
    return [54,108,108,162] if policy['identity_control'] else policy['groups_by_step']


def create_runtime(cache_path,policy_path):
    from step_budget_runtime import create_runtime as parent_runtime
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    policy = json.loads(Path(policy_path).read_text())
    counts = effective_groups(policy)
    for name,sha in policy['runtime_sources'].items():
        if digest(name) != sha:
            raise RuntimeError('Frozen UB2 runtime numerical source changed')
    parent = policy['parent_policy']
    if digest(parent['path']) != parent['sha256']:
        raise RuntimeError('Frozen UB1 parent policy changed')
    runtime = parent_runtime(cache_path,parent['path'])
    old_policy = runtime.policy
    if old_policy['identity_control'] or old_policy['cache_manifest'] != policy['cache_manifest']:
        raise RuntimeError('UB2 requires the active original UB1 cache and step0/2 parameters')
    binding = policy['step1_fit']
    if digest(binding['path']) != binding['sha256']:
        raise RuntimeError('UB2 changed-step coefficient report changed')
    fit,report,identity = open_verified(Path(binding['path']),'COMPLETED_UB2_STEP1_QUARTER40_FIT')
    if (fit.fingerprint != binding['fingerprint'] or len(report['positions']) != 40 or report['step'] != 1
            or report['groups'] != 54 or report['control']['status'] != 'PASS_OLD_C24_ALPHA_BIAS_RR_PARAMETER_BITS'
            or identity['parent_policy_sha256'] != old_policy['parent_policy']['sha256']
            or identity['frozen_UB1_policy_sha256'] != parent['sha256']
            or identity['cache_manifest_sha256'] != policy['cache_manifest']['sha256']
            or identity['protocol_sha256'] != policy['protocol']['sha256']
            or digest(policy['protocol']['path']) != policy['protocol']['sha256']):
        raise RuntimeError('Exact original recipe/control/cache and40 changed-step fits required')
    for name,sha in identity['sources'].items():
        if digest(name) != sha:
            raise RuntimeError('Frozen UB2 coefficient producer changed')
    cache = json.loads(Path(policy['cache_manifest']['path']).read_text())
    raw = policy['low_raw_checkpoint']
    if Path(raw['path']).stat().st_size != raw['bytes']:
        raise RuntimeError('Original D raw bias checkpoint changed')
    if not policy['identity_control']:
        with safe_open(raw['path'],framework='pt',device='cpu') as handle:
            for block in range(40):
                part = fit.load('positions',f'step1_block{block:02d}')
                if part is None:
                    raise RuntimeError('Missing UB2 changed-step position')
                arrays,info = part['arrays'],part['metadata']
                groups = torch.from_numpy(arrays['groups']).long()
                if (groups.shape != (54,) or len(groups.unique()) != 54 or not torch.equal(groups,groups.sort().values)
                        or bool((groups<0).any()) or bool((groups>=216).any())):
                    raise RuntimeError('Exact54 original whole64 groups required')
                channels = (groups[:,None]*64+torch.arange(64)).reshape(-1)
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    shard = next(v for v in cache['shards'] if v['expert'] == expert and v['block'] == block)
                    if digest(shard['path']) != shard['sha256']:
                        raise RuntimeError('Original full effective quantized shard changed')
                    payload = torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)[expert]
                    r0,r2 = [dict(payload[f'blocks.{block}.ffn.{slot}']) for slot in [0,2]]
                    original_bias_sha = runtime.positions[step,block]['original_b2_sha256']
                    if step == 1:
                        if tensor_sha(r2['weight_qdq']) != info['original_W2_sha256'] or original_bias_sha != info['original_b2_sha256']:
                            raise RuntimeError('UB2 changed-step effective weight/raw bias differs')
                        r0['weight_qdq'] = r0['weight_qdq'].index_select(0,channels).contiguous()
                        r2['weight_qdq'] = r2['weight_qdq'].index_select(1,channels).contiguous()
                        r2['smooth_scale'] = r2['smooth_scale'].index_select(0,channels).contiguous()
                        b2 = torch.from_numpy(arrays['b2']).bfloat16()
                        rr = {k[3:]:torch.from_numpy(v).bfloat16() for k,v in arrays.items() if k.startswith('rr_')}
                        if (tensor_sha(b2) != info['b2_sha256'] or {k:tensor_sha(v) for k,v in rr.items()} != info['RR_parameter_sha256']):
                            raise RuntimeError('UB2 native bias/residual encoding changed')
                        runtime.positions[step,block] = {'records':{0:r0,2:r2},'channels':channels,
                            'alpha':info['alpha'],'b2':b2,'original_b2_sha256':original_bias_sha}
                        runtime.residual_parameters[step,block] = rr
                    else:
                        b2 = handle.get_tensor(f'blocks.{block}.ffn.2.bias').bfloat16()
                        if tensor_sha(b2) != original_bias_sha:
                            raise RuntimeError('Dense terminal branch requires the original raw D bias exactly once')
                        runtime.positions[step,block] = {'records':{0:r0,2:r2},'channels':torch.arange(13824),
                            'alpha':1.,'b2':b2,'original_b2_sha256':original_bias_sha}
                        del runtime.residual_parameters[step,block]
        del payload
    old_apply,old_report = runtime.apply,runtime.report
    runtime.policy,runtime.policy_sha = policy,digest(policy_path)
    runtime.residual_fingerprint = hashlib.sha256(canonical_json({'parent':runtime.residual_fingerprint,
        'step1':fit.fingerprint,'policy':runtime.policy_sha}).encode()).hexdigest()
    runtime.channels_cuda.clear()
    expected_residuals = 160 if policy['identity_control'] else 120

    def apply(self,mm_weight,input_tensor,*,expert,step_index,block_index,module_name):
        if not module_name.endswith(('.ffn.0','.ffn.2')):
            return old_apply(mm_weight,input_tensor,expert=expert,step_index=step_index,
                             block_index=block_index,module_name=module_name)
        if expert != ('high_noise_model' if step_index < 2 else 'low_noise_model'):
            raise RuntimeError('UB2 expert/step contract differs')
        position = self.positions[step_index,block_index]
        slot = 0 if module_name.endswith('.ffn.0') else 2
        width = counts[step_index]*64
        expected_in,expected_out = (5120,width) if slot == 0 else (width,5120)
        if input_tensor.shape[-1] != expected_in:
            raise RuntimeError('UB2 input main width differs')
        record = position['records'][slot]
        x = (input_tensor.float()/record['smooth_scale'].to(input_tensor.device)).to(input_tensor.dtype)
        x = qdq(x,force_fp32=True)
        self.counters['activation_qdq_calls'] += 1
        key = (step_index,block_index)
        use_residual = policy['identity_control'] or step_index != 3
        if slot == 0 and use_residual:
            if key in self.pending_residual:
                raise RuntimeError('Duplicate UB2 residual prediction')
            parameters = {k:v.to(x.device) for k,v in self.residual_parameters[key].items()}
            self.pending_residual[key] = self.predict_residual(x,parameters)
            self.residual_coverage[f'step{step_index}_block{block_index}'] = {'rows':x.numel()//5120,
                'auxiliary_GEMM_shapes':[[256,5120],[5120,256]],'auxiliary_linear_calls':2,
                'auxiliary_parameter_H2D_bytes':sum(v.numel()*v.element_size() for v in parameters.values()),
                'auxiliary_MACs_over_original_dense_FFN':256/13824}
        if slot == 2:
            # Unit terminal gain is the original dense path; no extra correction.
            if use_residual:
                x = (x.float()*position['alpha']).to(x.dtype)
            bias = position['b2'].to(x.device)
        else:
            original = mm_weight._get_actual_bias()
            if not use_residual:
                bias = original.to(x.device) if original is not None else None
            else:
                ids = (step_index,block_index,str(x.device))
                if ids not in self.channels_cuda:
                    self.channels_cuda[ids] = position['channels'].to(x.device)
                bias = None if original is None else original.to(x.device).index_select(0,self.channels_cuda[ids])
        weight = self._weight(expert=expert,block_index=block_index,module_name=module_name,
                              record=record,device=x.device,dtype=x.dtype)
        result = F.linear(x,weight,bias)
        if slot == 2 and use_residual:
            if key not in self.pending_residual:
                raise RuntimeError('Missing paired UB2 residual')
            correction = self.pending_residual.pop(key)
            if correction.shape != result.shape or correction.dtype != result.dtype:
                raise RuntimeError('UB2 residual geometry/precision differs')
            result = result+correction
        if step_index == 3 and block_index == 39 and slot == 2:
            if len(self.residual_coverage) != expected_residuals or self.pending_residual:
                raise RuntimeError('UB2 actual residual call coverage differs from the registered schedule')
        name = f'step{step_index}_block{block_index}_ffn{slot}'
        if name in self.coverage or result.shape[-1] != expected_out:
            raise RuntimeError('Duplicate or wrong UB2 main output')
        self.coverage[name] = {'rows':input_tensor.numel()//expected_in,'actual_weight_shape':list(weight.shape),
                               'main_GEMM_density':width/13824}
        self.counters['linear_forward_calls'] += 1
        return result

    def report_runtime():
        result = old_report()
        packed = result['packed_reference']
        packed['parent_UB1_changed_fits'] = packed['changed_fit_identities']
        packed.update(source_sha256=digest(Path(__file__)),parent_runtime_source_sha256=digest(Path(__file__).with_name('step_budget_runtime.py')),
                      groups_by_step=counts,main_density_by_step=[v/216 for v in counts],
                      changed_fit_identities=[{'step':1,'fingerprint':fit.fingerprint}],identity_control=policy['identity_control'])
        packed['residual_correction']['expected_positions'] = expected_residuals
        packed['terminal_dense'] = {'active':not policy['identity_control'],'step':3,
            'full_main_positions':0 if policy['identity_control'] else 40,'expected_residual_positions':expected_residuals,
            'expected_auxiliary_GEMMs':expected_residuals*2,'nominal_auxiliary_MAC_density':expected_residuals*256/(160*13824),
            'identity_control':policy['identity_control']}
        return result

    runtime.apply = MethodType(apply,runtime)
    runtime.report = report_runtime
    return runtime
