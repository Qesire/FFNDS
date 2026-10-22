"""Native runtime hook for one paired R15 integration smoke.

Original LightX2V owns normalization, GELU, gate and residual writeback.
The hook replaces the two linear boundaries only at block20 in four steps.
"""
from types import MethodType

import torch
import torch.nn.functional as F

import mean_contrast_kernels as fused


class MeanContrastHook:
    def __init__(self,runtime,parameters,expected_grid):
        self.runtime,self.parameters=runtime,parameters
        self.expected_grid=tuple(expected_grid)
        self.pending={};self.events=[];self.grid=None

    def __enter__(self):
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.old_apply=self.runtime.apply
        self.old_block=WanTransformerInfer.infer_block
        self.infer=WanTransformerInfer
        owner=self

        def block(infer,weights,x,pre):
            before=owner.grid
            owner.grid=tuple(int(v) for v in pre.grid_sizes.tuple)
            try:return owner.old_block(infer,weights,x,pre)
            finally:owner.grid=before

        def apply(runtime,mm,input_tensor,*,expert,step_index,block_index,module_name):
            if block_index!=20 or not module_name.endswith(('.ffn.0','.ffn.2')):
                return owner.old_apply(mm,input_tensor,expert=expert,step_index=step_index,
                                       block_index=block_index,module_name=module_name)
            if (step_index not in range(4) or owner.grid!=owner.expected_grid or
                    owner.grid[-1]%4 or input_tensor.dtype!=torch.bfloat16 or
                    expert!=('high_noise_model' if step_index<2 else 'low_noise_model')):
                raise RuntimeError('R15 native position/grid/precision differs from registered smoke')
            record=runtime.records[expert][module_name]
            if not record['activation_qdq'] or not record['force_fp32']:
                raise RuntimeError('Original frozen native QDQ regime required')
            width=input_tensor.shape[-1]
            x=(input_tensor.float()/record['smooth_scale'].to(input_tensor.device,dtype=torch.float32)).to(input_tensor.dtype)
            x=adapter.native_hif4_activation_qdq(x,force_fp32=True).to(input_tensor.dtype)
            runtime.counters['activation_qdq_calls']+=1
            weight=runtime._weight(expert=expert,block_index=block_index,module_name=module_name,
                                   record=record,device=x.device,dtype=x.dtype)
            bias=mm._get_actual_bias()
            if bias is not None:bias=bias.to(device=x.device,dtype=x.dtype)
            flat=x.reshape(-1,width)
            if flat.shape[0]!=31668:
                raise RuntimeError('R15 one-source full native token count changed')
            if module_name.endswith('.ffn.0'):
                if step_index in owner.pending or width!=5120:
                    raise RuntimeError('Duplicate or malformed R15 W1 boundary')
                hh,mu=fused.prepare(flat)
                p=owner.parameters[expert]
                projected=F.linear(hh,p['A'])
                zmean=F.linear(mu,weight,bias)
                compressed=F.linear(hh,p['V'])
                ch3=F.linear(compressed,p['U'])
                output=fused.reconstruct(zmean,ch3)
                owner.pending[step_index]=projected
                shapes=[[len(hh),5120,5120],[len(mu),5120,13824],
                        [len(hh),5120,256],[len(hh),256,13824]]
            else:
                if step_index not in owner.pending or width!=13824:
                    raise RuntimeError('Missing R15 W1 projection before native W2')
                pooled=fused.pool(flat)
                response=F.linear(pooled,weight,bias)
                # Preserve the two BF16 boundaries used by the tested backend:
                # inverse contrast first, then add the pooled response. The
                # preactivation reconstruction kernel has only one final cast.
                projected=owner.pending.pop(step_index)
                a,b,c=projected.reshape(len(response),3,5120).float().unbind(1)
                centered=torch.stack(((a+b+c)*.5,(a-b-c)*.5,
                                      (-a+b-c)*.5,(-a-b+c)*.5),1).bfloat16()
                output=(centered+response[:,None]).reshape(-1,5120)
                shapes=[[len(pooled),13824,5120]]
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('Nonfinite R15 native candidate output')
            owner.events.append({'step':int(step_index),'block':20,'module':module_name,
                                 'grid':list(owner.grid),'original_QDQ_rows':len(flat),
                                 'physical_linear_MKH':shapes,'output_shape':list(output.shape)})
            runtime.counters['linear_forward_calls']+=1
            return output.reshape(*input_tensor.shape[:-1],output.shape[-1])

        WanTransformerInfer.infer_block=block
        self.runtime.apply=MethodType(apply,self.runtime)
        return self

    def __exit__(self,*error):
        self.runtime.apply=self.old_apply
        self.infer.infer_block=self.old_block
        if not any(error):
            expected={(step,slot) for step in range(4) for slot in (0,2)}
            actual={(e['step'],int(e['module'].split('.')[-1])) for e in self.events}
            if self.pending or len(self.events)!=8 or actual!=expected:
                raise RuntimeError('R15 candidate did not execute all four intended FFN positions')
