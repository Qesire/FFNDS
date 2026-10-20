"""Offline dense FFN calls through the unmodified HiF4 runtime at student states."""
import contextvars
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ar2_checkpoint import atomic_json,digest
from run_ar2_restart import tensor_sha


class FrozenBias:
    def __init__(self,bias):self.value=bias
    def _get_actual_bias(self):return self.value


def equal_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.int16),b.view(torch.int16))


class TeacherRecords:
    """One verified original block at a time; never retain a second full model."""
    def __init__(self,manifest):
        from HiFloat4.ptq.adapters.lightx2v_wan_distill import HIF4CacheRuntime
        self.manifest=manifest;self.verified={};self.payload=None
        # Initialize the original runtime's bookkeeping without loading its
        # 56GB bundle again. Its record lookup, smoothing, QDQ, bias casting,
        # weight transfer and F.linear implementations remain unmodified.
        r=self.runtime=HIF4CacheRuntime.__new__(HIF4CacheRuntime)
        r.cache_path=Path(manifest['path']);r.meta={'purpose':'offline native local dense teacher'}
        r.records={};r.activation_qdq_override=None;r.keep_fp=None
        r._resident_key=None;r._resident_weights={}
        r.counters={k:0 for k in ['linear_forward_calls','activation_qdq_calls','weight_transfers','cache_misses']}

    def configure(self,expert,block):
        shard=next(s for s in self.manifest['shards'] if s['expert']==expert and s['block']==block)
        p=Path(shard['path']);stat=p.stat();key=str(p);signature=(stat.st_size,stat.st_mtime_ns)
        if key not in self.verified:
            if digest(p)!=shard['sha256']:raise RuntimeError('Original teacher shard changed')
            self.verified[key]=signature
        elif self.verified[key]!=signature:raise RuntimeError('Teacher shard mutated during collection')
        self.payload=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        records=self.payload[expert]
        selected={f'blocks.{block}.ffn.{slot}':records[f'blocks.{block}.ffn.{slot}'] for slot in [0,2]}
        for name,r in selected.items():
            expected=(13824,5120) if name.endswith('.0') else (5120,13824)
            if (r['weight_qdq'].shape!=expected or r['weight_qdq'].dtype!=torch.bfloat16
                    or not r['activation_qdq'] or not r['force_fp32']):
                raise RuntimeError('Teacher is not the original full native W4A4 bridge')
        self.runtime.records={expert:selected}
        return shard

    def release(self):
        self.runtime._resident_weights.clear();self.runtime._resident_key=None
        self.runtime.records={};self.payload=None


class NativeTeacherCapture:
    def __init__(self,runtime,teacher,out,index,*,dense_control=False,rows=256):
        self.runtime,self.teacher,self.out,self.index=runtime,teacher,Path(out),index
        self.dense_control,self.rows=dense_control,rows
        self.scopes=[];self.context=contextvars.ContextVar('ffnds_native_local_teacher',default=None)

    def __enter__(self):
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.adapter,self.infer=adapter,WanTransformerInfer
        self.old_apply,self.old_qdq,self.old_linear=self.runtime.apply,adapter.native_hif4_activation_qdq,F.linear
        self.old_block,self.old_ffn=WanTransformerInfer.infer_block,WanTransformerInfer.infer_ffn
        self.old_predict=getattr(self.runtime,'predict_residual',None)
        if not self.dense_control and self.old_predict is None:raise RuntimeError('OP1 requires the frozen residual student')
        observer=self

        def block(infer,weights,x,pre):
            token=observer.context.set({'grid':[int(v) for v in pre.grid_sizes.tuple]})
            try:return observer.old_block(infer,weights,x,pre)
            finally:observer.context.reset(token)

        def ffn(infer,phase,x,attn,shift,scale,gate=None):
            if gate is None or gate.numel()!=5120:raise RuntimeError('Native row-constant gate required')
            token=observer.context.set({**observer.context.get(),'gate':gate.reshape(1,5120).detach().clone()})
            cpu_rng,cuda_rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            try:
                result=observer.old_ffn(infer,phase,x,attn,shift,scale,gate)
                if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
                    raise RuntimeError('Offline teacher changed the native RNG state')
                return result
            finally:observer.context.reset(token)

        def qdq(activation,*,force_fp32=True):
            result=observer.old_qdq(activation,force_fp32=force_fp32);s=observer.context.get()
            if s is not None:
                native=result.to(activation.dtype).detach()
                if s.get('teacher_active'):
                    stage=s['teacher_stage']
                    if stage==0:
                        s['Q1_teacher_equals_actual']=equal_bits(native,s['actual_v1'])
                        if not s['Q1_teacher_equals_actual']:raise RuntimeError('Native full teacher Q1 differs from actual V1')
                    elif observer.dense_control:
                        s['Q2_teacher_equals_actual']=equal_bits(native,s['actual_v2'])
                        if not s['Q2_teacher_equals_actual']:raise RuntimeError('Native full teacher Q2 differs from actual dense V2')
                elif observer.dense_control:
                    if s.get('slot')==0:s['actual_v1']=native
                    if s.get('slot')==2:s['actual_v2']=native
            return result

        def predict(x,parameters):
            result=observer.old_predict(x,parameters);s=observer.context.get()
            if s is None or s.get('slot')!=0:raise RuntimeError('Residual predictor escaped native FFN0')
            s['actual_v1'],s['actual_residual']=x.detach(),result.detach()
            return result

        def linear(value,weight,bias=None):
            result=observer.old_linear(value,weight,bias);s=observer.context.get()
            if s is not None and s.get('slot')==2 and not s.get('teacher_active'):
                expected=(5120,13824 if observer.dense_control else 6912)
                if weight.shape!=expected or result.dtype!=torch.bfloat16 or 'base_y' in s:
                    raise RuntimeError('Actual native main FFN2 capture shape/count differs')
                s['base_y']=result.detach();s['effective_base_bias_sha256']=tensor_sha(bias)
            return result

        def apply(mm,value,*,expert,step_index,block_index,module_name):
            kwargs=dict(expert=expert,step_index=step_index,block_index=block_index,module_name=module_name)
            if not module_name.endswith(('.ffn.0','.ffn.2')):return observer.old_apply(mm,value,**kwargs)
            s=observer.context.get()
            if s is None:raise RuntimeError('Teacher observation escaped the native FFN context')
            slot=0 if module_name.endswith('.ffn.0') else 2
            bias=mm._get_actual_bias()
            if bias is None:raise RuntimeError('The frozen distilled FFN has explicit original biases')
            if slot==0:
                s.update(expert=expert,step=step_index,block=block_index,norm_input=value.detach(),b0=bias.detach().clone())
                if value.numel()//5120!=int(np.prod(s['grid'])):raise RuntimeError('Actual native FFN rows/grid differ')
            else:
                if (expert,step_index,block_index)!=(s['expert'],s['step'],s['block']):raise RuntimeError('Unpaired FFN0/FFN2')
                s['b2']=bias.detach().clone()
            s['slot']=slot
            try:result=observer.old_apply(mm,value,**kwargs)
            finally:s.pop('slot',None)
            if slot==2:observer.observe(s,result)
            return result

        WanTransformerInfer.infer_block,WanTransformerInfer.infer_ffn=block,ffn
        self.runtime.apply,adapter.native_hif4_activation_qdq,F.linear=apply,qdq,linear
        if self.old_predict is not None:self.runtime.predict_residual=predict
        return self

    def observe(self,s,result):
        n=int(np.prod(s['grid']));key=(s['step'],s['block'])
        if any((r['step'],r['block'])==key for r in self.scopes):raise RuntimeError('Duplicate native teacher position')
        if not self.dense_control and not equal_bits(s['base_y']+s['actual_residual'],result):
            raise RuntimeError('Captured BaseY is not the actual pre-residual BF16 main output')
        shard=self.teacher.configure(s['expert'],s['block']);r=self.teacher.runtime;before=dict(r.counters)
        s['teacher_active']=True
        try:
            s['teacher_stage']=0
            z=r.apply(FrozenBias(s['b0']),s['norm_input'],expert=s['expert'],step_index=s['step'],
                block_index=s['block'],module_name=f'blocks.{s["block"]}.ffn.0')
            hidden=F.gelu(z,approximate='tanh');s['teacher_stage']=2
            teacher_y=r.apply(FrozenBias(s['b2']),hidden,expert=s['expert'],step_index=s['step'],
                block_index=s['block'],module_name=f'blocks.{s["block"]}.ffn.2')
        finally:s['teacher_active']=False
        delta={k:r.counters[k]-before[k] for k in before}
        if delta['linear_forward_calls']!=2 or delta['activation_qdq_calls']!=2 or delta['cache_misses']:
            raise RuntimeError('Offline teacher did not execute two full native FFN linears/QDQs')
        dense_equal=equal_bits(teacher_y,result) if self.dense_control else None
        if self.dense_control and not dense_equal:raise RuntimeError('Full native teacher Y differs from actual dense FFN output')
        metadata={'step':s['step'],'block':s['block'],'expert':s['expert'],'grid':s['grid'],'full_rows':n,
            'Q1_teacher_equals_actual':s.get('Q1_teacher_equals_actual',False),
            'Q2_teacher_equals_actual':s.get('Q2_teacher_equals_actual') if self.dense_control else None,
            'Y_teacher_equals_actual_dense':dense_equal,'extra_full_teacher_GEMMs':2,
            'effective_base_bias_sha256':s['effective_base_bias_sha256'],
            'original_b0_sha256':tensor_sha(s['b0']),'original_b2_sha256':tensor_sha(s['b2']),
            'gate_sha256':tensor_sha(s['gate']),'teacher_shard_sha256':shard['sha256']}
        if not self.dense_control:
            salt=f'FFNDS_CALIB24_UNIFORM_V1|{self.index}|{s["step"]}|{s["block"]}|{s["grid"]}'
            seed=int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8],'little')
            selected=np.sort(np.random.default_rng(seed).choice(n,self.rows,replace=False)).tolist()
            ids=torch.tensor(selected,device=result.device)
            values={k:v.reshape(-1,5120).index_select(0,ids).detach().cpu().contiguous() for k,v in
                [('V1',s['actual_v1']),('BaseY',s['base_y']),('TeacherY',teacher_y),('StudentY',result)]}
            values.update(b2=s['b2'].detach().cpu().contiguous(),gate=s['gate'].detach().cpu().contiguous())
            if any(v.dtype!=torch.bfloat16 or not bool(torch.isfinite(v).all()) for v in values.values()):
                raise RuntimeError('Nonfinite/non-BF16 native on-policy operands')
            path=self.out/f'step{s["step"]}_block{s["block"]:02d}.safetensors'
            if path.exists():raise RuntimeError('Refusing to overwrite completed native operands')
            temp=Path(str(path)+'.tmp');save_file(values,str(temp));temp.replace(path)
            metadata.update(rows=self.rows,selected_rows=selected,sampling_seed=seed,inclusion_probability=self.rows/n,
                inverse_inclusion_weight=n/self.rows,path=str(path),sha256=digest(path),bytes=path.stat().st_size,
                actual_native_BaseY_plus_old_residual_equals_StudentY=True)
        self.scopes.append(metadata)
        # Acceptance/recovery is per complete no-op input. Per-step metadata
        # checkpoints avoid160 shared-filesystem fsync/rename cycles, while
        # every operand payload is still written atomically before acceptance.
        if len(self.scopes)%40==0:atomic_json(self.out/'SCOPES.json',self.scopes)
        for k in ['norm_input','actual_v1','actual_v2','actual_residual','base_y','b0','b2']:s.pop(k,None)

    def __exit__(self,*error):
        self.infer.infer_block,self.infer.infer_ffn=self.old_block,self.old_ffn
        self.runtime.apply,self.adapter.native_hif4_activation_qdq,F.linear=self.old_apply,self.old_qdq,self.old_linear
        if self.old_predict is not None:self.runtime.predict_residual=self.old_predict
        self.teacher.release()
