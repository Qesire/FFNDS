"""Scoped raw-BF16 W2 factor dispatch and prefetch interception; no import-time Torch."""
from threading import Lock
import types

ORIGINAL_W2_BYTES=141568000


def need(value,message):
    if not value:raise ValueError(message)


class FactorW2Runtime:
    """Intercept four actual buffer MM objects after OriginalGuard is installed.

    Call synchronize before entering and after leaving a factor phase. The bank
    stays resident for the context; allocation of original CUDA buffers remains.
    The caller supplies the real factor_forward and paid per-GEMM meter.
    """
    def __init__(self,pipe,guard,bank,*,factor_forward,meter,synchronize,record_stream,emit=lambda event:None):
        self.pipe,self.guard,self.bank=pipe,guard,bank
        self.factor_forward,self.meter,self.synchronize,self.record_stream,self.emit=factor_forward,meter,synchronize,record_stream,emit
        self.mode='dense';self.markers={};self.dense_ready=set();self.lock=Lock();self.saved=[];self.objects={}
        for expert,model in enumerate(pipe.runner.model.model):
            buffers=model.transformer_infer.offload_manager.cuda_buffers
            need(len(buffers)==2,'two original CUDA buffers per expert required')
            for buf in buffers:
                mm=buf.compute_phases[2].ffn_2
                need(id(mm) not in self.objects and type(mm).__name__=='MMWeight' and not mm.has_lora_branch and not mm.has_diff and not hasattr(mm,'weight_diff') and not hasattr(mm,'bias_diff'),'unsupported W2 object/adapters')
                self.objects[id(mm)]=(mm,expert)
        need(len(self.objects)==4,'four distinct W2 buffers required')

    def _install(self,obj,name,fn):
        existing=obj.__dict__.get(name);present=name in obj.__dict__;self.saved.append((obj,name,present,existing));setattr(obj,name,types.MethodType(fn,obj))

    def __enter__(self):
        self.synchronize()
        for mm,expert in self.objects.values():
            old_apply=mm.apply;old_load=mm.load_state_dict;runtime=self
            def load(obj,destination,block_index,adapter_block_index=None,*,_load=old_load,_expert=expert):
                if runtime.mode=='dense':
                    result=_load(destination,block_index,adapter_block_index)
                    with runtime.lock:runtime.markers.pop(id(obj),None);runtime.dense_ready.add(id(obj))
                    return result
                need(type(block_index) is int and 0<=block_index<40 and adapter_block_index is None,'unsupported W2 prefetch block')
                prefix=f'blocks.{block_index}.ffn.2.';w=destination.get(prefix+'weight');b=destination.get(prefix+'bias')
                need(w is not None and b is not None and tuple(w.shape)==(13824,5120) and tuple(b.shape)==(5120,) and w.element_size()==b.element_size()==2,'original W2 source shape/size differs')
                with runtime.lock:runtime.markers[id(obj)]=(_expert,block_index)
                runtime.emit({'event':'original_W2_H2D_skipped','expert':_expert,'block':block_index,'bytes':ORIGINAL_W2_BYTES})
            def apply(obj,value,*,_apply=old_apply,_expert=expert):
                if runtime.mode=='dense':
                    need(id(obj) in runtime.dense_ready,'dense W2 buffer must be reloaded after mode switch')
                    return _apply(value)
                active=runtime.guard.active;need(active is not None and active.block is not None,'W2 outside native compute block')
                expert_now,block=active.block;step=active.step(expert_now)
                need(expert_now==_expert and expert_now==(0 if step<2 else 1),'W2 current expert/step differs')
                with runtime.lock:marker=runtime.markers.get(id(obj))
                need(marker==(_expert,block),'current compute buffer block not loaded')
                factors=runtime.bank[_expert,step,block];left,right,bias=(factors[k] for k in ('left.weight','right.weight','right.bias'))
                need(len(value.shape)==2 and value.shape[1]==13824 and len(left.shape)==2 and left.shape[1]==13824 and len(right.shape)==2 and right.shape[1]==left.shape[0] and right.shape[0]==5120 and tuple(bias.shape)==(5120,), 'factor dimensions differ')
                for tensor in (left,right,bias):runtime.record_stream(tensor)
                meta={'input':list(value.shape),'call_label':'factor_W2_two_GEMMs','factor_left':list(left.shape),'factor_right':list(right.shape),'bias':list(bias.shape),'original_full_W2_loaded':False}
                def factor_callback(_mm,input_value):return runtime.factor_forward(input_value,factors,runtime.meter)
                return active.mm_call(obj,value,meta,factor_callback)
            self._install(mm,'load_state_dict',load);self._install(mm,'apply',apply)
        with self.lock:self.dense_ready=set(self.objects)
        return self

    def set_mode(self,mode):
        need(mode in ('dense','factor'),'unsupported runtime mode');self.synchronize()
        with self.lock:
            if mode!=self.mode:self.dense_ready.clear()
            self.markers.clear();self.mode=mode
        self.emit({'event':'mode','mode':mode,'original_cuda_buffer_allocation_retained':True})

    def __exit__(self,*_):
        try:self.synchronize()
        finally:
            for obj,name,present,value in reversed(self.saved):
                if present:setattr(obj,name,value)
                else:delattr(obj,name)
            with self.lock:self.markers.clear();self.dense_ready.clear();self.mode='dense'
            self.saved.clear()
