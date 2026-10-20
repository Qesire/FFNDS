"""In-memory native bridge residency and conservative complete FFN profiling."""
from contextlib import contextmanager
import time
import types

import torch


@contextmanager
def unused_raw_placeholders(adapter):
    """Do not allocate/offload matrices which the strict QDQ bridge replaces."""
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    original_load,original_apply=MMWeight.load,MMWeight.apply
    audit={'calls':0,'names':set(),'replaced_raw_bytes':0,'fallback_calls':0}

    def load(self,weight_dict):
        name=adapter.normalize_module_name(self.weight_name)
        if not adapter.is_target_module(name):
            return original_load(self,weight_dict)
        if self.lazy_load or self.has_lora_branch or self.has_diff:
            raise RuntimeError('Placeholder loading does not support lazy/LoRA/diff weights')
        raw=weight_dict[self.weight_name]
        if raw.ndim!=2 or raw.dtype!=torch.bfloat16:
            raise RuntimeError('Only the exact bridge-covered BF16 raw matrices may be omitted')
        substitute=dict(weight_dict)
        substitute[self.weight_name]=torch.zeros((1,1),dtype=raw.dtype,device=raw.device)
        audit['calls']+=1;audit['names'].add(name)
        audit['replaced_raw_bytes']+=raw.numel()*raw.element_size()
        return original_load(self,substitute)

    def guard(self,input_tensor):
        if adapter.is_target_module(adapter.normalize_module_name(self.weight_name)):
            audit['fallback_calls']+=1
            raise RuntimeError('A covered target escaped the QDQ runtime; raw placeholder fallback forbidden')
        return original_apply(self,input_tensor)

    MMWeight.load,MMWeight.apply=load,guard
    try:
        yield audit
    finally:
        MMWeight.load=original_load
        # Keep guard installed: activate_lightx2v_hooks captures it as its
        # fallback. This isolated runner never executes with runtime=None.


def make_resident(runtime):
    """Move aliased tensors once, keeping all exact parameter values/layouts."""
    moved={};total_bytes=0

    def move(tensor):
        nonlocal total_bytes
        key=id(tensor)
        if key not in moved:
            result=tensor.to('cuda').contiguous()
            moved[key]=result;total_bytes+=result.numel()*result.element_size()
        return moved[key]

    records=[]
    for expert in runtime.records.values():records.extend(expert.values())
    for position in getattr(runtime,'positions',{}).values():
        records.extend(position['records'].values())
        position['channels']=move(position['channels'])
        position['b2']=move(position['b2'])
    seen=set()
    for record in records:
        if id(record) in seen:continue
        seen.add(id(record))
        for key in ['weight_qdq','smooth_scale']:
            if key in record and torch.is_tensor(record[key]):record[key]=move(record[key])
    for key,parameters in getattr(runtime,'residual_parameters',{}).items():
        runtime.residual_parameters[key]={n:move(v) for n,v in parameters.items()}
    torch.cuda.synchronize()
    original_report=runtime.report

    def weight(self,*,expert,block_index,module_name,record,device,dtype):
        tensor=record['weight_qdq']
        if tensor.device!=device or tensor.dtype!=dtype or not tensor.is_contiguous():
            raise RuntimeError('Resident operand escaped its device/dtype/layout contract')
        return tensor

    def report(self):
        result=original_report()
        result['resident_QDQ']={'unique_tensor_count':len(moved),'parameter_bytes':total_bytes,
            'steady_weight_H2D_bytes':0,'setup_weight_H2D_bytes':total_bytes,
            'raw_placeholder_fallback_allowed':False}
        if 'packed_reference' in result:
            residual=result['packed_reference']['residual_correction']
            if residual['active']:
                residual['timing_includes_auxiliary_cost']=bool(getattr(self,'full_ffn_profiler_active',False))
                # The unchanged mathematical runtime calls .to(cuda), which is
                # a no-op for these resident parameters. Correct its generic
                # transfer-byte estimate in this residency-specific report.
                for row in residual['coverage'].values():row['auxiliary_parameter_H2D_bytes']=0
        return result

    runtime._weight=types.MethodType(weight,runtime)
    runtime.report=types.MethodType(report,runtime)
    runtime._resident_weights.clear()
    return {'unique_tensor_count':len(moved),'parameter_bytes':total_bytes}


class FullFFNProfiler:
    """Charge complete native operations plus conservative shared preparation."""
    def __init__(self,adapter):
        self.adapter=adapter;self.rows=[];self.pending={};self.originals=[]

    def start_region(self):
        torch.cuda.synchronize()
        event=torch.cuda.Event(enable_timing=True);event.record()
        return time.perf_counter(),event

    def finish_region(self,start,kind,step,block):
        end=torch.cuda.Event(enable_timing=True);end.record();torch.cuda.synchronize()
        wall=(time.perf_counter()-start[0])*1000
        self.rows.append({'kind':kind,'step':int(step),'block':int(block),
            'wall_ms':wall,'cuda_ms':start[1].elapsed_time(end)})

    def begin_request(self):
        if self.pending:raise RuntimeError('Unmatched FFN timing boundary')
        self.rows=[]

    def install(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from lightx2v.common.offload.manager import WeightAsyncStreamManager
        profiler=self
        def patch(cls,name,wrapper):
            original=getattr(cls,name);self.originals.append((cls,name,original));setattr(cls,name,wrapper(original))

        def pre(original):
            def wrapped(obj,*args,**kwargs):
                start=profiler.start_region();result=original(obj,*args,**kwargs)
                profiler.finish_region(start,'shared_modulation_preparation',obj.scheduler.step_index,obj.block_idx)
                return result
            return wrapped

        def ffn(original):
            def wrapped(obj,*args,**kwargs):
                key=id(obj)
                if key in profiler.pending:raise RuntimeError('Nested FFN region')
                profiler.pending[key]=profiler.start_region()
                return original(obj,*args,**kwargs)
            return wrapped

        def post(original):
            def wrapped(obj,*args,**kwargs):
                result=original(obj,*args,**kwargs)
                start=profiler.pending.pop(id(obj))
                profiler.finish_region(start,'FFN_norm_through_gate_residual',obj.scheduler.step_index,obj.block_idx)
                return result
            return wrapped

        def transfer(kind):
            def wrapper(original):
                def wrapped(obj,*args,**kwargs):
                    start=profiler.start_region()
                    # Native prefetch switches to a separate stream internally;
                    # a device sync finishes all its work before closing time.
                    result=original(obj,*args,**kwargs)
                    torch.cuda.synchronize()
                    step=profiler.adapter._step_ctx.get()
                    block=0 if kind=='initial_raw_bias_buffer' else int(args[0])
                    profiler.finish_region(start,kind,step,block)
                    return result
                return wrapped
            return wrapper

        patch(WanTransformerInfer,'pre_process',pre)
        patch(WanTransformerInfer,'infer_ffn',ffn)
        patch(WanTransformerInfer,'post_process',post)
        patch(WeightAsyncStreamManager,'init_first_buffer',transfer('initial_raw_bias_buffer'))
        patch(WeightAsyncStreamManager,'prefetch_weights',transfer('all_raw_bias_prefetch_conservative_charge'))

    def report(self):
        core=[r for r in self.rows if r['kind']=='FFN_norm_through_gate_residual']
        modulation=[r for r in self.rows if r['kind']=='shared_modulation_preparation']
        expected={(s,b) for s in range(4) for b in range(40)}
        if (self.pending or len(core)!=160 or {(r['step'],r['block']) for r in core}!=expected
                or len(modulation)!=160 or {(r['step'],r['block']) for r in modulation}!=expected):
            raise RuntimeError('Incomplete full160 FFN timing coverage')
        return {'rows':self.rows,'wall_ms':sum(r['wall_ms'] for r in self.rows),
            'cuda_ms':sum(r['cuda_ms'] for r in self.rows),'actual_FFN_positions':160,
            'conservative_shared_preparation_and_all_prefetch_charged':True,
            'acceptance':False}

    def restore(self):
        for cls,name,original in reversed(self.originals):setattr(cls,name,original)
        self.originals=[]
