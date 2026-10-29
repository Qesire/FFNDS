"""Paid, same-master SharQ/Dense32 dispatch for the original Wan FFN.

No import-time numerical work or execution authorization. Intended to be
embedded only by a separately reviewed video producer. The original block
loader and both BF16 FFN H2D transfers are deliberately untouched.
"""
import time,types
import e0_sharq_wan_ffn_ops as ops
import e0_sharq_wan_author_stream_bridge as stream_bridge

MODES=('bf16','bf16_cache','dense32','sharq')
LINEARS={'ffn.0':(5120,13824),'ffn.2':(13824,5120)}
CACHE_LIMIT_BYTES=16 << 30
BRIDGE_COST={'sharq_stream_bridge_calls':1,'sharq_bridge_event_records':2,'sharq_bridge_event_waits':2}


def need(condition,message):
    if not condition:raise ValueError(message)


def operation_budget(mode):
    need(mode in ('dense32','sharq'),'bad FP4 mode')
    return ({'dense32_prepare':1,'output_scale':1,'dense32_gemm':1,'bias_add':1}
            if mode=='dense32' else
            {'sharq_prepare':1,'output_scale':1,'sparse_gemm':1,'dense_residual_gemm':1,'bias_add':1})


class Meter:
    """Per-entry invocation counts inside an already predebited whole linear."""
    def __init__(self,cost):self.remaining=cost.copy();self.invoked={};self.returned={}
    def call(self,key,fn,*args):
        need(self.remaining.get(key,0)>0,'operation without predebit');self.remaining[key]-=1
        self.invoked[key]=self.invoked.get(key,0)+1
        result=fn(*args);self.returned[key]=self.returned.get(key,0)+1;return result
    def finish(self):need(not any(self.remaining.values()),'missing physical entry')


class SameMasterFFNRuntime:
    """Intercept only the eight original offload-buffer MM objects.

    We use original GPU-buffer BF16 W at first use, transpose/materialize and
    call the author's shared-W quantizer once per expert/block/linear. Cached
    q, sparse/dense scale and global scale remain on GPU across both steps and
    later phases. Source bias is read from the *current* original CUDA buffer.
    All original block prefetch and W1/W2 copies remain paid and untouched.
    """
    def __init__(self,pipe,guard,api,quant,ledger,operand_check,*,sync,emit=lambda event:None,cache_limit=CACHE_LIMIT_BYTES):
        self.pipe,self.guard,self.api,self.quant,self.ledger=pipe,guard,api,quant,ledger
        self.operand_check,self.sync,self.emit=operand_check,sync,emit
        self.cache_limit=cache_limit;self.cache={};self.cache_bytes=0;self.saved=[];self.mode='bf16';self.rid=None;self.seen=set();self.objects={}
        models=pipe.runner.model.model;need(len(models)==2,'two experts required')
        for expert,model in enumerate(models):
            buffers=model.transformer_infer.offload_manager.cuda_buffers;need(len(buffers)==2,'original double-buffer required')
            for buffer in buffers:
                phase=buffer.compute_phases[2]
                for suffix,attribute in (('ffn.0','ffn_0'),('ffn.2','ffn_2')):
                    mm=getattr(phase,attribute)
                    need(id(mm) not in self.objects and type(mm).__name__=='MMWeight' and not mm.has_lora_branch and not mm.has_diff and not hasattr(mm,'weight_diff') and not hasattr(mm,'bias_diff'),'unsupported buffer/adapters')
                    self.objects[id(mm)]=(mm,expert,suffix)
        need(len(self.objects)==8,'eight distinct original FFN buffers required')

    def _pack(self,mm,key):
        torch=self.guard.torch;weight=mm.weight;k,n=LINEARS[key[2]]
        dense_sf_bytes=(n//128+1)*128*(k//16)
        conservative=n*(k//2)+2*dense_sf_bytes+4
        need(self.cache_bytes+conservative<=self.cache_limit,'packed GPU cache predebit exceeds budget')
        def materialize():return weight.T.contiguous()
        started=time.perf_counter();original=self.ledger.call(self.rid,{'sharq_static_W_materializations':1,'sharq_static_W_materialization_bytes':weight.numel()*weight.element_size()},materialize);materialize_submit_seconds=time.perf_counter()-started
        def quantize():
            value=self.ledger.call(self.rid,BRIDGE_COST,stream_bridge.author_call,torch,self.quant.quantize_weight_shared_nvfp4,original);self.sync();return value
        packed=self.ledger.call(self.rid,{'sharq_static_W_quantizations':1,'sharq_static_W_cache_predebit_bytes':conservative},quantize);pack_sync_seconds=time.perf_counter()-started
        need(isinstance(packed,(tuple,list)) and len(packed)==4 and all(torch.is_tensor(x) and x.device.type=='cuda' for x in packed),'bad shared weight result')
        q,ss,sd,scale=packed
        need(tuple(q.shape)==(n,k//2) and q.dtype==torch.uint8 and ss.dtype==sd.dtype==torch.uint8 and sd.numel()==dense_sf_bytes and scale.dtype==torch.float32 and scale.numel()==1 and all(x.is_contiguous() for x in packed),'shared packed layout differs from author')
        sizes=sum(x.numel()*x.element_size() for x in packed)
        need(sizes<=conservative and self.cache_bytes+sizes<=self.cache_limit,'packed GPU cache exceeds predebit')
        self.cache[key]=tuple(packed);self.cache_bytes+=sizes
        self.emit({'event':'static_W_cached','expert':key[0],'block':key[1],'linear':key[2],'bytes':sizes,'total_cache_bytes':self.cache_bytes,'materialize_buffer_alias':bool(original.data_ptr()==weight.data_ptr()),'materialize_shape':list(original.shape),'materialize_stride':list(original.stride()),'materialization_bytes_are_logical':'PHYSICAL_COPY_UNKNOWN_IF_NOT_ALIAS','materialize_submit_wall_seconds':materialize_submit_seconds,'materialize_and_quant_sync_wall_seconds':pack_sync_seconds,'source':'current original CUDA block buffer, original H2D paid'})
        return self.cache[key]

    def _apply_fp4(self,mm,value,expert,suffix):
        active=self.guard.active;need(active is not None and active.block is not None and self.rid is not None,'unregistered FFN call')
        current,block=active.block;step=active.step(current)
        need(expert==current==step//2 and 0<=block<40,'expert or block differs')
        need(active.mapping.get(id(mm))==('block',expert,suffix),'MM identity differs')
        target=(step,block,suffix);need(target not in self.seen,'duplicate FFN site')
        meta=self.operand_check(self.guard.torch,mm,value);k,n=LINEARS[suffix]
        need(value.shape[1]==k and tuple(mm.weight.shape)==(k,n) and tuple(mm.bias.shape)==(n,),'FFN geometry/bias differs')
        key=(expert,block,suffix);self.seen.add(target)
        def invoke(_mm,x):
            packed=self.cache.get(key)
            need(packed is not None,'BF16 neutral must complete shared W cache before candidates')
            q,ss,sd,sw=packed
            for tensor in packed:tensor.record_stream(self.guard.torch.cuda.current_stream())
            layer=types.SimpleNamespace(weight_q=q,scale_w_sparse=ss,scale_w_dense=sd,weight_scale=sw,bias=mm.bias,in_features=k,out_features=n)
            cost=operation_budget(self.mode);meter=Meter(cost)
            def linear():
                try:
                    result=stream_bridge.author_call(self.guard.torch,(ops.dense32_linear if self.mode=='dense32' else ops.sharq_linear),x,layer,self.api,meter)
                    meter.finish();return result
                finally:self.emit({'event':'linear_python_entries','step':step,'expert':expert,'block':block,'linear':suffix,'mode':self.mode,'predebited':cost,'invoked':dict(meter.invoked),'returned':dict(meter.returned),'CUDA_completion':'UNKNOWN_UNTIL_CALLER_SYNCHRONIZES'})
            return self.ledger.call(self.rid,{**cost,**BRIDGE_COST},linear)
        return active.mm_call(mm,value,meta,invoke)

    def _apply_bf16_cache(self,mm,value,expert,suffix,original):
        active=self.guard.active;need(active is not None and active.block is not None and self.rid is not None,'unregistered BF16 cache call')
        current,block=active.block;step=active.step(current);need(expert==current==step//2 and 0<=block<40 and active.mapping.get(id(mm))==('block',expert,suffix),'BF16 cache identity differs')
        target=(step,block,suffix);need(target not in self.seen,'duplicate BF16 cache site');self.operand_check(self.guard.torch,mm,value);k,n=LINEARS[suffix]
        need(value.shape[1]==k and tuple(mm.weight.shape)==(k,n) and tuple(mm.bias.shape)==(n,),'BF16 cache FFN geometry differs')
        self.seen.add(target);key=(expert,block,suffix)
        if key not in self.cache:self._pack(mm,key)
        return original(value)

    def __enter__(self):
        self.sync()
        try:
            for mm,expert,suffix in self.objects.values():
                old=mm.apply;present='apply' in mm.__dict__;prior=mm.__dict__.get('apply');self.saved.append((mm,present,prior));runtime=self
                def apply(obj,value,*,_old=old,_expert=expert,_suffix=suffix):
                    if runtime.mode=='bf16':return _old(value)
                    if runtime.mode=='bf16_cache':return runtime._apply_bf16_cache(obj,value,_expert,_suffix,_old)
                    return runtime._apply_fp4(obj,value,_expert,_suffix)
                mm.apply=types.MethodType(apply,mm)
        except BaseException as error:
            self.__exit__(type(error),error,error.__traceback__);raise
        return self

    def begin(self,rid,mode):
        need(self.rid is None and mode in MODES and rid,'invalid phase transition');self.sync();self.rid=rid;self.mode=mode;self.seen=set()

    def finish(self):
        need(self.rid is not None,'phase not begun');self.sync()
        if self.mode!='bf16':need(self.seen=={(s,b,n) for s in range(4) for b in range(40) for n in LINEARS},'incomplete all160 FFN site coverage')
        result={'run_id':self.rid,'mode':self.mode,'actual_linears':len(self.seen),'cache_entries':len(self.cache),'cache_bytes':self.cache_bytes,'original_W1_W2_prefetch':'UNCHANGED_AND_PAID'}
        self.emit({'event':'phase_complete',**result});self.rid=None;self.mode='bf16';return result

    def __exit__(self,typ,value,tb):
        try:self.sync()
        except BaseException as cleanup_error:
            if value is None:raise
            note='runtime cleanup synchronize failed; completion UNKNOWN: '+type(cleanup_error).__name__+': '+str(cleanup_error)
            if hasattr(value,'add_note'):value.add_note(note)
            else:value.__notes__=getattr(value,'__notes__',[])+[note]
        finally:
            for obj,present,prior in reversed(self.saved):
                if present:obj.__dict__['apply']=prior
                else:obj.__dict__.pop('apply',None)
            self.saved.clear();self.cache.clear();self.cache_bytes=0;self.rid=None;self.mode='bf16'

class OriginalFFNPrefetchAudit:
    """Observe, never bypass, each original W1/W2/bias block-buffer load.

    Install after the native tap has installed its own block-loader wrappers.
    Counts are requested CPU-to-GPU copy tensor bytes from actual source and
    target metadata; they are not a hardware-bus traffic counter.
    """
    def __init__(self,pipe,torch,ledger,rid,emit=lambda event:None):
        self.pipe,self.torch,self.ledger,self.rid,self.emit=pipe,torch,ledger,rid,emit
        self.saved=[];self.events=[]
    def observe(self,buffer,destination,block,expert,old,adapter=None):
        need(type(block) is int and 0<=block<40 and adapter is None,'unexpected original prefetch')
        def execute():
            result={}
            for suffix,attribute in (('ffn.0','ffn_0'),('ffn.2','ffn_2')):
                mm=getattr(buffer.compute_phases[2],attribute);k,n=LINEARS[suffix];total=0
                for field,shape in (('weight',(k,n)),('bias',(n,))):
                    source=destination[f'blocks.{block}.{suffix}.{field}'];target=getattr(mm,field+'_cuda_buffer')
                    need(tuple(source.shape)==tuple(target.shape)==shape and source.dtype==target.dtype==self.torch.bfloat16 and source.device.type=='cpu' and target.device.type=='cuda','original FFN copy geometry differs')
                    total+=source.numel()*source.element_size()
                result[suffix]=total
            value=old(destination,block,adapter);event={'expert':expert,'block':block,'requested_W1_H2D_bytes':result['ffn.0'],'requested_W2_H2D_bytes':result['ffn.2'],'interpretation':'requested tensor copy bytes; actual bus traffic UNKNOWN'}
            self.events.append(event);self.emit(event);return value
        return self.ledger.call(self.rid,{'original_FFN_prefetch_observations':1},execute)
    def __enter__(self):
        try:
            for expert,model in enumerate(self.pipe.runner.model.model):
                for buffer in model.transformer_infer.offload_manager.cuda_buffers:
                    old=buffer.load_state_dict;present='load_state_dict' in buffer.__dict__;prior=buffer.__dict__.get('load_state_dict');self.saved.append((buffer,present,prior));audit=self
                    def load(obj,destination,block,adapter_block_index=None,*,_old=old,_expert=expert):return audit.observe(obj,destination,block,_expert,_old,adapter_block_index)
                    buffer.load_state_dict=types.MethodType(load,buffer)
        except BaseException:
            self.__exit__(None,None,None);raise
        return self
    def finish(self):
        need(len(self.events)==162,'expected 162 actual original block prefetch calls')
        return {'original_FFN_prefetch_observations':len(self.events),'requested_W1_H2D_bytes':sum(x['requested_W1_H2D_bytes'] for x in self.events),'requested_W2_H2D_bytes':sum(x['requested_W2_H2D_bytes'] for x in self.events),'original_FFN_prefetch_skips':0}
    def __exit__(self,*_):
        for obj,present,prior in reversed(self.saved):
            if present:obj.__dict__['load_state_dict']=prior
            else:obj.__dict__.pop('load_state_dict',None)
        self.saved.clear()
