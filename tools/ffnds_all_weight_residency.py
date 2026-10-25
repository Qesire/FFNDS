"""Memory-bounded retention of exact decoded native quantized linear weights.

This independent control extends the FFN-only study without changing its code.
"""
import time


class AllWeightResidency:
    modes=('all_resident_no_flush',)

    def __init__(self,runtime,mode,max_retained_bytes):
        if mode not in self.modes:raise ValueError('Unknown full residency mode')
        self.runtime,self.mode,self.limit=runtime,mode,max_retained_bytes
        self.retained={};self.logical_key=None
        self.stats={'weight_requests':0,'FFN_weight_requests':0,'all_retained_hits':0,
                    'FFN_retained_hits':0,'retained_bytes':0,'peak_retained_bytes':0,
                    'block_key_switches':0,'avoided_explicit_empty_cache_calls':0,
                    'weight_fetch_wall_seconds':0.,'FFN_weight_fetch_wall_seconds':0.}

    def __enter__(self):
        import torch
        self.old=self.runtime._weight;owner=self
        def weight(*,expert,block_index,module_name,record,device,dtype):
            begin=time.perf_counter();is_ffn=module_name.endswith(('.ffn.0','.ffn.2'))
            owner.stats['weight_requests']+=1
            if is_ffn:owner.stats['FFN_weight_requests']+=1
            logical=(expert,int(block_index),str(device),str(dtype))
            if logical!=owner.logical_key:
                owner.logical_key=logical;owner.stats['block_key_switches']+=1
                if device.type=='cuda':owner.stats['avoided_explicit_empty_cache_calls']+=1
            key=(expert,module_name,str(device),str(dtype))
            if key in owner.retained:
                tensor=owner.retained[key];owner.stats['all_retained_hits']+=1
                if is_ffn:owner.stats['FFN_retained_hits']+=1
            else:
                source=record.get('weight_qdq')
                if not torch.is_tensor(source):raise TypeError('Native decoded weight missing')
                size=source.numel()*torch.empty((),dtype=dtype,device='cpu').element_size()
                if owner.stats['retained_bytes']+size>owner.limit:raise RuntimeError('Full decoded-weight residency budget exceeded')
                tensor=source.to(device=device,dtype=dtype,non_blocking=False).contiguous()
                owner.retained[key]=tensor;owner.stats['retained_bytes']+=size
                owner.stats['peak_retained_bytes']=max(owner.stats['peak_retained_bytes'],owner.stats['retained_bytes'])
                owner.runtime.counters['weight_transfers']+=1
            elapsed=time.perf_counter()-begin;owner.stats['weight_fetch_wall_seconds']+=elapsed
            if is_ffn:owner.stats['FFN_weight_fetch_wall_seconds']+=elapsed
            return tensor
        self.runtime._weight=weight;return self

    def snapshot(self):return dict(self.stats)

    def __exit__(self,*error):
        self.runtime._weight=self.old;self.retained.clear();self.stats['retained_bytes']=0
