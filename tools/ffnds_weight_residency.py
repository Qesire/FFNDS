"""Optional dense D-Q weight allocation/residency controls; no math changes.

The original one-block policy remains a comparator. No vendor source is edited.
"""
import time


class DenseWeightResidency:
    modes=('original','block_no_flush','ffn_resident_no_flush')

    def __init__(self,runtime,mode,max_retained_bytes=24*(1<<30)):
        if mode not in self.modes:raise ValueError('Unknown residency control')
        self.runtime,self.mode,self.limit=runtime,mode,max_retained_bytes
        self.retained={}
        self.stats={'weight_requests':0,'FFN_weight_requests':0,'FFN_retained_hits':0,
                    'retained_bytes':0,'peak_retained_bytes':0,'block_key_switches':0,
                    'avoided_explicit_empty_cache_calls':0,
                    'weight_fetch_wall_seconds':0.,'FFN_weight_fetch_wall_seconds':0.}

    def __enter__(self):
        import torch
        self.old=self.runtime._weight
        owner=self
        def weight(*,expert,block_index,module_name,record,device,dtype):
            start=time.perf_counter();is_ffn=module_name.endswith(('.ffn.0','.ffn.2'))
            owner.stats['weight_requests']+=1
            if is_ffn:owner.stats['FFN_weight_requests']+=1
            retain_key=(expert,module_name,str(device),str(dtype))
            if owner.mode=='ffn_resident_no_flush' and is_ffn and retain_key in owner.retained:
                tensor=owner.retained[retain_key];owner.stats['FFN_retained_hits']+=1
            elif owner.mode=='original':
                key=(expert,int(block_index),str(device),str(dtype))
                owner.stats['block_key_switches']+=int(key!=owner.runtime._resident_key)
                tensor=owner.old(expert=expert,block_index=block_index,module_name=module_name,
                                 record=record,device=device,dtype=dtype)
            else:
                # Same exact key, transfer, dtype and contiguous layout as the
                # pinned native _weight, omitting only its explicit cache flush.
                key=(expert,int(block_index),str(device),str(dtype))
                if key!=owner.runtime._resident_key:
                    owner.runtime._resident_weights.clear();owner.runtime._resident_key=key
                    owner.stats['block_key_switches']+=1
                    if device.type=='cuda':owner.stats['avoided_explicit_empty_cache_calls']+=1
                tensor=owner.runtime._resident_weights.get(module_name)
                if tensor is None:
                    source=record.get('weight_qdq')
                    if not torch.is_tensor(source):raise TypeError('Native cache weight missing')
                    tensor=source.to(device=device,dtype=dtype,non_blocking=False).contiguous()
                    owner.runtime._resident_weights[module_name]=tensor
                    owner.runtime.counters['weight_transfers']+=1
                if owner.mode=='ffn_resident_no_flush' and is_ffn:
                    size=tensor.numel()*tensor.element_size()
                    if owner.stats['retained_bytes']+size>owner.limit:raise RuntimeError('FFN retention budget exceeded')
                    owner.retained[retain_key]=tensor;owner.stats['retained_bytes']+=size
                    owner.stats['peak_retained_bytes']=max(owner.stats['peak_retained_bytes'],owner.stats['retained_bytes'])
            duration=time.perf_counter()-start;owner.stats['weight_fetch_wall_seconds']+=duration
            if is_ffn:owner.stats['FFN_weight_fetch_wall_seconds']+=duration
            return tensor
        self.runtime._weight=weight
        return self

    def snapshot(self):return dict(self.stats)

    def __exit__(self,*error):
        self.runtime._weight=self.old;self.retained.clear();self.stats['retained_bytes']=0
