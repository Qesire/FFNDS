"""Seven-phase sync-granularity tap; draft, no authorization at import."""
from chipmunk_input_video_tap import make_tap as original_tap, need
from chipmunk_p64_deferred_ledger import DeferredLedger

# Mask transfer/SHA, cache creation/release and cleanup stay on the original
# synchronous path. Only these same-stream candidate actions can be deferred.
DEFER_KEYS=frozenset(('weight_materializations','p64_score','p64_select','selected_old_h','packed_action'))

def make_tap(vm,native,*,mode,rows,kernels,emit,deferred,geometry=(31668,5120,13824,5120)):
    base=original_tap(vm,native,mode=mode,rows=rows,kernels=kernels,emit=emit,geometry=geometry)
    class Tap(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.deferred_transaction=None
        def charged(self,cost,fn):
            keys=set(cost)
            if not deferred:return super().charged(cost,fn)
            if keys & DEFER_KEYS:
                need(len(keys & DEFER_KEYS)==1,'ambiguous deferred action')
                stream=self.guard.torch.cuda.current_stream()
                if self.deferred_transaction is None:
                    self.deferred_transaction=DeferredLedger(self.guard.ledger,native.LIMITS,stream,self.guard.torch.cuda.synchronize,native.RUN_IDS)
                return self.deferred_transaction.call(self.rid,{k:v for k,v in cost.items() if v},fn,actual_stream=stream)
            if 'cache_release_sync' in keys and self.deferred_transaction is not None:
                self.guard.ledger.call(self.rid,{'deferred_completion_barriers':1},self.deferred_transaction.barrier)
                self.deferred_transaction=None
            return super().charged(cost,fn)
        def finish(self):
            need(self.deferred_transaction is None,'deferred transaction was not closed')
            return super().finish()
    return Tap
