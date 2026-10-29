"""Non-executable engineering integration for same-cache Dense32 ABBA.

Requires a new supervised owner, artifact verifier, bounded observers and real
review before model use. This module has no runtime authorization.
"""
import e0_sharq_wan_all160_runtime_v2 as original
from sharq_resident_ffn_load_draft import SelectiveFFNLoad
from sharq_resident_ffn_guard_draft import operand_guard, require_load_context, load_source_guard, require_compute_context

class ResidentDense32Runtime(original.SameMasterFFNRuntime):
 def __init__(self,pipe,guard,api,quant,ledger,*,sync,resolve,charge_load,original_operand,boundary_control,emit=lambda row:None):
  super().__init__(pipe,guard,api,quant,ledger,self._check_operand,sync=sync,emit=emit)
  original.need(callable(boundary_control),'boundary observer required')
  self._boundary_control=boundary_control
  self._boundary_seen=set()
  self._original_operand=original_operand
  self._selector=SelectiveFFNLoad([(mm,expert,suffix) for mm,expert,suffix in self.objects.values()],self.cache,resolve,charge_load,emit,self._check_load,lambda t:t.numel()*t.element_size(),self._check_source)
  self._resident_next=False
 def _check_operand(self,torch,mm,value):
  if self.mode=='bf16_cache':return self._original_operand(torch,mm,value)
  active=self.guard.active;expert,block=active.block
  obj, mapped_expert,suffix=self.objects[id(mm)]
  original.need(obj is mm and mapped_expert==expert,'dispatch owner')
  require_compute_context(torch,self.pipe.runner.model.model[expert].transformer_infer.offload_manager,mm,suffix)
  packed=self.cache.get((expert,block,suffix));k,n=original.LINEARS[suffix]
  return operand_guard(torch,mm,value,packed,k,n)
 def _check_source(self,mm,key,w,b):
  load_source_guard(self.guard.torch,mm,key,w,b,original.LINEARS)
 def _check_load(self,mm,key):
  model=self.pipe.runner.model.model[key[0]]
  require_load_context(self.guard.torch,model.transformer_infer.offload_manager,mm,key,self)
 def __enter__(self):
  super().__enter__()
  try:self._selector.__enter__()
  except BaseException:
   import sys
   super().__exit__(*sys.exc_info());raise
  return self
 def _apply_fp4(self,mm,value,expert,suffix):
  active=self.guard.active;step=active.step(expert);block=active.block[1]
  if self.mode!='dense32' or block!=0 or suffix!='ffn.0' or step not in (0,2):
   return super()._apply_fp4(mm,value,expert,suffix)
  target=(step,block,suffix);original.need(target not in self._boundary_seen,'duplicate boundary observer')
  output=super()._apply_fp4(mm,value,expert,suffix)
  # Supplied owner must charge duplicate prepare/GEMM/bias/compare and
  # synchronization; reference receives the SAME input/cache/current bias.
  self._boundary_control(self,mm,value,output,(expert,block,suffix))
  self._boundary_seen.add(target);return output
 def set_next_resident(self,enabled):
  original.need(self.rid is None and isinstance(enabled,bool),'phase selection')
  if enabled:original.need(set(self.cache)=={(e,b,s) for e in (0,1) for b in range(40) for s in original.LINEARS},'cache not fully built')
  self._resident_next=enabled
 def begin(self,rid,mode):
  original.need(not self._resident_next or mode=='dense32','resident only for Dense32')
  self._boundary_seen=set();super().begin(rid,mode);self._selector.enabled=self._resident_next;self._resident_next=False
 def finish(self):
  try:
   if self.mode=='dense32':original.need(self._boundary_seen=={(0,0,'ffn.0'),(2,0,'ffn.0')},'representative controls incomplete')
   resident=self._selector.enabled
   row=super().finish()
   row['original_W1_W2_prefetch']='RESIDENT_SELECTED_CHECK_COPY_JOURNAL' if resident else 'LEGACY_SELECTED_CHECK_COPY_JOURNAL'
   return row
  finally:self._selector.enabled=False
 def __exit__(self,typ,value,tb):
  self._selector.enabled=False
  try:self._selector.__exit__(typ,value,tb)
  finally:super().__exit__(typ,value,tb)
  return False
