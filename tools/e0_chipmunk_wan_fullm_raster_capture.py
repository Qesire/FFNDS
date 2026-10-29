"""Passive full-M native BF16 FFN capture; no model or GPU at import.

Captures actual W1 input, post-GELU W2 input, W2 output and FFN gate for
block20 at all four steps. The caller owns the separately reviewed scope,
source, authorization, synchronization, storage, and neutral comparison.
"""
import math
GRID=(21,29,52);M=math.prod(GRID);SITES=frozenset((s,20) for s in range(4));DTYPE='bfloat16'
OPERANDS={'V1':(M,5120),'h':(M,13824),'Y':(M,5120)}
OPERAND_BYTES={k:2*math.prod(shape) for k,shape in OPERANDS.items()};GATE_BYTES=5120*2;PAYLOAD_BYTES_PER_STEP=sum(OPERAND_BYTES.values())+GATE_BYTES

def need(x,msg):
 if not x:raise ValueError(msg)
def coordinate(row):
 need(type(row) is int and 0<=row<M,'raster row out of range');return (row//(29*52),(row//52)%29,row%52)

def header(source,step,expert,grid,values):
 need(type(source) is int and source in (898,96) and type(step) is int and 0<=step<4 and expert==(0 if step<2 else 1) and tuple(grid)==GRID,'invalid source/site/grid')
 need(set(values)=={'V1','h','Y','gate'},'capture incomplete')
 return {'schema':'WAN_FULLM_FFN_RASTER_BF16_V1','source_ordinal':source,'seed':42,'mode':'passive_original_bf16','step':step,'block':20,'expert':'high' if expert==0 else 'low','grid':list(GRID),'row_count':M,'row_mapping':'row=tau*29*52+y*52+x; tau=0..20,y=0..28,x=0..51','row_order':'actual native tensor row index; no reorder or subset','operands':values}

class CaptureTransaction:
 def __init__(self,copy,commit):self.copy,self.commit=copy,commit;self.pending=None;self.seen=set()
 def begin(self,step,block,expert,gate):
  site=(step,block)
  if site not in SITES:need(self.pending is None,'nested capture');return False
  need(self.pending is None and site not in self.seen and expert==(0 if step<2 else 1) and gate is not None,'wrong/duplicate site');self.pending={'site':site,'expert':expert,'gate_object':gate,'values':{},'Y_object':None};self.pending['values']['gate']=self.copy('gate',gate);return True
 def operand(self,name,value,shape,dtype,stride):
  need(self.pending is not None and name in OPERANDS and name not in self.pending['values'] and tuple(shape)==OPERANDS[name] and dtype==DTYPE,'operand geometry/order differs');self.pending['values'][name]=self.copy(name,value,tuple(stride))
  if name=='Y':self.pending['Y_object']=value
 def finish(self,step,block,expert,gate,y,grid):
  p=self.pending;site=(step,block)
  if site not in SITES:need(p is None,'unexpected writeback');return
  need(p is not None and p['site']==site and p['expert']==expert and gate is p['gate_object'] and y is p['Y_object'] and tuple(grid)==GRID and set(p['values'])=={'V1','h','Y','gate'},'actual FFN writeback identity differs');self.commit(p,tuple(grid));self.seen.add(site);self.pending=None
 def verify_complete(self):need(self.pending is None and self.seen==SITES,'missing actual full-M sites')

def make_tap_class(vm,native):
 class RasterTap(vm.LightTap):
  def __init__(self,*args,capture,**kw):super().__init__(*args,**kw);self.capture=capture
  def mm_call(self,mm,value,meta,original):
   slot=self.mapping.get(id(mm));suffix=slot[2] if slot is not None and slot[0]=='block' else None;p=self.capture.pending
   if p is not None and suffix in ('ffn.0','ffn.2'):
    need(self.block==(p['expert'],p['site'][1]),'FFN capture block mismatch');self.capture.operand('V1' if suffix=='ffn.0' else 'h',value,value.shape,str(value.dtype).removeprefix('torch.'),value.stride())
   result=super().mm_call(mm,value,meta,original)
   if p is not None and suffix=='ffn.2':self.capture.operand('Y',result,result.shape,str(result.dtype).removeprefix('torch.'),result.stride())
   return result
  def install(self):
   super().install()
   for expert,model in enumerate(self.runner.model.model):
    infer=model.transformer_infer;ffn0,post0=infer.infer_ffn,infer.post_process
    def ffn(phase,x,attn,shift,scale,gate=None,*,_expert=expert,_infer=infer,_old=ffn0):
     step,block=self.step(_expert),int(_infer.block_idx);active=self.capture.begin(step,block,_expert,gate)
     value=_old(phase,x,attn,shift,scale,gate)
     if active:need(value is self.capture.pending['Y_object'],'FFN output object changed')
     return value
    def post(x,y,gate,pre_infer_out=None,*,_expert=expert,_infer=infer,_old=post0):
     step,block=self.step(_expert),int(_infer.block_idx)
     if (step,block) in SITES:
      need(pre_infer_out is not None and self.block==(_expert,block),'grid/block unavailable');self.capture.finish(step,block,_expert,gate,y,tuple(pre_infer_out.grid_sizes.tuple))
     return _old(x,y,gate,pre_infer_out)
    self.stack.enter_context(native.patch_method(infer,'infer_ffn',ffn));self.stack.enter_context(native.patch_method(infer,'post_process',post))
  def finish(self):super().finish();self.capture.verify_complete()
 return RasterTap
