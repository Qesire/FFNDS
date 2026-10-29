"""Same-input Dense32 representative boundary control, separately charged.

Caller selects and freezes sites and supplies already-cached W and current bias.
An owner must predebit duplicate kernel/prepare, comparison, synchronization and
observer time; this helper deliberately performs no work at import.
"""
def need(v,m):
 if not v:raise ValueError(m)

def control(actual,run_reference,bitwise_compare,synchronize,charge,persist):
 """The reference must evaluate exactly the same x/cache/bias/author bridge.

Persist the comparison before applying the strict byte gate. The callback must
compare tensor storage bytes (not numeric equality, which conflates signed zero).
"""
 def observe():
  reference=run_reference();synchronize();return bitwise_compare(actual,reference)
 outcome=charge(observe);need(type(outcome) is dict and type(outcome.get('byte_equal')) is bool,'control record')
 persist(outcome);need(outcome['byte_equal'],'representative boundary byte mismatch');return outcome

def make_dense32_reference(*,torch,ops,bridge,api,charge,synchronize,persist):
 """Build a same-input packed-W/current-bias duplicate Dense32 control.

``charge`` must predebit the dense32 prepare/scale/GEMM/bias, stream bridge,
byte comparison, scalar readback and sync, then retain UNKNOWN on interruption.
 """
 from types import SimpleNamespace
 def callback(runtime,mm,value,actual,key):
  packed=runtime.cache[key];q,ss,sd,sw=packed;k,n=value.shape[1],mm.bias.shape[0]
  layer=SimpleNamespace(weight_q=q,scale_w_sparse=ss,scale_w_dense=sd,weight_scale=sw,bias=mm.bias,in_features=k,out_features=n)
  def reference():
   from e0_sharq_wan_all160_runtime_v2 import Meter,operation_budget
   meter=Meter(operation_budget('dense32'))
   result=bridge.author_call(torch,ops.dense32_linear,value,layer,api,meter);meter.finish();return result
  def compare(a,b):
   need(tuple(a.shape)==tuple(b.shape) and a.dtype==b.dtype and a.device==b.device,'control shape/dtype/device')
   finite=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
   equal=bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
   return {'key':key,'byte_equal':equal,'finite':finite,'logical_elements':a.numel()}
  result=control(actual,reference,compare,synchronize,charge,persist)
  need(result['finite'],'nonfinite boundary output');return result
 return callback
