"""Bridge all SharQ author preparation and native calls onto the default stream."""
from types import SimpleNamespace

def _tensors(value,torch):
 if torch.is_tensor(value):yield value
 elif isinstance(value,SimpleNamespace):yield from _tensors(vars(value),torch)
 elif isinstance(value,dict):
  for item in value.values():yield from _tensors(item,torch)
 elif isinstance(value,(tuple,list)):
  for item in value:yield from _tensors(item,torch)

def author_call(torch,fn,*args,**kwargs):
 source=torch.cuda.current_stream();target=torch.cuda.default_stream();ready=torch.cuda.Event();done=torch.cuda.Event();result=None
 try:
  ready.record(source);target.wait_event(ready)
  for t in _tensors((args,kwargs),torch):
   if t.device.type=='cuda':t.record_stream(target)
  with torch.cuda.stream(target):
   result=fn(*args,**kwargs);done.record(target)
  source.wait_event(done)
  for t in _tensors(result,torch):
   if t.device.type=='cuda':t.record_stream(source)
  return result
 except BaseException as error:
  # Keep both argument and result references live while all possibly issued
  # work drains.  Any failed drain is explicit but does not hide the original
  # error; the enclosing scope terminates and retains CUDA completion UNKNOWN.
  for label,stream in (('default',target),('caller',source)):
   try:stream.synchronize()
   except BaseException as sync_error:
    if hasattr(error,'add_note'):error.add_note(label+' stream drain failed: '+type(sync_error).__name__)
  raise
