"""Unregistered resident Dense32 runtime guard, with injected Torch API."""

def need(value,message):
 if not value:raise ValueError(message)

def operand_guard(torch,mm,x,packed,k,n):
 """Validate actual operands; BF16 W buffer only establishes layout, never data."""
 need(torch.is_tensor(x) and x.dtype==torch.bfloat16 and x.device.type=='cuda' and x.ndim==2 and x.shape[1]==k,'input')
 w,b=mm.weight,mm.bias
 need(torch.is_tensor(w) and w is mm.weight_cuda_buffer and w.dtype==torch.bfloat16 and w.device.type=='cuda' and tuple(w.shape)==(k,n) and tuple(w.stride())==(1,k),'stale weight buffer metadata')
 need(torch.is_tensor(b) and b is mm.bias_cuda_buffer and b.dtype==torch.bfloat16 and b.device.type=='cuda' and tuple(b.shape)==(n,),'bias buffer')
 need(isinstance(packed,(tuple,list)) and len(packed)==4,'packed tuple')
 q,ss,sd,scale=packed
 need(all(torch.is_tensor(t) and t.device.type=='cuda' and t.is_contiguous() for t in packed),'packed device/layout')
 need(q.dtype==torch.uint8 and tuple(q.shape)==(n,k//2) and ss.dtype==sd.dtype==torch.uint8 and sd.numel()==(n//128+1)*128*(k//16) and scale.dtype==torch.float32 and scale.numel()==1,'packed shape/dtype')
 return {'input':list(x.shape),'weight':list(w.shape),'input_stride':list(x.stride()),'weight_stride':list(w.stride()),'bias_stride':list(b.stride()),'dtype':str(x.dtype),'device':str(x.device),'weight_contents':'STALE_FORBIDDEN_AS_ARITHMETIC'}

def require_load_context(torch,manager,mm,key,runtime):
 need(runtime.mode=='dense32' and runtime.rid is not None,'resident scope/phase')
 need(not manager.lazy_load and len(manager.cuda_buffers)==2,'offload configuration')
 stream=torch.cuda.current_stream();need(stream in (manager.cuda_load_stream,manager.init_stream),'unexpected load stream')
 need(key in runtime.cache,'cache identity')
 expected_buffer=manager.cuda_buffers[0 if stream==manager.init_stream else 1]
 need(key[2] in ('ffn.0','ffn.2') and mm is getattr(expected_buffer.compute_phases[2],'ffn_0' if key[2]=='ffn.0' else 'ffn_2'),'wrong active load buffer')


def load_source_guard(torch,mm,key,w,b,linears):
 need(key[2] in linears,'linear key');k,n=linears[key[2]]
 need(torch.is_tensor(w) and torch.is_tensor(b) and w.device.type==b.device.type=='cpu' and w.dtype==b.dtype==torch.bfloat16 and tuple(w.shape)==(k,n) and tuple(b.shape)==(n,),'actual FFN source shape/dtype/device')
 need(torch.is_tensor(mm.weight_cuda_buffer) and torch.is_tensor(mm.bias_cuda_buffer) and mm.weight_cuda_buffer.device.type==mm.bias_cuda_buffer.device.type=='cuda' and mm.weight_cuda_buffer.dtype==mm.bias_cuda_buffer.dtype==torch.bfloat16 and tuple(mm.weight_cuda_buffer.shape)==(k,n) and tuple(mm.bias_cuda_buffer.shape)==(n,),'actual FFN target shape/dtype/device')

def require_compute_context(torch,manager,mm,suffix):
 need(suffix in ('ffn.0','ffn.2') and not manager.lazy_load and len(manager.cuda_buffers)==2,'compute configuration')
 need(torch.cuda.current_stream()==manager.compute_stream,'wrong compute stream')
 target=getattr(manager.cuda_buffers[0].compute_phases[2],'ffn_0' if suffix=='ffn.0' else 'ffn_2')
 need(mm is target,'wrong active compute buffer')
