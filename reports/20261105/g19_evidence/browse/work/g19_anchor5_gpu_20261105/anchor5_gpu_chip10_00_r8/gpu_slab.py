"""One current-expert CUDA slab; uninitialized slots must never be read."""
class Slab:
 def __init__(self,torch,dtype,device,rows=31668,width=5120):
  self.torch=torch; self.dtype=dtype; self.device=device; self.rows=rows; self.width=width
  self.data=torch.empty((80,rows,width),dtype=dtype,device=device); self.owner=None; self.valid=set()
  self.slot_bytes=rows*width*self.data.element_size();self.physical_bytes=80*self.slot_bytes
 def _index(self,branch,layer):
  if branch not in ('conditional','unconditional') or type(layer)!=int or not 0<=layer<40:raise ValueError('invalid cache key')
  return (0 if branch=='conditional' else 40)+layer
 def bind(self,expert):
  if expert not in ('high','low'):raise ValueError('invalid expert')
  if self.owner!=expert:self.valid.clear();self.owner=expert
 def put(self,expert,branch,layer,value):
  if self.owner!=expert:raise RuntimeError('wrong expert ownership')
  i=self._index(branch,layer)
  if value.shape!=(self.rows,self.width) or value.dtype!=self.dtype or value.device!=self.device:raise RuntimeError('cache payload mismatch')
  self.data[i].copy_(value);self.valid.add(i)
 def get(self,expert,branch,layer):
  i=self._index(branch,layer)
  if self.owner!=expert or i not in self.valid:raise RuntimeError('uninitialized/stale slot')
  return self.data[i]
 def release(self):
  self.valid.clear();self.owner=None;self.data=None
