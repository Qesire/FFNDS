"""Bounded durable observation sink, used only inside charged observer calls."""
import json,hashlib,struct,os
from pathlib import Path

def gate_metadata(torch,gate,width=5120):
 shape=list(gate.shape);stride=list(gate.stride())
 if gate.dtype!=torch.bfloat16 or shape!=[1,1,width] or stride!=[6*width,width,1]:raise ValueError('gate BF16 shape/stride differs')
 return {'gate':gate.squeeze().float().cpu().tolist(),'gate_shape':shape,'gate_stride':stride,'gate_dtype':'bfloat16'}

class Sink:
 def __init__(self,*,torch,counter,out,hf_root,save_file,max_bytes,host_guard,max_host_copy_bytes):
  self.torch=torch;self.counter=counter;self.out=Path(out);self.hf=Path(hf_root);self.save_file=save_file;self.max_bytes=max_bytes;self.host_guard=host_guard;self.max_host_copy_bytes=max_host_copy_bytes;self.records={};self.completed=set()
  if max_bytes<=0 or max_host_copy_bytes<=0 or not callable(host_guard) or self.out.exists():raise ValueError('fresh bounded output required')
  self.out.mkdir(parents=True)
 def _tensor_file(self,path,tensors):
  temporary=path.with_suffix(path.suffix+'.tmp')
  if path.exists() or temporary.exists():raise ValueError('duplicate artifact')
  n=sum(v.numel()*v.element_size() for v in tensors.values())
  if n>self.max_host_copy_bytes:raise ValueError('host copy bound')
  self.host_guard(n)
  if self.counter.summary()['peak_tracked_bytes_plus_reserved_and_atomic_temp']+n>self.max_bytes:raise ValueError('observer output cap')
  self.counter.reserve_payload(n)
  try:
   cpu={k:v.detach().cpu().contiguous() for k,v in tensors.items()}
   raw={k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.'),'raw_sha256':hashlib.sha256(memoryview(v.view(self.torch.uint8).numpy())).hexdigest()} for k,v in cpu.items()}
   self.save_file(cpu,str(temporary))
   os.replace(temporary,path)
  finally:
   # Retain and charge partial temp or final output against the reservation.
   if temporary.exists():self.counter.observe_external(temporary)
   elif path.exists():self.counter.observe_external(path)
  with path.open('rb') as f:
   head=f.read(8);size=struct.unpack('<Q',head)[0];header=json.loads(f.read(size))
  names=set(header)-{'__metadata__'}
  if names!=set(raw):raise ValueError('saved tensor keys differ')
  types={'bfloat16':'BF16','float32':'F32','int64':'I64'}
  for name,entry in raw.items():
   h=header[name]
   if h['shape']!=entry['shape'] or h['dtype']!=types.get(entry['dtype']):raise ValueError('saved tensor header differs')
   entry['data_offsets']=h['data_offsets']
  return {'file':self.counter.binding(path,self.hf),'tensors':raw}
 def metadata(self,block,mask,score,selector_rows,gate,eval_rows):
  if block in self.records:raise ValueError('duplicate site')
  path=self.out/f'block{block:02d}_metadata.json';v={'block':block,'mask':mask.detach().cpu().tolist(),'selector_score':score.detach().float().cpu().tolist(),'selector_rows':selector_rows.detach().cpu().tolist(),'evaluation_rows':eval_rows,**gate_metadata(self.torch,gate)};self.counter.write_json(path,v);self.records[block]={'metadata':self.counter.binding(path,self.hf),'chunks':0,'partitions':{}}
 def sample(self,block,selector_rows,eval_rows,x,old_h,current_h,old_y,dense_y,candidate_y):
  rows=self.torch.tensor(list(selector_rows.detach().cpu().tolist())+[z['row'] for z in eval_rows],device=x.device,dtype=self.torch.int64)
  tensors={k:v.index_select(0,rows) for k,v in {'V1':x,'old_h':old_h,'current_h':current_h,'old_Y':old_y,'current_dense_Y':dense_y,'candidate_Y':candidate_y}.items()};self.records[block]['samples']=self._tensor_file(self.out/f'block{block:02d}_samples.safetensors',tensors)
 def snapshot(self,block,x,old_h,current_h,old_y,dense_y,candidate_y,gate,mask,score,selector_rows,eval_rows):
  tensors={'V1':x,'old_h':old_h,'current_h':current_h,'old_Y':old_y,'current_dense_Y':dense_y,'candidate_Y':candidate_y}
  self.records[block]['snapshot']={}
  for name,value in tensors.items():
   self.records[block]['snapshot'][name]=self._tensor_file(self.out/f'block{block:02d}_full_{name}.safetensors',{name:value})
 def dense_metadata(self,block,eval_rows,selector,gate):
  if block in self.records:raise ValueError('duplicate site')
  p=self.out/f'block{block:02d}_metadata.json';self.counter.write_json(p,{'block':block,'mode':'dense_no_hypothetical_mask','evaluation_rows':eval_rows,'selector_rows':list(selector),**gate_metadata(self.torch,gate)});self.records[block]={'metadata':self.counter.binding(p,self.hf),'chunks':0,'partitions':{}}
 def dense_sample(self,block,selector,eval_rows,x,old_h,current_h,old_y,dense_y):
  rows=self.torch.tensor(list(selector)+[z['row'] for z in eval_rows],device=x.device,dtype=self.torch.int64)
  tensors={k:v.index_select(0,rows) for k,v in {'V1':x,'old_h':old_h,'current_h':current_h,'old_Y':old_y,'current_dense_Y':dense_y}.items()};self.records[block]['samples']=self._tensor_file(self.out/f'block{block:02d}_samples.safetensors',tensors);self.completed.add(block)
 def dense_snapshot(self,block,x,old_h,current_h,old_y,dense_y):
  self.records[block]['snapshot']={}
  for name,value in {'V1':x,'old_h':old_h,'current_h':current_h,'old_Y':old_y,'current_dense_Y':dense_y}.items():
   self.records[block]['snapshot'][name]=self._tensor_file(self.out/f'block{block:02d}_full_{name}.safetensors',{name:value})
 def gate_slice(self,gate,start,end):
  g=gate.squeeze()
  if g.ndim!=1:raise ValueError('per-row gate unsupported')
  return g
 def partition(self,block,start,end,stats):
  rec=self.records[block]; expected=rec['chunks']*256
  if start!=expected or end<=start:raise ValueError('chunk order differs')
  for key,value in stats.items():
   values=value.detach().cpu().tolist();previous=rec['partitions'].setdefault(key,[0]*len(values));
   if len(previous)!=len(values):raise ValueError('partition shape differs')
   rec['partitions'][key]=[a+b for a,b in zip(previous,values)]
  rec['chunks']+=1
  self.counter.write_json(self.out/f'block{block:02d}_stats_progress.json',rec)
 def finish_site(self,block):
  rec=self.records[block]
  if rec['chunks']!=124 or sum(rec['partitions']['count'])!=31668 or block in self.completed:raise ValueError('incomplete site')
  path=self.out/f'block{block:02d}_stats.json';progress=self.out/f'block{block:02d}_stats_progress.json';self.counter.write_json(path,rec);rec['stats']=self.counter.binding(path,self.hf);rec['stats_progress']=self.counter.binding(progress,self.hf);self.completed.add(block);return rec['stats']
 def finalize(self):
  if self.completed!=set(range(40)):raise ValueError('full 40 site coverage required')
  return {'sites':40,'records':self.records,'quality_acceptance':False}
