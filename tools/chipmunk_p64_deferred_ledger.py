"""Draft same-stream, block-scoped completion ledger; no CUDA imports."""
from contextlib import nullcontext
class DeferredLedger:
 def __init__(self,ledger,limits,stream,barrier,run_ids=None):
  self.ledger=ledger;self.limits=limits;self.stream=stream;self.barrier_fn=barrier;self.pending=[];self.poisoned=False;self.run_ids=run_ids
 def lock(self):return getattr(self.ledger,'lock',None) or nullcontext()
 def call(self,rid,costs,fn,*,actual_stream):
  if self.poisoned or actual_stream!=self.stream or (self.run_ids is not None and rid not in self.run_ids):raise RuntimeError('deferred stream or state differs')
  with self.lock():
   r=self.ledger.record
   if not costs or not all(k in self.limits and type(n) is int and n>0 and r['predebited'].get(k,0)+n<=self.limits[k] for k,n in costs.items()):raise ValueError('deferred budget exceeded')
   for k,n in costs.items():r['predebited'][k]=r['predebited'].get(k,0)+n
   item={'run_id':rid,'costs':dict(costs),'state':'RESERVED_UNKNOWN','keepalive_callable':fn};self.pending.append(item)
   try:self.ledger.journal({'kind':'reserved_before_call','run_id':rid,'costs':costs,'completion':'PENDING_DEVICE_BARRIER'})
   except BaseException:
    self.poisoned=True;raise
  try:
   value=fn();item['keepalive']=value;item['state']='ENQUEUED_UNKNOWN'
   with self.lock():self.ledger.journal({'kind':'enqueued_not_completed','run_id':rid,'costs':costs})
   return value
  except BaseException:
   self.poisoned=True;raise
 def barrier(self):
  if self.poisoned:raise RuntimeError('failed enqueue cannot become complete')
  try:self.barrier_fn()
  except BaseException:
   self.poisoned=True;raise
  with self.lock():
   while self.pending:
    item=self.pending[0]
    if item['state']!='ENQUEUED_UNKNOWN':raise RuntimeError('incomplete deferred action')
    try:self.ledger.journal({'kind':'observed_after_device_barrier','run_id':item['run_id'],'costs':item['costs']})
    except BaseException:
     self.poisoned=True;raise
    for k,n in item['costs'].items():self.ledger.record['observed'][k]=self.ledger.record['observed'].get(k,0)+n
    self.pending.pop(0)

def replay(events):
 """Replay new and original operation journal, rejecting impossible ordering."""
 pre={};done={};pending={}
 def add(dst,cost):
  for k,n in cost.items():
   if type(n) is not int or n<=0:raise ValueError('invalid count')
   dst[k]=dst.get(k,0)+n
 def token(row):return (row['run_id'],tuple(sorted(row['costs'].items())))
 for row in events:
  kind=row.get('kind')
  if kind not in ('reserved_before_call','enqueued_not_completed','observed_return','observed_after_device_barrier'):continue
  t=token(row)
  if kind=='reserved_before_call':
   add(pre,row['costs']);pending.setdefault(t,[]).append({'deferred':row.get('completion')=='PENDING_DEVICE_BARRIER','enqueued':False})
  elif kind=='enqueued_not_completed':
   candidates=[x for x in pending.get(t,[]) if x['deferred'] and not x['enqueued']]
   if not candidates:raise ValueError('enqueue without reservation')
   candidates[0]['enqueued']=True
  else:
   deferred=kind=='observed_after_device_barrier'; candidates=[x for x in pending.get(t,[]) if x['deferred']==deferred and (not deferred or x['enqueued'])]
   if not candidates:raise ValueError('completion without matching reservation/enqueue')
   pending[t].remove(candidates[0]);add(done,row['costs'])
 return {'predebited':pre,'observed':done,'pending_count':sum(len(v) for v in pending.values())}
