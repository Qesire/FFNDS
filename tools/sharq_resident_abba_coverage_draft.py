"""Enforce and persist actual issued FFN prefetch coverage for ABBA.

No authorizations, CUDA or tensor reads. Calls and expected coverage are
logical enqueue observations. The caller owns all durable worker semantics.
"""
ARMS=('warmup','legacy_A1','resident_B1','resident_B2','legacy_A2')
def need(x,m):
 if not x:raise ValueError(m)
class Coverage:
 def __init__(self,persist):self.persist=persist;self.arm=None;self.rows=[]
 def begin(self,arm):
  need(self.arm is None and arm in ARMS and arm not in {r['arm'] for r in self.rows},'phase coverage begin')
  self.arm={'arm':arm,'block_loads':0,'weight_issued':0,'weight_skipped':0,'bias_issued':0,'weight_bytes':0,'skipped_weight_bytes':0,'bias_bytes':0};self.persist(self.snapshot())
 def event(self,event):
  need(self.arm is not None,'phase not active');a=self.arm
  if event.get('event')=='block_load_returned':a['block_loads']+=1
  elif event.get('weight_copy') in ('ISSUED','SKIPPED'):
   skip=event['weight_copy']=='SKIPPED';need(event['bias_copy']=='ISSUED','bias copy absent')
   a['weight_skipped' if skip else 'weight_issued']+=1;a['bias_issued']+=1
   a['skipped_weight_bytes' if skip else 'weight_bytes']+=event['skipped_weight_logical_bytes' if skip else 'requested_weight_logical_bytes'];a['bias_bytes']+=event['requested_bias_logical_bytes']
  else:return
  need(a['block_loads']<=162 and a['weight_issued']+a['weight_skipped']<=324 and a['bias_issued']<=324,'coverage exceeds bound');self.persist(self.snapshot())
 def finish(self):
  a=self.arm;need(a is not None,'phase not active');resident=a['arm'].startswith('resident');need(a['block_loads']==162 and a['bias_issued']==324 and a['weight_skipped']==(324 if resident else 0) and a['weight_issued']==(0 if resident else 324),'actual enqueue coverage incomplete');self.rows.append(a);self.arm=None;self.persist(self.snapshot())
 def snapshot(self):return {'completed_phases':list(self.rows),'current_phase':dict(self.arm) if self.arm else None,'scope':'CPU enqueue counts and logical tensor bytes; GPU completion and physical bus UNKNOWN until separately measured'}
