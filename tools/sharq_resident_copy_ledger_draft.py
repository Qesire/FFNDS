"""Predebit/return journal adapter for FFN copy choices; GPU completion unknown."""

def need(v,m):
 if not v:raise ValueError(m)
class CopyLedger:
 def __init__(self,limit,persist):self.limit=limit;self.persist=persist;self.reserved=0;self.returned=0;self.inflight=None;self.records=[]
 def call(self,key,weight_bytes,bias_bytes,skip,fn):
  need(self.inflight is None and self.reserved<self.limit and type(skip) is bool and weight_bytes>0 and bias_bytes>0,'copy reservation')
  self.reserved+=1;self.inflight={'key':key,'skip':skip,'weight_bytes':weight_bytes,'bias_bytes':bias_bytes,'completion':'UNKNOWN'};self.persist(self.snapshot())
  result=fn()
  self.returned+=1;self.records.append({'key':key,'skip':skip,'weight_bytes':weight_bytes,'bias_bytes':bias_bytes,'GPU_copy_completion':'UNKNOWN_UNTIL_LOAD_STREAM_JOIN'});self.inflight=None;self.persist(self.snapshot());return result
 def snapshot(self):return {'reserved':self.reserved,'returned':self.returned,'inflight':self.inflight,'last_returned':self.records[-1] if self.records else None,'returned_records':len(self.records),'detail_policy':'per-call events must be appended by the owner separately'}
