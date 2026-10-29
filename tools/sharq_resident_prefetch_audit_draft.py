"""Bounded observer for original block-load entry and selective FFN copies.

Caller owns stream guards, per-call predebit, failure persistence and runtime
scope. This implementation reports logical operations only; no physical bus
traffic or asynchronous completion is inferred.
"""
import types

def need(v,m):
 if not v:raise ValueError(m)

class PrefetchAudit:
 def __init__(self,entries,charge,emit):
  self.entries=entries;self.charge=charge;self.emit=emit;self.saved=[];self.events=[]
 def __enter__(self):
  try:
   for buffer,expert in self.entries:
    need(expert in (0,1),'expert differs');old=buffer.load_state_dict;present='load_state_dict' in buffer.__dict__;previous=buffer.__dict__.get('load_state_dict');self.saved.append((buffer,present,previous));owner=self
    def load(obj,destination,block,adapter_block_index=None,*,_old=old,_expert=expert):
     need(type(block) is int and 0<=block<40 and adapter_block_index is None,'prefetch differs')
     def call():return _old(destination,block,adapter_block_index)
     value=owner.charge(_expert,block,call)
     row={'expert':_expert,'block':block,'event':'block_load_returned','CUDA_completion':'UNKNOWN_UNTIL_LOAD_STREAM_JOIN','physical_bus_bytes':'UNKNOWN'};owner.events.append(row);owner.emit(row);return value
    buffer.load_state_dict=types.MethodType(load,buffer)
   return self
  except BaseException:
   self.__exit__(None,None,None);raise
 def finish(self):
  need(len(self.events)==162,'original load coverage differs');return {'actual_block_load_calls_returned':len(self.events),'physical_bus_bytes':'UNKNOWN'}
 def __exit__(self,typ,value,tb):
  for obj,present,old in reversed(self.saved):
   if present:obj.__dict__['load_state_dict']=old
   else:obj.__dict__.pop('load_state_dict',None)
  self.saved.clear()
