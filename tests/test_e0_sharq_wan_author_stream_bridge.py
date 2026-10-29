import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_author_stream_bridge as b
class Tests(unittest.TestCase):
 def test_both_dependencies_and_lifetime_are_ordered(self):
  trace=[]
  class Stream:
   def __init__(self,name):self.name=name
   def wait_event(self,e):trace.append((self.name,'wait',e.name))
  class Event:
   count=0
   def __init__(self):self.name=Event.count;Event.count+=1
   def record(self,s):trace.append((s.name,'record',self.name))
  class Context:
   def __enter__(self):trace.append('enter-default')
   def __exit__(self,*a):trace.append('exit-default')
  class T:
   device=SimpleNamespace(type='cuda')
   def __init__(self,name):self.name=name
   def record_stream(self,s):trace.append((self.name,'lifetime',s.name))
  current,default=Stream('compute'),Stream('default');torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,T),cuda=SimpleNamespace(current_stream=lambda:current,default_stream=lambda:default,Event=Event,stream=lambda x:Context()));x,y=T('x'),T('y')
  def operation(z):trace.append('author');return {'out':y}
  self.assertEqual(b.author_call(torch,operation,x),{'out':y});self.assertEqual(trace,[('compute','record',0),('default','wait',0),('x','lifetime','default'),'enter-default','author',('default','record',1),'exit-default',('compute','wait',1),('y','lifetime','compute')])
 def test_wait_and_output_record_failure_drain_both_streams(self):
  for stage in ('wait','record'):
   calls=[]
   class S:
    def __init__(self,name):self.name=name
    def wait_event(self,e):
     if stage=='wait' and self.name=='caller':raise RuntimeError('injected wait')
    def synchronize(self):calls.append(self.name)
   class E:
    def record(self,s):pass
   class C:
    def __enter__(self):pass
    def __exit__(self,*a):pass
   class T:
    device=SimpleNamespace(type='cuda')
    def record_stream(self,s):
     if stage=='record' and s.name=='caller':raise RuntimeError('injected record')
   caller,default=S('caller'),S('default');torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,T),cuda=SimpleNamespace(current_stream=lambda:caller,default_stream=lambda:default,Event=E,stream=lambda _:C()))
   with self.assertRaisesRegex(RuntimeError,'injected'):b.author_call(torch,lambda:T())
   self.assertEqual(calls,['default','caller'])
 def test_exception_is_propagated(self):
  class E:
   def record(self,s):pass
  class S:
   def wait_event(self,e):pass
   def synchronize(self):calls.append('drain')
  class C:
   def __enter__(self):pass
   def __exit__(self,*a):pass
  calls=[];torch=SimpleNamespace(is_tensor=lambda x:False,cuda=SimpleNamespace(current_stream=S,default_stream=S,Event=E,stream=lambda _:C()))
  with self.assertRaisesRegex(RuntimeError,'injected'):b.author_call(torch,lambda:(_ for _ in ()).throw(RuntimeError('injected')))
  self.assertEqual(calls,['drain','drain'])
if __name__=='__main__':unittest.main()
