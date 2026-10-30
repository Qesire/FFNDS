import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from chipmunk_p64_deferred_ledger import DeferredLedger
class L:
 def __init__(self):self.record={'predebited':{},'observed':{}};self.events=[]
 def journal(self,e):self.events.append(e)
class Tests(unittest.TestCase):
 def test_complete_only_after_barrier(self):
  l=L();d=DeferredLedger(l,{'x':2},'s',lambda:None);v=object();self.assertIs(d.call('r',{'x':1},lambda:v,actual_stream='s'),v);self.assertEqual(l.record['observed'],{});d.barrier();self.assertEqual(l.record['observed'],{'x':1});self.assertEqual(d.pending,[])
 def test_failure_keeps_reserved_unknown(self):
  l=L();d=DeferredLedger(l,{'x':1},'s',lambda:(_ for _ in ()).throw(RuntimeError('barrier')));d.call('r',{'x':1},lambda:object(),actual_stream='s')
  with self.assertRaisesRegex(RuntimeError,'barrier'):d.barrier()
  self.assertEqual(l.record['observed'],{});self.assertEqual(d.pending[0]['state'],'ENQUEUED_UNKNOWN')
 def test_budget_and_foreign_stream_reject_before_dispatch(self):
  l=L();d=DeferredLedger(l,{'x':1},'s',lambda:None)
  with self.assertRaises(RuntimeError):d.call('r',{'x':1},lambda:self.fail(),actual_stream='other')
  with self.assertRaises(ValueError):d.call('r',{'x':2},lambda:self.fail(),actual_stream='s')
  self.assertEqual(l.record['predebited'],{})
 def test_enqueue_failure_stays_unknown(self):
  l=L();d=DeferredLedger(l,{'x':1},'s',lambda:None)
  with self.assertRaises(ZeroDivisionError):d.call('r',{'x':1},lambda:1/0,actual_stream='s')
  self.assertEqual(d.pending[0]['state'],'RESERVED_UNKNOWN')
  with self.assertRaises(RuntimeError):d.barrier()
 def test_initial_journal_failure_poisoned(self):
  l=L();l.journal=lambda _:(_ for _ in ()).throw(OSError('journal'));d=DeferredLedger(l,{'x':2},'s',lambda:None)
  with self.assertRaises(OSError):d.call('r',{'x':1},lambda:self.fail(),actual_stream='s')
  self.assertTrue(d.poisoned);self.assertEqual(d.pending[0]['state'],'RESERVED_UNKNOWN')
  with self.assertRaises(RuntimeError):d.call('r',{'x':1},lambda:self.fail(),actual_stream='s')
 def test_callable_capture_is_retained_only_until_barrier(self):
  import weakref,gc
  class Item:pass
  l=L();d=DeferredLedger(l,{'x':1},'s',lambda:None);item=Item();ref=weakref.ref(item)
  def make(value):return lambda:value
  fn=make(item);d.call('r',{'x':1},fn,actual_stream='s');del item,fn;gc.collect();self.assertIsNotNone(ref());d.barrier();gc.collect();self.assertIsNone(ref())
 def test_replay_deferred_and_regular_and_reject_fake_completion(self):
  from chipmunk_p64_deferred_ledger import replay
  l=L();d=DeferredLedger(l,{'x':1},'s',lambda:None);d.call('r',{'x':1},lambda:None,actual_stream='s');self.assertEqual(replay(l.events)['pending_count'],1);d.barrier();self.assertEqual(replay(l.events),{'predebited':{'x':1},'observed':{'x':1},'pending_count':0})
  with self.assertRaises(ValueError):replay([{'kind':'observed_after_device_barrier','run_id':'r','costs':{'x':1}}])
if __name__=='__main__':unittest.main()
