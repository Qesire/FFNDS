import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_p64_sync_abba_tap as tap
class Tests(unittest.TestCase):
 def test_only_candidate_actions_deferred(self):
  self.assertEqual(tap.DEFER_KEYS,{'weight_materializations','p64_score','p64_select','selected_old_h','packed_action'})
  self.assertFalse(tap.DEFER_KEYS & {'mask_sha_calls','cache_x_clone','cache_shadow','cache_y_clone','cache_release_sync','expert_boundary_cleanup'})
if __name__=='__main__':unittest.main()

class FakeLedger:
 def __init__(self): self.record={'predebited':{},'observed':{}};self.events=[]
 def journal(self,x):self.events.append(x)
 def call(self,rid,cost,fn):
  for k,n in cost.items():self.record['predebited'][k]=self.record['predebited'].get(k,0)+n
  value=fn()
  for k,n in cost.items():self.record['observed'][k]=self.record['observed'].get(k,0)+n
  return value
class Integration(unittest.TestCase):
 def test_candidate_single_block_release_barrier(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  class Base:
   def __init__(self,guard):self.guard=guard;self.rid='r'
   def charged(self,cost,fn):return self.guard.ledger.call(self.rid,cost,fn)
   def finish(self):return 'done'
  calls=[];ledger=FakeLedger();cuda=SimpleNamespace(current_stream=lambda:'stream',synchronize=lambda:calls.append('sync'));guard=SimpleNamespace(ledger=ledger,torch=SimpleNamespace(cuda=cuda));vm=SimpleNamespace(LIMITS={'p64_select':1,'cache_release_sync':1,'deferred_completion_barriers':1})
  with patch.object(tap,'original_tap',return_value=Base):
   cls=tap.make_tap(vm,SimpleNamespace(LIMITS=vm.LIMITS,RUN_IDS=('r',)),mode='high_x_recompute_p64',rows=None,kernels=None,emit=None,deferred=True,geometry=(1,1,1,1));x=cls(guard);value=object();self.assertIs(x.charged({'p64_select':1},lambda:value),value);self.assertEqual(ledger.record['observed'],{});x.charged({'cache_release_sync':1},lambda:None);self.assertEqual(ledger.record['observed'],{'p64_select':1,'cache_release_sync':1,'deferred_completion_barriers':1});self.assertIsNone(x.deferred_transaction);self.assertEqual(calls,['sync']);self.assertEqual(x.finish(),'done')
