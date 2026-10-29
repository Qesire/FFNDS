import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_rtlynx_wan_ffn_probe as m
class Tests(unittest.TestCase):
 def test_scope_and_call_budget(self):
  self.assertEqual(m.DESIGN['M'],[32,31668]);self.assertEqual(m.LIMITS['W1_bf16_addmm'],60);self.assertEqual(m.LIMITS['W2_bf16_addmm'],20);self.assertEqual(m.LIMITS['cutlass_sparse_W2'],40);self.assertEqual(m.LIMITS['ffn_event_pair'],60);self.assertEqual(m.LIMITS['reference_dense_W2'],8)
 def test_interruption_retains_predebit_and_never_completes(self):
  states=[];r={'predebited':{},'invoked':{},'returned':{},'inflight':None};l=m.Ledger(r,lambda:states.append((dict(r['predebited']),r['inflight'])))
  l.reserve({'W1_bf16_addmm':1});
  with self.assertRaises(RuntimeError):l.call('W1_bf16_addmm',lambda:(_ for _ in ()).throw(RuntimeError('injected')))
  self.assertEqual(r['predebited']['W1_bf16_addmm'],1);self.assertEqual(r['invoked']['W1_bf16_addmm'],1);self.assertNotIn('W1_bf16_addmm',r['returned']);self.assertIsNotNone(r['inflight'])
 def test_extra_sync_failure_keeps_completion_unknown(self):
  r={'predebited':{},'invoked':{},'returned':{},'inflight':None};l=m.Ledger(r,lambda:None,lambda:(_ for _ in ()).throw(RuntimeError('sync failed')))
  with self.assertRaisesRegex(RuntimeError,'sync failed'):l.extra('reference_dense_W2',lambda:'enqueued')
  self.assertEqual(r['invoked']['reference_dense_W2'],1);self.assertNotIn('reference_dense_W2',r['returned']);self.assertIsNotNone(r['inflight'])
 def test_reference_observer_failure_keeps_predebit(self):
  r={'predebited':{},'invoked':{},'returned':{},'inflight':None};l=m.Ledger(r,lambda:None,lambda:None)
  def dense_and_observe():raise RuntimeError('reference observer failed')
  with self.assertRaisesRegex(RuntimeError,'observer failed'):l.extra('reference_dense_W2',dense_and_observe)
  self.assertNotIn('reference_dense_W2',r['returned']);self.assertIsNotNone(r['inflight'])
 def test_overbudget_fails_before_dispatch(self):
  r={'predebited':{'reference_mask':m.LIMITS['reference_mask']},'invoked':{},'returned':{},'inflight':None};l=m.Ledger(r,lambda:None)
  with self.assertRaises(ValueError):l.reserve({'reference_mask':1})
 def test_rejects_unreviewed_config(self):
  with self.assertRaises((ValueError,KeyError)):m.validate({'id':'fake'})
if __name__=='__main__':unittest.main()
