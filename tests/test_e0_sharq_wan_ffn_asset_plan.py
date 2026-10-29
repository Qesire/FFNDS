import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_asset_plan.py';s=importlib.util.spec_from_file_location('ap',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def examples():
 packets=[{'run_id':'whole_ffn_bridge_ord911_seed42_dense','step':i,'block':20,'payload':{},'metadata':{},'tensor_metadata':{k:{} for k in ('V1','a','h','Y','gate','rows')}} for i in (1,2)]
 return {'status':'COMPLETE_NATIVE_WHOLE_FFN_BOUNDARY_BRIDGE','closed_verification':{'status':'PASS_CLOSED_NATIVE_BOUNDARY_BRIDGE'},'packets':packets},{'models':[{'kind':'high'},{'kind':'low'}]},{'files':[{'filename':'A_high.safetensors'},{'filename':'A_low.safetensors'}]}
class Tests(unittest.TestCase):
 def test_exact_paired_gate_and_expert(self):
  b,r,n=examples();v=m.select(b,r,n);self.assertEqual(v['high']['step'],1);self.assertEqual(v['low']['step'],2);self.assertEqual(v['high']['rows'],32)
 def test_missing_or_wrong_gate_fails(self):
  b,r,n=examples();del b['packets'][0]['tensor_metadata']['gate']
  with self.assertRaises(ValueError):m.select(b,r,n)
 def test_unclosed_or_ambiguous_parent_fails(self):
  b,r,n=examples();b['closed_verification']['status']='FAILED'
  with self.assertRaises(ValueError):m.select(b,r,n)
  b,r,n=examples();b['packets'].append(b['packets'][0])
  with self.assertRaises(ValueError):m.select(b,r,n)
if __name__=='__main__':unittest.main()
