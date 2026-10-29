import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import sharq_resident_dense32_runtime_draft as m
from tests.test_e0_sharq_wan_all160_runtime import build

class Tests(unittest.TestCase):
 def test_phase_requires_full_cache_and_dense32(self):
  pipe,active,_,quant,ledger,_=build();guard=type('Guard',(),{'torch':None,'active':active})();obj=m.ResidentDense32Runtime(pipe,guard,None,quant,ledger,sync=lambda:None,resolve=lambda n,*a:n,charge_load=lambda k,f:f(),original_operand=lambda *a:{},boundary_control=lambda *a:None)
  with self.assertRaises(ValueError):obj.set_next_resident(True)
  obj.cache.update({(e,b,s):(1,2,3,4) for e in (0,1) for b in range(40) for s in ('ffn.0','ffn.2')})
  obj.set_next_resident(True)
  with self.assertRaises(ValueError):obj.begin('test','bf16')
  obj.begin('test','dense32');self.assertTrue(obj._selector.enabled);self.assertFalse(obj._resident_next)
  obj.rid=None;obj._selector.enabled=False
 def test_first_expert_boundary_control_before_return(self):
  from unittest.mock import patch
  pipe,active,_,quant,ledger,_=build();guard=type('Guard',(),{'torch':None,'active':active})();events=[]
  obj=m.ResidentDense32Runtime(pipe,guard,None,quant,ledger,sync=lambda:None,resolve=lambda n,*a:n,charge_load=lambda *a:None,original_operand=lambda *a:{},boundary_control=lambda *a:events.append(a[-1]))
  obj.mode='dense32';active.s=0;active.block=(0,0)
  with patch.object(m.original.SameMasterFFNRuntime,'_apply_fp4',return_value='output'):
   self.assertEqual(obj._apply_fp4(object(),object(),0,'ffn.0'),'output');self.assertEqual(events,[(0,0,'ffn.0')])
   with self.assertRaises(ValueError):obj._apply_fp4(object(),object(),0,'ffn.0')
 def test_guard_used_before_cache_creation(self):
  pipe,active,_,quant,ledger,_=build();guard=type('Guard',(),{'torch':None,'active':active})();obj=m.ResidentDense32Runtime(pipe,guard,None,quant,ledger,sync=lambda:None,resolve=lambda n,*a:n,charge_load=lambda k,f:f(),original_operand=lambda *a:{'original':True},boundary_control=lambda *a:None)
  obj.mode='bf16_cache';self.assertEqual(obj._check_operand(None,None,None),{'original':True})
if __name__=='__main__':unittest.main()
