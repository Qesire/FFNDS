import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_wan_fullm_raster_capture as m
class Tests(unittest.TestCase):
 def test_geometry_and_cost(self):
  self.assertEqual(m.M,31668);self.assertEqual(m.coordinate(31667),(20,28,51));self.assertEqual(m.coordinate(1508),(1,0,0));self.assertEqual(m.PAYLOAD_BYTES_PER_STEP,1524127744);self.assertLess(max(m.OPERAND_BYTES.values()),1<<30)
  for x in (-1,31668,True):
   with self.assertRaises(ValueError):m.coordinate(x)
 def test_real_identity_and_order_preserved(self):
  seen=[];tx=m.CaptureTransaction(lambda name,v,*s:(name,v),lambda p,g:seen.append((p,g)))
  for step in range(4):
   gate=object();expert=step//2;tx.begin(step,20,expert,gate)
   for name,shape in m.OPERANDS.items():tx.operand(name,object(),shape,'bfloat16',(shape[1],1))
   y=tx.pending['Y_object'];tx.finish(step,20,expert,gate,y,m.GRID)
  tx.verify_complete();self.assertEqual(len(seen),4);self.assertEqual(seen[2][0]['expert'],1)
 def test_reject_bad_shapes_and_wrong_writeback(self):
  t=m.CaptureTransaction(lambda *v:v,lambda *v:None);gate=object();t.begin(0,20,0,gate)
  with self.assertRaises(ValueError):t.operand('V1',object(),(32,5120),'bfloat16',(5120,1))
  for name,shape in m.OPERANDS.items():t.operand(name,object(),shape,'bfloat16',(shape[1],1))
  with self.assertRaises(ValueError):t.finish(0,20,0,gate,object(),m.GRID)
 def test_mm_hook_preserves_original_values_and_capture_order(self):
  class T:
   dtype='torch.bfloat16'
   def __init__(self,shape):self.shape=shape
   def stride(self):return (self.shape[1],1)
  class Base:
   def __init__(self,*a,**k):self.mapping={};self.block=(0,20)
   def mm_call(self,mm,value,meta,original):return original(value)
  from types import SimpleNamespace
  tapcls=m.make_tap_class(SimpleNamespace(LightTap=Base),None);order=[];tx=m.CaptureTransaction(lambda n,v,*a:order.append(n) or v,lambda *a:None);gate=object();tx.begin(0,20,0,gate);tap=tapcls(capture=tx);w1,w2=object(),object();tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};x=T(m.OPERANDS['V1']);h=T(m.OPERANDS['h']);y=T(m.OPERANDS['Y']);self.assertIs(tap.mm_call(w1,x,{},lambda z:z),x);self.assertIs(tap.mm_call(w2,h,{},lambda z:y),y);self.assertEqual(order,['gate','V1','h','Y']);tx.finish(0,20,0,gate,y,m.GRID)
 def test_typed_header(self):
  h=m.header(898,3,1,m.GRID,{k:{} for k in ('V1','h','Y','gate')});self.assertEqual(h['expert'],'low');self.assertEqual(h['row_count'],31668)
  with self.assertRaises(ValueError):m.header(96,3,0,m.GRID,{k:{} for k in ('V1','h','Y','gate')})
if __name__=='__main__':unittest.main()
