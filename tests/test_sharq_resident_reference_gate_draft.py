import pathlib,sys,unittest,tempfile,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_reference_gate_draft import ReferenceGate
class T(unittest.TestCase):
 def test_compare_success_budget_and_failure(self):
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t);(p/'a').write_bytes(b'x');(p/'b').write_bytes(b'x');sha=hashlib.sha256(b'x').hexdigest();binding=lambda n:{'path':n,'bytes':1,'sha256':sha};snap=[]
   g=ReferenceGate(p,{'bf16':{'video':binding('a')},'dense32':{'video':binding('a')}},lambda n,fn:fn(),snap.append,max_compare_bytes=2)
   g.check('warmup','video',binding('b'));self.assertEqual(g.completed,2);self.assertTrue(g.checks[0]['byte_equal'])
   with self.assertRaises(ValueError):g.check('legacy_A1','video',binding('b'))
 def test_alias_rejected(self):
  import os
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t);(p/'a').write_bytes(b'x');os.link(p/'a',p/'b');sha=hashlib.sha256(b'x').hexdigest();old={'path':'a','bytes':1,'sha256':sha};new={**old,'path':'b'}
   g=ReferenceGate(p,{'bf16':{'video':old}},lambda n,fn:fn(),lambda x:None)
   with self.assertRaises(ValueError):g.check('warmup','video',new)
 def test_failure_keeps_inflight(self):
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t);(p/'a').write_bytes(b'x');(p/'b').write_bytes(b'x');sha=hashlib.sha256(b'x').hexdigest();b={'path':'a','bytes':1,'sha256':sha};new={**b,'path':'b'};snap=[]
   def fail(n,f):raise OSError('interrupted')
   g=ReferenceGate(p,{'bf16':{'video':b}},fail,snap.append)
   with self.assertRaises(OSError):g.check('warmup','video',new)
   self.assertEqual(g.inflight['completion'],'UNKNOWN');self.assertEqual(g.completed,0)
if __name__=='__main__':unittest.main()
