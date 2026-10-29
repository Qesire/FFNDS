import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_prefetch_audit_draft import PrefetchAudit
class B:
 def load_state_dict(self,d,b,a=None):return (d,b,a)
class T(unittest.TestCase):
 def test_coverage_and_failed_call(self):
  b=B();events=[];charges=[]
  def debit(e,n,f):charges.append((e,n));return f()
  with PrefetchAudit([(b,0)],debit,events.append) as audit:
   for i in range(162):b.load_state_dict({},i%40)
   self.assertEqual(audit.finish()['actual_block_load_calls_returned'],162)
   with self.assertRaises(ValueError):b.load_state_dict({},40)
  self.assertEqual(len(charges),162);self.assertNotIn('load_state_dict',b.__dict__)
 def test_exception_does_not_report_return(self):
  b=B();events=[]
  def debit(*a):raise RuntimeError('inflight')
  with PrefetchAudit([(b,0)],debit,events.append):
   with self.assertRaises(RuntimeError):b.load_state_dict({},0)
  self.assertEqual(events,[])
if __name__=='__main__':unittest.main()
