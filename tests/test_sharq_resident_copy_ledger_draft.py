import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_copy_ledger_draft import CopyLedger
class T(unittest.TestCase):
 def test_predebit_and_exception(self):
  records=[];l=CopyLedger(1,records.append)
  def fail():raise RuntimeError('copy failed')
  with self.assertRaises(RuntimeError):l.call((0,0,'ffn.0'),8,2,True,fail)
  self.assertEqual(l.reserved,1);self.assertEqual(l.returned,0);self.assertEqual(records[-1]['inflight']['completion'],'UNKNOWN')
  with self.assertRaises(ValueError):l.call((0,0,'ffn.0'),8,2,True,lambda:None)
 def test_return_means_enqueued_not_gpu_complete(self):
  l=CopyLedger(1,lambda x:None);l.call((0,0,'ffn.2'),8,2,False,lambda:None)
  self.assertEqual(l.records[0]['GPU_copy_completion'],'UNKNOWN_UNTIL_LOAD_STREAM_JOIN')
if __name__=='__main__':unittest.main()
