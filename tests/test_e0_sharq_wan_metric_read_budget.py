import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_metric_read_budget as m
class Tests(unittest.TestCase):
 def test_predebit_completion_and_failure(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'f';p.write_bytes(b'abc');journal=Path(d)/'log';old=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest();base=SimpleNamespace(sha=old)
   with m.HashReads(base,journal,'CPU'):
    self.assertEqual(base.sha(p),old(p))
   self.assertIs(base.sha,old);rows=[json.loads(x) for x in journal.read_text().splitlines()];self.assertEqual([r['event'] for r in rows],['PREDEBIT','COMPLETE','PHASE_END']);self.assertEqual(rows[-1]['completed_bytes'],3)
 def test_failure_retains_predebit_and_restores(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'f';p.write_bytes(b'abc');journal=Path(d)/'log'
   old=lambda p:'unused';base=SimpleNamespace(sha=old);original=Path.open
   def broken(path,*a,**k):
    if path==p:raise OSError('interrupted')
    return original(path,*a,**k)
   with patch.object(Path,'open',broken):
    with self.assertRaises(OSError):
     with m.HashReads(base,journal,'GPU'):base.sha(p)
   self.assertIs(base.sha,old);rows=[json.loads(x) for x in journal.read_text().splitlines()];self.assertEqual(rows[-1]['inflight'],1);self.assertEqual(rows[-1]['completed_bytes'],0)
 def test_worker_extra_output_budget_is_separate_and_enforced(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'f';p.write_bytes(b'abcd');base=SimpleNamespace(sha=lambda _:None)
   with patch.object(m,'MAX_WORKER_EXTRA_BYTES',3):
    with self.assertRaisesRegex(ValueError,'worker extra'):
     with m.HashReads(base,Path(d)/'log','CPU'):base.sha(p)
 def test_rejects_over_budget_before_read(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'f';p.write_bytes(b'abcd');base=SimpleNamespace(sha=lambda x:self.fail('not invoked'))
   with patch.object(m,'MAX_BYTES',3):
    with self.assertRaisesRegex(ValueError,'budget'):
     with m.HashReads(base,Path(d)/'log','CPU'):base.sha(p)
if __name__=='__main__':unittest.main()
