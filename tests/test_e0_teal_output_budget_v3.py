import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('budget_test',ROOT/'tools/e0_teal_output_budget_v3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_nfs_unlink_between_walk_and_lstat_is_counted(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True);p=root/'protocol';p.write_bytes(b'protocol');(snap/'protocol.json').write_bytes(b'protocol');transient=out/'.nfs0000002100a9dbee00045983';transient.write_bytes(b'old');(out/'live').write_bytes(b'12345')
   b=m.SnapshotBudget(out,{'repo_sources':[]},p,100);b.initialize();original=Path.lstat
   def vanish(path):
    if path==transient:path.unlink()
    return original(path)
   with patch.object(Path,'lstat',vanish):value=b.total()
   self.assertEqual(value['mutable_bytes'],5);self.assertEqual(value['vanished_mutable_entries'],1);self.assertEqual(value['enforcement'],'SOFT_SAMPLED_NOT_FILESYSTEM_QUOTA');b.final()
 def test_snapshot_missing_and_unrelated_errors_fail(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True);p=root/'p';p.write_bytes(b'p');(snap/'protocol.json').write_bytes(b'p');live=out/'live';live.write_bytes(b'x');b=m.SnapshotBudget(out,{'repo_sources':[]},p,100);b.initialize();original=Path.lstat
   def denied(path):
    if path==live:raise PermissionError('denied')
    return original(path)
   with patch.object(Path,'lstat',denied):
    with self.assertRaises(PermissionError):b.total()
   (snap/'protocol.json').unlink()
   with self.assertRaises(ValueError):b.final()
 def test_over_limit_and_presnapshot_disappearance(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';out.mkdir();p=root/'p';p.write_bytes(b'p');(out/'large').write_bytes(b'123');b=m.SnapshotBudget(out,{'repo_sources':[]},p,2)
   with self.assertRaises(ValueError):b.total_before_snapshot()
if __name__=='__main__':unittest.main()
