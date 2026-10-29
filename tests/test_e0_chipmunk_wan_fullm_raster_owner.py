import hashlib,json,struct,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_wan_fullm_raster_owner as m
class Tests(unittest.TestCase):
 def test_budget_and_fixed_counts(self):
  x=m.limits();self.assertEqual(x['video_generations'],2);self.assertEqual(x['native_full_model_flows'],8);self.assertEqual(x['capture_D2H'],16);self.assertEqual(x['capture_D2H_bytes'],6096510976);self.assertEqual(x['neutral_state_checks'],10);self.assertEqual(x['capture_file_hash_calls'],16);self.assertLess(x['capture_D2H_bytes']+(1000<<20),m.RES['new_HF_bytes'])
 def test_interrupted_state_raw_hash_preserves_predebit(self):
  record={};events=[]
  with patch.object(m.native,'RUN_IDS',('run',)),patch.object(m.native,'LIMITS',{'state_raw_sha_calls':1,'state_raw_sha_bytes':4}):
   ledger=m.vm.Ledger(record,events.append)
   with self.assertRaisesRegex(OSError,'interrupted'):ledger.call('run',{'state_raw_sha_calls':1,'state_raw_sha_bytes':4},lambda:(_ for _ in ()).throw(OSError('interrupted')))
  self.assertEqual(record['predebited'],{'state_raw_sha_calls':1,'state_raw_sha_bytes':4});self.assertEqual(record['observed'],{});self.assertEqual(events[0]['kind'],'reserved_before_call')
 def test_typed_payload_sha_and_corruption(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'tiny.safetensors';raw=b'\x01\x02\x03\x04';header=json.dumps({'h':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode();data=struct.pack('<Q',len(header))+header+raw;path.write_bytes(data);m.inspect_payload(path,'h',[2],hashlib.sha256(data).hexdigest(),hashlib.sha256(raw).hexdigest())
   with self.assertRaises(ValueError):m.inspect_payload(path,'h',[2],'0'*64,hashlib.sha256(raw).hexdigest())
   with self.assertRaises(ValueError):m.inspect_payload(path,'h',[1],hashlib.sha256(data).hexdigest(),hashlib.sha256(raw).hexdigest())
 def test_bad_or_duplicated_part_rejected(self):
  for x in (-1,2,True):
   with self.assertRaises(ValueError):m.sid(x)
 def test_source_paths_include_new_closed_helper(self):
  for i in (0,1):
   p=m.source_paths(i);self.assertIn(m.SPATIAL.format(i),p);self.assertIn(m.SELF,p);self.assertIn('tools/e0_chipmunk_wan_fullm_raster_capture.py',p)
if __name__=='__main__':unittest.main()
