import pathlib,sys,unittest,tempfile,json,struct,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_artifact_verifier_draft import state_identity,byte_equal
class T(unittest.TestCase):
 def test_single_pass_and_bad_raw(self):
  with tempfile.TemporaryDirectory() as x:
   p=pathlib.Path(x)/'x';b=pathlib.Path(x)/'b';raw=b'\x01\x02';header=json.dumps({'latents':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}}).encode();data=struct.pack('<Q',len(header))+header+raw;p.write_bytes(data);b.write_bytes(data)
   info={'name':'initial_noise','dtype':'bfloat16','shape':[1],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'raw_tensor_sha256':hashlib.sha256(raw).hexdigest()}
   self.assertEqual(state_identity(p,info),len(data));self.assertTrue(byte_equal(p,b,len(data)))
   with self.assertRaises(ValueError):state_identity(p,{**info,'raw_tensor_sha256':'0'*64})
   b.write_bytes(data[:-1]+b'\x09');self.assertFalse(byte_equal(p,b,len(data)))
if __name__=='__main__':unittest.main()
