import importlib.util,json,struct,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_asset_extract.py';s=importlib.util.spec_from_file_location('extract',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Ledger:
 def __init__(self):self.events=[]
 def reserve(self,k,n):self.events.append(('reserve',k,n))
 def complete(self,k,n):self.events.append(('complete',k,n))
class Tests(unittest.TestCase):
 def test_typed_range_predebit_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'sample';h={'w':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]}};raw=json.dumps(h).encode();f.write_bytes(struct.pack('<Q',len(raw))+raw+b'abcdefgh');header,n,state=m.read_header(f);ledger=Ledger();data,sha=m.checked_range(f,header,n,state,'w','BF16',[2,2],ledger)
   self.assertEqual(data,b'abcdefgh');self.assertEqual(len(sha),64);self.assertEqual([x[0] for x in ledger.events],['reserve','reserve','complete','complete'])
 def test_wrong_type_or_changed_master_fail(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'sample';h={'w':{'dtype':'U8','shape':[2],'data_offsets':[0,2]}};raw=json.dumps(h).encode();f.write_bytes(struct.pack('<Q',len(raw))+raw+b'ab');header,n,state=m.read_header(f)
   with self.assertRaises(ValueError):m.checked_range(f,header,n,state,'w','BF16',[2],Ledger())
   f.write_bytes(f.read_bytes()+b'!');ledger=Ledger()
   with self.assertRaises(ValueError):m.checked_range(f,header,n,state,'w','U8',[2],ledger)
   self.assertEqual([x[0] for x in ledger.events],['reserve','reserve'])
if __name__=='__main__':unittest.main()
