import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_assets.py';s=importlib.util.spec_from_file_location('asset_producer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_budget_predebit_and_interruption(self):
  r={'predebited':{},'completed':{}};events=[];l=m.Ledger(r,lambda:events.append(dict(r['predebited'])))
  l.reserve('tensor_bytes',3);self.assertEqual(events,[{'tensor_bytes':3}]);self.assertNotIn('tensor_bytes',r['completed']);l.complete('tensor_bytes',3);self.assertEqual(r['completed']['tensor_bytes'],3)
  with self.assertRaises(Exception):l.reserve('tensor_bytes',m.RES['max_tensor_bytes'])
 def test_no_scope_wrong_protocol(self):
  with self.assertRaises(Exception):m.validate({'id':m.ID})
 def test_exit_zero_is_required(self):
  with self.assertRaises(Exception):m.verify_parent({'id':'x'},{'id':'x','process_exited':True,'child_returncode':1},{'sha256':'a'})
 def test_overlapping_predebited_range_interrupted_at_each_boundary(self):
  import struct
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'small';h={'w':{'dtype':'U8','shape':[2],'data_offsets':[0,2]}};raw=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+b'ab');header,n,state=m.extract.read_header(path)
   for stop in (1,2,3,4):
    report={'predebited':{},'completed':{}};calls=[]
    def persist():
     calls.append(1)
     if len(calls)==stop:raise RuntimeError('synthetic interruption')
    ledger=m.Ledger(report,persist)
    with self.assertRaises(RuntimeError):m.extract.checked_range(path,header,n,state,'w','U8',[2],ledger)
    self.assertEqual(report['inflight']['tensor_bytes']['progress'],'UNKNOWN_UNTIL_COMPLETE' if stop<=2 else 'COMPLETE')
    if stop>=2:self.assertEqual(report['inflight']['tensor_ranges']['progress'],'COMPLETE' if stop==4 else 'UNKNOWN_UNTIL_COMPLETE')
 def test_tiny_two_expert_actual_staging_and_bundle_roundtrip(self):
  import torch
  from safetensors.torch import save_file,load_file
  def digest(t):return hashlib.sha256(memoryview(t.reshape(-1).view(torch.uint8).numpy())).hexdigest()
  def bind(path,root):return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=m.HF;m.HF=root;bundle={};r={'predebited':{},'completed':{}};ledger=m.Ledger(r,lambda:None)
   try:
    for expert,step in (('high',1),('low',2)):
     pkt=root/(expert+'_packet.safetensors');v={'V1':torch.arange(32,dtype=torch.bfloat16).reshape(2,16),'a':torch.ones((2,32),dtype=torch.bfloat16),'h':torch.ones((2,32),dtype=torch.bfloat16),'Y':torch.ones((2,16),dtype=torch.bfloat16),'gate':torch.ones(16,dtype=torch.bfloat16),'rows':torch.tensor([5,7],dtype=torch.int64)};save_file(v,str(pkt));tm={k:{'shape':list(t.shape),'dtype':str(t.dtype).removeprefix('torch.'),'raw_tensor_sha256':digest(t)} for k,t in v.items()};meta={'source_ordinal':911,'mode':'dense','source_state':'teacher_input','step':step,'block':20,'expert':expert,'grid':[21,29,52],'population_rows':31668,'payload':bind(pkt,root),'tensor_metadata':tm,'rows':[5,7]};meta_path=root/(expert+'_metadata.json');meta_path.write_text(json.dumps(meta))
     raw=root/(expert+'_raw.safetensors');nv=root/(expert+'_nv.safetensors');rw={};nw={}
     for number,n,k in ((0,32,16),(2,16,32)):
      prefix=f'blocks.20.ffn.{number}.';rw[prefix+'weight']=torch.ones((n,k),dtype=torch.bfloat16);rw[prefix+'bias']=torch.ones(n,dtype=torch.bfloat16);nw[prefix+'weight']=torch.ones((n,k//2),dtype=torch.uint8);nw[prefix+'weight_scale']=torch.ones((n,k//16),dtype=torch.float8_e4m3fn);nw[prefix+'bias']=torch.ones(n,dtype=torch.bfloat16);nw[prefix+'alpha']=torch.tensor(1.0);nw[prefix+'input_global_scale']=torch.tensor(1.0)
     save_file(rw,str(raw));save_file(nw,str(nv));sel={'step':step,'packet':{'payload':meta['payload'],'metadata':bind(meta_path,root),'tensor_metadata':tm},'original_bf16':{'path':str(raw)},'nvfp4_checkpoint':{'cache_path':str(nv.relative_to(root))}};states=(m.extract.stat_id(raw),m.extract.stat_id(nv),m.extract.header_digest(raw),m.extract.header_digest(nv));values,records=m.tensor_rows(expert,sel,ledger,torch,states,dims=(2,16,32));self.assertEqual(len(records),20);bundle.update(values)
    dest=root/'bundle.safetensors';save_file(bundle,str(dest));got=load_file(str(dest),device='cpu');self.assertEqual(set(got),set(bundle))
    for key in got:self.assertEqual(got[key].dtype,bundle[key].dtype);self.assertEqual(list(got[key].shape),list(bundle[key].shape));self.assertEqual(digest(got[key]),digest(bundle[key]))
    self.assertEqual(r['predebited'],r['completed'])
    sel['packet']['tensor_metadata']['gate']['raw_tensor_sha256']='0'*64
    with self.assertRaises(ValueError):m.tensor_rows('low',sel,ledger,torch,states,dims=(2,16,32))
    sel['packet']['tensor_metadata']['gate']['raw_tensor_sha256']=digest(v['gate'])
    nw['blocks.20.ffn.0.weight']=torch.ones((1,1),dtype=torch.uint8);save_file(nw,str(nv));badstates=(m.extract.stat_id(raw),m.extract.stat_id(nv),m.extract.header_digest(raw),m.extract.header_digest(nv))
    with self.assertRaises(ValueError):m.tensor_rows('low',sel,ledger,torch,badstates,dims=(2,16,32))
    oldstat=states;raw.write_bytes(raw.read_bytes()+b'!')
    with self.assertRaises(ValueError):m.tensor_rows('low',sel,ledger,torch,oldstat,dims=(2,16,32))
   finally:m.HF=old
if __name__=='__main__':unittest.main()
