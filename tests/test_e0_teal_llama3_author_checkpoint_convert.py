import importlib.util, json, pathlib, tempfile, unittest, os, fcntl, errno
from unittest import mock
P=pathlib.Path(__file__).resolve().parents[1]/'tools/e0_teal_llama3_author_checkpoint_convert.py';s=importlib.util.spec_from_file_location('teal_convert_wrapper',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_complete_real_shapes(self):
  a=m.source_expected();b=m.output_expected();self.assertEqual(len(a),291);self.assertEqual(len(b),227);self.assertEqual(b['layers.31.attention.wqkv.weight'],(6144,4096));self.assertEqual(a['model.layers.31.self_attn.k_proj.weight'],(1024,4096));m.check_shapes(a,a)
 def test_missing_or_wrong_shape_fails(self):
  with self.assertRaises(ValueError):m.check_shapes({'x':(2,)},{'x':(3,)})
  with self.assertRaises(ValueError):m.check_shapes({},{'x':(3,)})
 def test_worker_budget_and_pinned_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'logs').write_bytes(b'x'*40);report={};m.checked_persist(root/'RUN.json',report,root,2<<20)
   with self.assertRaises(ValueError):m.checked_persist(root/'RUN.json',{},root,100)
   old=root/'input';old.write_bytes(b'old');fd=m.pin_file(old)
   try:
    replacement=root/'new';replacement.write_bytes(b'new');os.replace(replacement,old);self.assertEqual(os.pread(fd,3,0),b'old');self.assertEqual(old.read_bytes(),b'new')
   finally:os.close(fd)
 def test_mutable_scanner_disappearing_entry_and_other_errors(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);gone=root/'gone';gone.write_bytes(b'abc');kept=root/'kept';kept.write_bytes(b'xy');original=pathlib.Path.lstat
   def vanish(path,*args,**kwargs):
    if path==gone:gone.unlink();raise FileNotFoundError(str(path))
    return original(path,*args,**kwargs)
   with mock.patch.object(pathlib.Path,'lstat',vanish):result=m.worker_usage(root)
   self.assertEqual(result['missing_entries'],1);self.assertEqual(result['apparent_bytes'],2)
   def vanished_dir(path,followlinks,onerror):
    onerror(FileNotFoundError(errno.ENOENT,'disappeared directory',str(root/'nested')));return iter(())
   with mock.patch.object(m.os,'walk',vanished_dir):self.assertEqual(m.worker_usage(root)['missing_entries'],1)
   def vanished_root(path,followlinks,onerror):
    onerror(FileNotFoundError(errno.ENOENT,'disappeared root',str(root)));return iter(())
   with mock.patch.object(m.os,'walk',vanished_root):
    with self.assertRaises(FileNotFoundError):m.worker_usage(root)
   def denied(path,*args,**kwargs):raise PermissionError(str(path))
   with mock.patch.object(pathlib.Path,'lstat',denied):
    with self.assertRaises(PermissionError):m.worker_usage(root)
 def test_failure_reserve_and_conflicting_lock(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);reserve=m.reserve_emergency(root);(root/'oversize').write_bytes(b'x'*100)
   with self.assertRaises(ValueError):m.checked_persist(root/'RUN.json',{},root,100)
   m.emergency_failure(reserve,{'id':'TEST','status':'FAILED','hash_reads':[{'predebited_bytes':3,'completed_bytes':0}]},'oversize');self.assertEqual(json.loads(reserve.read_text())['status'],'FAILED_EMERGENCY_RECEIPT');self.assertEqual(reserve.stat().st_size,m.EMERGENCY_BYTES)
   source=root/'locked';source.write_bytes(b'a');fd=os.open(source,os.O_RDONLY)
   try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):m.pin_file(source)
   finally:os.close(fd)
 def test_mutated_pinned_source_fails(self):
  with tempfile.TemporaryDirectory() as d:
   source=pathlib.Path(d)/'input';source.write_bytes(b'abc');fd=m.pin_file(source)
   try:
    before={'src':m.statid(os.fstat(fd))};source.write_bytes(b'abcd')
    with self.assertRaises(ValueError):m.check_pins({'src':fd},before)
   finally:os.close(fd)
 def test_unmodified_author_converter_two_tiny_safetensors_shards(self):
  import torch
  from safetensors.torch import save_file
  author=m.source_module()
  class Tiny:
   @classmethod
   def from_name(cls,_):return type('Cfg',(),{'dim':8,'head_dim':4,'n_head':2,'n_local_heads':1})()
  author.ModelArgs=Tiny
  with tempfile.TemporaryDirectory() as d:
   base=pathlib.Path(d);source=base/'source';source.mkdir();(source/'tokenizer.model').write_bytes(b'tiny token');root=base/'llama-3-tiny'
   q=torch.arange(64,dtype=torch.bfloat16).reshape(8,8);k=(torch.arange(32,dtype=torch.bfloat16)+20).reshape(4,8);v=(torch.arange(32,dtype=torch.bfloat16)+40).reshape(4,8);gate=(torch.arange(48,dtype=torch.bfloat16)+50).reshape(6,8);embed=torch.arange(24,dtype=torch.bfloat16).reshape(3,8)
   x={'model.embed_tokens.weight':embed,'model.layers.0.self_attn.q_proj.weight':q,'model.layers.0.self_attn.k_proj.weight':k};y={'model.layers.0.self_attn.v_proj.weight':v,'model.layers.0.mlp.gate_proj.weight':gate}
   save_file(x,str(source/'a.safetensors'));save_file(y,str(source/'b.safetensors'));mapping={**{k:'a.safetensors' for k in x},**{k:'b.safetensors' for k in y}};(source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}));files=['a.safetensors','b.safetensors','model.safetensors.index.json','tokenizer.model'];pins={n:m.pin_file(source/n) for n in files};states={n:m.statid(os.fstat(fd)) for n,fd in pins.items()}
   try:
    m.stage_pinned(root,[{'path':n,'filename':n} for n in files[:2]],{'path':files[2],'filename':files[2]},{'path':files[3]},pins);author.convert_hf_checkpoint(checkpoint_dir=root,model_name='llama-3-tiny');m.check_pins(pins,states);out=torch.load(root/'model.pth',map_location='cpu',weights_only=True)
   finally:
    for fd in pins.values():os.close(fd)
   expected_q=q.view(2,2,2,8).transpose(1,2).reshape(8,8);expected_k=k.view(1,2,2,8).transpose(1,2).reshape(4,8)
   self.assertEqual(set(out),{'tok_embeddings.weight','layers.0.feed_forward.w1.weight','layers.0.attention.wqkv.weight'});self.assertTrue(torch.equal(out['layers.0.attention.wqkv.weight'],torch.cat([expected_q,expected_k,v])));self.assertTrue(torch.equal(out['layers.0.feed_forward.w1.weight'],gate));self.assertTrue(all(t.dtype==torch.bfloat16 for t in out.values()));self.assertEqual((root/'tokenizer.model').read_bytes(),b'tiny token')
if __name__=='__main__':unittest.main()
