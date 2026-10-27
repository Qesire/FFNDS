import importlib.util
from pathlib import Path
import tempfile
import unittest
p=Path(__file__).resolve().parents[1]/'tools/e0_ub1_consumer_official_quality.py'
s=importlib.util.spec_from_file_location('quality',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_failed_call_remains_debited(self):
  r={'predebited_work':{}};m.debit(r,'dynamic_degree')
  try:raise RuntimeError('model failed')
  except RuntimeError:pass
  self.assertEqual(r['predebited_work']['dynamic_degree'],{'RAFT':240,'RAFT_iterations':4800})
  with self.assertRaises(ValueError):m.debit(r,'dynamic_degree')
 def test_video_identity(self):
  good=[{'video_path':'a'},{'video_path':'b'}];self.assertIs(m.validate_results(good,['a','b']),good)
  for bad in ([good[0]],[good[0],good[0]],[good[0],{'video_path':'c'}]):
   with self.assertRaises(ValueError):m.validate_results(bad,['a','b'])
 def test_transparent_observer(self):
  seen=[];obj=object()
  def original(self,*a,**kw):seen.append((self,a,kw));return 2.5
  scores=[];self.assertEqual(m.observe_score(original,scores)(obj,1,z=2),2.5);self.assertEqual(seen,[(obj,(1,),{'z':2})]);self.assertEqual(scores,[2.5])
 def test_hub_keyword_metadata_and_result(self):
  seen=[];result=object()
  def old(*a,**kw):seen.append((a,kw));return result
  loader=m.make_local_hub_loader(old,Path('/local/dino'),{'/local/dino'})
  self.assertIs(loader(**{'repo_or_dir':'/local/dino','model':'dino_vitb16','path':'/weights','read_frame':False,'resolution':'16-9','source':'local','pretrained':True}),result)
  self.assertEqual(seen,[(('/local/dino','dino_vitb16'),{'source':'local','pretrained':True})])
  with self.assertRaises(ValueError):loader(repo_or_dir='/bad',model='dino_vitb16')
 def test_raft_counter_preserves_return(self):
  result=object();seen=[];counts={'forwards':0,'iterations':0}
  def original(obj,*a,**kw):seen.append((obj,a,kw));return result
  obj=object();fn=m.observed_raft_forward(original,counts)
  self.assertIs(fn(obj,1,2,iters=20,test_mode=True),result);self.assertEqual(counts,{'forwards':1,'iterations':20});self.assertEqual(seen,[(obj,(1,2),{'iters':20,'test_mode':True})])
  with self.assertRaises(ValueError):fn(obj,1,2,iters=19,test_mode=True)
 def test_dreamsim_compat_and_active_extension_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg=root/'adapter_config.json';defaults={'alpha_pattern':{},'rank_pattern':{},'layer_replication':None,'loftq_config':{},'megatron_config':None,'megatron_core':'megatron.core','use_dora':False,'use_rslora':False,'r':16,'lora_alpha':1.,'lora_dropout':.3,'bias':'none','target_modules':['qkv']};cfg.write_text(__import__('json').dumps(defaults));digest=m.sha(cfg);seen=[];ret=object()
   def factory(**kw):return kw
   def original(model,path,*a,**kw):seen.append((model,path,a,kw));return ret
   fn=m.dreamsim_adapter_loader(original,factory,root,digest);model=object()
   self.assertIs(fn(model,str(root),device='cuda'),ret)
   self.assertEqual(seen[0][3],{'device':'cuda','config':{k:defaults[k] for k in ('r','lora_alpha','lora_dropout','bias','target_modules')}})
   with self.assertRaises(ValueError):fn(model,str(root/'bad'))
   defaults['use_dora']=True;cfg.write_text(__import__('json').dumps(defaults));fn=m.dreamsim_adapter_loader(original,factory,root,m.sha(cfg))
   with self.assertRaises(ValueError):fn(model,str(root))
 def test_dreamsim_context_restores_descriptor(self):
  class Fake:
   @classmethod
   def from_pretrained(cls,*a,**kw):return cls,a,kw
  saved=Fake.__dict__['from_pretrained']
  with tempfile.TemporaryDirectory() as d:
   with m.dreamsim_adapter_compat(Fake,dict,Path(d),'0'*64):self.assertIsNot(Fake.__dict__['from_pretrained'],saved)
  self.assertIs(Fake.__dict__['from_pretrained'],saved)
 def test_binding(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=root/'x';f.write_bytes(b'abc');item={'path':'x','bytes':3,'sha256':m.sha(f)};self.assertEqual(m.bind(root,item),f);f.write_bytes(b'bad')
   with self.assertRaises(ValueError):m.bind(root,item)
if __name__=='__main__':unittest.main()
