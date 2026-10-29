import importlib.util
import os
import subprocess
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_teal_llama3_grab_acts_launcher.py'
s=importlib.util.spec_from_file_location('grablauncher',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class FakeDataset:
 def __init__(self,n):self.n=n
 def skip(self,n):return self
 def take(self,n):return [{'text':str(i)} for i in range(min(n,self.n))]
class Tests(unittest.TestCase):
 def test_prefix_order_and_exact_count(self):
  out=m.read_prefix(FakeDataset(300));self.assertTrue(out.startswith('0\n\n1\n\n'));self.assertTrue(out.endswith('299\n\n'))
  with self.assertRaises(Exception):m.read_prefix(FakeDataset(299))
 def test_counts_must_all_complete(self):
  x={'sort':128,'histogram':128,'layer_inputs':32,'histogram_files':64,'layer_forward_pre':32,'layer_forward_post':32,'activation_finite_reductions':64};m.check_counts(x)
  x['sort']=127
  with self.assertRaises(Exception):m.check_counts(x)
 def test_wrong_segments_fail_before_numerical_work(self):
  class A:shape=(9,2048)
  with self.assertRaises(Exception):m.validate_segments(A(),A())
 def test_resource_budget_rejects_host_gpu_and_disk_overflow(self):
  r=m.RES;m.check_resources(1,1,r['minimum_free_disk_bytes'],1)
  for args in [(r['host_RSS_max_bytes']+1,1,r['minimum_free_disk_bytes'],1),(1,1,r['minimum_free_disk_bytes'],r['GPU_memory_max_bytes']+1),(1,r['new_HF_bytes'],r['minimum_free_disk_bytes'],1),(1,1,0,1)]:
   with self.assertRaises(Exception):m.check_resources(*args)
 def test_real_author_module_imports_without_gpu(self):
  code="""import sys, importlib
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
for name,rel in [('utils.utils','utils/utils.py'),('utils.data','utils/data.py'),('utils.eval_ppl','utils/eval_ppl.py'),('teal.mlp','teal/mlp.py'),('teal.self_attn','teal/self_attn.py'),('teal.model','teal/model.py')]:
 module=importlib.import_module(name)
 assert Path(module.__file__).resolve()==(root/rel).resolve(),name
"""
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';subprocess.run([str(m.ENV/'bin/python'),'-I','-B','-c',code,str(m.AUTHOR)],env=env,check=True,timeout=90)
 def test_frozen_and_scope_required(self):
  with self.assertRaises(Exception):m.validate({'id':m.ID,'status':'DRAFT_REVIEW_REQUESTED'})
if __name__=='__main__':unittest.main()
