import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('test_teal_visual',ROOT/'tools/e0_teal_wan_visual_cpu.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_full_frame_coverage(self):
  p0=m.visual.old.page_positions(0);p1=m.visual.old.page_positions(1)
  self.assertEqual([x for x in p0+p1 if x is not None],list(range(81)))
  self.assertEqual(m.LIMITS['videos'],3);self.assertEqual(m.LIMITS['JPEG_pages'],6);self.assertEqual(m.LIMITS['successful_decoded_frames'],243)
 def test_missing_parent_blocks_draft(self):
  with patch.object(m.line,'parent_data',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):m.draft(Path('/tmp'),0)
 def test_draft_and_run_execute_imported_execution_module(self):
  from types import SimpleNamespace
  parent={'protocol':{'path':'parent.json'}}
  with patch.object(m.line,'parent_data',return_value=parent),patch.object(m.line,'closed_rows',return_value=[]),patch.object(m.base,'read_bound',return_value={'repo_sources':[]}),patch.object(m.base,'binding',side_effect=lambda p,r:{'path':str(p)}),patch.object(m,'environment',return_value={}):
   p=m.draft(Path('/tmp'),0)
  self.assertEqual(p['execution_private_worker_config_sha256'],m.execution.PROFILE_SHA)
  self.assertTrue(any(x['path'].endswith('tools/e0_teal_quality_execution.py') for x in p['repo_sources']))
  args=SimpleNamespace(hf_root=Path('/tmp'),protocol=Path('/tmp/protocol'))
  with patch.object(m.base,'read',return_value=p),patch.object(m.base,'sha',return_value='a'*64),patch.object(m,'authorize',return_value={}),patch.object(m.execution,'verify_job',side_effect=RuntimeError('execution reached')) as verify:
   with self.assertRaisesRegex(RuntimeError,'execution reached'):m.run(args)
   verify.assert_called_once()
 def test_short_video_predebit(self):
  class Reader:
   def read(self):return False,None
  class Ledger:
   def __init__(self):self.pre=[]
   def call(self,key,fn):self.pre.append(key);return fn()
  ledger=Ledger()
  with self.assertRaises(ValueError):m.visual.extract(Reader(),None,{},ledger,lambda *_:None)
  self.assertEqual(ledger.pre,['successful_decoded_frames'])
if __name__=='__main__':unittest.main()

class FrameTests(unittest.TestCase):
 def test_exact_81_and_extra_frame_rejected(self):
  class Reader:
   def __init__(self,n):self.n=n
   def read(self):
    self.n-=1;return (True,object()) if self.n>=0 else (False,None)
  class Renderer:
   def add(self,*_):pass
   def finish(self,i):return i
  class Ledger:
   def __init__(self):self.count={}
   def call(self,k,fn):self.count[k]=self.count.get(k,0)+1;return fn()
  pages=[];ledger=Ledger();m.visual.extract(Reader(81),Renderer(),{'ordinal':1040,'mode':'dense'},ledger,lambda _img,i,positions:pages.append((i,positions)))
  self.assertEqual(ledger.count,{'successful_decoded_frames':81,'JPEG_pages':2,'eof_checks':1});self.assertEqual([p[0] for p in pages],[0,1])
  self.assertEqual([x for _,p in pages for x in p if x is not None],list(range(81)))
  with self.assertRaises(ValueError):m.visual.extract(Reader(82),Renderer(),{'ordinal':1040,'mode':'dense'},Ledger(),lambda *_:None)
