import importlib.util
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_teal_author_llm_scope_planner.py'
s=importlib.util.spec_from_file_location('planner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Checks(unittest.TestCase):
 def test_temporary_checkpoint_rejected(self):
  with self.assertRaises(ValueError):m.calibration('/proc/123/fd/4','out')
 def test_uniform_requires_hist(self):
  with self.assertRaises(ValueError):m.generation('/x/llama-3-8b/model.pth',None,'Hello','uniform50')
 def test_no_neutral_mislabel(self):
  with self.assertRaises(ValueError):m.generation('/x/llama-3-8b/model.pth','h','Hello','neutral')
 def test_scope_counts(self):
  p=m.calibration('/hf/snapshot','out');self.assertEqual(p['expected']['sorts'],128);self.assertFalse(p['execute'])
if __name__=='__main__':unittest.main()
