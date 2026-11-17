"""Exercise actual Q1 880-pair preflight without torch or data."""
import ast,unittest
from pathlib import Path
p=Path(__file__).with_name('q1_cal11_reproduction.py');t=ast.parse(p.read_text(encoding='utf8'))
nodes=[n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CAL11' for x in n.targets)]
nodes += [n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='validate_calibration_pairs']
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
class TestQ1Pairs(unittest.TestCase):
 def test_full_coverage_and_mismatch(self):
  items=[]
  for source in ns['CAL11']:
   for expert in ('high','low'):
    for layer in range(40):
     ends=[]
     for step in ((0,1) if expert=='high' else (2,3)):
      ends.append({'source':source,'layer':layer,'step':step,'seed':42,'kind':'v2_panel','input_identity':{'checkpoint_sha256':expert},'panel':{'sha256':'a'*64},'token_id_sha256_le_int64':'b'*64})
     items.append({'source':source,'expert':expert,'layer':layer,'anchor':ends[0],'hit':ends[1]})
  scope={'calibration_pairs':items,'checkpoints':{'high':{'sha256':'high'},'low':{'sha256':'low'}}}
  self.assertEqual(len(ns['validate_calibration_pairs'](scope)),880)
  scope['calibration_pairs']=items[:-1]
  with self.assertRaises(ValueError):ns['validate_calibration_pairs'](scope)
  scope['calibration_pairs']=items;items[0]['hit']['step']=3
  with self.assertRaises(ValueError):ns['validate_calibration_pairs'](scope)
if __name__=='__main__':unittest.main()
