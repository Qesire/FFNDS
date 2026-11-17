"""Exercise actual selection function bodies without requiring torch."""
import ast,math,unittest
from pathlib import Path
p=Path(__file__).with_name('icc_local_replay.py'); tree=ast.parse(p.read_text(encoding='utf8'))
names={'FAMILIES','RANKS','MODES','SITES','DEV11'}
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_grid','select_family')]
ns={'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
class TestSelection(unittest.TestCase):
 def test_equal_weight_complete_and_tie(self):
  sources,sites,fams,ranks,modes=(ns[k] for k in ('DEV11','SITES','FAMILIES','RANKS','MODES'))
  rows=[dict(source=s,expert=e,layer=l,family=f,rank=r,mode=m,risk=float(fams.index(f)+1)) for s in sources for e,l in sites for f in fams for r in ranks for m in modes]
  self.assertEqual(len(rows),792); self.assertEqual(ns['select_family'](rows)['family'],'SVD')
  with self.assertRaises(ValueError):ns['select_family'](rows[:-1])
  with self.assertRaises(ValueError):ns['select_family'](rows+[rows[0]])
  for row in rows:row['risk']=1.
  self.assertEqual(ns['select_family'](rows)['family'],'SVD')
  rows[0]['risk']=float('nan')
  with self.assertRaises(ValueError):ns['select_family'](rows)
if __name__=='__main__':unittest.main()
