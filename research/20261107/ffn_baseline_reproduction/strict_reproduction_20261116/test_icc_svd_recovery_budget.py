"""CPU-only orchestration test: no SVD, torch runtime or remote writes."""
import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch

class FactorBudgetTest(unittest.TestCase):
 def test_five_new_pairs_and_one_reused_pair(self):
  sites=tuple((e,l) for e in ('high','low') for l in (4,19,34)); seen=[]
  torch=types.ModuleType('torch'); torch.__version__=type('Version',(str,),{})('2.10.0+cu128'); torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)));torch.cuda=types.SimpleNamespace(empty_cache=lambda:None)
  local=types.ModuleType('icc_local_replay');local.SITES=sites;local.FAMILIES=('SVD','CA','CD')
  assets=types.ModuleType('icc_stage3_assets');assets.checked_path=lambda x:x;assets.sha=lambda p:'code-hash';assets.load_weights=lambda c,l,d:(1,2,3,4);assets.verify_live_weights=lambda *a:None
  assets.make_site_factors=lambda *a,**kw:(1,2)
  def save(path,first,second,identity):
   seen.append((identity['expert'],identity['layer']))
   assert type(identity['torch_version']) is str
   return {'path':str(path),'sha256':'a'*64,'bytes':1,'identity':identity}
  assets.save_factors=save
  grid=types.ModuleType('icc_stage3_grid');grid.exclusive_json=lambda p,v:Path(p).write_text(json.dumps(v))
  rec=types.ModuleType('icc_svd_high04_recovery'); rec.recover=lambda scope,out:({'source_job_id':'168881'}, {'path':str(out/'SVD__high__04__r1408.pt'),'identity':{'family':'SVD','expert':'high','layer':4},'sha256':'b'*64,'bytes':1})
  modules={'torch':torch,'icc_local_replay':local,'icc_stage3_assets':assets,'icc_stage3_grid':grid,'icc_svd_high04_recovery':rec}
  with patch.dict(sys.modules,modules):
   source=Path(__file__).with_name('icc_stage3_factor_build.py');spec=importlib.util.spec_from_file_location('factor_build_test',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   with tempfile.TemporaryDirectory() as d:
    out=Path(d)/'output';binder=str(source.resolve().with_name('icc_svd_high04_recovery.py'))
    scope={'mode':'six_site_svd_recover_high04','max_matrix_decompositions':10,'recovered_matrix_decompositions':2,'scales':[],'output':str(out),'implementation_sha256':{binder:'code-hash'},'checkpoints':{'high':{'sha256':'h'},'low':{'sha256':'l'}},'live_weight_manifest':{},'svd_driver':'gesvd'}
    result=mod.execute(scope,factorize=None,device='cpu',authorize=lambda s:None,check_resources=lambda:None)
    m=json.loads((out/'FACTOR_MANIFEST.json').read_text())
    self.assertEqual(set(seen),set(sites)-{('high',4)})
    self.assertEqual(result['new_matrix_decompositions'],10)
    self.assertEqual(result['recovered_matrix_decompositions'],2)
    self.assertEqual((m['matrix_decompositions'],len(m['factor_pairs'])),(12,6))
    self.assertEqual(m['mode'],'six_site_svd')

if __name__=='__main__':unittest.main()
