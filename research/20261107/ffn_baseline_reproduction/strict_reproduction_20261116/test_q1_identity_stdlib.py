import ast,json,unittest
from pathlib import Path
source=Path(__file__).with_name('q1_v4_fit_remaining_candidate.py').read_text(encoding='utf8')
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='same_q1_identity')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<identity>','exec'),ns);same=ns['same_q1_identity']
class Q1IdentityTest(unittest.TestCase):
 def test_torch_tuple_json_list_round_trip(self):
  artifact={'expert':'high','layer':4,'calibration_sources':('a','b'),'checkpoint_sha256':'abc'}
  manifest=json.loads(json.dumps(artifact));self.assertTrue(same(artifact,manifest));self.assertIsInstance(artifact['calibration_sources'],tuple)
 def test_no_extra_equivalences(self):
  x={'calibration_sources':('a','b'),'layer':4};self.assertFalse(same(x,{'calibration_sources':['b','a'],'layer':4}));self.assertFalse(same(x,{'calibration_sources':['a','b'],'layer':5}));self.assertFalse(same(x,{'calibration_sources':['a','b'],'layer':4,'extra':1}));self.assertFalse(same(x,{'calibration_sources':'ab','layer':4}))
 def test_admission_invokes_normalized_comparison(self):
  import tempfile,hashlib,types
  tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('read','same_q1_identity','admitted_reconstructions')]
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'receipt').mkdir();(root/'output').mkdir();sites=[('high',4),('high',19),('high',34),('low',4),('low',19),('low',34)];cal=('a','b');ck={'high':{'sha256':'H'},'low':{'sha256':'L'}};scope0={'max_new_pcg_fits':0,'checkpoints':ck,'calibration_manifest_sha256':'M'}
   def write(p,obj):p.write_text(json.dumps(obj),encoding='utf8')
   def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
   write(root/'scope.json',scope0);sh=sha(root/'scope.json');write(root/'receipt/RUN.json',{'scope_sha256':sh,'slurm_job_id':'7'});write(root/'receipt/EXIT.json',{'exit_code':0,'slurm_job_id':'7'})
   class Matrix: dtype='float32';shape=(5120,5120)
   class Result:
    def all(self):return self
    def item(self):return True
   objects={};entries=[]
   for expert,layer in sites:
    identity={'expert':expert,'layer':layer,'checkpoint_sha256':ck[expert]['sha256'],'calibration_sources':cal,'calibration_manifest_sha256':'M','calibration_target':'T','common_replay_sha256':'C','export_origin':'V4_SAVED_CONVERGED_ALPHA_RECONSTRUCTED_NO_PCG'}
    fit={'status':'CONVERGED','true_relative_residual':0.};path=root/'output'/f'{expert}_{layer}.pt';path.write_bytes(b'dummy');objects[str(path)]={'identity':identity,'q1_fit':fit,'A_q1_f32':Matrix()};entries.append({'path':str(path),'identity':identity,'q1_fit':fit})
   write(root/'output/RECONSTRUCTED_Q1_SIX.json',{'status':'RECONSTRUCTED_SIX_NO_PCG','sites':entries})
   torch=types.SimpleNamespace(load=lambda p,**kw:objects[str(p)],float32='float32',isfinite=lambda _:Result())
   env={'Path':Path,'json':json,'sha':sha,'checked_path':lambda r:Path(r['path']),'torch':torch,'SITES':sites,'CAL11':cal}
   exec(compile(ast.Module(body=nodes,type_ignores=[]),'<admission>','exec'),env)
   scope={'reconstruction':{'stage':str(root),'scope_sha256':sh,'job_id':'7'},'checkpoints':ck,'calibration_manifest_sha256':'M','target_identity':'T','common_replay_sha256':'C'}
   self.assertEqual(len(env['admitted_reconstructions'](scope)[0]),6)
   objects[str(root/'output/high_4.pt')]['identity']['calibration_sources']=('b','a')
   with self.assertRaises(ValueError):env['admitted_reconstructions'](scope)
if __name__=='__main__':unittest.main()
