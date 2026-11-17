"""Small filesystem fixtures for true-M scope provenance; no Torch or GPU."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

path=Path(__file__).with_name('chip_true_m_scope_prepare.py')
spec=importlib.util.spec_from_file_location('preparer_test',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class ScopePreparationTest(unittest.TestCase):
 def test_exact_grid_factor_cache_and_tamper_rejection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf8');return mod.spec(path)
   def producer(name,config,job):
    stage=root/name;(stage/'receipt').mkdir(parents=True);(stage/'output').mkdir()
    cs=put(stage/'scope.json',config)
    put(stage/'receipt/RUN.json',{'scope_sha256':cs['sha256'],'slurm_job_id':str(job)})
    put(stage/'receipt/EXIT.json',{'slurm_job_id':str(job),'exit_code':0})
    put(stage/'REVIEW.json',{'scope_sha256':cs['sha256'],'recheck_disposition':'APPROVE'})
    return stage,{'stage':str(stage),'scope_sha256':cs['sha256'],'job_id':str(job)}
   cp={'high':{'sha256':'h'},'low':{'sha256':'l'}};factors=[];producers=[]
   for j,family in enumerate(mod.FAMILIES):
    stage,ref=producer(f'p{j}',{'job':j},100+j);producers.append(ref);rows=[]
    for e,l in mod.SIX:
     identity={'family':family,'expert':e,'layer':l,'rank':1408,'checkpoint_sha256':cp[e]['sha256'],'calibration_manifest_sha256':None if family=='SVD' else 'cal'}
     fp=stage/'output'/f'{family}_{e}_{l}.pt';fp.write_bytes((family+e+str(l)).encode());row={'path':str(fp),'sha256':mod.sha(fp),'bytes':fp.stat().st_size,'identity':identity}
     put(Path(str(fp)+'.json'),row);rows.append(row);factors.append({'family':family,'expert':e,'layer':l,**row})
    put(stage/'output/FACTOR_MANIFEST.json',{'status':'FACTORS_WRITTEN_PENDING_ORACLE','matrix_decompositions':12,'factor_pairs':rows})
   pair=[{'source':s,'expert':e,'layer':l} for s in mod.DEV for e,l in mod.SIX]
   pairs=put(root/'pairs.json',{'input_pairs':pair});capture=put(root/'capture.json',{'fixed64_token_ids':list(range(64))})
   grid,gridref=producer('grid',{'input_pairs':pair,'factors':factors,'checkpoints':cp,'clean_cal11_manifest_sha256':'cal','decomposition_norm_floor':1e-12},200)
   rows=[]
   for src in mod.DEV:
    for e,l in mod.SIX:
     for f in mod.FAMILIES:
      fhash=next(x['sha256'] for x in factors if (x['family'],x['expert'],x['layer'])==(f,e,l))
      for rank in (512,1408):
       for mode in ('4S','2R'):
        rows.append({'source':src,'expert':e,'layer':l,'family':f,'rank':rank,'mode':mode,'risk':1.0,'scope_sha256':gridref['scope_sha256'],'factor_sha256':fhash})
   selected,means,_=mod.select_grid(rows,gridref['scope_sha256']);selection={'family':selected,'risk_by_family':means}
   score=put(grid/'output/scores.json',{'scope_sha256':gridref['scope_sha256'],'cell_count':792,'rows':rows})
   sel=put(grid/'output/selection.json',{'scope_sha256':gridref['scope_sha256'],**selection})
   cache=[];(grid/'output/cache').mkdir()
   for src in mod.DEV:
    for e,l in mod.SIX:
     file=grid/'output/cache'/f'native__{src}__{e}__{l:02d}.pt';file.write_bytes(b'cache');cache.append({'file':file.name,'bytes':5,'sha256':mod.sha(file)})
   cm=put(grid/'output/cache_manifest.json',{'scope_sha256':gridref['scope_sha256'],'status':'PRESERVED_PENDING_SELECTED_FAMILY_ERROR_DECOMPOSITION','files':cache})
   put(grid/'output/SUMMARY.json',{'scope_sha256':gridref['scope_sha256'],'status':'GRID_COMPLETE_PENDING_Q1_GP64_AND_ERROR_DECOMPOSITION','count':792,'selection':selection,'scores_sha256':score['sha256'],'selection_sha256':sel['sha256'],'cache_manifest_sha256':cm['sha256']})
   template={'dev_pairs':pairs,'capture_manifest':capture,'checkpoints':cp,'prerequisites':{}}
   recipe={'template':template,'grid':gridref,'factor_producers':producers}
   with patch.object(mod,'validate_template'):
    result=mod.build(recipe)
    self.assertEqual(result['status'],'CANDIDATE_REQUIRES_INDEPENDENT_REVIEW')
    self.assertEqual(result['readiness']['q1'],'NOT_BOUND')
    self.assertEqual(result['scope']['fixed64_token_ids'],list(range(64)))
    (grid/'output/cache'/cache[0]['file']).write_bytes(b'wrong')
    with self.assertRaisesRegex(ValueError,'cache changed'):mod.build(recipe)

class PreviewPathTest(unittest.TestCase):
 def test_only_exact_stage_relative_bytes_are_accepted(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'nested').mkdir();file=root/'nested/tail.py';file.write_bytes(b'candidate')
   digest=mod.sha(file);stage='/ffnds-test-stage-never-create';final=stage+'/nested/tail.py'
   scope={'remote_stage':stage,'output':stage+'/output','implementation_sha256':{final:digest}}
   mod.verify_stage_or_external(final,digest,scope,root)
   with self.assertRaises(ValueError):mod.verify_stage_or_external(final,'0'*64,scope,root)
   with self.assertRaises(ValueError):mod.verify_stage_or_external(stage+'/nested/../tail.py',digest,scope,root)
   with self.assertRaises(ValueError):mod.verify_stage_or_external('/outside/tail.py',digest,scope,root)
   file.write_bytes(b'changed')
   with self.assertRaises(ValueError):mod.verify_stage_or_external(final,digest,scope,root)

if __name__=='__main__':unittest.main()
