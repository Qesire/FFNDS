"""Stdlib-only Stage4 planning fixture; no model, videos or scheduler."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

path=Path(__file__).with_name('quality_stage4_prepare.py');spec=importlib.util.spec_from_file_location('quality_prepare_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Stage4PrepareTest(unittest.TestCase):
 def test_full_batches_order_duplicate_and_e6_gate(self):
  admission=[];conditions=[];clusters={};images={};fake_subs=[];submissions={}
  for src in m.DEV:
   prompt='prompt '+src;ph=hashlib.sha256(prompt.encode()).hexdigest();ih=hashlib.sha256(src.encode()).hexdigest();cluster='cluster-'+src;clusters[src]=cluster
   ident={'source':src,'seed':42,'image_sha256':ih,'prompt_sha256':ph,'source_cluster_id':cluster,'checkpoint_sha256':{'high':'h','low':'l'}}
   admission.append({'source':src,'seed':42,'quality_video_eligible':True,'main_slot_complete':True,'origin':'stage1_full_u' if src in m.HISTORICAL else 'legacy',
                     'input_identity':ident,'initial_noise':{'sha256':'noise'},'scheduler':[1,2,3,4],
                     'video':{'path':'/videos/'+src+'.mp4','sha256':'v-'+src}})
   conditions.append({'record_id':src,'seed':42,'prompt':prompt,'prompt_sha256':ph,'image_path':'/images/'+src+'.jpg','image_sha256':ih,'source_cluster_id':cluster})
   images['/project/vbench2_beta_i2v/data/crop/16-9/'+src+'.jpg']=ih
   images['/videos/'+src+'.mp4']='v-'+src
  values={'admission':{'slots':admission},'audit':{'status':'READ_ONLY_AUDIT_COMPLETE','exit_code':0,'output_sha256':'adm-sha'},
          'contract':{'sources':list(m.DEV),'arms':list(m.ARMS),'metric_seed':20261103,'generation_seed':42,'source_clusters':clusters,'dev_cohort_admission_sha256':'adm-sha'},
          'conditions':conditions,'decision':{'status':'E6_TAIL52_FROZEN','affects_tail_adaptation':True,'sources':list(m.DEV3)},
          'freeze_record':{'parameters':{'ICC':'icc-params','Q1r2':'q1-params'},'prerequisites':{'true_m':{'scope':{'path':'true_m'}}}},
          'true_m':{'fixed64_token_ids':[1,2],'provider':'provider','voxel':'voxel'}}
  j=0
  for src in m.DEV:
   for arm in m.ARMS:
    if arm=='Dense' and src in m.HISTORICAL:continue
    j+=1;token='sub'+str(j);fake_subs.append({'path':token});submissions[token]={'job_id':str(j),'stage':'/stage/'+str(j),'source':src,'arm':arm}
   if src in m.DEV3:
    for arm in ('ICC 2R E4','Chipmunk Author Safe E6'):
     j+=1;token='sub'+str(j);fake_subs.append({'path':token});submissions[token]={'job_id':str(j),'stage':'/stage/'+str(j),'source':src,'arm':arm}
  def checked(s):return submissions[s['path']] if s['path'].startswith('sub') else values[s['path']]
  def producer(sub,freeze):
   arm=sub['arm'];src=sub['source'];scope={'arm':{'ICC 2R E4':'ICC','Chipmunk Author Safe E6':'Chipmunk Author Safe'}.get(arm,arm)}
   if arm in ('ICC','ICC 2R E4'):scope['parameters']='icc-params'
   if arm=='Q1r2':scope['parameters']='q1-params'
   if arm=='Global-P64':scope['fixed64_token_ids']=[1,2]
   if arm in ('Chipmunk Author Safe','Chipmunk Author Safe E6'):scope.update(chip_provider='provider',voxel='voxel')
   if arm=='ICC 2R E4':scope.update(ablation='E4_SAME_FACTOR_2R',cache_mode='2R')
   if arm=='Chipmunk Author Safe E6':scope.update(ablation='E6_TAIL52',chip_tail_mode='valid52',tail_evidence={'decision':{'sha256':'e6-sha'}})
   return scope,{src:{'initial_noise':{'sha256':'noise'},'schedule':[1,2,3,4],'video':{'mp4_sha256':'v-'+src+arm}}},{'stage':sub['stage'],'files':{},'reference_sha256':{},'recipe_sha256':'r'}
  recipe={'dense_admission':{'path':'admission','sha256':'adm-sha'},'dense_audit':{'path':'audit'},'paired_contract':{'path':'contract'},'conditions':{'path':'conditions'},'e6_decision':{'path':'decision','sha256':'e6-sha'},'frozen_comparison':{'path':'freeze_record','sha256':'freeze'},'metric_project_root':'/project','submitted':fake_subs}
  with patch.object(m,'checked',side_effect=checked),patch.object(m,'producer',side_effect=producer),patch.object(m,'sha',side_effect=lambda path:images[str(path)]):
   records,bundles,comparisons=m.assemble(recipe)
   self.assertEqual(comparisons,['main','E4','E6'])
   for c in comparisons:
    sources=m.DEV if c=='main' else m.DEV3
    rows=[records[s][a] for s in sources for a in m.COMPARISONS[c]]
    self.assertEqual(len(rows),55 if c=='main' else 6)
   self.assertEqual(sum(records[s]['Dense']['generation_proof_key']=='historical_dense' for s in m.DEV),2)
   main_subs=[x for x in fake_subs if submissions[x['path']]['arm'] not in ('ICC 2R E4','Chipmunk Author Safe E6')]
   _records,_bundles,chosen=m.assemble({**recipe,'submitted':main_subs,'comparisons_requested':['main']})
   self.assertEqual(chosen,['main'])
   e4_subs=[x for x in fake_subs if submissions[x['path']]['arm'] in ('ICC','ICC 2R E4')]
   e4_records,_,e4_choice=m.assemble({**recipe,'submitted':e4_subs,'comparisons_requested':['E4']})
   self.assertEqual(e4_choice,['E4']);self.assertEqual(sum(len([a for a in m.COMPARISONS['E4'] if a in e4_records[src]]) for src in m.DEV3),6)
   duplicate={**recipe,'submitted':fake_subs+[fake_subs[0]]}
   with self.assertRaisesRegex(ValueError,'duplicate submitted'):m.assemble(duplicate)
   values['decision']={'status':'E6_DISABLED'}
   with self.assertRaisesRegex(ValueError,'E6 producer'):m.assemble(recipe)

class CandidateScopeTest(unittest.TestCase):
 def test_27_complete_guard_scope_shapes_and_shared_protocol(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);files={};records={};all_arms=set(m.ARMS)|{'ICC 2R E4','Chipmunk Author Safe E6'}
   for src in m.DEV:
    records[src]={}
    for arm in all_arms:
     video=root/(src+'_'+arm.replace(' ','_')+'.mp4');video.write_bytes((src+arm).encode())
     records[src][arm]={'source':src,'arm':arm,'video_path':str(video),'video_sha256':'v','generation_proof_sha256':'p','seed':42}
   old={'metric_project_root':'/project','vbench_working_directory':'/project','metric_seed':20261103,
        'python_executable':'/public/home/scc/pb23061276/miniconda3/envs/vbench_i2v_official/bin/python3.10',
        'runtime_versions':{},'metric_package_versions':{},'metric_asset_sha256':{},'metric_code_sha256':{},
        'ffmpeg_path':'/ffmpeg','ffmpeg_sha256':'x','metric_cache_root':'/cache','metric_site_packages':'/site'}
   decision={'status':'E6_TAIL52_FROZEN','true_m_summary_sha256':'tm'}
   contract={'metric_seed':20261103,'explicit_camera_instruction_by_source':{}}
   paired={'path':'contract','sha256':'pc'};e6={'path':'decision','sha256':'e6'}
   freeze={'seed':42,'source_order':list(m.DEV),'arms':list(m.ARMS),'main_family_selected':True,'main_icc_rank':1408,'main_icc_state':'4S','optional_cal22_enabled':False,'e4_same_factor_approved':True,'expected_main_slots':[{'arm':a,'source':s} for a in m.ARMS for s in m.DEV],'e6_decision':e6,'paired_contract':paired,'prerequisites':{'true_m':{'payload':{'sha256':'tm'}}}}
   mapping={'contract':contract,'decision':decision,'frozen':freeze,'old':old}
   budget={'gpus_per_job':1,'max_jobs':1,'cpus':4,'host_memory_gib':64,'gpu_memory_gib':60,'wall_seconds':3600,'soft_seconds':3400,'min_free_disk_gib':10,'output_bytes':1000000}
   budgets={c:{dim:dict(budget) for dim in m.DIMS if dim!='camera_motion'} for c in ('main','E4','E6')}
   runtime=Path(__file__).resolve().parent/'quality_guard.py'
   reference=Path(__file__).resolve().parent/'fixture_ref'
   reference.mkdir(exist_ok=True)
   try:
    for name in m.REF_SHA:(reference/name).write_text('fixture')
    recipe={'frozen_comparison':{'path':'frozen','sha256':'f'},'paired_contract':paired,'e6_decision':e6,
            'old_quality_scope':{'path':'old','sha256':'302afde8c3d038af5ca4f644aff9a3ecdcac30b3601e8c6f3fccf45494ff64fc'},'quality_reference_source':str(reference),'metric_project_root':'/project',
            'python_executable_sha256':'python-hash','metric_additional_source_roots':[],
            'planned_root':'/planned/quality','authors':['author'],'scheduler':{'partition':'P107-A100','account':'competition','qos':'qos_p107-a100'}, 'runtime_guard_source':str(runtime),
            'runtime_guard_sha256':m.sha(runtime),'budgets':budgets,'dense_admission':{'path':'adm','sha256':'a'},
            'dense_audit':{'path':'audit','sha256':'b'}}
    original=m.sha
    def sha_override(value):return 'python-hash' if str(value)==old['python_executable'] else original(value)
    with patch.object(m,'checked',side_effect=lambda x:mapping[x['path']]), patch.object(m,'assemble',return_value=(records,{},['main','E4','E6'])), patch.object(m,'verify_closure',return_value={'files_hashed':0,'dimensions':9}),patch.object(m,'sha',side_effect=sha_override):
     result=m.build(recipe,root/'out')
    self.assertEqual(len(result['scopes']),27)
    for c,n in [('main',55),('E4',6),('E6',6)]:
     p=root/'out'/'batches'/(c.lower()+'__i2v_subject')/'scope.json';scope=json.loads(p.read_text())
     self.assertEqual(len(scope['records']),n)
     self.assertEqual(scope['metric_protocol_sha256'],m.sha(root/'out'/'protocols'/'i2v_subject.json'))
     self.assertEqual(scope['python_executable'],old['python_executable'])
     self.assertTrue(set(m.PROTO)<=set(scope))
     self.assertEqual(scope['remote_stage'],str(Path(scope['output']).parent).replace('\\','/'))
     self.assertEqual(scope['resources']['gpus'],1);self.assertIn('quality_guard.py',scope['bundle_sha256'])
    self.assertFalse((root/'out'/'batches'/'main__camera_motion').exists())
    contact=json.loads((root/'out'/'contact'/'DEV3_INPUT.json').read_text())
    playback=json.loads((root/'out'/'contact'/'PLAYBACK_INPUT.json').read_text())
    self.assertEqual(len(contact['scope_fields']['records']),15);self.assertEqual(len(playback['videos']),15)
    self.assertEqual(contact['scope_fields']['frame_indices'],[0,20,40,60,80])
   finally:
    import shutil;shutil.rmtree(reference)

if __name__=='__main__':unittest.main()
