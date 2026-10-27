import importlib.util
from pathlib import Path
import tempfile
import unittest
p=Path(__file__).resolve().parents[1]/'tools/e0_ub1_consumer_disjoint_quality_history.py'
s=importlib.util.spec_from_file_location('quality',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_preflight_rejects_path_and_inode_aliases_before_scoring(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);items=[]
   for i in range(60):
    p=root/str(i);p.write_bytes(b'equal-content-is-possible');items.append({'source':i//3,'mode':str(i%3),'video':{'path':p.name,'bytes':p.stat().st_size,'sha256':m.sha(p)}})
   plan={'videos':items[:40],'historical_videos':items[40:]}
   self.assertEqual(len(m.validate_clip_paths(plan,root)[0]),60)
   saved=items[-1]['video'];items[-1]['video']=items[0]['video']
   with self.assertRaisesRegex(ValueError,'distinct'):m.validate_clip_paths(plan,root)
   items[-1]['video']=saved;(root/'59').unlink();(root/'59').hardlink_to(root/'0')
   with self.assertRaisesRegex(ValueError,'distinct'):m.validate_clip_paths(plan,root)
 def test_revoked_authority_and_preflight_failure_receipt(self):
  import json
  from types import SimpleNamespace
  from unittest import mock
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=m.ROOT;m.ROOT=root
   try:
    plan={'id':m.ID,'assets_finalized':True,'limits':m.LIMITS,'automatic_retry':False,'output_HF_relative':'result'}
    protocol=root/'protocol.json';protocol.write_text(json.dumps(plan));digest=m.sha(protocol)
    active={'user_continuation_authorization_20261027':{'id':'USER_RESUME_OFFLINE_IFC_20261027','status':'ACTIVE_CURRENT_USER_DIRECTED_SESSION'},'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':digest,'execution_status':'AUTHORIZED_NOT_STARTED'}]}}
    m.authorize(plan,digest,active,0)
    for part in (False,2):
     with self.assertRaises(ValueError):m.authorize(plan,digest,active,part)
    (root/'research').mkdir();path=root/'research/ACTIVE_RESEARCH.json';path.write_text(json.dumps(active));args=SimpleNamespace(protocol=protocol,hf_root=root,part=0)
    with mock.patch.object(m,'preflight',side_effect=ValueError('broken input')):
     with self.assertRaisesRegex(ValueError,'broken input'):m.run(args)
    self.assertTrue((root/'result/part0/ATTEMPT.json').is_file());record=json.loads((root/'result/part0/RUN.json').read_text());self.assertEqual(record['status'],'FAILED_NO_RETRY');self.assertEqual(record['predebited_work'],{})
    with self.assertRaises(FileExistsError):m.run(args)
    active['user_continuation_authorization_20261027']['status']='PAUSED';path.write_text(json.dumps(active));args.part=1
    with self.assertRaisesRegex(ValueError,'inactive'):m.run(args)
    self.assertFalse((root/'result/part1/ATTEMPT.json').exists())
   finally:m.ROOT=old
 def test_failed_call_remains_debited(self):
  r={'predebited_work':{}};m.debit(r,'dynamic_degree')
  try:raise RuntimeError('model failed')
  except RuntimeError:pass
  self.assertEqual(r['predebited_work']['dynamic_degree'],{'RAFT':1200,'RAFT_iterations':24000})
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
 def test_cohort_missing_and_mismatched_noise(self):
  import json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);videos=[]
   def binding(path):return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':m.sha(path)}
   for source in range(20):
    for mode in ('dense','candidate'):
     folder=root/str(source)/mode;folder.mkdir(parents=True);noise=folder/'noise';noise.write_bytes(b'noise'+str(source).encode());video=folder/'video';video.write_bytes(b'video'+str(source).encode()+mode.encode());v=binding(video)
     record={'status':'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE','mode':mode,'source_index':source,'source_id':str(source),'seed':42,'prompt_sha256':'p','initial_noise':{'sha256':m.sha(noise),'path':str(noise.relative_to(root)),'bytes':noise.stat().st_size},'video':{'size':[832,464],'sha256':v['sha256'],'frames':81,'fps':16}};gen=folder/'generation';gen.write_text(json.dumps(record))
     videos.append({'source':source,'mode':mode,'source_index':source,'source_id':str(source),'seed':42,'prompt_sha256':'p','scheduler_id':'s','image_sha256':'i','initial_noise':binding(noise),'video':v,'generation':binding(gen)})
   self.assertTrue(m.validate_pairs(videos,root))
   producers={i:{'id':'replacement' if i<5 else 'original','sha256':'new' if i<5 else 'old'} for i in range(20)}
   for item in videos:
    f=root/item['generation']['path'];record=json.loads(f.read_text());identity=producers[item['source']];record.update(id=identity['id'],protocol_sha256=identity['sha256'],registered_run_id=str(item['source'])+item['mode'],bank_manifest_sha256='bank' if item['mode']=='candidate' else None);f.write_text(json.dumps(record));item['generation']=binding(f)
   self.assertTrue(m.validate_pairs(videos,root,producers,'bank'))
   f=root/videos[0]['generation']['path'];record=json.loads(f.read_text());record['protocol_sha256']='old';f.write_text(json.dumps(record));videos[0]['generation']=binding(f)
   with self.assertRaises(ValueError):m.validate_pairs(videos,root,producers,'bank')
   with self.assertRaises(ValueError):m.validate_pairs(videos[:-1],root)
   target=root/videos[1]['initial_noise']['path'];target.write_bytes(b'other');videos[1]['initial_noise']=binding(target)
   with self.assertRaises(ValueError):m.validate_pairs(videos,root)
 def test_history_receipt_and_no_noise_fabrication(self):
  import json,hashlib
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); videos=[];cases=[];completed=[]
   def binding(p):return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':m.sha(p)}
   for i in range(20):
    prompt='prompt'+str(i);digest=hashlib.sha256(prompt.encode()).hexdigest();group='group'+str(i);source='image'+str(i);v=root/('v'+str(i));v.write_bytes(str(i).encode());vb=binding(v)
    record={'index':i,'status':'PASS_FORMAL_INPUT_GENERATION_ONLY','arm':'D-Q-PACKED-TRAINED-REFERENCE','input':{'seed':42,'source_group':group,'prompt':prompt,'prompt_sha256':digest,'input_provenance':{'image_sha256':source}},'video':{'sha256':vb['sha256'],'frames':81,'fps':16,'size':[832,464]}}
    f=root/('r'+str(i));f.write_text(json.dumps(record));ib=binding(f);completed.append({'index':i,'case_report_sha256':ib['sha256']});cases.append({'source_id':source,'source_group':group,'prompt_sha256':digest,'image':{'sha256':source}})
    videos.append({'source':i,'mode':'historical_neutral','source_id':source,'image_sha256':source,'prompt':prompt,'prompt_sha256':digest,'exposure':'HISTORICAL_SAME_SOURCE_SEED_NO_SAVED_INITIAL_NOISE','video':vb,'historical_input':ib})
   run={'status':'PASS_FORMAL_GENERATION_SLICE_ONLY','completed':completed};self.assertTrue(m.validate_historical(videos,root,{'cases':cases},run))
   videos[0]['source_id']='wrong'
   with self.assertRaises(ValueError):m.validate_historical(videos,root,{'cases':cases},run)
 def test_cluster_interface_preserves_twenty_sources(self):
  import json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=m.ROOT;m.ROOT=root
   try:
    cases=[{'semantic_source_id':str(i)} for i in range(20)];bank=[{'source_id':str(i),'image_id':'image'+str(i)} for i in range(20)];g=root/'generation.json';g.write_text(json.dumps({'cases':cases,'source_bank':{'sources':bank}}));plan={'generation_protocol':{'path':'generation.json','bytes':g.stat().st_size,'sha256':m.sha(g)},'videos':[],'historical_videos':[]};runs=[]
    for part in range(2):
     run={'part':part,'status':'COMPLETE_SEVEN_PART_DIMENSIONS','protocol_sha256':'protocol','completed':[]};selected=[]
     for i in range(part*10,part*10+10):
      for mode in ('dense','candidate','historical_neutral'):
       f=root/('%d_%s'%(i,mode));f.write_text(mode);item={'source':i,'mode':mode,'video':{'path':f.name,'bytes':f.stat().st_size,'sha256':m.sha(f)}};plan['historical_videos' if mode=='historical_neutral' else 'videos'].append(item);selected.append(item)
     for dim in list(m.DIMENSIONS[:6])+[m.DIMENSIONS[6+part]]:
      rows=[{'video_path':str(root/x['video']['path']),'video_results':1.0} for x in selected];entry={'dimension':dim,'result':[1.0,rows]}
      if dim=='dynamic_degree':entry['RAFT_observations']=[{'video_path':r['video_path'],'scores':[1.0]*40} for r in rows]
      run['completed'].append(entry)
     runs.append(run)
    output=m.cluster_rows(plan,runs,root,'protocol');self.assertEqual(len(output['dense']),160);self.assertEqual(len(output['historical_neutral']),160);self.assertEqual(len({r['image_id'] for r in output['dense']}),20)
    runs[0]['completed'][0]['result'][-1].pop()
    with self.assertRaises(ValueError):m.cluster_rows(plan,runs,root,'protocol')
   finally:m.ROOT=old
 def test_composite_producer_same_bank_and_source(self):
  import copy
  original={k:{'sha256':k} for k in ('bank_manifest','bank_protocol','cache_manifest','original_recipe')};original['repo_sources']=[{'path':'native','sha256':'abc'}];producer=copy.deepcopy(original)
  self.assertTrue(m.validate_producer_method(producer,original,['native']))
  producer['bank_manifest']['sha256']='wrong'
  with self.assertRaises(ValueError):m.validate_producer_method(producer,original,['native'])
  producer=copy.deepcopy(original);producer['repo_sources'][0]['sha256']='wrong'
  with self.assertRaises(ValueError):m.validate_producer_method(producer,original,['native'])
 def test_binding(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=root/'x';f.write_bytes(b'abc');item={'path':'x','bytes':3,'sha256':m.sha(f)};self.assertEqual(m.bind(root,item),f);f.write_bytes(b'bad')
   with self.assertRaises(ValueError):m.bind(root,item)
if __name__=='__main__':unittest.main()
