"""Prepare offline Stage4 metadata candidates from real completed generation.

Reads SHA-bound immutable inputs and writes candidate metadata only. Does not
submit jobs, run VBench, decode videos, or declare review/ACTIVE approval.
"""
import argparse,hashlib,importlib.util,json,os,shutil
from pathlib import Path,PurePosixPath

from quality_pair_report import DEV,ARMS,DIMS
DEV3=('vbi_1013','vbi_0886','vbi_0800')
HISTORICAL=('vbi_1013','vbi_0800')
COMPARISONS={'main':ARMS,'E4':('ICC','ICC 2R E4'),'E6':('Chipmunk Author Safe','Chipmunk Author Safe E6')}
REF_SHA={'runner_metrics.py':'090f5e483c1cb7765dbda296baf01c858e7eb733ec433889e1f480e5fbd5d287',
         'metric_contract.py':'1f77d121d86d8e9b661be7cdac80cdc7467558df51216fe01fe40b37a2f50944',
         'metric_loaders.py':'9699de62e9d06cb6c09544b0681352dfa46b222809e99ad7dbdd4879dd1dcbba',
         'runtime_contract.py':'3fb526c26119a6e99b6efa40673aa9c56fe50051be3208c87725fdbe9f14f224'}
CODE=('quality_guard.py','quality_vbench_batch.py','quality_pair_report.py','quality_wan_proof.py','quality_dense_proof.py','quality_stage4_job.sh')
PROTO=('dimension','metric_seed','runtime_versions','metric_package_versions','metric_asset_sha256','metric_code_sha256',
       'quality_reference_sha256','ffmpeg_path','ffmpeg_sha256','vbench_working_directory','metric_project_root',
       'metric_cache_root','metric_site_packages','metric_additional_source_roots')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def jbytes(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def checked(spec):
 p=Path(spec['path']).resolve()
 if sha(p)!=spec['sha256']:raise ValueError('input identity mismatch: '+str(p))
 return read(p)
def require(value,message):
 if not value:raise ValueError(message)


def producer(submission, expected_freeze):
 stage=Path(submission['stage']).resolve(); job=str(submission['job_id']);require(job.isdigit(),'bad job id')
 names=['scope.json','REVIEW.json','ACTIVE_RESEARCH.json','receipt/RUN.json','receipt/EXIT.json','output/SUMMARY.json','output/NFE.jsonl']
 files={name:sha(stage/name) for name in names}
 scope=read(stage/'scope.json');review=read(stage/'REVIEW.json');active=read(stage/'ACTIVE_RESEARCH.json')[scope['scope_id'].lower()]
 require(Path(scope['remote_stage']).resolve()==stage,'producer stage path mismatch')
 run=read(stage/'receipt/RUN.json');end=read(stage/'receipt/EXIT.json');summary=read(stage/'output/SUMMARY.json')
 require(files['scope.json']==submission['scope_sha256']==run['scope_sha256'] and str(run['allocation']['job_id'])==job==str(end['slurm_job_id']) and end['exit_code']==0,'generation producer not completed')
 require(review.get('reviewer_agent_id') and review['reviewer_agent_id']!=review.get('author_agent_id') and review['scope_sha256']==submission['scope_sha256'] and review['recheck_disposition']=='APPROVE' and active['runnable'] and active['scope_sha256']==submission['scope_sha256'] and active['review_sha256']==files['REVIEW.json'],'generation review/ACTIVE mismatch')
 require(scope['force_dense'] is False and not scope.get('forced_arms') and summary['status']=='COMPLETE' and summary['arm']==scope['arm'] and summary['nfe']==4*len(scope['sources'])==scope['max_nfe'],'generation summary or budget mismatch')
 require([x['source'] for x in summary['records']]==scope['sources'],'generation source order mismatch')
 if scope['arm']!='Dense':require(scope['frozen_comparison']['sha256']==expected_freeze,'generation freeze mismatch')
 for name,expected in scope['bundle_sha256'].items():
  path=(stage/name).resolve();require(path.is_relative_to(stage) and sha(path)==expected,'generation bundle changed')
 for name in ('wan_comparison_job.py','icc_native_job.py','wan_comparison_job.sh'):require(name in scope['bundle_sha256'],'generation code missing')
 events=[json.loads(x) for x in (stage/'output/NFE.jsonl').read_text(encoding='utf8').splitlines()];expected=[]
 for i,source in enumerate(scope['sources']):
  for step in range(4): expected.extend([{'event':'NFE_ATTEMPT','source':source,'step':step,'attempt':i*4+step+1},{'event':'NFE_COMPLETED','source':source,'step':step,'expert':step//2}])
 require(events==expected,'generation NFE transaction mismatch')
 ref=scope['v2_reference_dir']; refs={name:scope['reference_sha256'][name] for name in ('scope.json','v2_conditions.json')}
 for name,value in refs.items():require(sha(Path(ref)/name)==value,'generation reference changed')
 prior=read(Path(ref)/'scope.json');recipe=prior['recipe_binding']['recipe_sha256']
 require(recipe=='ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6','generation recipe differs')
 receipts={}
 for item in summary['records']:
  source=item['source'];name='output/'+source+'.json';files[name]=sha(stage/name)
  require(files[name]==item['receipt_sha256'],'per-source receipt mismatch')
  row=read(stage/name);video=stage/'output'/f'{source}.mp4';latent=stage/'output'/f'{source}_latent.pt'
  require(row['source']==source and row['arm']==scope['arm'] and row['seed']==42 and row['schedule']==scope['schedule'] and row['dispatch']==[{'step':i,'expert':i//2} for i in range(4)] and len(row['sites'])==160,'per-source trajectory mismatch')
  require(sha(video)==row['video']['mp4_sha256'] and Path(row['latent_file']['path']).resolve()==latent.resolve() and sha(latent)==row['latent_file']['sha256'],'video or latent bytes mismatch')
  meta=row['video'];require(meta['frames']==81 and meta['width']==832 and meta['height']==464 and abs(meta['fps']-16)<.01,'video metadata mismatch')
  for i,site in enumerate(row['sites']):
   step,layer=divmod(i,40);expert='high' if step<2 else 'low'
   kind='forced_dense' if scope['arm']=='Dense' else ('fixed_dense' if scope['arm']=='Chipmunk Author Safe' and layer<2 else ('anchor' if step%2==0 else 'hit'))
   require((site['step'],site['layer'],site['expert'],site['kind'])==(step,layer,expert,kind),'generation FFN site trajectory mismatch')
   if scope['arm']=='Global-P64':require(site['selected_k']==(5120 if step%2 else None),'Global-P64 selected count mismatch')
   if scope['arm']=='Chipmunk Author Safe':
    tail=scope.get('chip_tail_mode','dense');dense_rows=52 if layer>=2 and (tail=='dense' or step%2==0) else 0
    require(site['dense_tail_rows']==dense_rows and site['tail_mode']==(tail if layer>=2 else None),'Chip tail mode mismatch')
    counts=site['block_counts'];require((isinstance(counts,list) and len(counts)==247) if layer>=2 and step%2 else counts is None,'Chip head counts mismatch')
    if tail=='valid52' and layer>=2:require(site['tail_info']['valid_rows']==52 and site['tail_info']['kind']==('sparse' if step%2 else 'full'),'valid52 trace missing')
  receipts[source]=row
 return scope,receipts,{'stage':str(stage),'files':files,'reference_sha256':refs,'recipe_sha256':recipe}


def label(scope):
 arm=scope['arm'];abl=scope.get('ablation')
 if arm=='ICC' and abl=='E4_SAME_FACTOR_2R' and scope.get('cache_mode')=='2R':return 'ICC 2R E4'
 if arm=='Chipmunk Author Safe' and abl=='E6_TAIL52' and scope.get('chip_tail_mode')=='valid52':return 'Chipmunk Author Safe E6'
 require(not abl and arm in ARMS and (arm!='ICC' or scope.get('cache_mode','4S')=='4S') and scope.get('chip_tail_mode','dense')=='dense','unsupported arm or ablation')
 return arm


def assemble(recipe):
 admission=checked(recipe['dense_admission']);audit=checked(recipe['dense_audit']);contract=checked(recipe['paired_contract'])
 require(audit['status']=='READ_ONLY_AUDIT_COMPLETE' and audit['exit_code']==0 and audit['output_sha256']==recipe['dense_admission']['sha256'] and contract['dev_cohort_admission_sha256']==recipe['dense_admission']['sha256'],'Dense admission identity')
 require(tuple(contract['sources'])==DEV and tuple(contract['arms'])==ARMS and contract['metric_seed']==20261103 and contract['generation_seed']==42,'quality paired contract mismatch')
 slots={x['source']:x for x in admission['slots']};require(len(admission['slots'])==11 and set(slots)==set(DEV),'Dense source coverage')
 conditions=checked(recipe['conditions']); subset=[x for x in conditions if x['record_id'] in DEV]; by_source={x['record_id']:x for x in subset}
 require(len(subset)==len(by_source)==11,'missing or duplicate source conditions')
 project=PurePosixPath(recipe['metric_project_root']); records={};bundles={}
 for source in DEV:
  row=slots[source]; c=by_source[source]; identity=row['input_identity']
  require(row['seed']==42 and row['quality_video_eligible'] and c['seed']==42 and c['prompt_sha256']==identity['prompt_sha256'] and c['image_sha256']==identity['image_sha256'] and c['source_cluster_id']==identity['source_cluster_id']==contract['source_clusters'][source],'source inputs mismatch')
  image=str(project/'vbench2_beta_i2v/data/crop/16-9'/Path(c['image_path']).name)
  require(sha(image)==identity['image_sha256'] and digest(c['prompt'].encode('utf8'))==identity['prompt_sha256'],'metric image/prompt bytes mismatch')
  require('/' not in c['prompt'] and '\\' not in c['prompt'] and len((c['prompt']+'-4.mp4').encode())<=255,'unsafe prompt name')
  base={'source':source,'seed':42,'prompt':c['prompt'],'prompt_sha256':c['prompt_sha256'],
        'image_path':image,'image_sha256':c['image_sha256'],'source_cluster_id':c['source_cluster_id'],
        'input_identity':{**identity,'initial_noise':row['initial_noise'],'scheduler':row['scheduler']}}
  records[source]={}
  if source in HISTORICAL:
   require(row['origin']=='stage1_full_u' and row['main_slot_complete'] and sha(row['video']['path'])==row['video']['sha256'],'historical Dense incomplete')
   records[source]['Dense']={**base,'arm':'Dense','video_path':row['video']['path'],'video_sha256':row['video']['sha256'],
                             'generation_proof_key':'historical_dense','generation_proof_sha256':recipe['dense_admission']['sha256']}
 require(len({x['prompt'] for x in by_source.values()})==11,'prompts must be unique')
 decision=checked(recipe['e6_decision']);require(decision['status'] in ('E6_DISABLED','E6_TAIL52_FROZEN'),'unsupported or missing E6 decision')
 if decision['status']=='E6_TAIL52_FROZEN':
  require(decision.get('affects_tail_adaptation') is True and decision.get('sources')==list(DEV3),'invalid frozen E6 tail decision')
 freeze_record=checked(recipe['frozen_comparison'])
 true_m_scope=checked(freeze_record['prerequisites']['true_m']['scope'])
 freeze=recipe['frozen_comparison']['sha256']; seen_jobs=set()
 for submitted_spec in recipe['submitted']:
  submitted=checked(submitted_spec);job=str(submitted['job_id']);require(job not in seen_jobs,'duplicate submitted job');seen_jobs.add(job)
  scope,receipts,bundle=producer(submitted,freeze);arm=label(scope);key='producer_'+job
  if arm in ('ICC','ICC 2R E4'):
   require(scope['parameters']==freeze_record['parameters']['ICC'],'ICC producer does not use frozen factors')
  if arm=='Q1r2':require(scope['parameters']==freeze_record['parameters']['Q1r2'],'Q1 producer does not use frozen parameters')
  if arm=='Global-P64':require(scope['fixed64_token_ids']==true_m_scope['fixed64_token_ids'],'Global-P64 fixed64 differs')
  if arm in ('Chipmunk Author Safe','Chipmunk Author Safe E6'):
   require(scope['chip_provider']==true_m_scope['provider'] and scope['voxel']==true_m_scope['voxel'],'author Chip source/voxel differs from true-M')
  if arm in ('ICC 2R E4','Chipmunk Author Safe E6'):
   require(set(receipts)<=set(DEV3),'ablation source outside frozen DEV3')
  if arm=='Chipmunk Author Safe E6':
   require(decision['status']=='E6_TAIL52_FROZEN' and scope['tail_evidence']['decision']['sha256']==recipe['e6_decision']['sha256'],'E6 producer differs from frozen tail decision')
  payload=jbytes(bundle); bundles[key]={'name':key+'.json','bytes':payload,'sha256':digest(payload)}
  for source,row in receipts.items():
   require(source in DEV and arm not in records[source] and not (arm=='Dense' and source in HISTORICAL),'unexpected or duplicate generated source arm')
   slot=slots[source];c=by_source[source];identity=slot['input_identity'];require(row['initial_noise']==slot['initial_noise'] and row['schedule']==slot['scheduler'],'paired generation noise or scheduler mismatch')
   video=str(Path(submitted['stage'])/'output'/f'{source}.mp4')
   records[source][arm]={'source':source,'arm':arm,'seed':42,'prompt':c['prompt'],'prompt_sha256':c['prompt_sha256'],
       'image_path':str(project/'vbench2_beta_i2v/data/crop/16-9'/Path(c['image_path']).name),'image_sha256':c['image_sha256'],
       'source_cluster_id':c['source_cluster_id'],'input_identity':{**identity,'initial_noise':slot['initial_noise'],'scheduler':slot['scheduler']},
       'video_path':video,'video_sha256':row['video']['mp4_sha256'],'generation_proof_key':key,'generation_proof_sha256':digest(payload)}
 allowed=['main','E4']+(['E6'] if decision['status']=='E6_TAIL52_FROZEN' else [])
 comparisons=recipe.get('comparisons_requested',allowed)
 require(isinstance(comparisons,list) and comparisons and len(set(comparisons))==len(comparisons)
         and all(x in allowed for x in comparisons) and comparisons==[x for x in allowed if x in comparisons],
         'requested comparisons must use frozen order and be eligible')
 for comparison in comparisons:
  sources=DEV if comparison=='main' else DEV3
  for src in sources:
   for arm in COMPARISONS[comparison]:require(arm in records[src],'missing completed video '+comparison+' '+src+' '+arm)
 return records,bundles,comparisons


def verify_closure(old,reference,additional):
 require(isinstance(additional,list) and len(set(additional))==len(additional),'explicit additional source roots required')
 for name,expected in REF_SHA.items():require(sha(Path(reference)/name)==expected,'quality reference changed: '+name)
 module_spec=importlib.util.spec_from_file_location('frozen_metric_contract',Path(reference)/'metric_contract.py')
 module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 memo={}
 def stable_sha(path):
  p=Path(path);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns,st.st_ctime_ns)
  if key not in memo:memo[key]=sha(p)
  return memo[key]
 cfg={**old,'metric_additional_source_roots':additional}
 for dim in DIMS:
  if dim!='camera_motion':module.verify_closure(dim,cfg,stable_sha)
 require(sha(old['ffmpeg_path'])==old['ffmpeg_sha256'],'ffmpeg identity differs')
 return {'files_hashed':len(memo),'dimensions':9}


def check_budget(b):
 names=('gpus_per_job','max_jobs','cpus','host_memory_gib','gpu_memory_gib','wall_seconds','soft_seconds','min_free_disk_gib','output_bytes')
 require(set(b)==set(names),'explicit exact quality budget fields required')
 require(b['gpus_per_job']==1 and b['max_jobs']==1 and 1<=b['cpus']<=4 and 0<b['host_memory_gib']<=160
         and 0<b['gpu_memory_gib']<=78 and 0<b['wall_seconds']<=10800
         and 0<b['soft_seconds']<b['wall_seconds'] and b['min_free_disk_gib']>0 and b['output_bytes']>0,'quality resource ceiling')


def build(recipe, destination=None, *, materials=None):
 freeze=checked(recipe['frozen_comparison']);contract=checked(recipe['paired_contract']);decision=checked(recipe['e6_decision'])
 require(freeze['seed']==42 and tuple(freeze['source_order'])==DEV and tuple(freeze['arms'])==ARMS
         and freeze['main_family_selected'] is True and freeze['main_icc_rank']==1408 and freeze['main_icc_state']=='4S'
         and freeze['optional_cal22_enabled'] is False and freeze['e4_same_factor_approved'] is True
         and freeze['expected_main_slots']==[{'arm':arm,'source':src} for arm in ARMS for src in DEV]
         and freeze['e6_decision']==recipe['e6_decision'] and freeze['paired_contract']==recipe['paired_contract'],'frozen comparison identity')
 require(decision['true_m_summary_sha256']==freeze['prerequisites']['true_m']['payload']['sha256'],'E6 evidence differs from frozen true-M')
 require(contract['explicit_camera_instruction_by_source']=={},'camera axis applicability changed')
 records,bundles,comparisons=assemble(recipe)
 require(recipe['old_quality_scope']['sha256']=='302afde8c3d038af5ca4f644aff9a3ecdcac30b3601e8c6f3fccf45494ff64fc','old metric scope differs from pinned artifact')
 old=checked(recipe['old_quality_scope']);reference=Path(recipe['quality_reference_source']).resolve()
 require(old['metric_project_root']==recipe['metric_project_root'] and old['vbench_working_directory']==old['metric_project_root'],'old metric project differs')
 require(old['metric_seed']==contract['metric_seed']==20261103,'metric seed mismatch')
 require(old['python_executable']=='/public/home/scc/pb23061276/miniconda3/envs/vbench_i2v_official/bin/python3.10' and sha(old['python_executable'])==recipe['python_executable_sha256'],'VBench interpreter identity mismatch')
 closure=verify_closure(old,reference,recipe['metric_additional_source_roots'])
 remote=PurePosixPath(recipe['planned_root']);require(remote.is_absolute() and '..' not in remote.parts,'planned root must be absolute')
 require(recipe['authors'] and len(set(recipe['authors']))==len(recipe['authors']),'missing scope authors')
 require(recipe['scheduler']=={'partition':'P107-A100','account':'competition','qos':'qos_p107-a100'},'quality scheduler binding differs')
 source=Path(__file__).resolve().parent
 if materials is None:
  runtime=Path(recipe['runtime_guard_source']).resolve();require(sha(runtime)==recipe['runtime_guard_sha256'],'runtime guard identity')
  stage_code={name:(source/name).read_bytes() for name in CODE};stage_code['runtime_guard.py']=runtime.read_bytes()
  ref_data={name:(reference/name).read_bytes() for name in REF_SHA}
 else:
  stage_code=dict(materials['stage_code']);ref_data=dict(materials['reference'])
  require(set(stage_code)==set(CODE)|{'runtime_guard.py'} and set(ref_data)==set(REF_SHA),'preview material inventory differs')
  require(digest(stage_code['runtime_guard.py'])==recipe['runtime_guard_sha256'],'runtime guard preview mismatch')
  for name,expected in REF_SHA.items():require(digest(ref_data[name])==expected,'quality reference preview mismatch: '+name)
 for comparison in comparisons:
  for dim in DIMS:
   if dim=='camera_motion':continue
   check_budget(recipe['budgets'][comparison][dim])
 # Ensure source/arm keys and backing files are one-to-one before creating files.
 for comparison in comparisons:
  sources=DEV if comparison=='main' else DEV3
  items=[records[source][arm] for source in sources for arm in COMPARISONS[comparison]]
  paths=[Path(r['video_path']).resolve() for r in items];inodes=[(p.stat().st_dev,p.stat().st_ino) for p in paths]
  require(len(set(paths))==len(paths)==len(set(inodes)),'video path or inode alias within metric batch')
 artifacts={}
 for name,data in ref_data.items():artifacts['reference/'+name]=data
 for item in bundles.values():artifacts['bundles/'+item['name']]=item['bytes']
 remote_bundles={k:{'path':str(remote/'bundles'/v['name']),'sha256':v['sha256']} for k,v in bundles.items()}
 result=[]
 for dim in DIMS:
  if dim=='camera_motion':continue
  protocol={name:(dim if name=='dimension' else REF_SHA if name=='quality_reference_sha256' else recipe['metric_additional_source_roots'] if name=='metric_additional_source_roots' else old[name]) for name in PROTO}
  protocol_bytes=jbytes(protocol);artifacts['protocols/'+dim+'.json']=protocol_bytes
  for comparison in comparisons:
   sources=DEV if comparison=='main' else DEV3
   batch=f'{comparison.lower()}__{dim}';planned=remote/'batches'/batch
   scope={**protocol,'phase':'FFNDS_STAGE4_QUALITY','status':'REVIEW_CANDIDATE','scope_id':'FFNDS_STAGE4_'+batch.upper(),
          'active_key':('FFNDS_STAGE4_'+batch.upper()).lower(),'authors':recipe['authors'],'comparison':comparison,
          'remote_stage':str(planned),'scheduler':recipe['scheduler'],
          'resources':{'gpus':recipe['budgets'][comparison][dim]['gpus_per_job'],
                       'cpu_cores':recipe['budgets'][comparison][dim]['cpus'],
                       **{k:recipe['budgets'][comparison][dim][k] for k in ('host_memory_gib','gpu_memory_gib','wall_seconds','soft_seconds','min_free_disk_gib','output_bytes')},
                       **recipe['scheduler']},
          'bundle_sha256':{name:digest(data) for name,data in stage_code.items()},
          'records':[records[src][arm] for src in sources for arm in COMPARISONS[comparison]],
          'review_path':str(planned/'REVIEW.json'),'active_path':str(planned/'ACTIVE_RESEARCH.json'),
          'receipt_dir':str(planned/'receipt'),'output':str(planned/'output'),'budget':recipe['budgets'][comparison][dim],
          'paired_contract_path':recipe['paired_contract']['path'],'paired_contract_sha256':recipe['paired_contract']['sha256'],
          'dense_admission_path':recipe['dense_admission']['path'],'dense_admission_sha256':recipe['dense_admission']['sha256'],
          'dense_admission_receipt_path':recipe['dense_audit']['path'],'dense_admission_receipt_sha256':recipe['dense_audit']['sha256'],
          'quality_reference_dir':str(remote/'reference'),'runtime_guard_path':str(planned/'runtime_guard.py'),
          'runtime_guard_sha256':recipe['runtime_guard_sha256'],'python_executable':old['python_executable'],
          'python_executable_sha256':recipe['python_executable_sha256'],
          'metric_protocol_path':str(remote/'protocols'/(dim+'.json')),'metric_protocol_sha256':digest(protocol_bytes),
          'generation_verifier_path':str(planned/'quality_wan_proof.py'),
          'generation_verifier_sha256':digest(stage_code['quality_wan_proof.py']),
          'generation_proof_bundles':remote_bundles,
          'implementation_sha256':{str(planned/name):digest(data) for name,data in stage_code.items()}}
   if comparison in ('E4','E6'):scope['frozen_comparison_sha256']=recipe['frozen_comparison']['sha256']
   if comparison=='E6':scope.update(e6_variable='tail',e6_decision_sha256=recipe['e6_decision']['sha256'])
   prefix='batches/'+batch+'/'
   scoped=jbytes(scope);artifacts[prefix+'scope.json']=scoped
   for name,data in stage_code.items():artifacts[prefix+name]=data
   names=['scope.json',*stage_code]
   artifacts[prefix+'SHA256SUMS']=''.join(digest(artifacts[prefix+name])+'  '+name+'\n' for name in names).encode('utf8')
   result.append({'comparison':comparison,'dimension':dim,'scope_sha256':digest(scoped),
                  'video_count':len(scope['records']),'path':str(planned/'scope.json')})
 summary={'status':'CANDIDATE_NOT_REVIEWED','closure':closure,'scopes':result,
          'camera_motion':'NOT_APPLICABLE','generation_or_metric_jobs_started':0}
 if 'main' in comparisons:
  main_records=[records[src][arm] for src in DEV3 for arm in ARMS]
  contact_scope=json.loads(artifacts['batches/main__subject_consistency/scope.json'])
  contact_scope.update(sources=list(DEV3),arms=list(ARMS),frame_indices=[0,20,40,60,80],
                       records=main_records,output=str(remote/'contact'/'output'))
  artifacts['contact/DEV3_INPUT.json']=jbytes({'status':'INPUT_CANDIDATE_NOT_REVIEWED',
      'scope_fields':contact_scope,'frame_times_seconds':[0,1.25,2.5,3.75,5.0],
      'verifier_dependency':str(remote/'batches'/'main__subject_consistency'/'quality_wan_proof.py'),
      'requires':'independently reviewed contact guard and generation/video verification'})
  artifacts['contact/PLAYBACK_INPUT.json']=jbytes({'status':'PLAYBACK_INPUT_NOT_VERIFIED_FOR_DISPLAY',
      'sources':DEV3,'arms':ARMS,'same_times_seconds':[0,1.25,2.5,3.75,5.0],
      'videos':[{'source':r['source'],'arm':r['arm'],'path':r['video_path'],'sha256':r['video_sha256'],
                 'generation_proof_sha256':r['generation_proof_sha256']} for r in main_records]})
 artifacts['CANDIDATE_SUMMARY.json']=jbytes(summary)
 if destination is None:return summary,artifacts
 out=Path(destination).resolve();require(not out.exists(),'candidate destination exists');out.mkdir(parents=True)
 for name,data in artifacts.items():
  file=(out/name).resolve();require(file.is_relative_to(out),'artifact path escape');file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
 return summary


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('recipe');parser.add_argument('recipe_sha256');parser.add_argument('destination');a=parser.parse_args()
 result=build(checked({'path':a.recipe,'sha256':a.recipe_sha256}),a.destination)
 print(json.dumps({'status':result['status'],'scope_count':len(result['scopes'])}))
