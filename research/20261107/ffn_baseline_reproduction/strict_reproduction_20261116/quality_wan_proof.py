"""Proof for reviewed Wan comparison outputs; no research execution or mutations.

scope.generation_proof_bundles maps record.generation_proof_key to {path,sha256}.
The frozen bundle supplies stage and SHA256 of scope.json, REVIEW.json,
ACTIVE_RESEARCH.json, receipt/RUN.json, receipt/EXIT.json, output/SUMMARY.json,
output/NFE.jsonl and each source receipt under files (relative stage paths).
It also supplies prior_scope_sha256 and conditions_sha256, bound to the producer
scope's V2 reference_sha256. Historical Dense delegates to quality_dense_proof.
"""
import hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path
from quality_dense_proof import verify as verify_historical_dense

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def checked(stage,files,name):
 p=(stage/name).resolve();require(p.is_relative_to(stage) and name in files and sha(p)==files[name], 'producer file identity '+name);return p

def decoded(path,ffmpeg):
 h=hashlib.sha256();n=0;size=832*464*3
 with tempfile.TemporaryFile() as err:
  p=subprocess.Popen([str(ffmpeg),'-hide_banner','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-vsync','0','pipe:1'],stdout=subprocess.PIPE,stderr=err)
  try:
   while True:
    remaining=size
    while remaining:
     b=p.stdout.read(remaining)
     if not b:break
     h.update(b);remaining-=len(b)
    if remaining==size:break
    require(remaining==0,'partial decoded frame');n+=1;require(n<=81,'too many frames')
   require(p.wait(timeout=60)==0 and n==81,'video decode failed')
  finally:
   if p.poll() is None:p.kill();p.wait()
 return h.hexdigest()

def tensor_storage_sha(path):
 with zipfile.ZipFile(path) as z:
  names=[n for n in z.namelist() if n.endswith('/data/0')]
  require(len(names)==1 and len([n for n in z.namelist() if '/data/' in n])==1,'unexpected latent storage')
  h=hashlib.sha256()
  with z.open(names[0]) as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  return h.hexdigest()

def verify(record,scope):
 if record.get('generation_proof_key')=='historical_dense':
  dep=Path(__file__).resolve().with_name('quality_dense_proof.py')
  require(scope['implementation_sha256'].get(str(dep))==sha(dep),'historical verifier identity')
  return verify_historical_dense(record,scope)
 spec=scope['generation_proof_bundles'][record['generation_proof_key']]
 require(sha(spec['path'])==spec['sha256'] and spec['sha256']==record['generation_proof_sha256'],'proof bundle identity')
 bundle=read(spec['path']);stage=Path(bundle['stage']).resolve();files=bundle['files']
 source_scope=read(checked(stage,files,'scope.json'))
 require(source_scope['force_dense'] is False and not source_scope.get('forced_arms'),'forced Dense cannot prove approximate or production Dense')
 for name in ('wan_comparison_job.py','icc_native_job.py','wan_comparison_job.sh'):
  require(name in source_scope['bundle_sha256'],'missing producer code '+name)
 for name,digest in source_scope['bundle_sha256'].items():
  p=(stage/name).resolve();require(p.is_relative_to(stage) and sha(p)==digest,'producer bundle identity '+name)
 review=read(checked(stage,files,'REVIEW.json'));active_all=read(checked(stage,files,'ACTIVE_RESEARCH.json'))
 run=read(checked(stage,files,'receipt/RUN.json'));ex=read(checked(stage,files,'receipt/EXIT.json'))
 summary=read(checked(stage,files,'output/SUMMARY.json'));nfe=checked(stage,files,'output/NFE.jsonl')
 scope_sha=files['scope.json'];active=active_all[source_scope['scope_id'].lower()]
 require(review.get('reviewer_agent_id') and review['reviewer_agent_id']!=review.get('author_agent_id') and review['scope_sha256']==scope_sha and review['recheck_disposition']=='APPROVE' and active['runnable'] and active['scope_sha256']==scope_sha and active['review_sha256']==files['REVIEW.json'],'producer review/ACTIVE mismatch')
 require(run['scope_sha256']==scope_sha and ex['exit_code']==0 and str(run['allocation']['job_id'])==str(ex['slurm_job_id']),'producer job/run/exit mismatch')
 label=record['arm'];comparison=scope['comparison'];source=record['source']
 arm={'ICC 2R E4':'ICC','Chipmunk Author Safe E6':'Chipmunk Author Safe'}.get(label,label)
 require(source in source_scope['sources'] and source_scope['arm']==arm and arm in ('Dense','Q1r2','ICC','Chipmunk Author Safe','Global-P64'),'producer source/arm mismatch')
 require(comparison in ('main','E4','E6'),'unsupported production comparison')
 if comparison in ('E4','E6'):
  require(source_scope['frozen_comparison']['sha256']==scope['frozen_comparison_sha256'],'ablation parent/frozen comparison mismatch')
 mode=source_scope.get('cache_mode','4S')
 tail=source_scope.get('chip_tail_mode','dense')
 if comparison=='E6':
  require(scope.get('e6_variable')=='tail' and arm=='Chipmunk Author Safe','no verified producer for this E6 variable')
  if label=='Chipmunk Author Safe E6':
   require(tail=='valid52' and source_scope.get('ablation')=='E6_TAIL52'
           and source_scope['tail_evidence']['decision']['sha256']==scope['e6_decision_sha256'],'E6 child is not the verified frozen valid-tail producer')
  else:require(label==arm and tail=='dense' and not source_scope.get('ablation'),'E6 parent is not main dense-tail Chipmunk')
 elif comparison=='E4':
  require(not scope.get('e6_variable') and label in ('ICC','ICC 2R E4')
          and arm=='ICC' and mode==('2R' if label=='ICC 2R E4' else '4S'),'E4 scope mode or label unsupported')
 else:
  require(not scope.get('e6_variable') and label==arm and (arm!='ICC' or mode=='4S') and tail=='dense','main scope mode or label unsupported')
 ids=source_scope['sources'];require(summary['status']=='COMPLETE' and summary['arm']==arm and summary['nfe']==4*len(ids)==source_scope['max_nfe'],'incomplete summary')
 require([x['source'] for x in summary['records']]==ids,'summary source order')
 events=[json.loads(line) for line in nfe.read_text(encoding='utf8').splitlines()];expected=[]
 for i,s in enumerate(ids):
  for step in range(4):
   expected.extend([{'event':'NFE_ATTEMPT','source':s,'step':step,'attempt':i*4+step+1},{'event':'NFE_COMPLETED','source':s,'step':step,'expert':step//2}])
 require(events==expected,'NFE attempt/completion transaction mismatch')
 rel='output/'+source+'.json';rp=checked(stage,files,rel);row=read(rp)
 require(next(x['receipt_sha256'] for x in summary['records'] if x['source']==source)==files[rel],'summary receipt digest mismatch')
 require(row['source']==source and row['arm']==arm and row['seed']==42 and row['schedule']==source_scope['schedule'] and row['dispatch']==[{'step':i,'expert':i//2} for i in range(4)],'trajectory identity mismatch')
 sites=row['sites'];require(len(sites)==160,'site count')
 for i,site in enumerate(sites):
  step,layer=divmod(i,40);expert='high' if step<2 else 'low'
  kind='forced_dense' if arm=='Dense' else ('fixed_dense' if arm=='Chipmunk Author Safe' and layer<2 else ('anchor' if step%2==0 else 'hit'))
  require((site['step'],site['layer'],site['expert'],site['kind'])==(step,layer,expert,kind),'site trajectory mismatch')
  if arm=='Global-P64':require(site['selected_k']==(5120 if step%2 else None),'Global-P64 selection count')
  if arm=='Chipmunk Author Safe':
   dense_rows=52 if layer>=2 and (tail=='dense' or step%2==0) else 0
   require(site['dense_tail_rows']==dense_rows and site['tail_mode']==(tail if layer>=2 else None),'author Chip tail path mismatch')
   if tail=='valid52' and layer>=2:
    require(site['tail_info']['valid_rows']==52 and site['tail_info']['kind']==('sparse' if step%2 else 'full'),'E6 valid-tail actual action missing')
   counts=site['block_counts'];require((isinstance(counts,list) and len(counts)==247) if layer>=2 and step%2 else counts is None,'author Chip block count mismatch')
 ref=Path(source_scope['v2_reference_dir']);refs=source_scope['reference_sha256']
 for name in ('scope.json','v2_conditions.json'):
  require(sha(ref/name)==refs[name]==bundle['reference_sha256'][name],'producer V2 reference identity mismatch')
 prior=read(ref/'scope.json');conditions=read(ref/'v2_conditions.json');condition=next(c for c in conditions if c['record_id']==source)
 admission=Path(scope['dense_admission_path']);require(sha(admission)==scope['dense_admission_sha256'],'admission changed')
 contract=Path(scope['paired_contract_path']);audit=Path(scope['dense_admission_receipt_path'])
 require(sha(contract)==scope['paired_contract_sha256'] and read(contract)['dev_cohort_admission_sha256']==scope['dense_admission_sha256'] and sha(audit)==scope['dense_admission_receipt_sha256'],'admission proof identity')
 audit_data=read(audit);require(audit_data['status']=='READ_ONLY_AUDIT_COMPLETE' and audit_data['exit_code']==0 and audit_data['output_sha256']==scope['dense_admission_sha256'],'admission audit failed')
 dense=next(x for x in read(admission)['slots'] if x['source']==source);identity={**dense['input_identity'],'initial_noise':dense['initial_noise'],'scheduler':dense['scheduler']}
 require(record['input_identity']==identity and row['initial_noise']==dense['initial_noise'] and row['schedule']==dense['scheduler'],'noise/schedule mismatch')
 require(condition['image_sha256']==record['image_sha256']==identity['image_sha256'] and condition['prompt_sha256']==record['prompt_sha256']==identity['prompt_sha256'] and condition['source_cluster_id']==record['source_cluster_id']==identity['source_cluster_id'] and condition['seed']==42,'source condition mismatch')
 require({k:prior['checkpoints'][k]['sha256'] for k in ('high','low')}==identity['checkpoint_sha256'],'checkpoint recipe mismatch')
 require(prior['recipe_binding']['recipe_sha256']==bundle['recipe_sha256']=='ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6','generation recipe mismatch')
 noise=source_scope['dense_initial_noise'];require(sha(noise['path'])==noise['sha256'] and read(noise['path'])[source]==row['initial_noise'],'noise reference mismatch')
 video=(stage/'output'/f'{source}.mp4').resolve();require(Path(record['video_path']).resolve()==video and sha(video)==record['video_sha256']==row['video']['mp4_sha256'],'video identity mismatch')
 require(row['video']['frames']==81 and row['video']['width']==832 and row['video']['height']==464 and abs(row['video']['fps']-16)<.01,'video metadata mismatch')
 require(sha(scope['ffmpeg_path'])==scope['ffmpeg_sha256'] and decoded(video,scope['ffmpeg_path'])==row['video']['decoded_rgb24_sha256'],'decoded RGB mismatch')
 latent=stage/'output'/f'{source}_latent.pt';require(Path(row['latent_file']['path']).resolve()==latent and sha(latent)==row['latent_file']['sha256'],'latent file mismatch')
 require(tensor_storage_sha(latent)==row['final_latent']['sha256'],'final latent tensor mismatch')
 return {'status':'PASS','proof_sha256':spec['sha256']}
