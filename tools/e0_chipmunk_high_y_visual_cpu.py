"""Two actual matched Chipmunk P64 Wan videos: 81-frame contact sheets, CPU only."""
import argparse,copy,hashlib,importlib.metadata,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
line=module('_sharq_visual_lineage','tools/e0_chipmunk_high_y_quality_lineage.py')
visual=module('_teal_visual_private_existing_v2','tools/e0_s2_sm120_fixed_depth_visual_cpu_v2.py')
execution=module('_sharq_visual_execution_v2','tools/e0_chipmunk_p64_quality_execution.py')
base=visual.base;need=base.require
SELF='tools/e0_chipmunk_high_y_visual_cpu.py';TEST='tests/test_e0_chipmunk_high_y_metric_adapter.py'
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':300,'outer_seconds':600,'new_HF_bytes':32<<20,'max_RSS_bytes':4<<30,'automatic_retry':False}
LIMITS={'videos':1,'successful_decoded_frames':81,'eof_checks':1,'JPEG_pages':2,'model_calls':0,'GPU_calls':0,'RAFT_calls':0,'video_generations':0,'video_encodes':0}
visual.LIMITS=LIMITS
LAYOUT=visual.LAYOUT;INTERPRETATION='ALL_81_FRAMES_CONTACT_SHEETS_FOR_SEPARATE_AI_VISUAL_REVIEW_NOT_HUMAN_FULL_PLAYBACK'

def identity(part):need(type(part) is int and part in (0,1),'Invalid part');return f'E0_CHIPMUNK_HIGH_Y_ALL81_VISUAL_CPU_20261030_PART{part}'
def output(part):return f'ffnds-derived/e0_chipmunk_high_y_all81_visual_cpu_20261030/part{part}/attempt1/worker/results'

def draft(hf,part):
 parent,all_rows=line.parent_data(hf,part);rows=[r for r in all_rows if r['mode']=='high_y_reuse'];p=base.read_bound(ROOT,parent['protocol'])
 paths={b['path'] for b in p['repo_sources']}|{SELF,TEST,'tools/e0_chipmunk_high_y_quality_lineage.py','tests/test_e0_chipmunk_high_y_quality_lineage.py',
  'tools/e0_s2_sm120_fixed_depth_visual_cpu_v2.py','tests/test_e0_s2_sm120_fixed_depth_visual_cpu_v2.py',
  'tools/e0_s2_sm120_fixed_depth_motion_metrics.py','tools/e0_s2_new12_dense_visual_cpu.py','tools/e0_s2_new12_dense_motion_metrics.py',
  'tools/worker_identity.py','tools/e0_chipmunk_high_y_metric_freeze.py','tools/e0_chipmunk_high_y_metric_adapter.py','tests/test_e0_chipmunk_high_y_metric_adapter.py','tools/e0_sharq_wan_metric_read_budget.py','tests/test_e0_sharq_wan_metric_read_budget.py','tools/e0_chipmunk_high_y_metric_verify.py','tests/test_e0_chipmunk_high_y_metric_adapter.py','tools/e0_chipmunk_high_y_metric_jobs.py','tests/test_e0_chipmunk_high_y_metric_adapter.py',line.parent_path(part),line.parent_entry(part)['owner_source']['path']}|{line.factorial.CONFIGS[part],line.factorial.OWNER,line.prior.parent_path(part),line.prior.parent_entry(part)['owner_source']['path']}|execution.HELPERS|{'tools/e0_chipmunk_p64_quality_lineage.py','tests/test_e0_chipmunk_p64_quality_lineage.py','tools/e0_chipmunk_p64_metric_verify.py','tools/e0_chipmunk_p64_metric_adapter.py','tools/e0_chipmunk_p64_visual_cpu.py'}
 paths |= {b['path'] for b in base.read(ROOT/f'config/e0_chipmunk_p64_visual_20261030_part{part}_v1.json')['repo_sources']} | {f'config/e0_chipmunk_p64_visual_20261030_part{part}_v1.json'}
 p={'id':identity(part),'part':part,'status':'DRAFT','assets_finalized':False,'resources':RESOURCES,'limits':LIMITS,
  'output_HF_relative':output(part),'automatic_retry':False,'read_budget':{'sha_bytes_max':16<<30,'journal_bytes_max':16<<20,'worker_extra_bytes_max':64<<20,'worker_extra_enforcement':'SOFT_SAMPLED_AT_SHA_EVENTS','unit':'PER_FILE_SHA_PREDEBIT_AND_COMPLETE','parent_identity_nested_SHA':'UNKNOWN_OUTSIDE_HASH_READ_JOURNAL_BEFORE_JOB_GATE'},'execution_private_worker_config_sha256':execution.PROFILE_SHA,
  'execution_environment':environment(),'parent':parent,'videos':rows,'page_layout':LAYOUT,'interpretation':INTERPRETATION,
  'repo_sources':[base.binding(ROOT/x,ROOT) for x in sorted(paths)]}
 unique=inputs(p);p['known_hash_work']={'unique_input_files':len(unique),'unique_input_bytes':sum(b['bytes'] for _,b in unique),'extra_video_hash_calls':1,'extra_video_bytes':sum(v['video']['bytes'] for v in rows),'additional_output_hashes':'ACTUALLY_JOURNALED_COUNTS_NOT_PREASSERTED'};return p

def environment():
 info={}
 for name in ('numpy','opencv-python-headless'):
  d=importlib.metadata.distribution(name);meta=d.read_text('METADATA');record=d.read_text('RECORD')
  need(meta is not None and record is not None,'Package metadata missing')
  info[name]={'version':d.version,'metadata_sha256':hashlib.sha256(meta.encode()).hexdigest(),'record_sha256':hashlib.sha256(record.encode()).hexdigest()}
 py=(ROOT/'.venv/bin/python').resolve();need(Path(sys.executable).resolve()==py,'Repository venv required')
 return {'python_sha256':base.sha(py),'distributions':info}


def fingerprint(p):return base.state_sha({k:v for k,v in p.items() if k not in ('status','assets_finalized')})

def validate(p,hf,frozen=False):
 part=p['part'];need(type(part) is int and part in (0,1),'part differs')
 need(p['id']==identity(part) and p['resources']==RESOURCES and p['limits']==LIMITS and p['output_HF_relative']==output(part) and p['automatic_retry'] is False and p['execution_private_worker_config_sha256']==execution.PROFILE_SHA and p['page_layout']==LAYOUT and p['interpretation']==INTERPRETATION,'scope differs')
 need(p['parent']['protocol']==line.parent_entry(part)['protocol'],'bound parent or reused visual differs')
 need([(v['ordinal'],v['mode']) for v in p['videos']]==[(line.ORDINALS[part],m) for m in ('high_y_reuse',)],'video coverage differs')
 refs=p['repo_sources'];paths=[v['path'] for v in refs];need(len(paths)==len(set(paths)) and all(set(v)=={'path','bytes','sha256'} for v in refs),'source bindings malformed')
 required={SELF,TEST,'tools/e0_chipmunk_high_y_quality_lineage.py','tests/test_e0_chipmunk_high_y_quality_lineage.py','tools/e0_chipmunk_p64_quality_execution.py','tests/test_e0_chipmunk_p64_quality_execution.py','tools/e0_chipmunk_high_y_metric_verify.py','tools/e0_chipmunk_high_y_metric_jobs.py',line.parent_entry(part)['protocol']['path'],line.parent_entry(part)['owner_source']['path']}
 parent_meta=base.read(ROOT/line.parent_entry(part)['protocol']['path']);expected={v['path'] for v in parent_meta['repo_sources']}|required|{'tools/e0_s2_sm120_fixed_depth_visual_cpu_v2.py','tests/test_e0_s2_sm120_fixed_depth_visual_cpu_v2.py','tools/e0_s2_sm120_fixed_depth_motion_metrics.py','tools/e0_s2_new12_dense_visual_cpu.py','tools/e0_s2_new12_dense_motion_metrics.py','tools/worker_identity.py','tools/e0_chipmunk_high_y_metric_freeze.py','tools/e0_chipmunk_high_y_metric_adapter.py','tools/e0_sharq_wan_metric_read_budget.py','tests/test_e0_sharq_wan_metric_read_budget.py'}|{line.factorial.CONFIGS[part],line.factorial.OWNER,line.prior.parent_path(part),line.prior.parent_entry(part)['owner_source']['path']}|execution.HELPERS|{'tools/e0_chipmunk_p64_quality_lineage.py','tests/test_e0_chipmunk_p64_quality_lineage.py','tools/e0_chipmunk_p64_metric_verify.py','tools/e0_chipmunk_p64_metric_adapter.py','tools/e0_chipmunk_p64_visual_cpu.py'}
 expected |= {v['path'] for v in base.read(ROOT/f'config/e0_chipmunk_p64_visual_20261030_part{part}_v1.json')['repo_sources']} | {f'config/e0_chipmunk_p64_visual_20261030_part{part}_v1.json'}
 need(expected==set(paths),'source closure differs')
 need(p['read_budget']=={'sha_bytes_max':16<<30,'journal_bytes_max':16<<20,'worker_extra_bytes_max':64<<20,'worker_extra_enforcement':'SOFT_SAMPLED_AT_SHA_EVENTS','unit':'PER_FILE_SHA_PREDEBIT_AND_COMPLETE','parent_identity_nested_SHA':'UNKNOWN_OUTSIDE_HASH_READ_JOURNAL_BEFORE_JOB_GATE'},'read budget differs')
 unique=inputs(p);video_bytes=sum(v['video']['bytes'] for v in p['videos']);need(p['known_hash_work']=={'unique_input_files':len(unique),'unique_input_bytes':sum(b['bytes'] for _,b in unique),'extra_video_hash_calls':1,'extra_video_bytes':video_bytes,'additional_output_hashes':'ACTUALLY_JOURNALED_COUNTS_NOT_PREASSERTED'},'hash work differs')
 if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True,'frozen scope required')

def authorize(p,digest,hf):
 validate(p,hf,True);a=base.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']]
 need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(rows)==1 and rows[0].get('protocol_sha256')==digest
  and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and rows[0].get('resources')==RESOURCES
  and rows[0].get('limits')==LIMITS and rows[0].get('max_gpu_cards')==0,'Exact ACTIVE scope required')
 execution.require_review(p,rows[0],base,fingerprint,('launch_spec_sha256',))
 return rows[0]

def inputs(p):
 rows=[('HF',p['parent']['high_p64_parent'][k]) for k in ('RUN','exit','verification_RUN','verification_exit')]+[('REPO',p['parent']['high_p64_parent']['protocol'])]+[('REPO',b) for b in p['repo_sources']]+[('REPO',p['parent']['protocol'])]+[('HF',p['parent'][x]) for x in ('RUN','exit','verification_RUN','verification_exit')]
 nested=p['parent']['high_p64_parent']['previous_parent'];rows.extend([('REPO',nested['protocol'])]+[('HF',nested[k]) for k in ('RUN','exit','verification_RUN','verification_exit')])
 for v in p['videos']:
  rows += [('HF',v['video']),('HF',v['generation']),('HF',v['case']['image'])]+[('HF',x) for x in v['states'].values()]
 unique={}
 for root,b in rows:
  v={k:b[k] for k in ('path','bytes','sha256')};key=(root,v['path']);need(key not in unique or unique[key]==v,'Binding conflict');unique[key]=v
 return [(k[0],v) for k,v in sorted(unique.items())]

def _run(a):
 hf=a.hf_root.resolve();p=base.read(a.protocol);sha=base.sha(a.protocol);scope=authorize(p,sha,hf)
 execution.verify_job(a,p,scope,base,'visual','run',SELF,RESOURCES)
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2 and not {'torch','cv2','numpy'}&set(sys.modules),'Fresh CPU worker required')
 from worker_identity import require_worker,private_config_path
 require_worker();need(base.sha(private_config_path())==p['execution_private_worker_config_sha256'],'Execution worker differs');out=(hf/output(p['part'])).resolve();need(out.is_relative_to(hf) and not out.exists(),'Fresh output required');out.mkdir(parents=True)
 started=time.monotonic();report={'id':p['id'],'protocol_sha256':sha,'status':'STARTED','part':p['part'],'parent':p['parent'],
  'automatic_retry':False,'job_spec_sha256':scope['launch_spec_sha256'],'videos':[],'visual_review':'PENDING_SEPARATE_AI_INSPECTION','human_full_playback_review':'NOT_PERFORMED','excess_frames_detected':0}
 def persist():
  base.write(out/'RUN.json',report);need(sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<=RESOURCES['new_HF_bytes'],'Output cap')
 ledger=visual.JournalLedger(report,out/'OPERATIONS.jsonl');persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Visual wall')));signal.alarm(RESOURCES['wall_seconds'])
 no_torch=visual.old.NoTorch();sys.meta_path.insert(0,no_torch)
 try:
  records=[]
  for root,b in inputs(p):
   directory=ROOT if root=='REPO' else hf;path=directory/b['path'];before=visual.stat(path);base.bound(directory,b);need(visual.stat(path)==before,'Changed input');records.append({'root':root,'binding':b,'stat':before})
  need(environment()==p['execution_environment'],'Execution environment differs');need(line.visual_rows(p['parent'],hf,p['part'])==p['videos'],'Parent changed')
  base.write(out/'INPUT_PROOF.json',{'id':p['id'],'protocol_sha256':sha,'records':records});report['input_proof']=base.binding(out/'INPUT_PROOF.json',hf)
  import cv2,numpy as np
  cv2.setNumThreads(2);need('torch' not in sys.modules,'Torch forbidden')
  report['environment']={'opencv_version':cv2.__version__,'numpy_version':np.__version__,'opencv_origin':str(Path(cv2.__file__).resolve()),
   'numpy_origin':str(Path(np.__file__).resolve()),'CPU_affinity':sorted(os.sched_getaffinity(0))}
  for source in p['videos']:
   item={**copy.deepcopy(source),'pages':[],'decoded_frames':0,'status':'STARTED'};report['videos'].append(item);persist()
   path=base.bound(hf,source['video']);before=visual.stat(path)
   def open_video():
    cap=cv2.VideoCapture(str(path));need(cap.isOpened(),'Video open failed');return cap
   cap=ledger.call('videos',open_video)
   try:
    geom={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          'fps':cap.get(cv2.CAP_PROP_FPS),'reported_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    need(geom=={'width':832,'height':464,'fps':16.0,'reported_frames':81},'Video geometry differs');item['decoded_geometry']=geom
    renderer=visual.Sheets(cv2,np)
    class Reader:
     def read(self):
      ok,frame=cap.read()
      if ok:item['decoded_frames']+=1
      if item['decoded_frames']>81:report['excess_frames_detected']+=1
      return ok,frame
    def page(img,index,positions):
     ok,enc=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,90]);need(ok,'JPEG encode failed');data=enc.tobytes()
     need(len(data)<=3<<20 and sum(x.stat().st_size for x in out.rglob('*') if x.is_file())+len(data)+(4<<20)<=RESOURCES['new_HF_bytes'],'JPEG output cap')
     pathout=out/'pages'/f'source{source["ordinal"]}_{source["mode"]}_page{index}.jpg';pathout.parent.mkdir(exist_ok=True)
     with pathout.open('xb') as f:f.write(data)
     labels=[None if n is None else {'source_ordinal':source['ordinal'],'mode':source['mode'],'frame':n,'timestamp_seconds':n/16} for n in positions]
     item['pages'].append({'page':index,'frames':positions,'labels':labels,'jpeg':base.binding(pathout,hf),'width':1920,'height':1414,'quality':90})
    visual.extract(Reader(),renderer,source,ledger,page,persist);need(item['decoded_frames']==81 and visual.stat(path)==before,'Decoded or changed input')
    item['status']='COMPLETE';persist()
   finally:cap.release()
  ledger.finish();visual.check_stats(records,hf);need(len(report['videos'])==1 and report['excess_frames_detected']==0,'Incomplete decoding')
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;need(rss<=RESOURCES['max_RSS_bytes'],'RSS exceeded')
  manifest={'schema':'ffnds.chipmunk.high_y.all81.contact_sheets.v1','id':p['id'],'protocol_sha256':sha,'parent':p['parent'],'videos':report['videos'],'interpretation':INTERPRETATION}
  base.write(out/'MANIFEST.json',manifest);report.update(status='COMPLETE_CHIPMUNK_HIGH_Y_ALL81_CONTACT_SHEETS_NOT_HUMAN_REVIEW',manifest=base.binding(out/'MANIFEST.json',hf),max_RSS_bytes=rss)
 except BaseException as e:report.update(status='FAILED_VISUAL_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  signal.alarm(0);ledger.close();report['journal']=base.binding(out/'OPERATIONS.jsonl',hf)
  need(visual.replay_journal(out/'OPERATIONS.jsonl')=={k:report[k] for k in ('predebited','invoked','completed')},'Ledger differs')
  report['wall_seconds']=time.monotonic()-started;persist();sys.meta_path.remove(no_torch)

def run(a):
 from e0_sharq_wan_metric_read_budget import HashReads
 saved=execution.verify_job;state={'guard':None}
 def checked_job(args,plan,scope,base_module,kind,phase,source,limits):
  value=saved(args,plan,scope,base_module,kind,phase,source,limits);worker=(args.hf_root/plan['output_HF_relative']).parent;state['guard']=HashReads(base_module,worker/'HASH_READS.jsonl','VISUAL');state['guard'].__enter__();return value
 execution.verify_job=checked_job
 try:return _run(a)
 finally:
  import sys
  execution.verify_job=saved;guard=state['guard']
  if guard is not None:
   typ,value,tb=sys.exc_info();failure=None
   try:guard.__exit__(typ,value,tb)
   except BaseException as error:failure=error
   p=base.read(a.protocol);out=a.hf_root/p['output_HF_relative']/'RUN.json'
   if out.is_file():
    report=base.read(out);data=guard.path.read_bytes();report['sha_read_ledger']={'journal':{'path':str(guard.path.resolve().relative_to(a.hf_root.resolve())),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},'predebited_bytes':guard.predebited,'completed_bytes':guard.completed,'inflight':guard.inflight,'scope':'all visual base SHA calls including repeated video and JPEG identities; small metadata separately UNKNOWN'};
    if failure:report.update(status='FAILED_HASH_LEDGER_PRESERVED_NO_RETRY',error_type=type(failure).__name__,error=str(failure)[:500])
    base.write(out,report)
   if failure:raise failure

def _verify_closed(a):
 hf=a.hf_root.resolve();p=base.read(a.protocol);validate(p,hf,True);sha=base.sha(a.protocol);out=hf/output(p['part'])
 need(a.run.resolve()==out/'RUN.json' and a.exit.resolve()==out.parent/'EXIT_RECEIPT.json','Incorrect run/exit')
 binding=base.binding(a.run,hf);r=base.read(a.run);receipt=base.read(a.exit);base.exited(receipt,binding,sha,p['id'])
 need(receipt.get('spec_sha256')==r.get('job_spec_sha256'),'Supervisor spec mismatch')
 need(r['id']==p['id'] and r['protocol_sha256']==sha and r['status']=='COMPLETE_CHIPMUNK_HIGH_Y_ALL81_CONTACT_SHEETS_NOT_HUMAN_REVIEW'
  and all(r[k]==LIMITS for k in ('predebited','invoked','completed')) and 0<=r['wall_seconds']<=RESOURCES['wall_seconds']
  and 0<=r['max_RSS_bytes']<=RESOURCES['max_RSS_bytes'] and r['excess_frames_detected']==0,'Incomplete visual scope')
 proof=base.read_bound(hf,r['input_proof']);need([(z['root'],z['binding']) for z in proof['records']]==inputs(p),'Input proof differs')
 visual.check_stats(proof['records'],hf);need(line.visual_rows(p['parent'],hf,p['part'])==p['videos'],'Parent changed')
 need(visual.replay_journal(base.bound(hf,r['journal']))=={k:r[k] for k in ('predebited','invoked','completed')},'Ledger differs')
 manifest=base.read_bound(hf,r['manifest']);need(manifest.get('schema')=='ffnds.chipmunk.high_y.all81.contact_sheets.v1' and manifest['videos']==r['videos'] and manifest['parent']==p['parent']
  and manifest['interpretation']==INTERPRETATION and len(manifest['videos'])==1,'Manifest differs')
 paths=[]
 for actual,expected in zip(manifest['videos'],p['videos']):
  need(all(actual.get(k)==v for k,v in expected.items()) and actual['decoded_frames']==81 and actual['status']=='COMPLETE'
   and actual['decoded_geometry']=={'width':832,'height':464,'fps':16.0,'reported_frames':81} and len(actual['pages'])==2,'Video identity differs')
  for i,page in enumerate(actual['pages']):
   positions=visual.old.page_positions(i);labels=[None if n is None else {'source_ordinal':actual['ordinal'],'mode':actual['mode'],'frame':n,'timestamp_seconds':n/16} for n in positions]
   need(page['page']==i and page['frames']==positions and page['labels']==labels and page['width']==1920 and page['height']==1414
    and page['quality']==90 and page['jpeg']['bytes']<=3<<20,'Page geometry or labels differ')
   path=base.bound(hf,page['jpeg']);need(path==(out/'pages'/f'source{actual["ordinal"]}_{actual["mode"]}_page{i}.jpg').resolve(),'Wrong page path');paths.append(path)
 need(len(paths)==len(set(paths))==len({(x.stat().st_dev,x.stat().st_ino) for x in paths})==2,'Page alias')
 return {'id':p['id'],'status':'PASS_CLOSED_CHIPMUNK_HIGH_Y_VISUAL','RUN':binding,'exit':base.binding(a.exit,hf),'manifest':r['manifest']}

def verify_closed(a):
 need(globals().get('_authorized_verifier',False),'independent registered verifier required');
 from e0_sharq_wan_metric_read_budget import require_complete
 require_complete(base,a.hf_root,base.read(a.run));return _verify_closed(a)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('draft','inspect','run','verify-closed'));ap.add_argument('--part',type=int);ap.add_argument('--protocol',type=Path)
 ap.add_argument('--out',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);base.write(a.out,draft(a.hf_root,a.part))
 elif a.action=='inspect':validate(base.read(a.protocol),a.hf_root);print('STATIC_ONLY_NO_MODEL')
 elif a.action=='run':run(a)
 else:print(json.dumps(verify_closed(a),indent=2))
if __name__=='__main__':main()
