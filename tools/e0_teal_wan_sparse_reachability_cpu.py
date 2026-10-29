"""Bounded sampled-row TEAL mask reachability; no model, W2, or GPU.

Requires exact independent reviewer recheck and ACTIVE before run. Samples give
upper bounds on full-M cold columns, not actual contiguous token tiles.
"""
import argparse, hashlib, importlib.util, json, math, os, resource, signal, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('teal_prior',ROOT/'tools/e0_teal_wan_all160_cpu_local.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
import e0_teal_llama3_author_checkpoint_convert as envutil
native=old.native;HF=Path('/home/xiang/.cache/huggingface')
ID='E0_TEAL_WAN_MASK_REACHABILITY_CPU_20261030';SELF='tools/e0_teal_wan_sparse_reachability_cpu.py';TEST='tests/test_e0_teal_wan_sparse_reachability_cpu.py'
THRESHOLD='config/e0_teal_wan_all160_calibration_thresholds_20261029_v1.json';TSHA='e6a9cefad04d4482f9b13ae90feefa8d24eee13323900c8a3061626f98782e13'
OUT='ffnds-derived/e0_teal_wan_mask_reachability_cpu_20261030/attempt1/worker/results'
SOURCES=(894,256);D=13824;ROWS=256
RES={'GPU':0,'CPU_threads':4,'wall_seconds':1800,'outer_seconds':1920,'host_RSS_max_bytes':8589934592,'new_HF_bytes':67108864,'max_payload_hash_bytes':5631065600,'max_h_tensor_bytes':2264924160,'max_row_tensor_bytes':655360,'automatic_retry':False}
LIMITS={'packets':320,'site_rate_evaluations':640,'output_site_records':320,'W2_reads':0,'GPU_calls':0,'native_FFN_calls':0,'training_updates':0}
DESIGN={'sources':[894,256],'source_role':'historically exposed development teacher Dense seed42','sites':'4 steps x 40 blocks','sample_rows':256,'population_rows':31668,'h_width':13824,'thresholds':[0.25,0.5],'cold_column':'no kept value in 256 real rows; upper bound for the same-state full-M cold-column fraction','structure':'per-row K4 at most two kept, including 0/1','forbidden':'no pseudo-contiguous tile; no imputed full-M, no W2/model/GPU','cpu_affinity':[4,5,6,7]}
need=native.need

def threshold_map():
 p=ROOT/THRESHOLD;need(native.sha(p)==TSHA,'threshold SHA');x=native.read(p);need(x['schema']=='ffnds.teal.wan.calibration.thresholds.v1' and x['source']=='SIX_CALIBRATION_SOURCES_ONLY' and x['rates']==[0.25,0.5],'threshold identity');rows=x['threshold_rows'];need(len(rows)==160 and {(r['step'],r['block']) for r in rows}=={(s,b) for s in range(4) for b in range(40)},'threshold sites');need(all(r['expert']==('high' if r['step']<2 else 'low') and len(r['thresholds'])==2 and all(math.isfinite(t) and t>=0 for t in r['thresholds']) for r in rows),'threshold values');return {(r['step'],r['block']):r['thresholds'] for r in rows}

def draft():
 result,manifest=old.parent(HF);cases=[next(c for c in manifest['cases'] if c['native_source_ordinal']==i) for i in SOURCES];need(len({c['image_sha256'] for c in cases})==2 and all(c['role']=='development' and c['source_state']=='teacher_input' and len(c['packets'])==160 for c in cases),'source cohort')
 for c in cases:need({(x['step'],x['block']) for x in c['packets']}=={(s,b) for s in range(4) for b in range(40)},'site coverage')
 need(sum(x['payload']['bytes'] for c in cases for x in c['packets'])==RES['max_payload_hash_bytes'],'payload budget')
 previous=old.draft(HF);src={x['path'] for x in previous['repo_sources']}|{SELF,TEST,THRESHOLD,'tools/e0_teal_wan_all160_video_v3.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py'}
 threshold_map();return {'id':ID,'status':'FROZEN_PENDING_REGISTRATION','review':None,'resources':RES,'limits':LIMITS,'design':DESIGN,'output':OUT,'cases':cases,'manifest':result['manifest'],'audit_protocol':result['protocol'],'audit_RUN':result['RUN'],'audit_exit':result['exit'],'threshold':native.binding(ROOT/THRESHOLD,ROOT),'environment':old.environment(),'private_worker_sha256':previous['private_worker_sha256'],'repo_sources':[native.binding(ROOT/x,ROOT) for x in sorted(src)]}

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(p):
 need(p.get('id')==ID and p.get('status')=='FROZEN_PENDING_REGISTRATION' and p.get('review') is None and p.get('resources')==RES and p.get('limits')==LIMITS and p.get('design')==DESIGN and p.get('output')==OUT and p.get('threshold',{}).get('sha256')==TSHA,'scope identity')
 need([c['native_source_ordinal'] for c in p['cases']]==list(SOURCES) and all(len(c['packets'])==160 for c in p['cases']),'cases')
 need(p==draft(),'corpus/source/runtime closure changed')

def authorize(p,digest):
 validate(p);a=native.read(ROOT/'research/ACTIVE_RESEARCH.json');sc=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(sc)==1,'ACTIVE missing');s=sc[0];need(s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==RES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==0,'ACTIVE mismatch')
 r=native.read(native.bound(ROOT,s['review']));need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('reviewer_disposition') in ('REVISE','APPROVE') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('responses_to_review') and r.get('adversarial_counterexample') and r.get('approval_scope_fingerprint')==fingerprint(p),'review mismatch')
 need(r.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['repo_sources']} and r.get('input_cohort_manifest_sha256')==p['manifest']['sha256'],'review bindings')
 for key in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,r[key])
 return s

def predebit_evaluation(report,persist,rate):
 need(report['predebited']['site_rate_evaluations']<LIMITS['site_rate_evaluations'],'site evaluation budget')
 report['predebited']['site_rate_evaluations']+=1;report['inflight']['rate']=rate;report['inflight']['rate_progress']='UNKNOWN_UNTIL_COMPLETE';persist()

def complete_evaluation(report,persist):
 report['completed']['site_rate_evaluations']+=1;report['inflight']['rate_progress']='COMPLETE';persist()

def compute(h,t,torch):
 need(h.shape==(ROWS,D) and h.dtype==torch.bfloat16 and bool(torch.isfinite(h).all()),'h shape/dtype/finite')
 keep=h.abs()>t; counts=keep.reshape(ROWS,D//4,4).sum(dim=-1);bins=torch.bincount(counts.reshape(-1),minlength=5);eligible=counts<=2
 return {'threshold':t,'elements':h.numel(),'deleted':int((~keep).sum()),'preexisting_zero':int((h==0).sum()),'threshold_ties':int((h.abs()==t).sum()),'cold_columns_sample_upper_bound':int((~keep.any(dim=0)).sum()),'hidden_columns':D,'K4_nonzero_histogram':[int(x) for x in bins.tolist()],'K4_groups':counts.numel(),'K4_groups_eligible_at_most2':int(eligible.sum()),'rows_entirely_eligible':int(eligible.all(dim=1).sum()),'rows':ROWS}

def run(a):
 need(not {'torch','numpy','safetensors'} & set(sys.modules),'fresh worker required');p=native.read(a.protocol);digest=native.sha(a.protocol);scope=authorize(p,digest);need(a.job_spec and native.sha(a.job_spec)==scope.get('launch_spec_sha256'),'spec mismatch');j=native.read(a.job_spec);argv=j['argv'];need(len(argv)==18 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'argv');profile=Path(argv[1].split('=',1)[1]);need(native.sha(profile)==p['private_worker_sha256'],'worker profile')
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','taskset','-c','4-7',p['environment']['python_path'],'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())]
 out=HF/OUT;need(argv==expected and j['id']==ID and j['protocol_sha256']==digest and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and j['hf_root']==str(HF) and Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'job identity');need(sys.flags.isolated and sorted(os.sched_getaffinity(0))==[4,5,6,7] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and old.environment()==p['environment'],'worker environment')
 for b in p['repo_sources']:native.bound(ROOT,b)
 old.parent(HF);import worker_identity;worker_identity.require_worker();need(not out.exists(),'fresh attempt');out.mkdir(parents=True,mode=0o700);start=time.monotonic();cpu=time.process_time();r={'id':ID,'status':'STARTED','protocol_sha256':digest,'launch_spec_sha256':scope['launch_spec_sha256'],'automatic_retry':False,'predebited':{'payload_hash_bytes':0,'h_tensor_bytes':0,'row_tensor_bytes':0,'packets':0,'site_rate_evaluations':0},'completed':{'payload_hash_bytes':0,'h_tensor_bytes':0,'row_tensor_bytes':0,'packets':0,'site_rate_evaluations':0},'other_input_bytes':'metadata, manifest, source and package identity reads separately unmetered; in-memory raw tensor SHA reads additional memory bandwidth','GPU_calls':0,'W2_reads':0,'native_FFN_calls':0}
 def persist():old.write(out/'RUN.json',r)
 persist();stop=threading.Event();monitor={'samples':0,'peak_bytes':0,'exceeded':False,'error':None}
 def watch():
  try:
   while not stop.wait(2):
    val=int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024;monitor['samples']+=1;monitor['peak_bytes']=max(monitor['peak_bytes'],val)
    if val>RES['host_RSS_max_bytes']:monitor['exceeded']=True;os.kill(os.getpid(),signal.SIGUSR1);return
  except BaseException as e:monitor['error']=type(e).__name__;os.kill(os.getpid(),signal.SIGUSR1)
 signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(MemoryError('RSS watchdog')));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall')));signal.alarm(RES['wall_seconds']);thread=threading.Thread(target=watch,daemon=True);thread.start()
 try:
  import torch
  from safetensors import safe_open
  torch.set_num_threads(4);torch.set_num_interop_threads(1);need(not torch.cuda.is_initialized(),'GPU initialized');thresholds=threshold_map();records=[]
  for case in p['cases']:
   for item in sorted(case['packets'],key=lambda x:(x['step'],x['block'])):
    step=item['step'];block=item['block'];expert='high' if step<2 else 'low';pb=item['payload'];meta=native.read(native.bound(HF,item['metadata']))
    need(meta['native_source_ordinal']==case['native_source_ordinal'] and meta['source_id']==case['source_id'] and meta['image_sha256']==case['image_sha256'] and meta['prompt_sha256']==case['prompt_sha256'] and meta['step']==step and meta['block']==block and meta['expert']==expert and meta['source_state']=='teacher_input' and meta['grid']==[21,29,52] and len(meta['selected_rows'])==ROWS and len(set(meta['selected_rows']))==ROWS and all(0<=x<31668 for x in meta['selected_rows']),'metadata identity')
    r['inflight']={'source':case['native_source_ordinal'],'step':step,'block':block,'packet_hash_progress':'UNKNOWN_UNTIL_COMPLETE','tensor_read_progress':'NOT_STARTED'};r['predebited']['packets']+=1;r['predebited']['payload_hash_bytes']+=pb['bytes'];need(r['predebited']['packets']<=LIMITS['packets'] and r['predebited']['payload_hash_bytes']<=RES['max_payload_hash_bytes'],'packet budget');persist();candidate=(HF/pb['path']).resolve();need(candidate.is_relative_to(HF) and candidate.is_file(),'payload path');state=envutil.statid(candidate.stat());path=native.bound(HF,pb);need(envutil.statid(path.stat())==state,'payload changed during hash');r['completed']['payload_hash_bytes']+=pb['bytes'];r['inflight']['packet_hash_progress']='COMPLETE'
    hbytes=ROWS*D*2;rowbytes=ROWS*8;r['predebited']['h_tensor_bytes']+=hbytes;r['predebited']['row_tensor_bytes']+=rowbytes;need(r['predebited']['h_tensor_bytes']<=RES['max_h_tensor_bytes'] and r['predebited']['row_tensor_bytes']<=RES['max_row_tensor_bytes'],'tensor budget');r['inflight']['tensor_read_progress']='UNKNOWN_UNTIL_COMPLETE';persist()
    with safe_open(str(path),framework='pt',device='cpu') as f:h=f.get_tensor('h');rows=f.get_tensor('rows')
    need(old.sha_tensor(h,torch)==item['tensor_metadata']['h']['raw_tensor_sha256'] and old.sha_tensor(rows,torch)==item['tensor_metadata']['rows']['raw_tensor_sha256'] and rows.dtype==torch.int64 and rows.tolist()==meta['selected_rows'],'tensor identity');r['completed']['h_tensor_bytes']+=hbytes;r['completed']['row_tensor_bytes']+=rowbytes;r['inflight']['tensor_read_progress']='COMPLETE'
    rates={}
    for i,t in enumerate(thresholds[step,block]):
     rate=str((0.25,0.5)[i]);predebit_evaluation(r,persist,rate);rates[rate]=compute(h,t,torch);complete_evaluation(r,persist)
    need(envutil.statid(path.stat())==state,'payload changed while reading h');records.append({'source_ordinal':case['native_source_ordinal'],'step':step,'block':block,'expert':expert,'rates':rates});r['completed']['packets']+=1;r['inflight']=None;persist()
  need(len(records)==320 and r['completed']['site_rate_evaluations']==640,'coverage incomplete');summary={}
  for source in SOURCES:
   summary[str(source)]={}
   for rate in ('0.25','0.5'):
    points=[x['rates'][rate] for x in records if x['source_ordinal']==source];need(len(points)==160,'source summary coverage')
    metrics={'deleted_fraction':'deleted','cold_column_upper_bound':'cold_columns_sample_upper_bound','eligible_K4_fraction':'K4_groups_eligible_at_most2','all_K4_eligible_row_fraction':'rows_entirely_eligible'};den={'deleted_fraction':'elements','cold_column_upper_bound':'hidden_columns','eligible_K4_fraction':'K4_groups','all_K4_eligible_row_fraction':'rows'}
    summary[str(source)][rate]={name:sum(v[key]/v[den[name]] for v in points)/len(points) for name,key in metrics.items()}
  old.write(out/'SITE_RESULTS.json',{'id':ID,'records':records,'source_site_equal_weight_summary':summary,'interpretation':'sampled-row bound; not actual full-M or M16 tiles'});r['site_results']=native.binding(out/'SITE_RESULTS.json',HF);r['status']='COMPLETE_SAMPLED_ROW_REACHABILITY_NOT_FULL_M'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);stop.set();thread.join(10);r.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,rss_watchdog=monitor)
  if monitor['samples']<1 or monitor['error'] or monitor['exceeded'] or thread.is_alive() or r['peak_RSS_bytes']>RES['host_RSS_max_bytes']:r['status']='FAILED_MONITOR_PRESERVED'
  try:usage=envutil.worker_usage(out.parent);used=usage['apparent_bytes'];r['worker_usage']=usage
  except BaseException as e:r.update(status='FAILED_OUTPUT_SCAN',output_scan_error=type(e).__name__);persist();raise
  if used>RES['new_HF_bytes']-(1<<20):r['status']='FAILED_OUTPUT_BUDGET_PRESERVED'
  persist()
  if r['status'] in ('FAILED_MONITOR_PRESERVED','FAILED_OUTPUT_BUDGET_PRESERVED'):raise RuntimeError('resource bound failed')

def inspect_closed(protocol):
 p=native.read(protocol);validate(p);rr=HF/OUT/'RUN.json';ee=(HF/OUT).parent/'EXIT_RECEIPT.json';r=native.read(rr);e=native.read(ee)
 need(r.get('id')==ID and r.get('status')=='COMPLETE_SAMPLED_ROW_REACHABILITY_NOT_FULL_M' and r.get('protocol_sha256')==native.sha(protocol) and e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==native.sha(rr) and e.get('protocol_sha256')==r['protocol_sha256'] and e.get('spec_sha256')==r['launch_spec_sha256'],'actual EXIT0 required')
 need(r['predebited']=={k:r['completed'][k] for k in r['predebited']} and r['completed']['packets']==320 and r['completed']['site_rate_evaluations']==640 and r['completed']['payload_hash_bytes']==RES['max_payload_hash_bytes'] and r['completed']['h_tensor_bytes']==RES['max_h_tensor_bytes'] and r['completed']['row_tensor_bytes']==RES['max_row_tensor_bytes'],'coverage/budget incomplete')
 result=native.read(native.bound(HF,r['site_results']));need(len(result['records'])==320 and {(x['source_ordinal'],x['step'],x['block']) for x in result['records']}=={(source,step,block) for source in SOURCES for step in range(4) for block in range(40)},'result sites missing')
 need(r['GPU_calls']==r['W2_reads']==r['native_FFN_calls']==0 and r['wall_seconds']<=RES['wall_seconds'] and r['rss_watchdog']['samples']>=1 and not r['rss_watchdog']['error'] and not r['rss_watchdog']['exceeded'] and r['worker_usage']['apparent_bytes']<=RES['new_HF_bytes'],'resource evidence')
 return {'id':ID,'status':'PASS_CLOSED_SAMPLED_ROW_REACHABILITY_NOT_FULL_M','RUN':native.binding(rr,HF),'exit':native.binding(ee,HF),'site_results':r['site_results']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['draft','inspect','run','inspect-closed']);ap.add_argument('--out',type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out and not a.out.exists(),'fresh draft');old.write(a.out,draft())
 elif a.action=='inspect':p=native.read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'resources':RES,'limits':LIMITS},indent=2))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol),indent=2))
if __name__=='__main__':main()
