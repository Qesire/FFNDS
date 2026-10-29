"""CPU-only paired asset staging for source911, both experts, block20.

Draft reads only report/receipt bytes. Run requires exact reviewed ACTIVE,
private job, and real parent receipts. No model forward, GPU or training.
"""
import argparse,hashlib,importlib.metadata,json,os,resource,signal,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_sharq_wan_ffn_asset_plan as plan
import e0_sharq_wan_ffn_asset_extract as extract
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_llama3_author_checkpoint_convert as output
HF=Path('/home/xiang/.cache/huggingface');ID='E0_SHARQ_WAN_FFN_PAIRED_ASSETS_CPU_20261030'
SELF='tools/e0_sharq_wan_ffn_assets.py';OUT='ffnds-derived/e0_sharq_wan_ffn_paired_assets_20261030/attempt1/worker/results'
NV_PROBE=ROOT/'research/20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.json'
SOURCES=(SELF,'tools/e0_sharq_wan_ffn_asset_plan.py','tools/e0_sharq_wan_ffn_asset_extract.py','tests/test_e0_sharq_wan_ffn_assets.py','tests/test_e0_sharq_wan_ffn_asset_plan.py','tests/test_e0_sharq_wan_ffn_asset_extract.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/worker_identity.py','tools/durable_worker_job.py')
RES={'GPU':0,'CPU_threads':4,'wall_seconds':3600,'outer_seconds':3720,'host_RSS_max_bytes':8589934592,'new_HF_bytes':2147483648,'max_payload_hash_bytes':8388608,'max_tensor_bytes':805306368,'max_tensor_ranges':64,'max_bundle_hash_bytes':1073741824,'automatic_retry':False}
DESIGN={'source':911,'block':20,'cases':{'high':1,'low':2},'rows':32,'inputs':'actual matched V1, a, h, Y, gate, rows; original BF16 master W1/W2 and NVFP4 checkpoint W1/W2 from same expert/block','no_model_GPU':'CPU tensor staging only, no model forward or GPU','master_identity':'prior full-SHA parent receipts; header+fd/path stat; noncooperative writers not excluded'}
EXPECTED_RANGES=40;EXPECTED_TENSOR_BYTES=730505760;EXPECTED_PACKET_HASH_BYTES=6182512
need=base.need

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bound(root,b):return base.bound(root,b)
def runtime_identity():
 packages={}
 for name in ('torch','numpy','safetensors'):
  d=importlib.metadata.distribution(name);metadata=d.read_text('METADATA');record=d.read_text('RECORD');need(metadata is not None and record is not None,'package metadata missing')
  packages[name]={'version':d.version,'metadata_sha256':hashlib.sha256(metadata.encode()).hexdigest(),'record_sha256':hashlib.sha256(record.encode()).hexdigest()}
 return {'python_sha256':base.sha(Path(sys.executable).resolve()),'packages':packages,'scope':'metadata and RECORD; package implementation bytes not fully hashed'}
def verify_parent(run,exit,binding):
 need(exit.get('process_exited') is True and exit.get('child_returncode')==0 and exit.get('timeout') is False and exit.get('automatic_retry') is False and exit.get('run_sha256')==binding['sha256'] and exit.get('id')==run.get('id'),'parent actual EXIT0 required')

def parent_receipts():
 bridge=base.read(plan.BRIDGE);nv=base.read(plan.NV);raw=base.read(plan.RAW);bp=bridge['closed_verification'];np=nv;rp=raw['execution_raw_master_CPU_proof'];cp=base.read(NV_PROBE)['CPU_assets']
 for proof,rkey,ekey in ((bp,'RUN','exit'),(np,'RUN','exit'),(rp,'RUN','launch_exit'),(cp,'RUN','exit')):
  rr=base.read(bound(HF,proof[rkey]));ee=base.read(bound(HF,proof[ekey]));verify_parent(rr,ee,proof[rkey])
  need(ee.get('protocol_sha256')==rr.get('protocol_sha256'),'parent protocol identity')
  if 'protocol' in proof:bound(ROOT,proof['protocol']);need(ee.get('protocol_sha256')==proof['protocol']['sha256'],'parent protocol bytes differ')
 return {'bridge':bp,'nvfp4':{k:np[k] for k in ('protocol','RUN','exit')},'raw':rp,'nvfp4_stat_proof':cp}

def draft(profile):
 parents=parent_receipts();p=plan.plan();need(profile.is_file(),'worker profile missing')
 return {'id':ID,'status':'FROZEN_PENDING_REGISTRATION','review':None,'resources':RES,'design':DESIGN,'output':OUT,'selected':p['selected'],'parent_receipts':parents,'repo_sources':[base.binding(ROOT/x,ROOT) for x in SOURCES]+[base.binding(x,ROOT) for x in (plan.BRIDGE,plan.RAW,plan.NV,NV_PROBE)],'worker_config_sha256':base.sha(profile),'runtime':runtime_identity()}

def validate(p):
 need(p.get('id')==ID and p.get('status')=='FROZEN_PENDING_REGISTRATION' and p.get('review') is None and p.get('resources')==RES and p.get('design')==DESIGN and p.get('output')==OUT,'scope differs')
 need(p.get('selected')==plan.plan()['selected'] and p.get('parent_receipts')==parent_receipts(),'parent identity changed')
 need({x['path'] for x in p['repo_sources']}==set(SOURCES)|{str(x.relative_to(ROOT)) for x in (plan.BRIDGE,plan.RAW,plan.NV,NV_PROBE)},'source closure differs')

def authorize(p,d):
 validate(p);a=base.read(ROOT/'research/ACTIVE_RESEARCH.json');sc=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(len(sc)==1 and a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True,'ACTIVE required');s=sc[0]
 need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('max_gpu_cards')==0 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'),'scope differs')
 r=base.read(bound(ROOT,s['review']));need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('responses_to_review') and r.get('adversarial_counterexample'),'independent review required')
 need(r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'source review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,r[k])
 return s

class Ledger:
 def __init__(self,report,persist):self.r=report;self.persist=persist
 def reserve(self,k,n):
  limits={'tensor_bytes':RES['max_tensor_bytes'],'tensor_ranges':RES['max_tensor_ranges'],'payload_hash_bytes':RES['max_payload_hash_bytes'],'bundle_hash_bytes':RES['max_bundle_hash_bytes']};v=self.r['predebited'].get(k,0)+n;need(v<=limits[k],'predebit limit '+k);self.r['predebited'][k]=v;self.r.setdefault('inflight',{})[k]={'amount':n,'progress':'UNKNOWN_UNTIL_COMPLETE'};self.persist()
 def complete(self,k,n):self.r['completed'][k]=self.r['completed'].get(k,0)+n;self.r['inflight'][k]={'amount':n,'progress':'COMPLETE'};self.persist()

def recorded_states(p,expert):
 raw=base.read(bound(HF,p['parent_receipts']['raw']['RUN']));nr=base.read(bound(HF,p['parent_receipts']['nvfp4_stat_proof']['RUN']))
 rm=[x for x in raw['model_payloads'] if x['kind']==expert];nm=[x for x in nr['models'] if x['kind']==expert];need(len(rm)==len(nm)==1,'proof expert mismatch');rm,nm=rm[0],nm[0]
 selected=p['selected'][expert];need(rm['status']=='PASS_FRESH_FULL_SHA' and rm['sha256']==rm['expected_sha256']==selected['original_bf16']['sha256'] and rm['predebited_bytes']==rm['actual_read_bytes']==selected['original_bf16']['bytes'] and nm['sha256']==selected['nvfp4_checkpoint']['sha256'] and nm['bytes']==selected['nvfp4_checkpoint']['bytes'] and nm['cache_path']==selected['nvfp4_checkpoint']['cache_path'],'master proof differs')
 def state(v):return tuple(v[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns'))
 return state(rm['stat']),state(nm['stat']),selected['original_bf16']['header_sha256'],nm['header']['header_sha256']

def tensor_rows(expert,selected,ledger,torch,master_states,dims=(32,5120,13824)):
 row_count,dim,hidden=dims
 result={};records=[]
 def get(path,key,dtype,shape,name):
  hdr,n,state=extract.read_header(path);raw,sha=extract.checked_range(path,hdr,n,state,key,dtype,shape,ledger);types={'BF16':torch.bfloat16,'U8':torch.uint8,'F8_E4M3':torch.float8_e4m3fn,'F32':torch.float32,'I64':torch.int64};tensor=torch.frombuffer(bytearray(raw),dtype=types[dtype]).reshape(shape).clone();need(hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()==sha,'staged tensor bytes differ');result[name]=tensor;records.append({'key':key,'output_name':name,'dtype':dtype,'shape':shape,'bytes':len(raw),'sha256':sha})
 packet=selected['packet'];meta=base.read(bound(HF,packet['metadata']));need(meta['source_ordinal']==911 and meta['mode']=='dense' and meta['source_state']=='teacher_input' and meta['step']==selected['step'] and meta['block']==20 and meta['expert']==expert and meta['grid']==[21,29,52] and meta['population_rows']==31668 and meta['payload']==packet['payload'] and all(meta['tensor_metadata'][k]==packet['tensor_metadata'][k] for k in ('V1','a','h','Y','gate','rows')),'packet metadata identity differs')
 pb=packet['payload'];path=HF/pb['path'];ledger.reserve('payload_hash_bytes',pb['bytes']);need(base.bound(HF,pb)==path,'packet binding');ledger.complete('payload_hash_bytes',pb['bytes'])
 for key,dtype,shape in (('V1','BF16',[row_count,dim]),('a','BF16',[row_count,hidden]),('h','BF16',[row_count,hidden]),('Y','BF16',[row_count,dim]),('gate','BF16',[dim]),('rows','I64',[row_count])):
  get(path,key,dtype,shape,expert+'_'+key);actual=records[-1]['sha256'];need(actual==selected['packet']['tensor_metadata'][key]['raw_tensor_sha256'],'packet raw tensor differs')
 need(result[expert+'_rows'].tolist()==meta['rows'] and len(set(meta['rows']))==row_count and all(0<=x<31668 for x in meta['rows']),'packet coordinates differ')
 raw=Path(selected['original_bf16']['path']);nv=HF/selected['nvfp4_checkpoint']['cache_path'];need(extract.stat_id(raw)==master_states[0] and extract.stat_id(nv)==master_states[1] and extract.header_digest(raw)==master_states[2] and extract.header_digest(nv)==master_states[3],'master differs from prior full-SHA state/header')
 for number,n,k in ((0,hidden,dim),(2,dim,hidden)):
  prefix=f'blocks.20.ffn.{number}.'
  for name,dtype,shape in (('weight','BF16',[n,k]),('bias','BF16',[n])):get(raw,prefix+name,dtype,shape,f'{expert}_raw_{number}_{name}')
  for name,dtype,shape in (('weight','U8',[n,k//2]),('weight_scale','F8_E4M3',[n,k//16]),('bias','BF16',[n]),('alpha','F32',[]),('input_global_scale','F32',[])):get(nv,prefix+name,dtype,shape,f'{expert}_nv_{number}_{name}')
 need(extract.stat_id(raw)==master_states[0] and extract.stat_id(nv)==master_states[1],'master changed during staging');return result,records

def verify_job(a,p,s):
 need(a.job_spec and base.sha(a.job_spec)==s['launch_spec_sha256'],'job SHA differs');j=base.read(a.job_spec);argv=j['argv'];need(len(argv)==18 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'job argv');profile=Path(argv[1].split('=',1)[1]);need(base.sha(profile)==p['worker_config_sha256'],'worker profile changed')
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','taskset','-c','0,1,2,3',str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];out=HF/OUT
 need(argv==expected and j['id']==ID and j['protocol_sha256']==base.sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and j['hf_root']==str(HF) and Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'job identity')
 need(sys.flags.isolated and sorted(os.sched_getaffinity(0))==[0,1,2,3] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and runtime_identity()==p['runtime'] and not {'torch','numpy','safetensors'}&set(sys.modules),'fresh isolated CPU worker required')
 import worker_identity;worker_identity.require_worker();return out

def run(a):
 p=base.read(a.protocol);d=base.sha(a.protocol);s=authorize(p,d);out=verify_job(a,p,s)
 for b in p['repo_sources']:bound(ROOT,b)
 need(not out.exists(),'fresh attempt required');out.mkdir(parents=True,mode=0o700);started=time.monotonic();cpu=time.process_time();r={'id':ID,'status':'STARTED_NO_RETRY','protocol_sha256':d,'launch_spec_sha256':s['launch_spec_sha256'],'predebited':{},'completed':{},'model_calls':0,'GPU_calls':0,'automatic_retry':False,'master_identity':'prior full SHA receipt and same fd/path stat; noncooperative writes not excluded','input_read_scope':'packet full hash, selected typed tensor ranges; metadata and source reads additional'}
 emergency=output.reserve_emergency(out)
 def persist():output.checked_persist(out/'RUN.json',r,out.parent,RES['new_HF_bytes'])
 persist();signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('resource monitor failure')));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(RES['wall_seconds']);mon=output.UsageMonitor(out.parent,RES['new_HF_bytes'],4<<30);mon.start();stop=threading.Event();watch={'samples':0,'peak_rss':0,'fault':None}
 def watch_rss():
  try:
   while not stop.wait(2):
    rss=int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024;watch['samples']+=1;watch['peak_rss']=max(watch['peak_rss'],rss)
    if rss>RES['host_RSS_max_bytes']:watch['fault']='RSS exceeds budget';os.kill(os.getpid(),signal.SIGUSR1);return
  except BaseException as e:watch['fault']=type(e).__name__;os.kill(os.getpid(),signal.SIGUSR1)
 thread=threading.Thread(target=watch_rss,daemon=True);thread.start()
 try:
  resource.setrlimit(resource.RLIMIT_FSIZE,(RES['max_bundle_hash_bytes'],)*2)
  import torch
  from safetensors.torch import save_file
  torch.set_num_threads(4);torch.set_num_interop_threads(1);need(not torch.cuda.is_initialized(),'unexpected CUDA initialization');ledger=Ledger(r,persist);tensors={};r['source_records']={}
  for expert in ('high','low'):
   state=recorded_states(p,expert);values,rows=tensor_rows(expert,p['selected'][expert],ledger,torch,state);tensors.update(values);r['source_records'][expert]=rows;persist()
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['host_RSS_max_bytes'],'RSS exceeds budget')
  target=out/'BUNDLE.safetensors';r['bundle_write_predebit_bytes']=sum(v.numel()*v.element_size() for v in tensors.values())+(1<<20);need(r['bundle_write_predebit_bytes']<=RES['max_bundle_hash_bytes'],'bundle budget');persist();save_file(tensors,str(target));size=target.stat().st_size;need(size<=r['bundle_write_predebit_bytes'],'bundle exceeds bound')
  ledger.reserve('bundle_hash_bytes',size);bundle_sha=base.sha(target);ledger.complete('bundle_hash_bytes',size);r['bundle']={'path':str(target.relative_to(HF)),'bytes':size,'sha256':bundle_sha};r['status']='COMPLETE_PAIRED_WAN_FFN_CPU_ASSETS_EXIT_PENDING'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);stop.set();thread.join(10);r.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,rss_monitor=watch,output_monitor=mon.stop())
  if watch['fault'] or thread.is_alive() or r['peak_RSS_bytes']>RES['host_RSS_max_bytes'] or r['output_monitor']['fault'] or not r['output_monitor']['thread_stopped']:r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as e:output.emergency_failure(emergency,r,'final persist failed '+type(e).__name__);raise
  if r['status']=='FAILED_MONITOR_PRESERVED':raise RuntimeError('monitor failure')

def inspect_closed(protocol):
 p=base.read(protocol);validate(p);out=HF/OUT;rr=out/'RUN.json';ee=out.parent/'EXIT_RECEIPT.json';r=base.read(rr);e=base.read(ee);need(r.get('id')==ID and r.get('status')=='COMPLETE_PAIRED_WAN_FFN_CPU_ASSETS_EXIT_PENDING' and r.get('protocol_sha256')==base.sha(protocol) and e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==base.sha(rr) and e.get('protocol_sha256')==r['protocol_sha256'] and e.get('spec_sha256')==r['launch_spec_sha256'],'actual CPU EXIT0 required')
 need(r['predebited']==r['completed'] and r['completed'].get('tensor_ranges')==EXPECTED_RANGES and r['completed'].get('tensor_bytes')==EXPECTED_TENSOR_BYTES and r['completed'].get('payload_hash_bytes')==EXPECTED_PACKET_HASH_BYTES and r['model_calls']==r['GPU_calls']==0 and len(r['source_records'])==2 and r['wall_seconds']<=RES['wall_seconds'] and r['peak_RSS_bytes']<=RES['host_RSS_max_bytes'] and not r['rss_monitor']['fault'] and not r['output_monitor']['fault'] and r['output_monitor']['thread_stopped'],'asset ledger/resources incomplete');b=r['bundle'];path=HF/b['path'];need(path.is_file() and not path.is_symlink() and path.stat().st_size==b['bytes'] and len(b['sha256'])==64 and r['completed'].get('bundle_hash_bytes')==b['bytes'],'bundle stat differs')
 return {'id':ID,'status':'PASS_ACTUAL_EXIT_PAIRED_ASSETS','RUN':base.binding(rr,HF),'exit':base.binding(ee,HF),'bundle':b,'bundle_identity':'producer SHA; inspect stat only; GPU must rehash within own budget'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','inspect-closed'));ap.add_argument('--out',type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out and a.worker_profile and not a.out.exists(),'fresh draft/profile');base.write(a.out,draft(a.worker_profile))
 elif a.action=='inspect':p=base.read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'status':p['status'],'execution_requires':'independent recheck and ACTIVE'}))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol),indent=2))
if __name__=='__main__':main()
