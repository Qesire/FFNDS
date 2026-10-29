"""Bounded, independently reviewable A100 Wan BF16 2:4 FFN probe.

No model forward, LoRA, video, or GPU import at module import. This is an
RT-Lynx-inspired Wan port; author implementation and norm axis are unknown.
"""
import argparse,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_rtlynx_wan_ffn_ops as ops
import e0_sharq_wan_ffn_assets as parent
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_llama3_author_checkpoint_convert as support
import e0_sharq_wan_ffn_probe_v2 as paired
HF=Path('/home/xiang/.cache/huggingface');ID='E0_RTLYNX_WAN_BF16_2TO4_A100_PAIR_20261030';OUT='ffnds-derived/e0_rtlynx_wan_bf16_2to4_a100_pair_20261030/attempt1/worker/results'
SELF='tools/e0_rtlynx_wan_ffn_probe.py';TEST='tests/test_e0_rtlynx_wan_ffn_probe.py';CPU_PROTOCOL='config/e0_sharq_wan_ffn_paired_assets_cpu_20261030_pending_review_v4.json';C5='config/e0_activation24_fused_cutlass_probe_20261027_v1.json';PYTHON=ROOT/'.venv/bin/python'
RES={'GPU':1,'CPU_threads':4,'wall_seconds':1800,'outer_seconds':1920,'new_HF_bytes':256<<20,'host_RSS_max_bytes':16<<30,'GPU_reserved_max_bytes':64<<30,'bundle_hash_bytes_max':1<<30,'software_hash_bytes_max':2<<30,'automatic_retry':False}
DESIGN={'source_ordinal':911,'expert_step':{'high':1,'low':2},'block':20,'actual_rows':32,'M':[32,31668],'M31668':'actual V1 rows repeated modulo32; cost-only, no full-native-quality claim','arms':list(ops.ARMS),'warmups':1,'measured':3,'correctness':1,'tie':'finite BF16 magnitude descending; lower index first','norm':'rowwise FP32 sqrt(sum(h^2)/(sum(top2(h)^2)+1e-8)); scale selected values then BF16 round before CUTLASS pack; author reduction axis unknown','physical_path':'W1+b1/GELU(tanh)/top2+pack/real CUTLASS sparse W2/b2/gate','neutral_max_relative_l2':0.03,'same_mask_max_relative_l2':0.02,'single_source_exposed':True,'video_acceptance':False}
SOURCES=(SELF,TEST,'tools/e0_rtlynx_wan_ffn_ops.py','tests/test_e0_rtlynx_wan_ffn_ops.py','tools/activation24_fused_cutlass_pack.py','tools/e0_activation24_fused_cutlass_probe.py','tools/e0_activation24_cutlass_backend_probe.py','tools/e0_activation24_spmm_probe.py','tools/e0_sharq_wan_ffn_assets.py','tools/e0_sharq_wan_ffn_probe_v2.py','tools/e0_sharq_wan_ffn_asset_plan.py','tools/e0_sharq_wan_ffn_asset_extract.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/worker_identity.py','tools/durable_worker_job.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','scripts/worker_teal_assigned_uuid_exec.sh',CPU_PROTOCOL,C5)
SOURCES=tuple(sorted(set(SOURCES)|set(parent.SOURCES)|set(paired.SOURCES)))
LIMITS={}
for a in ops.ARMS:
 for k,n in ops.expected_calls(a).items():LIMITS[k]=LIMITS.get(k,0)+20*n
LIMITS.update(ffn_event_pair=60,finite_input_check=2,independent_select_M32=4,reference_mask=8,reference_converter=8,reference_dense_W2=8,reference_comparisons=12,bundle_hash=1,fullM_repeat=2)
need=base.need

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def draft(profile):
 proof=parent.inspect_closed(ROOT/CPU_PROTOCOL);c=base.read(ROOT/C5);need(profile.is_file() and proof['bundle']['bytes']==730509120,'actual closed bundle/profile required');return {'id':ID,'status':'FROZEN_PENDING_REGISTRATION','review':None,'design':DESIGN,'resources':RES,'limits':LIMITS,'output':OUT,'bundle_parent':proof,'worker_config_sha256':base.sha(profile),'source_bindings':[base.binding(ROOT/x,ROOT) for x in SOURCES],'backend_bindings':c['bindings'],'backend_identity':'C5 bound PyTorch CUTLASS and Triton; rehash before first tensor','automatic_retry':False}
def validate(p):
 need(p['id']==ID and p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and p['design']==DESIGN and p['resources']==RES and p['limits']==LIMITS and p['output']==OUT and p['automatic_retry'] is False,'fixed scope differs');need({b['path'] for b in p['source_bindings']}==set(SOURCES) and p['backend_bindings']==base.read(ROOT/C5)['bindings'] and p['bundle_parent']==parent.inspect_closed(ROOT/CPU_PROTOCOL),'parent/source closure differs')
def authorize(p,d):
 validate(p);a=base.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(len(rows)==1 and a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'ACTIVE required');s=rows[0];need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==1 and s.get('session_max_GPUs')==4 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'),'ACTIVE differs');r=base.read(base.bound(ROOT,s['review']));need(r.get('scope_id')==ID and r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id')!=r.get('reviewer_agent_id') and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['source_bindings']} and r.get('launch_spec_sha256')==s.get('launch_spec_sha256') and r.get('responses_to_review') and r.get('adversarial_counterexample'),'actual independent exact review required')
 for k in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,r[k])
 return s
class Ledger:
 def __init__(self,r,persist,sync=None):self.r,self.persist,self.current,self.sync=r,persist,None,sync
 def reserve(self,cost):
  need(self.current is None,'overlapping work');self.current=cost.copy()
  for k,n in cost.items():need(self.r['predebited'].get(k,0)+n<=LIMITS[k],'predebit limit '+k)
  for k,n in cost.items():self.r['predebited'][k]=self.r['predebited'].get(k,0)+n
  self.r['inflight']=dict(cost);self.persist()
 def call(self,k,f,*args):
  need(self.current is not None and self.current.get(k,0)>0,'call not predebited');self.current[k]-=1;self.r['invoked'][k]=self.r['invoked'].get(k,0)+1;v=f(*args);self.r['returned'][k]=self.r['returned'].get(k,0)+1;return v
 def finish(self):need(self.current is not None and not any(self.current.values()),'partial operation');self.current=None;self.r['inflight']=None;self.persist()
 def extra(self,k,f,*args):
  self.reserve({k:1});start=time.monotonic()
  def completed():
   v=f(*args)
   if self.sync is not None:self.sync()
   return v
  v=self.call(k,completed);self.r.setdefault('extra_wall_seconds_including_sync',{}).setdefault(k,[]).append(time.monotonic()-start);self.finish();return v
class API:
 def __init__(self,t):
  self.t=t
  from activation24_fused_cutlass_pack import fused_pack,materialize_masked,wrap_cutlass
  self.pack,self.materialize,self.wrap=fused_pack,materialize_masked,wrap_cutlass
 def addmm(self,b,x,w):return self.t.addmm(b,x,w)
 def gelu(self,x):return self.t.nn.functional.gelu(x,approximate='tanh')
 def compensate(self,h,s):return ops.row_compensated(self.t,h,s)
 def sparse_mm(self,a,b,m):return self.t.mm(a,b)[:m].contiguous()
 def bias(self,y,b):return y+b
 def mul(self,a,b):return a*b

def verify_job(a,p,s):
 need(a.job_spec and base.sha(a.job_spec)==s['launch_spec_sha256'],'job SHA differs');j=base.read(a.job_spec);v=j['argv'];need(len(v)>10 and v[1].startswith('FFNDS_WORKER_CONFIG='),'job profile');profile=Path(v[1].split('=',1)[1]);need(profile.is_absolute() and base.sha(profile)==p['worker_config_sha256'],'profile differs');gpu=os.environ.get('FFNDS_PHYSICAL_GPU','');uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'),'one device required');expected=['env',v[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin','bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid,'taskset','-c','0-3',str(PYTHON),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];need(v==expected and j['id']==ID and j['protocol_sha256']==base.sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and j['hf_root']==str(HF) and Path(j['output_dir']).resolve()==(HF/OUT).parent and Path(j['run_json']).resolve()==HF/OUT/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'] and sys.flags.isolated and sorted(os.sched_getaffinity(0))==[0,1,2,3] and not {'torch','numpy','safetensors'}&set(sys.modules),'exact private job/isolation differs');import worker_identity;worker_identity.require_worker()

def relative(t,a,b):
 af=a.float();bf=b.float();den=t.linalg.vector_norm(bf);num=t.linalg.vector_norm(af-bf);need(bool(t.isfinite(af).all()) and bool(t.isfinite(bf).all()),'nonfinite comparison');return {'relative_l2':float((num/den).item()) if den.item()>0 else None,'absolute_l2':float(num.item()),'zero_reference':bool(den.item()==0)}
def compare_bits(t,a,b):return a.shape==b.shape and a.dtype==b.dtype and bool(t.equal(a.contiguous().view(t.uint8),b.contiguous().view(t.uint8)))

def run(a):
 p=base.read(a.protocol);digest=base.sha(a.protocol);scope=authorize(p,digest);verify_job(a,p,scope);out=HF/OUT;need(not out.exists(),'fresh output required');out.mkdir(parents=True,mode=0o700);start=time.monotonic();cpu=time.process_time();r={'id':ID,'protocol_sha256':digest,'launch_spec_sha256':scope['launch_spec_sha256'],'status':'STARTED_NO_RETRY','automatic_retry':False,'predebited':{},'invoked':{},'returned':{},'inflight':None,'cases':[],'model_calls':0,'video_generations':0,'author_code':'NOT_VERIFIED','timing_scope':'whole FFN, CUDA events; setup, correctness, references, compilation separately charged; fixed-order diagnostic not fair SYS','GPU_internal_launches':'UNKNOWN','fullM_identity':DESIGN['M31668']};emergency=support.reserve_emergency(out)
 def persist():support.checked_persist(out/'RUN.json',r,out.parent,RES['new_HF_bytes'])
 persist();signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor failure')));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(RES['wall_seconds']);monitor=support.UsageMonitor(out.parent,RES['new_HF_bytes'],4<<30);monitor.start();gpu=None
 try:
  software=p['source_bindings']+p['backend_bindings'];need(len({x['path'] for x in p['backend_bindings']})==len(p['backend_bindings']),'duplicate backend binding');total=sum(x['bytes'] for x in software);need(total<=RES['software_hash_bytes_max'],'software hash budget');r['software_hash_predebited_bytes']=total;r['software_hash_completed_bytes']=0;r['software_hash_inflight']='UNKNOWN_UNTIL_ALL_COMPLETE';persist();st=time.monotonic()
  for b in software:base.bound(ROOT,b)
  r.update(software_hash_completed_bytes=total,software_hash_inflight='NONE',software_hash_wall_seconds=time.monotonic()-st);persist()
  import torch
  from safetensors.torch import load_file
  import startup_gpu_reservation
  import activation24_fused_cutlass_pack as pack
  from torch.sparse.semi_structured import SparseSemiStructuredTensorCUTLASS
  need(os.environ.get('TRITON_INTERPRET','0')=='0','actual CUDA Triton required');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;hold,gpu,reservation=startup_gpu_reservation.reserve_before_assets(torch);r['reservation']=reservation;need(torch.cuda.get_device_capability()==(8,0) and 'A100' in torch.cuda.get_device_name(),'actual A100 required')
  ledger=Ledger(r,persist,torch.cuda.synchronize);b=p['bundle_parent']['bundle'];r['bundle_hash_predebited_bytes']=b['bytes'];r['bundle_hash_completed_bytes']=0;persist();st=parent.extract.stat_id(HF/b['path']);path=ledger.extra('bundle_hash',base.bound,HF,b);need(parent.extract.stat_id(path)==st,'bundle stat changed during hash');r['bundle_hash_completed_bytes']=b['bytes'];persist();t=load_file(str(path),device='cpu');need(parent.extract.stat_id(path)==st,'bundle changed during load');paired.validate_tensors(t,torch)
  keys=[x for x in t if '_raw_' in x or x.endswith(('_V1','_a','_h','_Y','_gate'))];need(len(keys)==18,'selected tensor coverage differs');total=sum(t[k].numel()*t[k].element_size() for k in keys);r['H2D_predebited_bytes']=total;r['H2D_completed_bytes']=0;persist();st=time.monotonic();d={k:t[k].to('cuda') for k in keys};torch.cuda.synchronize();r['H2D_completed_bytes']=total;r['H2D_wall_seconds']=time.monotonic()-st;del t;persist();api=API(torch)
  def check():gpu.check();need(time.monotonic()-start<RES['wall_seconds'] and torch.cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['host_RSS_max_bytes'],'resource limit')
  for expert in ('high','low'):
   w1,b1,w2,b2=(d[f'{expert}_raw_{n}_{field}'] for n,field in ((0,'weight'),(0,'bias'),(2,'weight'),(2,'bias')));gate=d[expert+'_gate']
   def finite_inputs():
    need(all(bool(torch.isfinite(z).all()) for z in (d[expert+'_V1'],w1,b1,w2,b2,gate)),'nonfinite native input');torch.cuda.synchronize()
   ledger.extra('finite_input_check',finite_inputs)
   for m in (32,31668):
    x=d[expert+'_V1'] if m==32 else ledger.extra('fullM_repeat',lambda:d[expert+'_V1'].repeat((m+31)//32,1)[:m].contiguous());baseline=None
    for arm in ops.ARMS:
     timings=[];result=None;cold=None;warm=None
     for j in range(5):
      ledger.reserve({**ops.expected_calls(arm),'ffn_event_pair':1});begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);wall=time.monotonic()
      def timed_ffn():
       begin.record();z=ops.forward(arm,x,w1,b1,w2,b2,gate,api,ledger);end.record();torch.cuda.synchronize();return z
      value=ledger.call('ffn_event_pair',timed_ffn);elapsed=time.monotonic()-wall;cuda_ms=begin.elapsed_time(end);ledger.finish();check()
      if j==0:result=value;cold={'cuda_ms':cuda_ms,'wall_seconds':elapsed}
      elif j==1:warm={'cuda_ms':cuda_ms,'wall_seconds':elapsed}
      else:timings.append({'cuda_ms':cuda_ms,'wall_seconds':elapsed})
     if arm=='dense':baseline=result
     def compare():
      metrics={k:relative(torch,result[k],baseline[k]) for k in ('a','h','y','gated')};neutral={}
      if arm=='dense' and m==32:neutral={k:relative(torch,result[k],d[expert+'_'+src]) for k,src in (('a','a'),('h','h'),('y','Y'))};neutral['gated']=relative(torch,result['gated'],d[expert+'_Y']*gate)
      return metrics,neutral
     metrics,neutral=ledger.extra('reference_comparisons',compare);need(all(z['relative_l2'] is not None and z['relative_l2']<=DESIGN['neutral_max_relative_l2'] for z in neutral.values()),'captured neutral differs')
     exact={}
     if arm!='dense':
      def reference_mask():
       masked=pack.materialize_masked(result['h']);value=masked[:m]
       if arm=='top2_row_norm':value=ops.row_compensated(torch,result['h'],value)[0]
       return value
      selected=ledger.extra('reference_mask',reference_mask)
      if m==32:
       def independent_select():
        from e0_activation24_spmm_probe import select_top2
        independent=select_top2(result['h'])
        if arm=='top2_row_norm':independent=ops.row_compensated(torch,result['h'],independent)[0]
        need(compare_bits(torch,selected,independent),'independent stable top2/row norm differs')
       ledger.extra('independent_select_M32',independent_select)
      def reference_converter():
       padded=torch.nn.functional.pad(selected,(0,0,0,pack.padded_rows(m)-m)).contiguous();converted=SparseSemiStructuredTensorCUTLASS.from_dense(padded);torch.cuda.synchronize();need(compare_bits(torch,result['packed'],converted.packed) and compare_bits(torch,result['metadata'],converted.meta),'CUTLASS packed/meta mismatch');return {'packed_byte_equal':True,'metadata_byte_equal':True}
      exact=ledger.extra('reference_converter',reference_converter)
      def same_mask_reference():
       ref=torch.addmm(b2,selected,w2.T);return relative(torch,result['y'],ref)
      exact['same_mask_W2']=ledger.extra('reference_dense_W2',same_mask_reference);need(exact['same_mask_W2']['relative_l2'] is not None and exact['same_mask_W2']['relative_l2']<=DESIGN['same_mask_max_relative_l2'],'same-mask W2 numerical mismatch')
     r['cases'].append({'expert':expert,'M':m,'arm':arm,'relative_to_original_BF16':metrics,'captured_BF16_neutral':neutral,'same_mask_reference':exact,'correctness_cold':cold,'warmup':warm,'measurements':timings});persist();check()
    del x,baseline,result
  need(r['predebited']==r['invoked']==r['returned']==LIMITS and len(r['cases'])==12,'operation coverage differs');r['status']='COMPLETE_RTLYNX_WAN_PROBE_EXIT_PENDING'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500],error_notes=list(getattr(e,'__notes__',[])));raise
 finally:
  signal.alarm(0)
  if gpu is not None:gpu.close();r['GPU_monitor']=gpu.report();r['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved();r['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
  r['output_monitor']=monitor.stop();r['wall_seconds']=time.monotonic()-start;r['cpu_seconds']=time.process_time()-cpu;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  if r['output_monitor']['fault'] or not r['output_monitor']['thread_stopped'] or (gpu is not None and not r['GPU_monitor']['sampled_isolation_pass']):r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as e:support.emergency_failure(emergency,r,'final persist failed '+type(e).__name__);raise
  if r['status']=='FAILED_MONITOR_PRESERVED':raise RuntimeError('monitor failure')

def inspect_closed(protocol):
 p=base.read(protocol);validate(p);rr=HF/OUT/'RUN.json';ee=rr.parent.parent/'EXIT_RECEIPT.json';r=base.read(rr);e=base.read(ee);d=base.sha(protocol);need(r.get('id')==ID and r.get('status')=='COMPLETE_RTLYNX_WAN_PROBE_EXIT_PENDING' and r.get('protocol_sha256')==d and e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==base.sha(rr) and e.get('protocol_sha256')==d and e.get('spec_sha256')==r['launch_spec_sha256'],'actual GPU EXIT0 required');need(r['predebited']==r['invoked']==r['returned']==LIMITS and len(r['cases'])==12 and r['bundle_hash_predebited_bytes']==r['bundle_hash_completed_bytes']==730509120 and r['software_hash_predebited_bytes']==r['software_hash_completed_bytes'] and r['software_hash_inflight']=='NONE' and r['H2D_predebited_bytes']==r['H2D_completed_bytes'] and r['GPU_monitor']['sampled_isolation_pass'] and not r['output_monitor']['fault'],'incomplete resource/operation proof');return {'id':ID,'status':'PASS_CLOSED_RTLYNX_WAN_PAIR','RUN':base.binding(rr,HF),'exit':base.binding(ee,HF),'quality_acceptance':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','inspect-closed'));ap.add_argument('--worker-profile',type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out and a.worker_profile and not a.out.exists(),'fresh output and worker profile required');base.write(a.out,draft(a.worker_profile))
 elif a.action=='inspect':p=base.read(a.protocol);validate(p);print(json.dumps({'id':ID,'fingerprint':fingerprint(p),'status':p['status']}))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol)))
if __name__=='__main__':main()
