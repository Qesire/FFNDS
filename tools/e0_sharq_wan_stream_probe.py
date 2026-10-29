"""Bounded SM120 stream diagnostic over the actual paired M32 Wan assets.

SAFE and UNBRIDGED are separate processes/scopes. Draft is byte metadata only;
GPU execution needs actual parent EXIT0, exact independent review and ACTIVE.
"""
import argparse,copy,importlib,json,os,resource,signal,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_sharq_wan_ffn_probe_v2 as prior
import e0_sharq_wan_ffn_ops as ops
import e0_sharq_wan_author_stream_bridge as bridge
base=prior.base;need=base.need;HF=prior.HF
SELF='tools/e0_sharq_wan_stream_probe.py';TEST='tests/test_e0_sharq_wan_stream_probe.py'
PARENT='config/e0_sharq_sm120_wan_ffn_pair_probe_v2_20261030_pending_review_v2.json'
AUTHOR_SRCS=tuple('kernels/src/'+x for x in ('shared_weight_nvfp4.cu','fused_sparse_prepare.cu','nvfp4.cu','sparse_nvfp4.cu','reorder.cu','down.cu','bindings.cpp'))+tuple('kernels/include/'+x for x in ('shared_weight_nvfp4.h','fused_sparse_prepare.h','nvfp4.h','sparse_nvfp4.h','reorder.cuh','sharq_blackwell_arch.h'))
RES={'GPU':1,'CPU_threads':4,'wall_seconds':1800,'outer_seconds':1920,'new_HF_bytes':256<<20,'host_RSS_max_bytes':16<<30,'GPU_reserved_max_bytes':32<<30,'bundle_hash_bytes_max':1<<30,'source_hash_bytes_max':512<<20,'failure_receipt_hash_bytes_max':1<<20,'packed_digest_D2H_bytes_max':1<<30,'automatic_retry':False,'session_max_GPUs':4}
DESIGN={'experts':['high','low'],'source':911,'block':20,'expert_steps':{'high':1,'low':2},'rows':32,'matrices':[0,2],'safe':['default_reference','bridged_nondefault'],'unsafe':['unbridged_nondefault_once'],'unsafe_separate_process':True,'weight_identity':'raw BF16 W [N,K], GPU buffer K,N transpose view; W.T.contiguous() alias checked','comparison':'W and real activation bitwise unchanged, BF16 linear before/after, packed reference equality; bridged author output relative L2 <= 0.001','no_full_model_video':True}
COUNTS={'safe':{'bundle_hash':1,'weight_quant':8,'bf16_linear':24,'post_author_weight_equal':8,'post_author_input_equal':8,'post_author_linear_equal':8,'post_author_finite_checks':8,'dense32_linear':8,'sharq_linear':8,'weight_equal':8,'input_equal':8,'pack_equal':4,'linear_equal':8,'pack_digest':8,'logical_scale_extract':16,'prepare_case':8,'reference_pack_clone':4,'reference_output_clone':4,'BF16_finite_checks':8,'author_finite_checks':16,'device_sync':49,'author_output_comparisons':8,'dense32_prepare':8,'dense32_gemm':8,'sharq_prepare':8,'sparse_gemm':8,'dense_residual_gemm':8,'output_scale':16,'bias_add':16},'unsafe':{'bundle_hash':1,'weight_quant':4,'bf16_linear':8,'dense32_linear':0,'sharq_linear':0,'weight_equal':4,'input_equal':4,'pack_equal':0,'linear_equal':4,'pack_digest':4,'logical_scale_extract':8,'prepare_case':4,'reference_pack_clone':0,'reference_output_clone':0,'BF16_finite_checks':4,'author_finite_checks':0,'device_sync':17,'author_output_comparisons':0}}

def identity(kind):need(kind in COUNTS,'kind differs');return 'E0_SHARQ_SM120_STREAM_'+kind.upper()+'_20261030'
def output(kind):return 'ffnds-derived/e0_sharq_sm120_stream_'+kind+'_20261030/attempt1/worker/results'
def fingerprint(p):return base.canonical({k:v for k,v in p.items() if k not in ('status','review')}) if hasattr(base,'canonical') else prior.fingerprint(p)
def sources(parent):return sorted({x['path'] for x in parent['repo_sources']}|{SELF,TEST,'tools/e0_sharq_wan_author_stream_bridge.py','tests/test_e0_sharq_wan_author_stream_bridge.py',PARENT,'tools/e0_sharq_wan_all160_video_owner.py','tools/e0_sharq_wan_all160_video_driver.py','tools/e0_sharq_wan_all160_runtime.py','scripts/worker_teal_assigned_uuid_exec.sh'})
def failed_proofs():
 result=[]
 for part in (0,1):
  worker=HF/f'ffnds-derived/e0_sharq_wan_all160_video_20261030/part{part}/attempt1/worker';run=worker/'results/RUN.json';ex=worker/'EXIT_RECEIPT.json';r=base.read(run);e=base.read(ex);rb=base.binding(run,HF);need(r['id']==f'E0_SHARQ_WAN_ALL160_VIDEO_20261030_PART{part}' and r['error']=='nonfinite state' and e['id']==r['id'] and e['protocol_sha256']==r['protocol_sha256'] and e['process_exited'] is True and e['child_returncode']==1 and e['timeout'] is False and e['automatic_retry'] is False and e['run_sha256']==rb['sha256'] and r['operation_ledger']['observed']['video_generations']==1 and r['operation_ledger']['observed']['sharq_static_W_quantizations']==80,'actual failure evidence differs');result.append({'RUN':rb,'exit':base.binding(ex,HF)})
 return result

def draft(kind,profile):
 identity(kind);p=base.read(ROOT/PARENT);closed=prior.inspect_closed(ROOT/PARENT);need(closed['status']=='PASS_CLOSED_SINGLE_SOURCE_PAIRED_FFN' and profile.is_file(),'actual paired operator proof/profile required');need(base.sha(profile)==p['worker_config_sha256'],'worker profile differs')
 return {'id':identity(kind),'kind':kind,'status':'FROZEN_PENDING_REGISTRATION','review':None,'resources':RES,'limits':COUNTS[kind],'design':DESIGN,'output':output(kind),'prior_operator':{'protocol':base.binding(ROOT/PARENT,ROOT),'RUN':closed['RUN'],'exit':closed['exit']},'failure':failed_proofs(),'assets':copy.deepcopy(p['assets']),'software':copy.deepcopy(p['software']),'author_stream_sources':[base.binding(prior.SRC/x,HF) for x in AUTHOR_SRCS],'runtime':copy.deepcopy(p['runtime']),'worker_config_sha256':base.sha(profile),'repo_sources':[base.binding(ROOT/x,ROOT) for x in sources(p)]}
def validate(p):
 kind=p['kind'];parent=base.read(ROOT/PARENT);need(p['id']==identity(kind) and p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and p['resources']==RES and p['limits']==COUNTS[kind] and p['design']==DESIGN and p['output']==output(kind),'scope differs');need(p['assets']==parent['assets'] and p['software']==parent['software'] and p['runtime']==parent['runtime'] and p['worker_config_sha256']==parent['worker_config_sha256'] and {x['path'] for x in p['repo_sources']}==set(sources(parent)),'asset/source identity differs');need(len(p['failure'])==2 and [x['path'] for x in p['author_stream_sources']]==[str((prior.SRC/x).relative_to(HF)) for x in AUTHOR_SRCS],'failure/author source identity differs')

def authorize(p,d):
 validate(p);a=base.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']];need(len(rows)==1 and a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'ACTIVE required');s=rows[0];need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==p['limits'] and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('max_gpu_cards')==1 and s.get('session_max_GPUs')==4,'ACTIVE differs');r=base.read(base.bound(ROOT,s['review']));need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==p['id'] and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']} and r.get('launch_spec_sha256')==s.get('launch_spec_sha256'),'independent review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,r[k])
 return s

class Ledger:
 def __init__(self,r,persist,limits):self.r=r;self.persist=persist;self.limits=limits
 def call(self,key,fn,*args):
  p=self.r['predebited'];i=self.r['invoked'];d=self.r['returned'];need(p.get(key,0)<self.limits[key],'operation budget exceeded: '+key);p[key]=p.get(key,0)+1;self.r['inflight'][key]='UNKNOWN';self.persist();i[key]=i.get(key,0)+1
  value=fn(*args);d[key]=d.get(key,0)+1;self.r['inflight'].pop(key);self.persist();return value

def verify_job(a,p,s):
 need(a.job_spec and base.sha(a.job_spec)==s['launch_spec_sha256'],'job spec differs');j=base.read(a.job_spec);v=j['argv'];need(len(v)>4 and v[0]=='env' and v[1].startswith('FFNDS_WORKER_CONFIG='),'profile missing');profile=Path(v[1].split('=',1)[1]);need(profile.is_absolute() and base.sha(profile)==p['worker_config_sha256'],'profile differs');aff=','.join(str(x) for x in sorted(os.sched_getaffinity(0)));need(len(os.sched_getaffinity(0))==4 and sys.flags.isolated and not {'torch','numpy','safetensors'}&set(sys.modules) and prior.asset_parent.runtime_identity()==p['runtime'],'CPU/interpreter/runtime differs');gpu=os.environ.get('FFNDS_PHYSICAL_GPU');uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'),'assigned device differs');expected=['env',v[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin','bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid,'taskset','-c',aff,str(prior.ENV),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];need(v==expected and j['id']==p['id'] and j['protocol_sha256']==base.sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and Path(j['output_dir']).resolve()==(HF/p['output']).parent and Path(j['run_json']).resolve()==HF/p['output']/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'job mismatch');import worker_identity;worker_identity.require_worker()

def bitwise_equal(torch,a,b):
 need(a.shape==b.shape and a.dtype==b.dtype,'bitwise compare geometry differs');return bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))

def sparse_logical_scale(torch,scale,rows,k):
 # shared_weight_nvfp4.cu coordinates: row (32,4,tiles) and group
 # (4,tiles), 512 elements per 128-row x 4-group CUTE tile.
 need(k%128==0 and rows>0,'sparse scale geometry differs');groups=k//32;need(groups%4==0,'incomplete sparse group tile');row=torch.arange(rows,device=scale.device,dtype=torch.int64).reshape(-1,1);group=torch.arange(groups,device=scale.device,dtype=torch.int64).reshape(1,-1);offset=(row//128)*(groups//4)*512+(group//4)*512+(row%32)*16+((row%128)//32)*4+group%4;need(scale.numel()>=(rows+127)//128*128*groups,'sparse scale buffer too small');return scale.reshape(-1)[offset].contiguous()

def logical_packed(torch,packed,rows,k,ledger):
 sparse=ledger.call('logical_scale_extract',sparse_logical_scale,torch,packed[1],rows,k);dense=ledger.call('logical_scale_extract',prior.logical_scale_bytes,torch,packed[2],rows,k,'author');return (packed[0],sparse,dense,packed[3])

def require_BF16_gate(item,persist):
 need(item['weight_bitwise_unchanged'] and item['activation_bitwise_unchanged'] and item['BF16_pre_post_bitwise_equal'] and item['BF16_finite'],'BF16 input/weight/output integrity differs');item['BF16_gate']='PASS_BEFORE_ANY_CANDIDATE';persist()

def execute_cases(torch,t,quant,ext,kind,ledger,report,persist,check):
 import hashlib
 api=prior.AuthorAPI(torch,quant,ext,None);compute=torch.cuda.Stream();default=torch.cuda.default_stream();results=[]
 def plain_pack(value):return quant.quantize_weight_shared_nvfp4(value)
 def digest_packed(packed):
  ledger.call('device_sync',torch.cuda.synchronize);h=hashlib.sha256();n=0
  for part in packed:
   raw=part.detach().reshape(-1).view(torch.uint8).cpu().contiguous().numpy().tobytes();n+=len(raw);h.update(raw)
  return {'bytes':n,'sha256':h.hexdigest(),'scope':'Q payload and logical consumed sparse/dense scale coordinates and global scale; padding excluded'}
 for expert in DESIGN['experts']:
  for number in DESIGN['matrices']:
   raw=t[f'{expert}_raw_{number}_weight'];bias=t[f'{expert}_raw_{number}_bias'];source=t[expert+('_V1' if number==0 else '_h')];modes=DESIGN['safe'] if kind=='safe' else DESIGN['unsafe'];reference=None
   for mode in modes:
    check();before,source_copy,buffer,x=ledger.call('prepare_case',lambda:(raw.clone(),source.clone(),torch.empty_like(raw),source.clone()));ledger.call('device_sync',torch.cuda.synchronize)
    selected=default if mode=='default_reference' else compute
    with torch.cuda.stream(selected):
     buffer.copy_(raw,non_blocking=True);mm_weight=buffer.T;material=mm_weight.T.contiguous();need(material.data_ptr()==buffer.data_ptr() and mm_weight.stride()==(1,raw.shape[1]) and material.stride()==raw.stride(),'GPU buffer view/stride identity differs');before_linear=ledger.call('bf16_linear',lambda:torch.addmm(bias,x,mm_weight))
     if mode=='bridged_nondefault':packed=ledger.call('weight_quant',bridge.author_call,torch,plain_pack,material)
     else:packed=ledger.call('weight_quant',plain_pack,material)
    ledger.call('device_sync',torch.cuda.synchronize)
    with torch.cuda.stream(selected):after_linear=ledger.call('bf16_linear',lambda:torch.addmm(bias,x,mm_weight))
    ledger.call('device_sync',torch.cuda.synchronize)
    item={'expert':expert,'linear':number,'mode':mode,'weight_view_shape':list(mm_weight.shape),'weight_view_stride':list(mm_weight.stride()),'materialized_is_original_buffer_alias':True,'stream_contract':'UNBRIDGED_INDEPENDENT_PROCESS' if kind=='unsafe' else 'AUTHOR_DEFAULT_SINGLE_STREAM' if mode=='default_reference' else 'EVENT_BRIDGED_DEFAULT_WITH_INPUT_OUTPUT_RECORD_STREAM','status':'STARTED_BF16_GATE'};results.append(item);report['cases']=results;persist()
    item['weight_bitwise_unchanged']=ledger.call('weight_equal',lambda:bitwise_equal(torch,buffer,before));item['activation_bitwise_unchanged']=ledger.call('input_equal',lambda:bitwise_equal(torch,x,source_copy));item['BF16_pre_post_bitwise_equal']=ledger.call('linear_equal',lambda:bitwise_equal(torch,before_linear,after_linear));item['BF16_finite']=ledger.call('BF16_finite_checks',lambda:bool(torch.isfinite(before_linear).all().item() and torch.isfinite(after_linear).all().item()));require_BF16_gate(item,persist)
    logical=logical_packed(torch,packed,raw.shape[0],raw.shape[1],ledger);pack_bytes=sum(z.numel()*z.element_size() for z in logical);report['packed_digest_D2H_predebited_bytes']=report.get('packed_digest_D2H_predebited_bytes',0)+pack_bytes;need(report['packed_digest_D2H_predebited_bytes']<=RES['packed_digest_D2H_bytes_max'],'packed digest byte budget exceeded');persist();item['packed_digest']=ledger.call('pack_digest',digest_packed,logical);need(item['packed_digest']['bytes']==pack_bytes,'packed digest bytes differ');report['packed_digest_D2H_completed_bytes']=report.get('packed_digest_D2H_completed_bytes',0)+pack_bytes;persist()
    if kind=='unsafe':
     safe_ref=next(row for row in report['safe_reference_cases'] if row['expert']==expert and row['linear']==number and row['mode']=='default_reference');item['packed_digest_equal_safe_reference']=item['packed_digest']['sha256']==safe_ref['packed_digest']['sha256'];item['status']='COMPLETE';persist();continue
    if reference is None:reference=ledger.call('reference_pack_clone',lambda:tuple(z.clone() for z in logical))
    else:
     item['packed_equal_default']=ledger.call('pack_equal',lambda:all(torch.equal(a,b) for a,b in zip(reference,logical)));need(item['packed_equal_default'],'logical packed identity differs')
    with torch.cuda.stream(selected):
     layer=SimpleNamespace(weight_q=packed[0],scale_w_sparse=packed[1],scale_w_dense=packed[2],weight_scale=packed[3],bias=bias,in_features=raw.shape[1],out_features=raw.shape[0]);meter=ledger
     def run_dense():return ops.dense32_linear(x,layer,api,meter)
     def run_sparse():return ops.sharq_linear(x,layer,api,meter)
     dense=ledger.call('dense32_linear',bridge.author_call,torch,ops.dense32_linear,x,layer,api,meter) if mode=='bridged_nondefault' else ledger.call('dense32_linear',run_dense)
     sparse=ledger.call('sharq_linear',bridge.author_call,torch,ops.sharq_linear,x,layer,api,meter) if mode=='bridged_nondefault' else ledger.call('sharq_linear',run_sparse)
    ledger.call('device_sync',torch.cuda.synchronize);item['dense_finite']=ledger.call('author_finite_checks',lambda:bool(torch.isfinite(dense).all().item()));item['sharq_finite']=ledger.call('author_finite_checks',lambda:bool(torch.isfinite(sparse).all().item()));need(item['dense_finite'] and item['sharq_finite'],'safe author result nonfinite')
    if mode=='default_reference':reference_dense,reference_sparse=ledger.call('reference_output_clone',lambda:(dense.clone(),sparse.clone()))
    else:
     item['dense_vs_default']=ledger.call('author_output_comparisons',prior.relative,torch,dense,reference_dense);item['sharq_vs_default']=ledger.call('author_output_comparisons',prior.relative,torch,sparse,reference_sparse);need(all(v['finite'] and v['relative_l2']<=0.001 for v in (item['dense_vs_default'],item['sharq_vs_default'])),'bridged author output differs')
    with torch.cuda.stream(selected):after_author=ledger.call('bf16_linear',lambda:torch.addmm(bias,x,mm_weight))
    ledger.call('device_sync',torch.cuda.synchronize);item['post_author_weight_bitwise_unchanged']=ledger.call('post_author_weight_equal',lambda:bitwise_equal(torch,buffer,before));item['post_author_input_bitwise_unchanged']=ledger.call('post_author_input_equal',lambda:bitwise_equal(torch,x,source_copy));item['post_author_BF16_bitwise_equal']=ledger.call('post_author_linear_equal',lambda:bitwise_equal(torch,before_linear,after_author));item['post_author_BF16_finite']=ledger.call('post_author_finite_checks',lambda:bool(torch.isfinite(after_author).all().item()));need(all(item[k] for k in ('post_author_weight_bitwise_unchanged','post_author_input_bitwise_unchanged','post_author_BF16_bitwise_equal','post_author_BF16_finite')),'author call altered BF16 operands/result');item['status']='COMPLETE';persist()
 return results

def run(a):
 p=base.read(a.protocol);d=base.sha(a.protocol);scope=authorize(p,d);verify_job(a,p,scope);safe=None
 out=HF/p['output'];need(not out.exists(),'fresh output required');out.mkdir(parents=True);started=time.monotonic();r={'id':p['id'],'protocol_sha256':d,'launch_spec_sha256':scope['launch_spec_sha256'],'status':'STARTED_NO_RETRY','predebited':{},'invoked':{},'returned':{},'inflight':{},'cases':[],'automatic_retry':False,'model_calls':0,'generation_calls':0,'counter_semantics':'Python entry and return; case CUDA completion after explicit device synchronize; intermediate async completion unknown','safe_reference_cases':[],'pre_RUN_small_identity_reads':'protocol, ACTIVE/review/job/profile/runtime metadata; failure before RUN retained by supervisor EXIT; byte count UNKNOWN'};emergency=prior.support.reserve_emergency(out)
 def persist():prior.support.checked_persist(out/'RUN.json',r,out.parent,RES['new_HF_bytes'])
 persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor failure')));signal.alarm(RES['wall_seconds']);monitor=prior.support.UsageMonitor(out.parent,RES['new_HF_bytes'],4<<30);monitor.start();gpu=None
 try:
  sources_to_hash=p['repo_sources']+p['software']['author_modules']+[p['software']['sharq_elf']]+p['software']['lightx_files']+p['author_stream_sources'];r['source_hash_predebited_bytes']=sum(x['bytes'] for x in sources_to_hash);r['source_hash_file_calls']=len(sources_to_hash);need(r['source_hash_predebited_bytes']<=RES['source_hash_bytes_max'],'source hash budget exceeded');r['source_hash_inflight']='UNKNOWN';persist();prior.verify_software(p);[base.bound(HF,b) for b in p['author_stream_sources']];r['source_hash_completed_bytes']=r['source_hash_predebited_bytes'];r['source_hash_inflight']='NONE';persist()
  r['failure_receipt_hash_predebited_bytes']=sum(b['bytes'] for proof in p['failure'] for b in proof.values());need(r['failure_receipt_hash_predebited_bytes']<=RES['failure_receipt_hash_bytes_max'],'failure proof hash budget exceeded');r['failure_receipt_hash_calls']=4;r['failure_receipt_hash_inflight']='UNKNOWN';persist();need(p['failure']==failed_proofs(),'actual V6 failures changed');r['failure_receipt_hash_completed_bytes']=r['failure_receipt_hash_predebited_bytes'];r['failure_receipt_hash_inflight']='NONE';persist();prior.check_parent(p);need(prior.inspect_closed(ROOT/PARENT)['RUN']==p['prior_operator']['RUN'],'parent closed proof differs')
  if p['kind']=='unsafe':
   proof=scope.get('safe_scope_exit');need(isinstance(proof,dict) and scope.get('safe_protocol'),'SAFE actual closed scope required before unsafe');closed=inspect_closed(ROOT/scope['safe_protocol']);need(closed['id']==identity('safe') and closed['RUN']==proof['RUN'] and closed['exit']==proof['exit'],'SAFE actual closed identity differs');safe=base.read(base.bound(HF,proof['RUN']));r['safe_reference_cases']=safe['cases'];persist()
  import torch
  from safetensors.torch import load_file
  import startup_gpu_reservation
  torch.set_num_threads(4);torch.set_num_interop_threads(1);hold,gpu,reservation=startup_gpu_reservation.reserve_before_assets(torch);r['reservation']=reservation;need(torch.cuda.get_device_capability()==(12,0) and 'RTX PRO 6000' in torch.cuda.get_device_name(),'SM120 device differs');torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  ledger=Ledger(r,persist,p['limits']);b=p['assets']['bundle'];need(b['bytes']==730509120 and b['bytes']<=RES['bundle_hash_bytes_max'],'bundle bytes differ');bundle_path=HF/b['path'];st=prior.asset_parent.extract.stat_id(bundle_path);r['bundle_hash_predebit_bytes']=b['bytes'];persist();bundle=ledger.call('bundle_hash',base.bound,HF,b);need(prior.asset_parent.extract.stat_id(bundle)==st,'bundle changed');r['bundle_hash_completed_bytes']=b['bytes'];persist();cpu=load_file(str(bundle),device='cpu');prior.validate_tensors(cpu,torch);need(prior.asset_parent.extract.stat_id(bundle)==st,'bundle changed while mapping')
  keys=[f'{e}_{k}' for e in DESIGN['experts'] for k in ('V1','h','raw_0_weight','raw_2_weight','raw_0_bias','raw_2_bias')];r['H2D_predebit_bytes']=sum(cpu[k].numel()*cpu[k].element_size() for k in keys);need(r['H2D_predebit_bytes']<=RES['bundle_hash_bytes_max'],'H2D byte budget exceeded');persist();t={k:cpu[k].to('cuda') for k in keys};ledger.call('device_sync',torch.cuda.synchronize);r['H2D_completed_bytes']=r['H2D_predebit_bytes'];persist()
  os.environ['SHARQ_KERNEL_BUILD']=str(prior.ELF.parent);sys.path.insert(0,str(prior.SRC));sys.path.insert(0,str(prior.SRC/'model'));quant=importlib.import_module('quantize');ext=quant.load_sharq_ops();need(Path(quant.__file__).resolve()==(prior.SRC/'model/quantize.py').resolve() and Path(ext.__file__).resolve()==prior.ELF.resolve(),'actual author origin differs')
  def check():gpu.check();need(time.monotonic()-started<RES['wall_seconds'] and torch.cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['host_RSS_max_bytes'],'resource limit exceeded')
  execute_cases(torch,t,quant,ext,p['kind'],ledger,r,persist,check);need(all(r[x]=={k:v for k,v in p['limits'].items() if v} for x in ('predebited','invoked','returned')) and not r['inflight'],'operation count differs');r['status']='COMPLETE_STREAM_PROBE_EXIT_PENDING'
 except BaseException as e:r.update(status='FAILED_STREAM_PROBE_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500],error_notes=list(getattr(e,'__notes__',[])));raise
 finally:
  signal.alarm(0)
  if gpu:gpu.close();r['GPU_monitor']=gpu.report();r['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
  r['output_monitor']=monitor.stop();r['wall_seconds']=time.monotonic()-started;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  if r['output_monitor']['fault'] or not r['output_monitor']['thread_stopped'] or (gpu and not r['GPU_monitor']['sampled_isolation_pass']):r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as e:prior.support.emergency_failure(emergency,r,'final persist failed '+type(e).__name__);raise
  if r['status']=='FAILED_MONITOR_PRESERVED':raise RuntimeError('monitor failure')

def inspect_closed(protocol):
 p=base.read(protocol);validate(p);out=HF/p['output'];run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';r=base.read(run);e=base.read(ex);binding=base.binding(run,HF);need(r['id']==p['id']==e['id'] and r['status']=='COMPLETE_STREAM_PROBE_EXIT_PENDING' and e['process_exited'] is True and e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and e['run_sha256']==binding['sha256'] and e['protocol_sha256']==r['protocol_sha256']==base.sha(protocol) and e['spec_sha256']==r['launch_spec_sha256'],'actual EXIT0 required');need(all(r[x]=={k:v for k,v in p['limits'].items() if v} for x in ('predebited','invoked','returned')) and len(r['cases'])==(8 if p['kind']=='safe' else 4) and r['source_hash_predebited_bytes']==r['source_hash_completed_bytes']<=RES['source_hash_bytes_max'] and r['source_hash_inflight']=='NONE' and r['failure_receipt_hash_predebited_bytes']==r['failure_receipt_hash_completed_bytes']<=RES['failure_receipt_hash_bytes_max'] and r['failure_receipt_hash_calls']==4 and r['failure_receipt_hash_inflight']=='NONE' and r['bundle_hash_predebit_bytes']==r['bundle_hash_completed_bytes']==730509120 and r['H2D_predebit_bytes']==r['H2D_completed_bytes'] and r['packed_digest_D2H_predebited_bytes']==r['packed_digest_D2H_completed_bytes']<=RES['packed_digest_D2H_bytes_max'] and r['GPU_monitor']['sampled_isolation_pass'] and not r['output_monitor']['fault'] and r['wall_seconds']<=RES['wall_seconds'],'incomplete resource/accounting');return {'id':p['id'],'status':'PASS_ACTUAL_EXIT_STREAM_DIAGNOSTIC','RUN':binding,'exit':base.binding(ex,HF),'no_video_or_system_acceptance':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','inspect-closed'));ap.add_argument('--kind',choices=('safe','unsafe'));ap.add_argument('--profile',type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.kind and a.profile and a.out and not a.out.exists(),'fresh draft arguments required');base.write(a.out,draft(a.kind,a.profile))
 elif a.action=='inspect':p=base.read(a.protocol);validate(p);print(json.dumps({'id':p['id'],'fingerprint':fingerprint(p),'status':p['status']}))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol),indent=2))
if __name__=='__main__':main()
