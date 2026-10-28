"""B1 saved-state full-M terminal-head finite perturbation scope (draft)."""
import argparse,hashlib,json,os,shutil,signal,sys,time,traceback,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_B_TERMINAL_HEAD_PROBE_20261028'
LIMITS={'head_calls':72,'layer_norm_calls':72,'qr_calls':4,'direction_rows':1024,'CPU_response_rows':16384,'full_model':0,'FFN':0,'QDQ':0,'fit':0,'video':0}
RESOURCE={'GPU':1,'CPU_threads':2,'compute_seconds':600,'outer_seconds':900,'output_bytes':512<<20,'free_bytes':10<<30,'GPU_reserved_bytes':8<<30}


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()


def bound(root,item):
 p=(root/item['path']).resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=item['sha256'] or ('bytes' in item and p.stat().st_size!=item['bytes']):raise ValueError('Asset binding mismatch')
 return p


def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def authorize(p,digest,active):
 if p.get('id')!=ID or p.get('assets_finalized') is not True or p.get('limits')!=LIMITS or p.get('resources')!=RESOURCE:raise ValueError('Unfrozen or altered scope')
 u=active.get('user_continuation_authorization_20261027',{})
 if u.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or u.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION' or u.get('new_scope_registration_allowed') is not True or u.get('max_simultaneously_assigned_idle_GPUs')!=4:raise ValueError('Inactive user authorization')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Unregistered exact scope')


def reserve(report,cost,persist):
 for k,v in cost.items():
  if k not in LIMITS or type(v)!=int or v<0 or report['predebited'][k]+v>LIMITS[k]:raise ValueError('Budget exceeded before work')
 for k,v in cost.items():report['predebited'][k]+=v
 persist()


def native_head(torch,u,weight,bias,scale,shift,eps=1e-6,observe=lambda *_:None):
 observe('layer_norm_calls','invoked')
 x=torch.nn.functional.layer_norm(u,(u.shape[1],),None,None,eps)
 observe('layer_norm_calls','completed')
 x.mul_(1+scale.reshape(-1)).add_(shift.reshape(-1))
 out=torch.empty((u.shape[0],weight.shape[1]),device=u.device,dtype=u.dtype)
 observe('head_calls','invoked')
 torch.addmm(bias,x,weight,out=out)
 observe('head_calls','completed')
 return out,x


def same_bits(torch,a,b):
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def strict_rows(torch,rows,total,expected=256):
 if rows.ndim!=1 or len(rows)!=expected or rows.dtype!=torch.int64 or len(torch.unique(rows))!=expected or bool((rows<0).any()) or bool((rows>=total).any()):raise ValueError('Original rows invalid')


def restore_layout(torch,t,stride,device):
 if len(stride)!=t.ndim or any(type(x)!=int or x<0 for x in stride):raise ValueError('Invalid recorded stride')
 out=torch.empty_strided(tuple(t.shape),tuple(stride),dtype=t.dtype,device=device);out.copy_(t);return out


def variants():
 return [(kind,a,sign) for kind in ('visible','norm','null') for a in (1/16,1/4) for sign in (-1,1)]+[('constant',1/4,s) for s in (-1,1)]+[('pregate',1/4,s) for s in (-1,1)]


def verify_versions(p):
 expected={'torch':'2.10.0+cu128','numpy':'2.4.6','safetensors':'0.8.0'}
 if p.get('runtime_versions')!=expected:raise ValueError('Runtime contract differs')
 actual={k:importlib.metadata.version(k) for k in expected}
 if actual!=expected:raise ValueError('Runtime versions differ')
 return actual

def response_readout(np,u,delta,T,native,finite,linear):
 z=u-u.mean(axis=1,keepdims=True);D=u.shape[1]
 l2=np.linalg.norm(delta,axis=1);unorm=np.linalg.norm(u,axis=1);znorm=np.linalg.norm(z,axis=1);tiny=np.finfo(np.float64).tiny
 head=np.linalg.norm(delta@T.T,axis=1)/(max(np.linalg.norm(T),tiny)*np.maximum(l2,tiny))
 mean=np.abs(delta.sum(axis=1))/(np.sqrt(D)*np.maximum(l2,tiny))
 radial=np.abs((delta*z).sum(axis=1))/(np.maximum(znorm,tiny)*np.maximum(l2,tiny))
 def comparison(a,b):
  na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));return {'native_norm':na,'prediction_norm':nb,'error_SSE':float(np.square(a-b).sum()),'cosine':float(np.sum(a*b)/(na*nb)) if na and nb else None}
 def summary(x):return {'mean':float(np.mean(x)),'max':float(np.max(x))}
 return {'delta_l2':summary(l2),'delta_relative_l2':summary(l2/np.maximum(unorm,tiny)),'mean_leakage':summary(mean),'head_leakage':summary(head),'z_leakage':summary(radial),'native_vs_finite':comparison(native,finite),'native_vs_linear':comparison(native,linear),'finite_vs_linear':comparison(finite,linear)}


def preflight_only(protocol,hf):
 before='torch' in sys.modules
 p=json.loads(protocol.read_text());hf=hf.resolve()
 if p.get('id')!=ID or p.get('assets_finalized') is not True or p.get('limits')!=LIMITS or p.get('resources')!=RESOURCE:raise ValueError('Unfrozen scope')
 if len(p['cases'])!=4 or {(c['source_index'],c['step']) for c in p['cases']}!={(1,0),(1,3),(4,0),(4,3)}:raise ValueError('Wrong cases')
 verify_versions(p)
 for x in p['repo_sources']:bound(ROOT,x)
 for x in p['hf_assets']:bound(hf,x)
 assets={x['path']:x['sha256'] for x in p['hf_assets']}
 for item in [p['b0_run'],p['b0_exit']]+[c[k] for c in p['cases'] for k in ('u','capture','case','head')]:
  if assets.get(item['path'])!=item['sha256']:raise ValueError('Missing asset binding')
 br=json.loads(bound(hf,p['b0_run']).read_text());be=json.loads(bound(hf,p['b0_exit']).read_text())
 if br['status']!=p['b0_success_status'] or be.get('launcher_returncode')!=0 or be.get('process_exited') is not True or be.get('protocol_sha256')!=br.get('protocol_sha256'):raise ValueError('B0 producer not complete')
 return {'status':'PASS_CPU_BINDINGS_ONLY','cases':len(p['cases']),'assets':len(p['hf_assets']),'torch_loaded_before':before,'torch_loaded_after':'torch' in sys.modules,'QR_calls':0}


def run(args):
 p=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(p,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
 if len(p['cases'])!=4 or {(x['source_index'],x['step']) for x in p['cases']}!={(1,0),(1,3),(4,0),(4,3)}:raise ValueError('Wrong fixed cases')
 hf=args.hf_root.resolve();out=(hf/p['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output outside HF')
 sys.path.insert(0,str(ROOT/'tools'));from worker_identity import require_worker
 require_worker();out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'predebited':dict.fromkeys(LIMITS,0),'observed':dict.fromkeys(LIMITS,0),'invoked':{'head_calls':0,'layer_norm_calls':0,'qr_calls':0},'cases':[]};monitor=None;start=time.monotonic();torch=None
 def persist():write(out/'RUN.json',report)
 def check():
  if monitor:monitor.check()
  if shutil.disk_usage(out).free<RESOURCE['free_bytes'] or sum(x.stat().st_size for x in out.rglob('*') if x.is_file())>RESOURCE['output_bytes']:raise RuntimeError('Disk/output budget')
  if torch is not None and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>RESOURCE['GPU_reserved_bytes']:raise RuntimeError('GPU memory budget')
 def timer(*_):raise TimeoutError('Compute wall exhausted')
 previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in previous_handlers:signal.signal(sig,timer)
 signal.alarm(RESOURCE['compute_seconds'])
 try:
  report['runtime_package_versions']=verify_versions(p)
  for x in p['repo_sources']:bound(ROOT,x)
  for x in p['hf_assets']:bound(hf,x)
  assets={x['path']:x['sha256'] for x in p['hf_assets']}
  for key in ('b0_run','b0_exit'):
   if assets.get(p[key]['path'])!=p[key]['sha256']:raise ValueError('B0 receipt not bound')
  br=json.loads(bound(hf,p['b0_run']).read_text());be=json.loads(bound(hf,p['b0_exit']).read_text())
  if br['status']!=p['b0_success_status'] or be.get('launcher_returncode')!=0 or be.get('process_exited') is not True or be.get('protocol_sha256')!=br.get('protocol_sha256'):raise ValueError('B0 producer not complete')
  for c in p['cases']:
   for key in ('u','capture','case','head'):
    if assets.get(c[key]['path'])!=c[key]['sha256']:raise ValueError('Missing case binding')
  snap=out/'EXECUTED_SOURCES';snap.mkdir();shutil.copyfile(args.protocol,snap/'protocol.json')
  for x in p['repo_sources']:
   dst=snap/x['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/x['path'],dst)
  import torch as torch_api
  torch=torch_api
  import numpy as np
  from safetensors.torch import load_file,save_file
  from startup_gpu_reservation import reserve_before_assets
  import b_terminal_head_math as math
  reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
  if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Single A100 required')
  torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
  if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=br['BF16_reduced_precision_reduction']:raise ValueError('B0 BF16 reduction setting mismatch')
  report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'precision':torch.get_float32_matmul_precision(),'TF32':torch.backends.cuda.matmul.allow_tf32,'BF16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
  os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
  for c in p['cases']:
   check();case=json.loads(bound(hf,c['case']).read_text());op=case['actual_head_operator'];raw=load_file(str(bound(hf,c['u'])),device='cpu');v=load_file(str(bound(hf,c['capture'])),device='cpu');head=load_file(str(bound(hf,c['head'])),device='cpu')
   if set(raw)!={'u'} or raw['u'].dtype!=torch.bfloat16 or tuple(raw['u'].shape)!=(31668,5120):raise ValueError('Bad full-M u')
   rows=v['rows'];strict_rows(torch,rows,31668)
   owner=[x for x in br['cases'] if x['index']==c['source_index']]
   if len(owner)!=1:raise ValueError('B0 source identity mismatch')
   captured=[x for x in owner[0]['step_captures'] if x['step']==c['step']]
   if len(captured)!=1 or captured[0]['run_id']!=case['run_id']:raise ValueError('B0 capture identity mismatch')
   for key,old in (('u','u_file'),('capture','capture_file')):
    if captured[0][old]['sha256']!=c[key]['sha256'] or case[old]['sha256']!=c[key]['sha256']:raise ValueError('B0 capture bytes mismatch')
   weight=head['head_weight_runtime_KN'];bias=head['head_bias'];scale=v['head_scale'];shift=v['head_shift']
   def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
   if tensor_sha(weight)!=op['weight']['raw_tensor_sha256'] or tensor_sha(bias)!=op['bias']['raw_tensor_sha256']:raise ValueError('Actual head tensor bytes differ')
   if weight.shape!=(5120,64) or weight.dtype!=torch.bfloat16 or bias.shape!=(64,) or bias.dtype!=torch.bfloat16 or op['norm_eps']!=1e-6:raise ValueError('Bad actual head binding')
   actual_u=case['actual_pre_head_u'];mod=case['actual_combined_head_modulation']
   if tensor_sha(raw['u'])!=actual_u['raw_tensor_sha256'] or c['u_stride']!=actual_u['stride']:raise ValueError('Actual u binding differs')
   if op['head_class']!='lightx2v.common.ops.mm.mm_weight.MMWeight' or op['norm_class']!='lightx2v.common.ops.norm.layer_norm_weight.LNWeight' or op['head_has_lora_branch'] or op['norm_weight'] is not None or op['norm_bias'] is not None or op['infer_dtype']!='torch.bfloat16' or op['sensitive_dtype']!='torch.bfloat16':raise ValueError('Native operator branch differs')
   for value,spec in ((scale,mod['scale']),(shift,mod['shift'])):
    if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype'] or tensor_sha(value)!=spec['raw_tensor_sha256']:raise ValueError('Modulation differs')
   u=restore_layout(torch,raw['u'],actual_u['stride'],'cuda');w=restore_layout(torch,weight,op['weight']['stride'],'cuda');bias=restore_layout(torch,bias,op['bias']['stride'],'cuda');scale=restore_layout(torch,scale,mod['scale']['stride'],'cuda');shift=restore_layout(torch,shift,mod['shift']['stride'],'cuda');idx=rows.to('cuda');original=v['head_output'].to('cuda');expected_input=v['head_input_rows'].to('cuda');row={'source_index':c['source_index'],'step':c['step'],'variants':[]};report['cases'].append(row)
   def call(value):
    check();authorize(p,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()));reserve(report,{'head_calls':1,'layer_norm_calls':1},persist)
    def observe(key,phase):
     target=report['invoked'] if phase=='invoked' else report['observed'];target[key]+=1;persist()
    return native_head(torch,value,w,bias,scale,shift,observe=observe)
   for _ in range(2):
    baseline,hin=call(u)
    if not same_bits(torch,baseline,original) or not same_bits(torch,hin.index_select(0,idx),expected_input):raise RuntimeError('Neutral full-M byte replay failed')
   if not same_bits(torch,u.index_select(0,idx),v['u_rows'].to('cuda')):raise ValueError('u row binding mismatch')
   base_rows=u.index_select(0,idx);yr=v['Y'].to('cuda');gate=v['gate'].to('cuda');r=v['r_rows'].to('cuda');replayed=r.clone();replayed.add_(yr*gate)
   if not same_bits(torch,replayed,base_rows):raise RuntimeError('BF16 original gate writeback mismatch')
   effective=(1+scale).float().cpu().numpy().reshape(-1).astype(np.float64);scale_fp=scale.float().cpu().numpy().reshape(-1).astype(np.float64);report.setdefault('effective_gain_max_abs_diff',[]).append(float(np.max(np.abs(effective-(1+scale_fp)))))
   reserve(report,{'qr_calls':1,'direction_rows':256},persist);report['invoked']['qr_calls']+=1;persist()
   try:geo=math.head_geometry(weight.float().numpy().T,effective-1)
   except ValueError as error:
    if str(error)!='rank-deficient head geometry':raise
    report['status']='INCONCLUSIVE_RANK_DEFICIENT';row['status']='INCONCLUSIVE_RANK_DEFICIENT';persist();return
   report['observed']['qr_calls']+=1;persist()
   rows_np=base_rows.float().cpu().numpy().astype(np.float64);dirs=[]
   for vec in rows_np:
    dirs.append(math.directions(vec,geo));report['observed']['direction_rows']+=1
   if any(any(d[k] is None for k in ('visible','norm','null')) for d in dirs):
    report['status']='INCONCLUSIVE_DEGENERATE_DIRECTION';row['status']='INCONCLUSIVE_DEGENERATE_DIRECTION';persist();return
   pair_results={}
   for vi,(kind,amplitude,sign) in enumerate(variants()):
    check();dname='visible' if kind=='pregate' else kind;directions=np.stack([z[dname] for z in dirs]);target=yr.float().cpu().numpy().astype(np.float64) if kind=='pregate' else rows_np
    delta=np.stack([sign*math.nominal_delta(target[i],directions[i],amplitude) for i in range(256)])
    if kind=='pregate':
     new_y=(yr.float()+torch.from_numpy(delta).to('cuda',dtype=torch.float32)).to(torch.bfloat16);repl=r.clone();repl.add_(new_y*gate)
    else:repl=(base_rows.float()+torch.from_numpy(delta).to('cuda',dtype=torch.float32)).to(torch.bfloat16)
    modified=restore_layout(torch,raw['u'],c['u_stride'],'cuda');modified.index_copy_(0,idx,repl);output,_=call(modified)
    mask=torch.ones(31668,dtype=torch.bool,device='cuda');mask[idx]=False
    if not same_bits(torch,output[mask],baseline[mask]):raise RuntimeError('Unchanged rows differ')
    outrows=output.index_select(0,idx);actual=repl.cpu().float().numpy().astype(np.float64)-rows_np
    reserve(report,{'CPU_response_rows':256},persist);finite=[];linear=[]
    for i in range(256):
     q=math.finite_and_linear(rows_np[i],actual[i],geo['T']);finite.append(q['finite']);linear.append(q['linear']);report['observed']['CPU_response_rows']+=1
    finite=np.stack(finite);linear=np.stack(linear);native=outrows.cpu().float().numpy().astype(np.float64)-baseline.index_select(0,idx).cpu().float().numpy().astype(np.float64)
    stats=response_readout(np,rows_np,actual,geo['T'],native,finite,linear)
    pair_results[(kind,amplitude,sign)]=native
    path=out/(f"source{c['source_index']:02d}_step{c['step']}_variant{vi:02d}.safetensors")
    projected=sum(x.stat().st_size for x in out.rglob('*') if x.is_file())+repl.numel()*2+outrows.numel()*2+2*256*64*8+65536
    if projected>RESOURCE['output_bytes']:raise RuntimeError('Output budget before write')
    save_file({'u_rows':repl.cpu().contiguous(),'head_rows':outrows.cpu().contiguous(),'finite_prediction':torch.from_numpy(finite),'linear_prediction':torch.from_numpy(linear)},str(path))
    row['variants'].append({'kind':kind,'amplitude':amplitude,'sign':sign,'path':str(path.relative_to(hf)),'sha256':sha(path),'bytes':path.stat().st_size,'readout':stats});persist()
   row['paired_response']={}
   for kind in ('visible','norm','null'):
    vals={}
    for amp in (1/16,1/4):
     plus=pair_results[kind,amp,1];minus=pair_results[kind,amp,-1];vals[str(amp)]={'odd_l2':float(np.linalg.norm((plus-minus)/2)),'even_l2':float(np.linalg.norm((plus+minus)/2))}
    for metric in ('odd_l2','even_l2'):
     low=vals[str(1/16)][metric];vals[metric+'_high_over_low']=vals[str(1/4)][metric]/low if low else None
    row['paired_response'][kind]=vals
   row['neutral_repeat_difference_l2']=0.0;row['neutral_note']='Exact repeat-noop difference only; BF16 quantization floor not measured.';persist()
   del u,w,baseline,hin,raw,v,head
  if report['predebited']!=LIMITS or report['observed']!=LIMITS:raise RuntimeError('Incomplete operation ledger')
  report.update(status='COMPLETE_B1_SAVED_HEAD_DIAGNOSTIC',elapsed_seconds=time.monotonic()-start);check()
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),exception_traceback=traceback.format_exc());raise
 finally:
  original=sys.exc_info()[0] is not None;signal.alarm(0)
  for sig,handler in previous_handlers.items():signal.signal(sig,handler)
  err=None
  if monitor:
   try:
    monitor.close();report['assigned_gpu_resources']=monitor.report()
    if report['assigned_gpu_resources'].get('violations'):raise RuntimeError('Monitor violation')
   except BaseException as e:report['status']='FAILED_GPU_MONITOR';err=e
  persist()
  if err is not None and not original:raise err

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--hf-root',type=Path,required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args()
 if args.preflight:print(json.dumps(preflight_only(args.protocol,args.hf_root)))
 else:run(args)
