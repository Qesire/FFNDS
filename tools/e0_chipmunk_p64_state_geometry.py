"""Fixed paired saved-state geometry; no Torch/model/video operations."""
import argparse,filecmp,hashlib,importlib.metadata,json,math,os,pathlib,resource,signal,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];ID='E0_CHIPMUNK_P64_STATE_GEOMETRY_20261030';SHAPE=(16,21,58,104)
LIMITS={'cpu_threads':2,'wall_seconds':900,'outer_seconds':960,'read_bytes':512<<20,'output_bytes':16<<20,'memory_bytes':2<<30,'gpu':0,'model_calls':0,'attempts':1}
def need(x,m):
 if not x:raise ValueError(m)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def ref(root,b):
 p=(root/b['path']).resolve();need(p.is_relative_to(root.resolve()) and p.is_file() and p.stat().st_size==b['bytes'] and digest(p)==b['sha256'],'asset identity');return p
def components(dense,candidate):
 import numpy as np
 a=np.asarray(dense,dtype=np.float64);b=np.asarray(candidate,dtype=np.float64);need(a.shape==b.shape and a.ndim==4 and np.isfinite(a).all() and np.isfinite(b).all(),'shape/finite');e=b-a;spatial=e.mean(axis=(2,3),keepdims=True);dc=spatial.mean(axis=1,keepdims=True);temporal=spatial-dc;residual=e-spatial
 terms=[np.broadcast_to(dc,e.shape),np.broadcast_to(temporal,e.shape),residual];dot=lambda x,y:float(np.sum(x*y,dtype=np.float64));energies=[dot(x,x) for x in terms];cross=[2*dot(terms[i],terms[j]) for i,j in ((0,1),(0,2),(1,2))];total=dot(e,e);den=dot(a,a);closure=total-sum(energies)-sum(cross);bound=float(128*np.finfo(np.float64).eps*e.size*max(1.,total,sum(energies),sum(abs(c) for c in cross)));need(abs(closure)<=bound,'energy closure')
 return {'error_energy':total,'reference_energy':den,'relative_sse':total/den if den else None,'relative_l2':math.sqrt(total/den) if den else None,'zero_reference':den==0,'dc_energy':energies[0],'latent_time_ac_energy':energies[1],'spatial_ac_energy':energies[2],'signed_twice_cross_dc_time':cross[0],'signed_twice_cross_dc_space':cross[1],'signed_twice_cross_time_space':cross[2],'closure_error':closure,'closure_tolerance':bound,'axes':'channel, latent_time, height, width'}
def authorize(p,d):
 need(p.get('id')==ID and p.get('assets_finalized') is True and p.get('limits')==LIMITS,'scope not frozen');a=json.loads((R/'research/ACTIVE_RESEARCH.json').read_text());need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'],'authorization inactive');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(len(rows)==1 and rows[0].get('execution_status')=='RUNNING_FIXED_SCOPE' and rows[0].get('protocol_sha256')==d and rows[0].get('limits')==LIMITS and rows[0].get('max_gpu_cards')==0,'ACTIVE missing');q=rows[0];review=ref(R,q['review']);v=json.loads(review.read_text());need(v.get('state')=='REVIEWER_RECHECK_APPROVED' and v.get('scope_id')==ID and v.get('protocol_sha256')==d and v.get('job_spec_sha256')==q.get('launch_spec_sha256') and v.get('author_agent_id') and v.get('reviewer_agent_id') and v['author_agent_id']!=v['reviewer_agent_id'] and v.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'review');return q
def save(p,obj):
 raw=json.dumps(obj,indent=2,allow_nan=False).encode();need(len(raw)<LIMITS['output_bytes'],'output limit');tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
def load_tensor(path,info):
 import numpy as np
 with open(path,'rb') as f:
  n=int.from_bytes(f.read(8),'little');need(0<n<=1<<20,'header');h=json.loads(f.read(n));key='flow' if info['dtype']=='float32' else 'latents';t=h[key];data=f.read()
 need(set(h)-{'__metadata__'}=={key} and t['shape']==list(SHAPE) and t['dtype']==('F32' if key=='flow' else 'BF16') and t['data_offsets']==[0,len(data)] and hashlib.sha256(data).hexdigest()==info['raw_tensor_sha256'],'tensor header/hash')
 if key=='flow':a=np.frombuffer(data,dtype='<f4').reshape(SHAPE)
 else:a=(np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(SHAPE)
 need(np.isfinite(a).all(),'nonfinite');return a

def check_rlimit(soft,hard):
 need(0<=soft<=LIMITS['memory_bytes'] and 0<=hard<=LIMITS['memory_bytes'],'hard address-space limit')

def attach_case(report,case):
 row={'source':case['source'],'seed':42,'steps':[]};report['cases'].append(row);return row

def run(args):
 p=json.loads(args.protocol.read_text());d=digest(args.protocol);scope=authorize(p,d);need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and digest(args.job_spec)==scope['launch_spec_sha256'],'job/CPU isolation');j=json.loads(args.job_spec.read_text());need(j['driver_argv']==sys.argv and j['protocol_sha256']==d and pathlib.Path(j['cwd']).resolve()==R and sorted(os.sched_getaffinity(0))==j['cpu_affinity'] and len(j['cpu_affinity'])==2 and pathlib.Path(sys.executable).resolve()==(R/'.venv/bin/python').resolve() and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'job identity');sys.path.insert(0,str(R/'tools'));import worker_identity;worker_identity.require_worker();need(digest(worker_identity.private_config_path())==p['worker_config_sha256'],'worker identity');profile=str(worker_identity.private_config_path().resolve());expected=['env','FFNDS_WORKER_CONFIG='+profile,'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c',','.join(map(str,j['cpu_affinity'])),'prlimit','--as=2147483648:2147483648',str(R/'.venv/bin/python'),'-I','-B']+sys.argv;need(j['argv']==expected and j['external_timeout_seconds']==LIMITS['outer_seconds'],'exact launcher');out=(args.hf_root/p['output_HF_relative']).resolve();need(out.is_relative_to(args.hf_root.resolve()) and not out.exists() and pathlib.Path(j['output_dir']).resolve()==out.parent and pathlib.Path(j['run_json']).resolve()==out/'RUN.json' and pathlib.Path(j['hf_root']).resolve()==args.hf_root.resolve(),'exclusive output');out.mkdir(parents=True);r={'id':ID,'protocol_sha256':d,'status':'ATTEMPT','predebited_bytes':0,'completed_bytes':0,'inflight':None,'cases':[],'automatic_retry':False};save(out/'ATTEMPT.json',r);start=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall')));signal.alarm(LIMITS['wall_seconds'])
 def charged(item,fn):
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=LIMITS['memory_bytes'],'RSS budget');need(sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink())<=LIMITS['output_bytes'],'sampled worker output budget');n=item['bytes'];need(r['predebited_bytes']+n<=LIMITS['read_bytes'],'read budget');r['predebited_bytes']+=n;r['inflight']={'bytes':n,'completion':'UNKNOWN'};save(out/'RUN.json',r);x=fn();r['completed_bytes']+=n;r['inflight']=None;save(out/'RUN.json',r);return x
 try:
  check_rlimit(*resource.getrlimit(resource.RLIMIT_AS));need(charged({'bytes':p['python_bytes']},lambda:digest(pathlib.Path(sys.executable)))==p['python_sha256'] and importlib.metadata.version('numpy')==p['numpy_version'],'numpy/python identity');need(len(p['cases'])==2 and {x['source'] for x in p['cases']}=={898,96},'case coverage')
  for b in p['repo_sources']:charged(b,lambda b=b:ref(R,b))
  for parent in p['parents']:
   def meta(root,b):return charged({'bytes':2*b['bytes']},lambda:json.loads(ref(root,b).read_text()))
   proto=meta(R,parent['protocol']);pr=meta(args.hf_root,parent['RUN']);ex=meta(args.hf_root,parent['exit']);vr=meta(args.hf_root,parent['verification_RUN']);ve=meta(args.hf_root,parent['verification_exit']);need(pr['id']==proto['id']==ex['id']==parent['id'] and pr['status']=='COMPLETE_CHIPMUNK_P64_VIDEO_V2_EXIT_PENDING' and ex['child_returncode']==0 and ex['process_exited'] and ex['run_sha256']==parent['RUN']['sha256'] and ex['protocol_sha256']==parent['protocol']['sha256'] and vr['status']=='PASS_CLOSED_CHIPMUNK_P64_VIDEO_V2' and vr['RUN']==parent['RUN'] and vr['exit']==parent['exit'] and vr['predebited_bytes']==vr['completed_bytes'] and vr['inflight']=='NONE' and ve['child_returncode']==0 and ve['process_exited'] and ve['run_sha256']==parent['verification_RUN']['sha256'],'parent actual exit')
   case=next(x for x in p['cases'] if x['source']==pr['source_ordinal']);need(case['parent_id']==proto['id'] and [x['mode'] for x in pr['phases']]==['original_dense','hook_dense','cache_dense','global_p64'],'parent phase');
   for mode,idx in (('original_dense',0),('global_p64',3)):
    need(all(case['arms'][mode][key]==pr['phases'][idx]['states'][key] for key in case['arms'][mode]),'state not in parent')
  for case in p['cases']:
   row=attach_case(r,case);save(out/'RUN.json',r);modes=case['arms'];paths={}
   for mode in ('original_dense','global_p64'):
    for key,b in modes[mode].items():paths[mode,key]=charged(b,lambda b=b:ref(args.hf_root,b))
   x=modes['original_dense']['initial_noise'];y=modes['global_p64']['initial_noise'];need(x['sha256']==y['sha256'] and x['raw_tensor_sha256']==y['raw_tensor_sha256'] and x['bytes']==y['bytes'],'noise metadata differs');charged({'bytes':x['bytes']+y['bytes']},lambda:need(filecmp.cmp(paths['original_dense','initial_noise'],paths['global_p64','initial_noise'],shallow=False),'actual initial noise differs'))
   for i in range(4):
    step={'step':i};row['steps'].append(step);save(out/'RUN.json',r)
    for k in ('flow','latent'):
     key=f'{k}_step{i}';
     if i==0:
      budget=modes['original_dense'][key]['bytes']+modes['global_p64'][key]['bytes'];step.setdefault('step0_byte_controls',{})[k]=charged({'bytes':budget},lambda:filecmp.cmp(paths['original_dense',key],paths['global_p64',key],shallow=False));save(out/'RUN.json',r)
     a=charged(modes['original_dense'][key],lambda:load_tensor(paths['original_dense',key],modes['original_dense'][key]));b=charged(modes['global_p64'][key],lambda:load_tensor(paths['global_p64',key],modes['global_p64'][key]));step[k]=components(a,b)
    save(out/'RUN.json',r)
   save(out/'RUN.json',r)
  summary=[]
  for i in range(4):
   step={'step':i}
   for k in ('flow','latent'):
    vals=[x['steps'][i][k]['relative_sse'] for x in r['cases']];step[k]={'source_equal_relative_sse':sum(vals)/len(vals) if all(v is not None for v in vals) else None,'independent_sources':2}
   summary.append(step)
  r['source_equal_summary']=summary;need(r['predebited_bytes']==r['completed_bytes']==p['expected_total_read_bytes'] and r['inflight'] is None,'read ledger incomplete');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=LIMITS['memory_bytes'],'RSS budget');need(sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink())<=LIMITS['output_bytes'],'sampled output budget')
  r.update(status='COMPLETE_PAIRED_STATE_GEOMETRY',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,interpretation='Observed prefix-dependent trajectories; step contrasts are not isolated action causality.');save(out/'RUN.json',r)
 except BaseException as e:r.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());save(out/'RUN.json',r);raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=pathlib.Path,required=True);ap.add_argument('--hf-root',type=pathlib.Path,required=True);ap.add_argument('--job-spec',type=pathlib.Path,required=True);run(ap.parse_args())
if __name__=='__main__':main()
