"""Independently authorized, bounded CPU verifier for SharQ quality scopes."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_p64_metric_adapter as adapter
from e0_sharq_wan_metric_read_budget import HashReads,replay
HF=Path('/home/xiang/.cache/huggingface')
RES={'GPU':0,'CPU_threads':1,'wall_seconds':1800,'outer_seconds':1920,'whole_file_sha_bytes_max':16<<30,'new_HF_bytes':32<<20,'worker_extra_bytes_max':64<<20,'worker_extra_enforcement':'SOFT_SAMPLED_AT_SHA_EVENTS','automatic_retry':False}

def load(kind):
 if kind=='visual':
  import e0_chipmunk_p64_visual_cpu as e
  return e
 return adapter.install(kind)

def fingerprint(kind,e,p):return e.fingerprint(p) if kind=='visual' else e.base.state_sha({k:v for k,v in p.items() if k not in ('status','assets_finalized','binding_status')})
def need(x,msg):
 if not x:raise ValueError(msg)
def verify(a):
 import signal
 e=load(a.kind);base=e.base;p=base.read(a.protocol);digest=base.sha(a.protocol);uid=p['id']+'_VERIFY';a.hf_root=a.hf_root.resolve();need(a.hf_root==HF,'HF root differs');a.protocol=a.protocol.resolve();a.job_spec=a.job_spec.resolve();active=base.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==uid];need(len(rows)==1,'verifier ACTIVE required');row=rows[0]
 need(row.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and row.get('protocol_sha256')==digest and row.get('parent_scope_id')==p['id'] and row.get('resources')==RES and row.get('max_gpu_cards')==0,'verifier scope differs');review=base.read_bound(ROOT,row['review']);need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==uid and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE' and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id'] and review.get('approval_scope_fingerprint')==fingerprint(a.kind,e,p) and review.get('verification_spec_sha256')==row.get('verification_spec_sha256'),'verifier independent review differs')
 for key in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,review[key])
 need(base.sha(a.job_spec)==row['verification_spec_sha256'],'verifier spec differs');j=base.read(a.job_spec);argv=j['argv'];need(len(argv)>1 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'verifier profile missing');profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and base.sha(profile)==p['execution_private_worker_config_sha256'],'verifier worker differs')
 cpus=sorted(os.sched_getaffinity(0));need(len(cpus)==1 and sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU isolation differs');out=HF/p['output_HF_relative'];dest=out.parent.parent/'verification/worker/results';script=ROOT/'tools/e0_chipmunk_p64_metric_verify.py';python=ROOT/'.venv/bin/python' if a.kind=='visual' else HF/p['import_environment']['python_HF_relative'];flags=['-I','-B'] if a.kind=='visual' else ['-I','-S','-B'];expected=['env',argv[1],'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(cpus[0]),str(python),*flags,str(script),'verify','--kind',a.kind,'--protocol',str(a.protocol),'--hf-root',str(HF),'--job-spec',str(a.job_spec)]
 need(argv==expected and j['id']==uid and j['protocol_sha256']==digest and Path(j['protocol_path']).resolve()==a.protocol and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'verifier exact job differs')
 import worker_identity;worker_identity.require_worker();parent_run=out/'RUN.json';parent_exit=out.parent/'EXIT_RECEIPT.json';b=base.binding(parent_run,HF);receipt=base.read(parent_exit);base.exited(receipt,b,digest,p['id']);need(receipt.get('spec_sha256')==base.read(parent_run).get('job_spec_sha256'),'parent actual EXIT differs')
 need(not dest.exists(),'fresh verification required');dest.mkdir(parents=True);r={'id':uid,'parent_scope_id':p['id'],'protocol_sha256':digest,'job_spec_sha256':row['verification_spec_sha256'],'status':'STARTED_VERIFY_NO_RETRY','automatic_retry':False};base.write(dest/'RUN.json',r);start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verifier wall limit')));signal.alarm(RES['wall_seconds']);guard=HashReads(base,dest.parent/'HASH_READS.jsonl','VERIFY');snapshot=None;old=None;entered=False
 try:
  guard.__enter__();entered=True
  if a.kind!='visual':
   import e0_teal_output_budget_v3 as snapshots
   snapshot=snapshots.old.base;old=snapshot.sha;snapshot.sha=guard.sha
  e._authorized_verifier=True;result=e.verify_closed(SimpleNamespace(protocol=a.protocol,hf_root=HF,run=parent_run,exit=parent_exit));r.update(status='PASS_VERIFICATION_EXIT_PENDING',result=result)
 except BaseException as error:r.update(status='FAILED_VERIFY_NO_RETRY',error_type=type(error).__name__,error=str(error)[:500]);raise
 finally:
  signal.alarm(0)
  if snapshot is not None:snapshot.sha=old
  typ,value,tb=sys.exc_info();failure=None
  if entered:
   try:guard.__exit__(typ,value,tb)
   except BaseException as error:failure=error;r.update(status='FAILED_VERIFY_HASH_LEDGER_NO_RETRY',error_type=type(error).__name__,error=str(error)[:500])
  data=guard.path.read_bytes() if entered else b'';r.update(wall_seconds=time.monotonic()-start,predebited_bytes=guard.predebited,completed_bytes=guard.completed,inflight=guard.inflight,hash_journal={'path':str(guard.path.relative_to(HF)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},metadata_reads='small receipt/config and parent byte-verifier subprocess separately UNKNOWN');base.write(dest/'RUN.json',r)
  if failure:raise failure
 return r

def inspect(a):
 e=load(a.kind);base=e.base;p=base.read(a.protocol);digest=base.sha(a.protocol);out=HF/p['output_HF_relative'];run=out.parent.parent/'verification/worker/results/RUN.json';ex=run.parent.parent/'EXIT_RECEIPT.json';r=base.read(run);receipt=base.read(ex);binding=base.binding(run,HF);base.exited(receipt,binding,digest,p['id']+'_VERIFY');need(r['status']=='PASS_VERIFICATION_EXIT_PENDING' and receipt.get('spec_sha256')==r['job_spec_sha256'] and r['predebited_bytes']==r['completed_bytes'] and r['inflight'] is None,'actual verifier EXIT0 required');journal=base.bound(HF,r['hash_journal']);account=replay(journal);need(account=={'predebited_bytes':r['predebited_bytes'],'completed_bytes':r['completed_bytes'],'inflight':None,'status':'COMPLETE','pending':0},'verifier hash ledger differs');return {'status':'PASS_ACTUAL_EXIT_CHIPMUNK_P64_QUALITY_CLOSED','parent_id':p['id'],'verification_RUN':binding,'verification_exit':base.binding(ex,HF)}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','inspect'));p.add_argument('--kind',choices=('quality','raft','visual'),required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--job-spec',type=Path);p.add_argument('--hf-root',type=Path,default=HF);a=p.parse_args();print(json.dumps(verify(a) if a.action=='verify' else inspect(a),indent=2))
if __name__=='__main__':main()
