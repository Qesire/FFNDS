"""Two-source CPU matched-output diagnostics; independently reviewed scope only."""
import argparse, copy, hashlib, json, os, resource, signal, stat, struct, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_student_observation_owner as parent
from chipmunk_student_observation_coordinates import coordinates
native=parent.native;HF=parent.HF;PYTHON=ROOT/'.venv/bin/python'
SELF='tools/e0_chipmunk_p64_local_prefix_v2_owner.py';JOBS='tools/e0_chipmunk_p64_local_prefix_v2_jobs.py';MATH='tools/e0_chipmunk_p64_local_prefix_v2_math.py';TEST='tests/test_e0_chipmunk_p64_local_prefix_v2.py'
RES={'GPU':0,'CPU_threads':2,'wall_seconds':3600,'outer_seconds':3720,'receipt_wait_max_seconds':30,'receipt_poll_interval_ms':50,'read_bytes_max':6<<30,'metadata_read_bytes_max':256<<20,'max_RSS_bytes':4<<30,'address_space_max_bytes':4<<30,'new_HF_bytes':32<<20,'automatic_retry':False}
SOURCES=(898,979);SNAPSHOTS=(0,13,26,39);MODES=parent.MODES
AUTHORS=('/root','/root/chipmunk_student_discrimination_terra')
DATA_BYTES=40*3*320*5120*2+4*3*31668*5120*2
HEADER_MAX=1<<20;META_MAX=2<<20

def need(v,msg):
 if not v:raise ValueError(msg)
def sid(part):
 need(type(part) is int and part in (0,1),'part differs');return f'E0_CHIPMUNK_P64_LOCAL_PREFIX_V2_20261030_PART{part}'
def outpath(part):return f'ffnds-derived/e0_chipmunk_p64_local_prefix_v2_20261030/part{part}/attempt1/worker/results'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def binding(path,root):
 path=Path(path).resolve();root=Path(root).resolve();need(path.is_relative_to(root) and path.is_file(),'binding escaped root');return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)}
def read_bound(root,b):
 p=(Path(root)/b['path']).resolve();need(p.is_relative_to(Path(root).resolve()) and p.is_file() and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],'bound metadata differs');return native.read(p)

class MetadataBudget:
 """Predebit and durably persist each bootstrap metadata read."""
 def __init__(self,persist=lambda:None,initial=0):self.pre=initial;self.done=initial;self.calls=int(initial>0);self.inflight=None;self.persist=persist
 def wrap(self,fn):
  def counted(path,*args,**kwargs):
   size=Path(path).stat().st_size;need(size>=0 and self.pre+size<=RES['metadata_read_bytes_max'],'bootstrap metadata read limit')
   self.pre+=size;self.inflight={'bytes':size,'completed':'UNKNOWN'};self.persist()
   value=fn(path,*args,**kwargs);self.done+=size;self.calls+=1;self.inflight=None;self.persist();return value
  return counted
 def __enter__(self):
  global sha
  self.old_sha=sha;self.old_native_sha=native.sha;self.old_native_read=native.read
  sha=self.wrap(sha);native.sha=self.wrap(native.sha);native.read=self.wrap(native.read);return self
 def __exit__(self,*_):
  global sha
  sha=self.old_sha;native.sha=self.old_native_sha;native.read=self.old_native_read
def fingerprint(p):return native.canonical({k:v for k,v in p.items() if k not in ('status','assets_finalized','review')})

def require_verified_parent_bindings(verifier,run_binding,exit_binding):
 need(verifier.get('RUN')==run_binding and verifier.get('exit')==exit_binding,'verifier references different parent RUN/EXIT')

class ReceiptWaitTimeout(TimeoutError):
 def __init__(self,checks,elapsed):
  super().__init__('supervisor start receipt did not appear before bounded deadline');self.checks=checks;self.elapsed=elapsed

def wait_for_start_receipt(path,scope_start,*,clock=time.monotonic,sleep=time.sleep):
 """Bounded stat-only wait; the caller performs exactly one budgeted read."""
 begin=clock();end=min(begin+RES['receipt_wait_max_seconds'],scope_start+RES['wall_seconds']);checks=0
 while True:
  now=clock()
  if now>=end:raise ReceiptWaitTimeout(checks,now-begin)
  checks+=1
  try:
   info=Path(path).lstat();need(stat.S_ISREG(info.st_mode),'start receipt is not a regular file')
   return {'checks':checks,'elapsed_seconds':clock()-begin}
  except FileNotFoundError:
   remaining=end-clock()
   if remaining<=0:raise ReceiptWaitTimeout(checks,clock()-begin)
   sleep(min(RES['receipt_poll_interval_ms']/1000,remaining))

def parent_proof(part):
 sid(part);protocol=ROOT/f'config/e0_chipmunk_student_observation_20261030_part{part}_pending_review_v2.json'
 need(protocol.is_file(),'parent V2 protocol missing');plan=native.read(protocol);parent.validate(plan,True)
 # This checks actual parent verifier RUN/EXIT, never reads operand payloads.
 verified=parent.inspect_verification(SimpleNamespace(protocol=protocol))
 out=HF/plan['output_HF_relative'];run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json'
 report=native.read(run);receipt=native.read(ex);rb=binding(run,HF)
 parent.vm.exited(receipt,rb,native.sha(protocol),plan['id'])
 vr=read_bound(HF,verified['verification_RUN']);ve=read_bound(HF,verified['verification_exit'])
 eb=binding(ex,HF);require_verified_parent_bindings(vr,rb,eb)
 need(verified['status']=='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_STUDENT_OBSERVATION' and
      vr['artifact_count']==310 and vr['parent_scope_id']==plan['id'] and vr['predebited_bytes']==vr['completed_bytes'] and vr['inflight']=='NONE' and
      ve['process_exited'] is True and ve['child_returncode']==0 and report['source_ordinal']==SOURCES[part] and report['seed']==42 and
      report['status']=='COMPLETE_CHIPMUNK_STUDENT_OBSERVATION_EXIT_PENDING' and receipt['child_returncode']==0 and
      [x['mode'] for x in report['phases']]==list(MODES),'parent completion differs')
 parent.verify_reference_metadata(plan['source_parent'],report['phases'],report['prefix_byte_controls'],
     [{'mode':k,'equal':v} for k,v in report['neutral_video_checks'].items()])
 rows=[]
 for block in range(40):
  records=[ph['observer_manifest']['records'][str(block)] for ph in report['phases']]
  for name in ('old_h','old_Y'):
   need(records[0]['samples']['tensors'][name]['raw_sha256']==records[1]['samples']['tensors'][name]['raw_sha256'],'old cache sample differs')
  if block==0:
   for name in ('V1','current_h','current_dense_Y'):
    need(records[0]['samples']['tensors'][name]['raw_sha256']==records[1]['samples']['tensors'][name]['raw_sha256'],'block0 current sample differs')
  if block in SNAPSHOTS:
   for name in ('old_h','old_Y'):
    need(records[0]['snapshot'][name]['tensors'][name]['raw_sha256']==records[1]['snapshot'][name]['tensors'][name]['raw_sha256'],'old cache snapshot differs')
   if block==0:
    for name in ('V1','current_h','current_dense_Y'):
     need(records[0]['snapshot'][name]['tensors'][name]['raw_sha256']==records[1]['snapshot'][name]['tensors'][name]['raw_sha256'],'block0 current snapshot differs')
  rows.append({'block':block,'dense_metadata':records[0]['metadata'],'student_metadata':records[1]['metadata'],
   'dense_sample':records[0]['samples'],'student_sample':records[1]['samples'],
   'dense_snapshot':{n:records[0]['snapshot'][n] for n in ('current_dense_Y',)} if block in SNAPSHOTS else {},
   'student_snapshot':{n:records[1]['snapshot'][n] for n in ('current_dense_Y','candidate_Y')} if block in SNAPSHOTS else {}})
 return {'protocol':binding(protocol,ROOT),'RUN':rb,'EXIT':eb,'verification_RUN':verified['verification_RUN'],
         'verification_EXIT':verified['verification_exit'],'parent_id':plan['id'],'source_ordinal':SOURCES[part], 'sites':rows}

def draft(part,profile,affinity):
 entry=parent_proof(part);need(profile.is_file() and len(affinity)==2 and len(set(affinity))==2 and set(affinity)<=os.sched_getaffinity(0),'profile/affinity differs')
 source_paths=(SELF,JOBS,MATH,TEST,'tools/chipmunk_student_observation_coordinates.py','tools/worker_identity.py')
 return {'id':sid(part),'part':part,'status':'DRAFT_REVIEW_REQUESTED','assets_finalized':False,'review':None,'source_ordinal':SOURCES[part],
  'seed':42,'parent':entry,'resources':RES,'cpu_affinity':affinity,'execution_private_worker_config_sha256':sha(profile),
  'output_HF_relative':outpath(part),'source_bindings':[binding(ROOT/x,ROOT) for x in source_paths],
  'logical_Y_bytes':DATA_BYTES,'header_max_bytes':HEADER_MAX,'metadata_max_bytes':META_MAX,'snapshot_blocks':list(SNAPSHOTS),
  'source_role':'EXPOSED_DEVELOPMENT','authors':list(AUTHORS),'quality_acceptance':False,'automatic_retry':False,
  'interpretation':'FP64 output-level local/prefix decomposition; no H3 causality or cross-trajectory residual-write claim'}

def validate(p,frozen=False):
 part=p['part'];need(p['id']==sid(part) and p['source_ordinal']==SOURCES[part] and p['seed']==42 and p['resources']==RES and
 p['logical_Y_bytes']==DATA_BYTES and p['header_max_bytes']==HEADER_MAX and p['metadata_max_bytes']==META_MAX and
 p['snapshot_blocks']==list(SNAPSHOTS) and p['output_HF_relative']==outpath(part) and p['quality_acceptance'] is False and
 p['automatic_retry'] is False and p['authors']==list(AUTHORS) and len(p['cpu_affinity'])==len(set(p['cpu_affinity']))==2,'scope differs')
 need((p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True if frozen else p['status']=='DRAFT_REVIEW_REQUESTED' and p['assets_finalized'] is False) and p['review'] is None,'scope status differs')
 need(p['parent']==parent_proof(part),'closed parent changed')
 expected={SELF,JOBS,MATH,TEST,'tools/chipmunk_student_observation_coordinates.py','tools/worker_identity.py'}
 need({b['path'] for b in p['source_bindings']}==expected and len(p['source_bindings'])==len(expected),'source set differs')
 for b in p['source_bindings']:
  path=(ROOT/b['path']).resolve();need(path.is_relative_to(ROOT) and path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'source changed')
 return True

def authorize(p,digest):
 validate(p,True);active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']]
 need(len(rows)==1 and active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'ACTIVE required');scope=rows[0]
 need(scope.get('execution_status')=='RUNNING_FIXED_SCOPE' and scope.get('protocol_sha256')==digest and scope.get('resources')==RES and scope.get('max_gpu_cards')==0,'scope differs')
 review=read_bound(ROOT,scope['review'])
 need(review.get('scope_id')==p['id'] and review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE' and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id'] and review.get('approval_scope_fingerprint')==fingerprint(p) and review.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['source_bindings']} and review.get('launch_spec_sha256')==scope.get('launch_spec_sha256') and review.get('responses_to_review') and review.get('adversarial_counterexample'),'review differs')
 for key in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,review[key])
 need(review['reviewer_agent_id'] not in AUTHORS,'reviewer must be independent of all authors')
 return scope

def ident(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read_slice(root,item,names,expected_rows,charge):
 """Read named BF16 spans once; fd/path identity and each raw SHA checked."""
 b=item['file'];p=(Path(root)/b['path']).resolve();need(p.is_relative_to(Path(root).resolve()),'payload path escapes root')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size==b['bytes'] and ident(p.stat())==ident(before),'payload identity differs')
  prefix=charge('header_prefix',8,lambda:os.read(fd,8));need(len(prefix)==8,'truncated header prefix')
  n=struct.unpack('<Q',prefix)[0];need(0<n<=HEADER_MAX and 8+n<=before.st_size,'header bound')
  header=json.loads(charge('header',n,lambda:os.read(fd,n)))
  spans=[]
  for name in names:
   recorded=item['tensors'][name];entry=header[name];shape=[expected_rows,5120]
   need(recorded['dtype']=='bfloat16' and recorded['shape']==shape and entry['dtype']=='BF16' and entry['shape']==shape and entry['data_offsets']==recorded['data_offsets'],'tensor header differs')
   a,z=entry['data_offsets'];need(type(a) is int and type(z) is int and 0<=a<=z<=before.st_size-8-n and z-a==expected_rows*5120*2,'tensor span differs');spans.append((a,z,name,recorded['raw_sha256']))
  ordered=sorted(spans);need(all(ordered[i][1]<=ordered[i+1][0] for i in range(len(ordered)-1)),'overlapping selected spans')
  values={}
  for a,z,name,expected_sha in spans:
   def read():
    os.lseek(fd,8+n+a,os.SEEK_SET);result=bytearray();h=hashlib.sha256();remaining=z-a
    while remaining:
     chunk=os.read(fd,min(8<<20,remaining));need(bool(chunk),'truncated payload');remaining-=len(chunk);h.update(chunk);result.extend(chunk)
    need(h.hexdigest()==expected_sha,'raw SHA differs');return result
   values[name]=charge('Y_payload',z-a,read,detail=[{'path':b['path'],'tensor':name,'bytes':z-a}])
  need(ident(os.fstat(fd))==ident(before) and ident(p.stat())==ident(before),'payload changed during read');return values
 finally:os.close(fd)

def stream_triplet(root,items,names,rows,width,chunk_rows,charge,process):
 """Hash three selected spans in one pass while processing matched row chunks."""
 files=[];file_hashes=[]
 try:
  need(len(items)==len(names)==3 and chunk_rows>0,'triplet identity differs')
  for item,name in zip(items,names):
   b=item['file'];path=(Path(root)/b['path']).resolve();need(path.is_relative_to(Path(root).resolve()),'payload path escapes root')
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd);files.append((fd,path,before))
   need(stat.S_ISREG(before.st_mode) and before.st_size==b['bytes'] and ident(path.stat())==ident(before),'payload identity differs')
   need(len({(s.st_dev,s.st_ino) for _,_,s in files})==len(files),'triplet files alias')
   prefix=charge('header_prefix',8,lambda fd=fd:os.read(fd,8));need(len(prefix)==8,'truncated header prefix')
   n=struct.unpack('<Q',prefix)[0];need(0<n<=HEADER_MAX and 8+n<=before.st_size,'header bound')
   header_bytes=charge('header',n,lambda fd=fd,n=n:os.read(fd,n));need(len(header_bytes)==n,'truncated header');header=json.loads(header_bytes);need(set(header)=={name},'full snapshot tensor set differs');entry=header[name];recorded=item['tensors'][name]
   need(entry['dtype']=='BF16' and entry['shape']==[rows,width] and recorded['dtype']=='bfloat16' and recorded['shape']==[rows,width] and entry['data_offsets']==recorded['data_offsets'],'tensor header differs')
   a,z=entry['data_offsets'];need(type(a) is int and type(z) is int and a==0 and z==before.st_size-8-n and z==rows*width*2,'singleton tensor span or trailing bytes differs')
   filehash=hashlib.sha256();filehash.update(prefix);filehash.update(header_bytes);file_hashes.append(filehash)
   os.lseek(fd,8+n+a,os.SEEK_SET)
  hashes=[hashlib.sha256() for _ in files]
  def read_and_process():
   for lo in range(0,rows,chunk_rows):
    hi=min(lo+chunk_rows,rows);size=(hi-lo)*width*2;chunks=[]
    for (fd,_,_),h,fh in zip(files,hashes,file_hashes):
     buf=bytearray()
     while len(buf)<size:
      block=os.read(fd,size-len(buf));need(bool(block),'truncated payload');buf.extend(block);h.update(block);fh.update(block)
     chunks.append(buf)
    process(lo,hi,chunks)
   for (_,path,before),h,fh,item,name in zip(files,hashes,file_hashes,items,names):
    need(h.hexdigest()==item['tensors'][name]['raw_sha256'],'raw SHA differs')
    need(fh.hexdigest()==item['file']['sha256'],'whole file SHA differs')
    need(ident(path.stat())==ident(before),'payload path changed')
   for fd,_,before in files:need(ident(os.fstat(fd))==ident(before),'payload fd changed')
  return charge('Y_payload',3*rows*width*2,read_and_process,
                detail=[{'path':item['file']['path'],'tensor':name,'bytes':rows*width*2} for item,name in zip(items,names)])
 finally:
  for fd,_,_ in files:os.close(fd)

def load_metadata(root,b,charge):
 need(0<b['bytes']<=META_MAX,'metadata size differs');path=(Path(root)/b['path']).resolve();need(path.is_relative_to(Path(root).resolve()),'metadata escape')
 def read():
  with path.open('rb') as f:
   before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size==b['bytes'],'metadata identity differs');data=f.read();need(ident(os.fstat(f.fileno()))==ident(before) and ident(path.stat())==ident(before),'metadata changed')
  need(len(data)==b['bytes'] and hashlib.sha256(data).hexdigest()==b['sha256'],'metadata SHA differs');return json.loads(data)
 return charge('metadata',b['bytes'],read)

def require_distinct_files(root,bindings):
 identities=[]
 for binding in bindings:
  path=(Path(root)/binding['path']).resolve();need(path.is_relative_to(Path(root).resolve()),'payload path escapes root')
  info=path.stat();need(stat.S_ISREG(info.st_mode) and info.st_size==binding['bytes'],'payload identity differs');identities.append((info.st_dev,info.st_ino))
 need(len(set(identities))==len(identities),'paired payload files alias')

def atomic(path,obj):
 data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode();need(len(data)<RES['new_HF_bytes']//2,'report output cap')
 worker=path.parent.parent;used=sum(x.stat().st_size for x in worker.rglob('*') if x.is_file() and not x.is_symlink());need(used+len(data)<=RES['new_HF_bytes'],'worker atomic output cap')
 tmp=path.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,path)

def run(a):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall budget')));signal.alarm(RES['wall_seconds'])
 old_as=resource.getrlimit(resource.RLIMIT_AS);need(old_as[1]==resource.RLIM_INFINITY or old_as[1]>=RES['address_space_max_bytes'],'address-space hard limit');resource.setrlimit(resource.RLIMIT_AS,(RES['address_space_max_bytes'],old_as[1]))
 # Minimal untrusted job peek identifies only a fixed, fresh worker directory.
 # Its <=1MiB read is predebited in memory and backfilled immediately to STARTUP.
 need(a.job_spec is not None and a.job_spec.is_file() and a.job_spec.stat().st_size<=1<<20,'job peek bound')
 peek_before=a.job_spec.stat();peek_size=peek_before.st_size
 with a.job_spec.open('rb') as stream:
  need(ident(os.fstat(stream.fileno()))==ident(peek_before),'job peek fd identity');peek_data=stream.read((1<<20)+1)
  need(len(peek_data)==peek_size and ident(os.fstat(stream.fileno()))==ident(peek_before) and ident(a.job_spec.stat())==ident(peek_before),'job peek changed')
 peek=json.loads(peek_data);parts=[i for i in (0,1) if peek.get('id')==sid(i)];need(len(parts)==1,'job peek identity')
 out=HF/outpath(parts[0]);need(Path(peek['output_dir']).resolve()==out.parent and Path(peek['run_json']).resolve()==out/'RUN.json' and Path(peek['hf_root']).resolve()==HF and not out.exists(),'job peek output differs');out.mkdir(parents=True)
 startup={'id':sid(parts[0]),'status':'STARTING_UNVERIFIED','minimal_job_read_predebited_bytes':peek_size,'minimal_job_read_completed_bytes':peek_size,'metadata_predebited_bytes':peek_size,'metadata_completed_bytes':peek_size,'metadata_calls':1,'metadata_inflight':None,'scientific_read_authorized':False,'automatic_retry':False}
 def persist_startup():atomic(out/'STARTUP.json',startup)
 persist_startup()
 bootstrap=None
 try:
  def bootstrap_persist():
   startup.update(metadata_predebited_bytes=bootstrap.pre,metadata_completed_bytes=bootstrap.done,metadata_calls=bootstrap.calls,metadata_inflight=bootstrap.inflight);persist_startup()
  bootstrap=MetadataBudget(bootstrap_persist,peek_size)
  with bootstrap:
   p=native.read(a.protocol);digest=sha(a.protocol);scope=authorize(p,digest);need(a.job_spec and sha(a.job_spec)==scope['launch_spec_sha256'] and sys.flags.isolated,'job identity')
   job=native.read(a.job_spec);profile=Path(job['argv'][1].split('=',1)[1]);need(sha(profile)==p['execution_private_worker_config_sha256'],'profile differs')
   startup['receipt_wait']={'status':'WAITING','checks':'UNKNOWN','elapsed_seconds':'UNKNOWN'};persist_startup()
   try:
    wait_result=wait_for_start_receipt(out.parent/'START_RECEIPT.json',start)
   except ReceiptWaitTimeout as e:
    startup['receipt_wait']={'status':'TIMEOUT','checks':e.checks,'elapsed_seconds':e.elapsed};persist_startup();raise
   startup['receipt_wait']={'status':'PRESENT',**wait_result};persist_startup()
   start_receipt=native.read(out.parent/'START_RECEIPT.json');need(start_receipt.get('id')==p['id'] and start_receipt.get('spec_sha256')==scope['launch_spec_sha256'],'supervisor start receipt differs')
  aff=','.join(map(str,p['cpu_affinity']));expected=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','CUDA_VISIBLE_DEVICES=','PATH=/usr/bin:/bin','taskset','-c',aff,str(PYTHON),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--job-spec',str(a.job_spec.resolve())]
  need(profile.is_absolute() and job['argv']==expected and job['id']==p['id']==startup['id'] and job['protocol_sha256']==digest and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and Path(job['hf_root']).resolve()==HF and Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RES['outer_seconds'] and sorted(os.sched_getaffinity(0))==sorted(p['cpu_affinity']) and os.environ.get('CUDA_VISIBLE_DEVICES')=='','exact job differs')
  import worker_identity;worker_identity.require_worker();startup.update(status='AUTHORIZED_BOOTSTRAP_COMPLETE',scientific_read_authorized=True,wall_seconds=time.monotonic()-start);persist_startup()
 except BaseException as e:
  startup.update(status='FAILED_BOOTSTRAP_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500],wall_seconds=time.monotonic()-start);persist_startup();signal.alarm(0);resource.setrlimit(resource.RLIMIT_AS,old_as);raise
 report={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':scope['launch_spec_sha256'],'status':'STARTED_NO_RETRY','source_ordinal':p['source_ordinal'],'predebited_bytes':0,'completed_bytes':0,'by_kind':{},'calls_by_kind':{},'metadata_predebited_bytes':bootstrap.pre,'metadata_completed_bytes':bootstrap.done,'metadata_calls':bootstrap.calls,'metadata_inflight':bootstrap.inflight,'metadata_accounting':'bootstrap predebits durably persisted in STARTUP; minimal bounded job peek backfilled; output-monitor and interpreter system I/O not measured','CPU_predebited':0,'CPU_completed':0,'CPU_inflight':None,'inflight':None,'sites':[],'automatic_retry':False,'quality_acceptance':False,'sampling_semantics':'selector64 observed separately; weighted eval estimates finite population excluding selector64; no overall estimator or causal attribution','full_file_hash':'snapshot singleton whole-file and raw SHA checked in one pass; sample whole-file SHA inherited from closed parent VERIFY'}
 def persist():atomic(out/'RUN.json',report)
 def check():
  need(time.monotonic()-start<=RES['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['max_RSS_bytes'],'resource limit')
  need(sum(x.stat().st_size for x in out.parent.rglob('*') if x.is_file() and not x.is_symlink())<=RES['new_HF_bytes']-(1<<20),'output budget')
 def charge(kind,n,fn,detail=None):
  check();is_metadata=kind=='metadata';pre='metadata_predebited_bytes' if is_metadata else 'predebited_bytes';done='metadata_completed_bytes' if is_metadata else 'completed_bytes';limit=RES['metadata_read_bytes_max'] if is_metadata else RES['read_bytes_max'];need(report['inflight'] is None and n>0 and report[pre]+n<=limit,'read budget');report[pre]+=n;report['inflight']={'kind':kind,'bytes':n,'detail':detail,'completed':'UNKNOWN'};persist();value=fn();report[done]+=n;report['by_kind'][kind]=report['by_kind'].get(kind,0)+n;report['calls_by_kind'][kind]=report['calls_by_kind'].get(kind,0)+1;report['inflight']=None;persist();return value
 def compute(fn):
  check();need(report['CPU_inflight'] is None and report['CPU_predebited']<576,'CPU operation budget');report['CPU_predebited']+=1;report['CPU_inflight']={'kind':'FP64_partition_chunk','completed':'UNKNOWN'};persist();value=fn();report['CPU_completed']+=1;report['CPU_inflight']=None;return value
 persist()
 try:
  import numpy as np
  from e0_chipmunk_p64_local_prefix_v2_math import bf16_values,aggregate,add_groups,ratios
  matched={}
  for site in p['parent']['sites']:
   block=site['block'];dm=load_metadata(HF,site['dense_metadata'],charge);sm=load_metadata(HF,site['student_metadata'],charge)
   selector=dm['selector_rows'];parent.validate_site_metadata(dm,MODES[0],block,selector);parent.validate_site_metadata(sm,MODES[1],block,selector,hashlib.sha256(struct.pack('<'+'q'*len(sm['mask']),*sm['mask'])).hexdigest())
   need(dm['selector_rows']==sm['selector_rows'] and dm['evaluation_rows']==sm['evaluation_rows']==coordinates(selector) and dm['gate_shape']==sm['gate_shape'] and dm['gate_stride']==sm['gate_stride'] and dm['gate_dtype']==sm['gate_dtype'] and struct.pack('<5120f',*dm['gate'])==struct.pack('<5120f',*sm['gate']),'gate or coordinates differ')
   matched[block]={'gate':dm['gate'],'selector':selector,'evaluation_rows':dm['evaluation_rows']}
  need(len(matched)==40,'metadata coverage differs');report['all_sites_identity_checks_passed']=True;persist()
  for site in p['parent']['sites']:
   block=site['block'];dm=matched[block];selector=dm['selector']
   gate=np.asarray(dm['gate'],dtype=np.float64);evalrows=dm['evaluation_rows'];row_ids=selector+[x['row'] for x in evalrows]
   require_distinct_files(HF,[site['dense_sample']['file'],site['student_sample']['file']]);dense=read_slice(HF,site['dense_sample'],['current_dense_Y'],320,charge);student=read_slice(HF,site['student_sample'],['current_dense_Y','candidate_Y'],320,charge)
   arrays=[dense['current_dense_Y'],student['current_dense_Y'],student['candidate_Y']]
   result={'block':block,'groups':{}}
   for label,startidx,endidx in (('selector',0,64),('eval',64,320)):
    idx=np.asarray(row_ids[startidx:endidx]);groups=(idx//(29*52))*4+((idx%(29*52))//52>=14)*2+(idx%52>=26)
    weights=np.ones(len(idx),dtype=np.float64) if label=='selector' else np.asarray([1/x['inclusion_probability'] for x in evalrows],dtype=np.float64)
    def sample_compute():
     decoded=[bf16_values(memoryview(a)[startidx*5120*2:endidx*5120*2],endidx-startidx,5120) for a in arrays]
     return [ratios(x) for x in aggregate(*decoded,gate,groups,weights)]
    result['groups'][label]=compute(sample_compute)
   del arrays,dense,student
   if block in SNAPSHOTS:
    totals=[{} for _ in range(84)]
    def process(lo,hi,chunks):
     idx=np.arange(lo,hi);groups=(idx//(29*52))*4+((idx%(29*52))//52>=14)*2+(idx%52>=26)
     def full_compute():
      decoded=[bf16_values(a,hi-lo,5120) for a in chunks]
      add_groups(totals,aggregate(*decoded,gate,groups,np.ones(hi-lo)))
     compute(full_compute)
    stream_triplet(HF,[site['dense_snapshot']['current_dense_Y'],site['student_snapshot']['current_dense_Y'],site['student_snapshot']['candidate_Y']],['current_dense_Y','current_dense_Y','candidate_Y'],31668,5120,256,charge,process)
    result['groups']['full_M']=[ratios(x) for x in totals]
   report['sites'].append(result);persist()
  need(report['by_kind'].get('Y_payload')==DATA_BYTES and report['calls_by_kind']=={'metadata':80,'header_prefix':92,'header':92,'Y_payload':124} and report['metadata_predebited_bytes']==report['metadata_completed_bytes'] and report['metadata_inflight'] is None and report['CPU_predebited']==report['CPU_completed']==576 and report['CPU_inflight'] is None and len(report['sites'])==40 and report['inflight'] is None,'coverage differs')
  report.update(status='COMPLETE_LOCAL_PREFIX_EXIT_PENDING',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);check();persist()
 except BaseException as e:
  report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);persist();raise
 finally:
  signal.alarm(0)
  if 'old_as' in locals():resource.setrlimit(resource.RLIMIT_AS,old_as)

def inspect_exit(a):
 p=native.read(a.protocol);validate(p,True);digest=sha(a.protocol);out=HF/p['output_HF_relative'];runfile=out/'RUN.json';exfile=out.parent/'EXIT_RECEIPT.json';report=native.read(runfile);receipt=native.read(exfile);rb=binding(runfile,HF)
 parent.vm.exited(receipt,rb,digest,p['id'])
 startup=native.read(out/'STARTUP.json');need(startup['id']==p['id'] and startup['status']=='AUTHORIZED_BOOTSTRAP_COMPLETE' and startup['scientific_read_authorized'] is True and startup['metadata_predebited_bytes']==startup['metadata_completed_bytes'] and startup['metadata_inflight'] is None,'startup evidence differs')
 need(report['status']=='COMPLETE_LOCAL_PREFIX_EXIT_PENDING' and report['id']==p['id'] and
      receipt['spec_sha256']==report['job_spec_sha256'] and report['predebited_bytes']==report['completed_bytes'] and
      report['inflight'] is None and report['CPU_inflight'] is None and report['CPU_predebited']==report['CPU_completed']==576 and report['metadata_predebited_bytes']==report['metadata_completed_bytes'] and report['metadata_inflight'] is None and report['by_kind']['Y_payload']==DATA_BYTES and len(report['sites'])==40 and
      report['wall_seconds']<=RES['wall_seconds'] and report['peak_RSS_bytes']<=RES['max_RSS_bytes'],'actual EXIT differs')
 return {'status':'PASS_LOCAL_PREFIX_ACTUAL_EXIT','RUN':rb,'EXIT':binding(exfile,HF),'quality_acceptance':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','freeze','inspect','inspect-exit','run'));ap.add_argument('--part',type=int);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--cpu-affinity',default='0,1');ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--job-spec',type=Path);a=ap.parse_args();need(a.hf_root.resolve()==HF,'HF root differs')
 if a.action=='draft':need(a.out and a.worker_profile and not a.out.exists(),'fresh output required');native.write(a.out,draft(a.part,a.worker_profile,[int(x) for x in a.cpu_affinity.split(',')]))
 elif a.action=='freeze':
  p=native.read(a.protocol);validate(p);need(a.out and not a.out.exists(),'fresh output required');p['status']='FROZEN_PENDING_REGISTRATION';p['assets_finalized']=True;native.write(a.out,p)
 elif a.action=='inspect':p=native.read(a.protocol);validate(p,p['status']=='FROZEN_PENDING_REGISTRATION');print(json.dumps({'id':p['id'],'status':p['status'],'fingerprint':fingerprint(p)}))
 elif a.action=='inspect-exit':print(json.dumps(inspect_exit(a)))
 else:run(a)
if __name__=='__main__':main()
