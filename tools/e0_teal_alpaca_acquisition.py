"""Pending independent review: acquire one immutable Alpaca parquet via HF Hub.

No execution authority is conferred by this file or its draft protocol.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import signal
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_llama3_public_weight_acquisition as common

HF = Path('/home/xiang/.cache/huggingface')
ID = 'E0_TEAL_ALPACA_PARQUET_ACQUISITION_20261029'
REPO = 'tatsu-lab/alpaca'
REV = 'dce01c9b08f87459cf36a430d809084718273017'
NAME = 'data/train-00000-of-00001-a09b74b3ef9c3b56.parquet'
DIGEST = '06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073'
SIZE = 24246638
CACHE = 'ffnds-derived/e0_teal_alpaca_parquet_acquisition_20261029/attempt1/worker/scoped_hf_cache'
OUT = 'ffnds-derived/e0_teal_alpaca_parquet_acquisition_20261029/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 1, 'wall_seconds': 900, 'outer_seconds': 1020, 'peak_address_space_bytes': 4294967296, 'minimum_free_disk_bytes': 268435456, 'new_HF_bytes': 134217728, 'max_logical_hash_bytes': SIZE * 2, 'automatic_retry': False}
LIMITS = {'asset_files': 1, 'hub_api_calls_max': 1, 'model_calls': 0, 'GPU_calls': 0, 'dataset_reads': 0}
SOURCES = ('tools/e0_teal_alpaca_acquisition.py', 'tests/test_e0_teal_alpaca_acquisition.py', 'tools/e0_teal_llama3_public_weight_acquisition.py', 'tools/e0_teal_llama3_tokenizer_byte_bridge.py', 'tools/worker_identity.py', 'tools/durable_worker_job.py')
need = base.need

def runtime():
 # Narrow fingerprint: distribution METADATA and RECORD plus selected Hub entry
 # points, not a full dependency or native-library integrity closure.
 packages=('huggingface-hub','httpx','httpcore','certifi','anyio','h11','idna','filelock','fsspec','packaging','pyyaml','tqdm','typing-extensions','click','typer','hf-xet','rich','shellingham')
 records=[]
 for name in packages:
  d=importlib.metadata.distribution(name); files=[]
  for f in d.files or []:
   t=str(f)
   if t.endswith(('.dist-info/METADATA','.dist-info/RECORD')) or (name=='huggingface-hub' and t in ('huggingface_hub/__init__.py','huggingface_hub/file_download.py','huggingface_hub/utils/_http.py')):
    path=Path(d.locate_file(f)).resolve()
    if path.is_file(): files.append([t,base.sha(path)])
  need(len(files)>=2, 'package identity missing: '+name)
  records.append({'name':name,'version':d.version,'files':sorted(files)})
 return {'python':str(Path(sys.executable).resolve()),'python_sha256':base.sha(Path(sys.executable).resolve()),'version':sys.version.split()[0],'prefix':sys.prefix,'coverage':'METADATA, RECORD and selected Hub Python entrypoints; not full package or ELF byte closure','packages':records}

def draft():
 return {'id': ID, 'status': 'DRAFT_REVIEW_REQUESTED', 'repo': REPO, 'revision': REV, 'filename': NAME, 'sha256': DIGEST, 'size': SIZE, 'cache': CACHE, 'output': OUT, 'resources': RESOURCES, 'limits': LIMITS, 'source_bindings': [base.binding(ROOT / p, ROOT) for p in SOURCES], 'runtime': runtime(), 'worker_config_sha256': None, 'review': None}

def fingerprint(p):
 return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k != 'review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate(p, frozen=False):
 for k,v in {'id': ID, 'repo': REPO, 'revision': REV, 'filename': NAME, 'sha256': DIGEST, 'size': SIZE, 'cache': CACHE, 'output': OUT, 'resources': RESOURCES, 'limits': LIMITS}.items(): need(p.get(k)==v, 'protocol mismatch: '+k)
 need({x['path'] for x in p['source_bindings']} == set(SOURCES), 'source closure')
 if frozen: need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64, 'unfrozen')

def authorize(p,d):
 validate(p,True)
 active=base.read(ROOT/'research/ACTIVE_RESEARCH.json')
 matches=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
 need(active.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(matches)==1, 'ACTIVE required')
 s=matches[0]
 need(s.get('protocol_sha256')==d and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==RESOURCES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==0, 'ACTIVE mismatch')
 r=base.read(base.bound(ROOT,s['review']))
 need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('reviewer_disposition') in ('APPROVE','REVISE') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('responses_to_review') and r.get('adversarial_counterexample') and r.get('approval_scope_fingerprint')==fingerprint(p), 'review mismatch')
 need(r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['source_bindings']}, 'reviewed code mismatch')
 for k in ('author_review','independent_review','reviewer_recheck'): base.bound(ROOT,r[k])
 return s

def worker_usage(path):
 root=Path(os.path.abspath(path)); initial=root.lstat(); need(stat.S_ISDIR(initial.st_mode),'worker root invalid'); identity=(initial.st_dev,initial.st_ino); total=0; count=0; missing=0
 def onerror(e):
  nonlocal missing
  if isinstance(e,FileNotFoundError) and e.filename and Path(os.path.abspath(e.filename))!=root and Path(os.path.abspath(e.filename)).is_relative_to(root): missing+=1; return
  raise e
 for parent,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
  for name in files:
   try: st=(Path(parent)/name).lstat()
   except FileNotFoundError: missing+=1; continue
   if stat.S_ISREG(st.st_mode): total+=st.st_size; count+=1
 end=root.lstat();need(stat.S_ISDIR(end.st_mode) and (end.st_dev,end.st_ino)==identity,'worker root changed')
 return {'apparent_bytes':total,'regular_files':count,'missing_entries':missing}

def validate_closed(p,protocol_sha,run,exit_receipt,run_sha,spec_sha,asset_path=None):
 validate(p,True)
 need(run.get('id')==ID and run.get('status')=='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED' and run.get('protocol_sha256')==protocol_sha and run.get('launch_spec_sha256')==spec_sha,'run incomplete')
 need(exit_receipt.get('id')==ID and exit_receipt.get('process_exited') is True and exit_receipt.get('child_returncode')==0 and exit_receipt.get('timeout') is False and exit_receipt.get('automatic_retry') is False and exit_receipt.get('run_sha256')==run_sha and exit_receipt.get('protocol_sha256')==protocol_sha and exit_receipt.get('spec_sha256')==spec_sha,'actual EXIT0 required')
 need(run.get('download_predebit_bytes')==SIZE and run.get('hash_predebit_bytes')==SIZE*2 and run.get('first_hash_completed_bytes')==SIZE and run.get('second_hash_completed_bytes')==SIZE and run.get('result',{}).get('sha256')==DIGEST and run.get('result',{}).get('bytes')==SIZE and run.get('worker_usage',{}).get('apparent_bytes',RESOURCES['new_HF_bytes']+1)<=RESOURCES['new_HF_bytes'] and run.get('wall_seconds',RESOURCES['wall_seconds']+1)<=RESOURCES['wall_seconds'],'budget/identity incomplete')
 need(run.get('hub_api_calls')==1 and run.get('model_calls')==0 and run.get('gpu_calls')==0 and run.get('automatic_retry') is False and isinstance(run.get('cpu_seconds'),(int,float)) and run['cpu_seconds']>=0,'execution counts differ')
 expected=HF/CACHE/'datasets--tatsu-lab--alpaca'/'blobs'/DIGEST
 path=asset_path if asset_path is not None else HF/run.get('cache_path','')
 need(path==expected and path.is_file() and not path.is_symlink() and path.stat().st_size==SIZE,'asset path/stat differs')
 return {'status':'PASS_ACTUAL_EXIT_AND_TWO_HASHES_SAME_PROCESS','id':ID,'run_sha256':run_sha,'protocol_sha256':protocol_sha,'spec_sha256':spec_sha}

def run(a):
 p=base.read(a.protocol); d=base.sha(a.protocol); s=authorize(p,d)
 need(a.job_spec and base.sha(a.job_spec)==s['launch_spec_sha256'], 'launch spec mismatch')
 job=base.read(a.job_spec); argv=job['argv']; need(len(argv)==15 and argv[1].startswith('FFNDS_WORKER_CONFIG='), 'job argv')
 profile=Path(argv[1].split('=',1)[1]); need(base.sha(profile)==p['worker_config_sha256'], 'worker profile')
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0',str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/SOURCES[0]),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())]
 out=HF/OUT; cache=HF/CACHE
 need(argv==expected and job['id']==ID and job['protocol_sha256']==d and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RESOURCES['outer_seconds'], 'job identity')
 need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sorted(os.sched_getaffinity(0))==[0] and runtime()==p['runtime'], 'runtime mismatch')
 for b in p['source_bindings']: base.bound(ROOT,b)
 import worker_identity; worker_identity.require_worker()
 need(not out.exists() and not cache.exists(), 'fresh attempt required')
 out.mkdir(parents=True,mode=0o700); cache.mkdir(mode=0o700)
 start=time.monotonic(); cpu_start=time.process_time(); r={'id':ID,'status':'STARTED','protocol_sha256':d,'launch_spec_sha256':s['launch_spec_sha256'],'automatic_retry':False,'hub_api_calls':0,'model_calls':0,'gpu_calls':0,'hash_predebit_bytes':0,'actual_network_bytes':'UNKNOWN_BACKEND_NOT_EXPOSED','transport':'http; xet disabled; token=False; hub retry internals bounded by wall only'}
 def save(): base.write(out/'RUN.json',r)
 save()
 try:
  need(shutil.disk_usage(HF).free>=RESOURCES['minimum_free_disk_bytes'],'disk reserve')
  resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['peak_address_space_bytes'],)*2)
  resource.setrlimit(resource.RLIMIT_FSIZE,(SIZE,SIZE))
  signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit'))); signal.alarm(RESOURCES['wall_seconds'])
  for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'): os.environ.pop(k,None)
  os.environ.update(HF_ENDPOINT='https://huggingface.co',HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  need(not {'torch','triton','transformers'} & set(sys.modules),'numerical modules imported before backend')
  from huggingface_hub import hf_hub_download
  need(not {'torch','triton','transformers'} & set(sys.modules),'numerical modules imported by backend')
  r.update(hub_api_calls=1,download_predebit_bytes=SIZE,actual_network_bytes='UNKNOWN_BACKEND_NOT_EXPOSED'); save()
  path=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',filename=NAME,revision=REV,cache_dir=str(cache),token=False,force_download=False,local_files_only=False)).resolve()
  expected_blob=cache/'datasets--tatsu-lab--alpaca'/'blobs'/DIGEST
  need(path==expected_blob.resolve() and path.is_relative_to(cache),'unexpected blob')
  r.update(hash_predebit_bytes=SIZE,first_hash_progress='UNKNOWN_UNTIL_COMPLETE');save()
  item={'filename':NAME,'bytes':SIZE,'sha256':DIGEST,'git_blob':None}
  r['result']=common.verify_file(path,item); r['first_hash_completed_bytes']=SIZE;r['first_hash_progress']='COMPLETE'; save()
  r['hash_predebit_bytes']=SIZE*2;r['second_hash_progress']='UNKNOWN_UNTIL_COMPLETE';save()
  second=common.verify_file(path,item); need(second==r['result'],'second hash differs');r['second_hash_completed_bytes']=SIZE;r['second_hash_progress']='COMPLETE'
  r['cache_path']=str(path.relative_to(HF)); r['status']='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED'
 except BaseException as e:
  r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500] if isinstance(e,(ValueError,TimeoutError)) else 'REDACTED'); raise
 finally:
  signal.alarm(0); r.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  try: r['worker_usage']=worker_usage(out.parent)
  except BaseException as e:
   r.update(status='FAILED_WORKER_USAGE_PRESERVED',usage_error_type=type(e).__name__);save();raise
  exceeded=r['worker_usage']['apparent_bytes'] > RESOURCES['new_HF_bytes']
  if exceeded:r['status']='FAILED_WORKER_BUDGET_PRESERVED'
  save()
  if exceeded:raise RuntimeError('worker budget exceeded')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','inspect-closed'));ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);a=ap.parse_args()
 if a.action=='draft': need(a.out and not a.out.exists(),'fresh draft required');base.write(a.out,draft())
 elif a.action=='inspect': p=base.read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'resources':RESOURCES,'limits':LIMITS},indent=2))
 elif a.action=='run': run(a)
 else:
  need(a.run.resolve()==(HF/OUT/'RUN.json').resolve() and a.exit.resolve()==(HF/OUT).parent.joinpath('EXIT_RECEIPT.json').resolve() and a.job_spec is not None,'receipt identity')
  print(json.dumps(validate_closed(base.read(a.protocol),base.sha(a.protocol),base.read(a.run),base.read(a.exit),base.sha(a.run),base.sha(a.job_spec)),indent=2))
if __name__=='__main__': main()
