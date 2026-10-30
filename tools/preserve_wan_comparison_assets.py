"""Maintenance-only immutable content copies; no model or scientific evaluation.

Every distinct source path is rehashed, then each new independent copy is
rehash-verified. HF: is a logical prefix in the preserved historical manifest.
The vault is a same-host copy, not a claim of offsite or disk-failure protection.
"""
import argparse,hashlib,json,os,re,signal,stat,time
from pathlib import Path

READ_MAX=10<<30;NEW_MAX=3<<30;WALL=600
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def normalized(ref):
 value=ref.removeprefix('HF:');p=Path(value)
 if p.is_absolute() or '..' in p.parts or not value:raise ValueError('manifest path escape')
 return value
def identity(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def preserve(manifest,hf,vault,runfile):
 started=time.monotonic();hf=hf.resolve();vault=vault.resolve()
 records=json.loads(manifest.read_text())['assets'];sources={};content={}
 for row in records:
  rel=normalized(row['relative_path']);expected={'bytes':row['bytes'],'sha256':row['sha256']}
  if type(expected['bytes']) is not int or expected['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',expected['sha256']):raise ValueError('manifest identity')
  if rel in sources and sources[rel]!=expected:raise ValueError('conflicting source binding')
  if expected['sha256'] in content and content[expected['sha256']]!=expected['bytes']:raise ValueError('conflicting content binding')
  sources[rel]=expected;content[expected['sha256']]=expected['bytes']
 total=sum(x['bytes'] for x in sources.values());new=sum(content.values())
 if total+new>READ_MAX or new>NEW_MAX or len(records)!=868:raise ValueError('finite maintenance budget')
 if vault.exists() or runfile.exists() or hf==vault or vault.is_relative_to(hf):raise ValueError('fresh independent vault required')
 if os.statvfs(vault.parent).f_bavail*os.statvfs(vault.parent).f_frsize < NEW_MAX+(1<<30):raise ValueError('vault free space')
 vault.mkdir(mode=0o700);runfile.parent.mkdir(parents=True,exist_ok=True)
 report={'status':'STARTED_MAINTENANCE_NO_RETRY','manifest':{'path':str(manifest),'bytes':manifest.stat().st_size,'sha256':digest(manifest)},'records':len(records),'distinct_source_paths':len(sources),'distinct_content':len(content),'planned_source_bytes':total,'planned_independent_copy_bytes':new,'read_predebited_bytes':0,'read_completed_bytes':0,'source_paths_verified':0,'independent_objects_verified':0,'new_copy_predebited_bytes':0,'new_copy_bytes':0,'inflight':None,'automatic_retry':False,'science_or_model_execution':False,'offsite_backup':False,'scope':'Immutable same-host independent-inode copies; source and destination raw SHA verified. Original sources unchanged; source/copy payload accounting excludes small manifest/index/ledger metadata and system/interpreter I/O.'}
 def persist():save(runfile,report)
 def check():
  if time.monotonic()-started>WALL or report['read_predebited_bytes']>READ_MAX or report['new_copy_predebited_bytes']>NEW_MAX:raise ValueError('maintenance budget exceeded')
 def debit(kind,rel,size):
  check();report['read_predebited_bytes']+=size;report['inflight']={'kind':kind,'relative_path':rel,'bytes':size,'completed':'UNKNOWN'};persist();check()
 persist();verified=set()
 try:
  for rel,b in sources.items():
   source=(hf/rel).resolve()
   if not source.is_relative_to(hf):raise ValueError('source root escape')
   before=source.stat()
   if not stat.S_ISREG(before.st_mode) or before.st_size!=b['bytes']:raise ValueError('source size/type')
   target=vault/'objects'/b['sha256'][:2]/b['sha256'];fresh=b['sha256'] not in verified;partial=target.with_suffix('.partial')
   target.parent.mkdir(parents=True,exist_ok=True);debit('SOURCE_HASH_AND_OPTIONAL_COPY',rel,b['bytes']);h=hashlib.sha256()
   if fresh:report['new_copy_predebited_bytes']+=b['bytes'];persist();check()
   output=partial.open('xb') if fresh else None
   try:
    with source.open('rb') as stream:
     if identity(os.fstat(stream.fileno()))!=identity(before):raise ValueError('source fd changed')
     remaining=b['bytes']
     while remaining:
      chunk=stream.read(min(1<<20,remaining))
      if not chunk:raise ValueError('truncated source')
      remaining-=len(chunk)
      h.update(chunk)
      if output:output.write(chunk)
      check()
     if identity(os.fstat(stream.fileno()))!=identity(before) or identity(source.stat())!=identity(before):raise ValueError('source changed')
    if h.hexdigest()!=b['sha256']:raise ValueError('source SHA mismatch')
    if output:output.flush();os.fsync(output.fileno())
   finally:
    if output:output.close()
   report['read_completed_bytes']+=b['bytes'];report['source_paths_verified']+=1;report['inflight']=None;persist()
   if fresh:
    partial.rename(target);report['new_copy_bytes']+=b['bytes'];debit('INDEPENDENT_COPY_REHASH',rel,b['bytes'])
    if target.stat().st_size!=b['bytes'] or target.stat().st_ino==before.st_ino and target.stat().st_dev==before.st_dev or digest(target)!=b['sha256']:raise ValueError('independent copy verification')
    report['read_completed_bytes']+=b['bytes'];report['independent_objects_verified']+=1;report['inflight']=None;verified.add(b['sha256']);persist()
  index={'manifest_sha256':report['manifest']['sha256'],'source_HF_relative_to_object':{rel:'objects/'+b['sha256'][:2]+'/'+b['sha256'] for rel,b in sources.items()},'logical_HF_prefixes_normalized':sum(x['relative_path'].startswith('HF:') for x in records)};save(vault/'INDEX.json',index)
  if report['source_paths_verified']!=len(sources) or len(verified)!=len(content) or report['read_predebited_bytes']!=report['read_completed_bytes'] or report['new_copy_predebited_bytes']!=report['new_copy_bytes'] or report['inflight'] is not None:raise ValueError('incomplete maintenance')
  report.update(status='COMPLETE_MAINTENANCE_EXIT_PENDING',wall_seconds=time.monotonic()-started,index_sha256=digest(vault/'INDEX.json'),index_bytes=(vault/'INDEX.json').stat().st_size);persist();return report
 except BaseException as error:
  report.update(status='FAILED_MAINTENANCE_NO_RETRY',error_type=type(error).__name__,error=str(error),wall_seconds=time.monotonic()-started);persist();raise
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--hf-root',type=Path,required=True);parser.add_argument('--vault',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('maintenance wall limit')));signal.alarm(WALL)
 result=preserve(a.manifest,a.hf_root,a.vault,a.run);signal.alarm(0);print(json.dumps({k:result[k] for k in ('status','records','source_paths_verified','independent_objects_verified','new_copy_bytes','wall_seconds')}))
if __name__=='__main__':main()
