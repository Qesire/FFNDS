"""Per-call whole-file digest accounting for one SharQ quality phase."""
import hashlib,json,os
from pathlib import Path
MAX_BYTES=16<<30
MAX_JOURNAL_BYTES=16<<20
MAX_WORKER_EXTRA_BYTES=64<<20

def need(v,msg):
 if not v:raise ValueError(msg)
class HashReads:
 def __init__(self,base,path,phase):
  self.base=base;self.path=Path(path);self.phase=phase;self.entries=0;self.predebited=0;self.completed=0;self.inflight=None;self.previous=None
 def event(self,kind,**data):
  extra=0
  for entry in os.scandir(self.path.parent):
   try:
    if entry.is_file(follow_symlinks=False):extra+=entry.stat(follow_symlinks=False).st_size
   except FileNotFoundError:continue
  need(extra<=MAX_WORKER_EXTRA_BYTES,'worker extra output budget exceeded')
  row={'event':kind,'phase':self.phase,**data};payload=(json.dumps(row,separators=(',',':'))+'\n').encode();need(self.path.stat().st_size+len(payload)<=MAX_JOURNAL_BYTES and extra+len(payload)<=MAX_WORKER_EXTRA_BYTES,'hash journal or worker extra output budget exceeded');self.stream.write(payload);self.stream.flush();os.fsync(self.stream.fileno())
 def __enter__(self):
  self.path.parent.mkdir(parents=True,exist_ok=True);self.stream=self.path.open('xb',buffering=0);self.previous=self.base.sha;self.base.sha=self.sha;return self
 def sha(self,path):
  path=Path(path);size=path.stat().st_size;need(size>=0 and self.predebited+size<=MAX_BYTES,'SHA read budget exceeded');self.entries+=1;self.predebited+=size;tag=hashlib.sha256(os.fsencode(str(path.resolve()))).hexdigest();self.inflight=self.entries;self.event('PREDEBIT',sequence=self.entries,path_identity_sha256=tag,bytes=size,completion='UNKNOWN')
  digest=hashlib.sha256();count=0
  with path.open('rb') as stream:
   while True:
    data=stream.read(min(4<<20,size-count+1));count+=len(data);need(count<=size,'file grew during hashed read')
    if not data:break
    digest.update(data)
  need(count==size and path.stat().st_size==size,'hashed input changed');result=digest.hexdigest();self.completed+=count;self.event('COMPLETE',sequence=self.entries,bytes=size,sha256=result);self.inflight=None;return result
 def __exit__(self,typ,value,tb):
  self.base.sha=self.previous
  try:self.event('PHASE_END',predebited_bytes=self.predebited,completed_bytes=self.completed,inflight=self.inflight,status='COMPLETE' if typ is None else 'FAILED_UNKNOWN')
  finally:self.stream.close()

def replay(path):
 rows=[json.loads(x) for x in Path(path).read_text().splitlines()];need(rows and rows[-1].get('event')=='PHASE_END','hash journal incomplete');pre=done=0;pending={}
 for row in rows[:-1]:
  seq=row['sequence']
  if row['event']=='PREDEBIT':
   need(seq not in pending and row['bytes']>=0,'bad hash predebit');pending[seq]=row['bytes'];pre+=row['bytes']
  elif row['event']=='COMPLETE':
   need(seq in pending and pending.pop(seq)==row['bytes'] and len(row['sha256'])==64,'bad hash completion');done+=row['bytes']
  else:raise ValueError('unknown hash event')
 last=rows[-1];need(last['predebited_bytes']==pre and last['completed_bytes']==done and pre<=MAX_BYTES,'hash ledger totals differ');return {'predebited_bytes':pre,'completed_bytes':done,'inflight':last['inflight'],'status':last['status'],'pending':len(pending)}

def require_complete(base,hf,report):
 value=report.get('sha_read_ledger');need(isinstance(value,dict) and value.get('inflight') is None and value.get('predebited_bytes')==value.get('completed_bytes'),'completed hash ledger required');path=base.bound(hf,value['journal']);result=replay(path);need(result=={'predebited_bytes':value['predebited_bytes'],'completed_bytes':value['completed_bytes'],'inflight':None,'status':'COMPLETE','pending':0},'hash journal disagrees with result');return value
