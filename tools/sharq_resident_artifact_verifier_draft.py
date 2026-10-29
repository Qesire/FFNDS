"""Non-executable single-pass typed state verifier and byte comparator.

Caller must reserve/persist read budgets before invoking these helpers. No
artifact access occurs at module import and no scope is authorized here.
"""
import hashlib,json,struct

def need(v,m):
 if not v:raise ValueError(m)

def state_identity(path,expected,read_bound=1<<20):
 dtype={'float32':('F32',4),'bfloat16':('BF16',2)}[expected['dtype']]
 key='flow' if expected['name'].startswith('flow_step') else 'latents';h=hashlib.sha256();raw=hashlib.sha256();total=0
 with path.open('rb') as f:
  before=__import__('os').fstat(f.fileno());prefix=f.read(8);need(len(prefix)==8,'header prefix');n=struct.unpack('<Q',prefix)[0];need(0<n<=read_bound,'header size');data=f.read(n);need(len(data)==n,'header incomplete');doc=json.loads(data);h.update(prefix);h.update(data);total=8+n
  elements=1
  for x in expected['shape']:elements*=x
  need(set(doc)=={key} and doc[key]['dtype']==dtype[0] and doc[key]['shape']==expected['shape'] and doc[key]['data_offsets']==[0,elements*dtype[1]],'state header')
  while True:
   chunk=f.read(4<<20)
   if not chunk:break
   h.update(chunk);raw.update(chunk);total+=len(chunk)
  after=__import__('os').fstat(f.fileno())
 ident=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 need(ident(before)==ident(after)==ident(path.stat()) and total==before.st_size==8+n+elements*dtype[1] and total==expected['bytes'] and h.hexdigest()==expected['sha256'] and raw.hexdigest()==expected['raw_tensor_sha256'],'state identity')
 return total

def byte_equal(a,b,expected_bytes):
 with a.open('rb') as x,b.open('rb') as y:
  ax=__import__('os').fstat(x.fileno());by=__import__('os').fstat(y.fileno());need(ax.st_size==by.st_size==expected_bytes,'size')
  same=True
  while True:
   p=x.read(4<<20);q=y.read(4<<20)
   if p!=q:same=False
   if not p and not q:break
  az=__import__('os').fstat(x.fileno());bz=__import__('os').fstat(y.fileno())
  identity=lambda z:(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
  need(identity(ax)==identity(az)==identity(a.stat()) and identity(by)==identity(bz)==identity(b.stat()),'changed while comparing')
 return same
