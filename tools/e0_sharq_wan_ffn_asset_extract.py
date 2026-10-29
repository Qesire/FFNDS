"""Exact, budgeted safetensors range selection for the future SharQ bundle.

This module has no CLI, no model/GPU calls, and no default filesystem access.
It is a reusable primitive, not an authorized CPU acquisition scope.
"""
import hashlib,json,os,struct
from pathlib import Path

DTYPE_BYTES={'BF16':2,'U8':1,'F8_E4M3':1,'F32':4,'I64':8}

def stat_values(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def stat_id(path):return stat_values(Path(path).stat())

def read_header(path,max_header=32<<20):
 path=Path(path);before=stat_id(path)
 with path.open('rb') as f:
  if stat_values(os.fstat(f.fileno()))!=before:raise ValueError('master fd differs')
  b=f.read(8)
  if len(b)!=8:raise ValueError('short header prefix')
  n=struct.unpack('<Q',b)[0]
  if not 0<n<=max_header:raise ValueError('header length')
  raw=f.read(n)
  if len(raw)!=n:raise ValueError('short header')
  if stat_values(os.fstat(f.fileno()))!=before:raise ValueError('master fd changed')
  header=json.loads(raw)
 if stat_id(path)!=before:raise ValueError('master changed during header')
 return header,n,before

def header_digest(path,max_header=32<<20):
 path=Path(path);before=stat_id(path)
 with path.open('rb') as f:
  if stat_values(os.fstat(f.fileno()))!=before:raise ValueError('master fd differs')
  prefix=f.read(8)
  if len(prefix)!=8:raise ValueError('short header prefix')
  n=struct.unpack('<Q',prefix)[0]
  if not 0<n<=max_header:raise ValueError('header length')
  raw=f.read(n)
  if len(raw)!=n or stat_values(os.fstat(f.fileno()))!=before:raise ValueError('master header changed')
 if stat_id(path)!=before:raise ValueError('master header path changed')
 return hashlib.sha256(raw).hexdigest()

def checked_range(path,header,header_bytes,expected_stat,key,expected_dtype,expected_shape,ledger):
 row=header.get(key)
 if not isinstance(row,dict) or row.get('dtype')!=expected_dtype or row.get('shape')!=list(expected_shape):raise ValueError('typed tensor mismatch '+key)
 start,end=row.get('data_offsets',(-1,-1));count=1
 for v in expected_shape:count*=v
 length=count*DTYPE_BYTES[expected_dtype]
 if not (type(start) is int and type(end) is int and 0<=start<=end and end-start==length and 8+header_bytes+end<=expected_stat[2]):raise ValueError('tensor offsets or size differ')
 ledger.reserve('tensor_bytes',length);ledger.reserve('tensor_ranges',1)
 if stat_id(path)!=expected_stat:raise ValueError('master changed before read')
 with Path(path).open('rb') as f:
  if stat_values(os.fstat(f.fileno()))!=expected_stat:raise ValueError('master fd differs')
  f.seek(8+header_bytes+start);raw=f.read(length)
  if stat_values(os.fstat(f.fileno()))!=expected_stat:raise ValueError('master fd changed')
 if len(raw)!=length or stat_id(path)!=expected_stat:raise ValueError('master changed or short tensor')
 ledger.complete('tensor_bytes',length);ledger.complete('tensor_ranges',1)
 return raw,hashlib.sha256(raw).hexdigest()
