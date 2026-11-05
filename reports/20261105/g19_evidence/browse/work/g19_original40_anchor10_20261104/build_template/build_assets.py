"""Raw BF16 input-output matrix assets. Numeric imports occur only behind a guard."""
import hashlib,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def save_matrix(path,w1_native_bf16,w2_native_bf16,alpha_f64,provenance,require_compute):
 require_compute()
 import torch,numpy as np
 if w1_native_bf16.dtype!=torch.bfloat16 or w2_native_bf16.dtype!=torch.bfloat16 or w1_native_bf16.shape!=(13824,5120) or w2_native_bf16.shape!=(5120,13824) or alpha_f64.shape!=(13824,) or alpha_f64.dtype!=torch.float64:raise ValueError('matrix operands')
 if not all(bool(torch.isfinite(x).all()) for x in (w1_native_bf16,w2_native_bf16,alpha_f64)):raise ValueError('nonfinite operands')
 A=((w1_native_bf16.double()*alpha_f64[:,None]).T@w2_native_bf16.double().T).float().to(torch.bfloat16)
 if A.shape!=(5120,5120) or not bool(torch.isfinite(A).all()):raise ValueError('nonfinite A')
 bits=A.detach().cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False)
 p=Path(path)
 with p.open('xb') as f:bits.tofile(f);f.flush();os.fsync(f.fileno())
 if p.stat().st_size!=5120*5120*2:raise ValueError('raw matrix bytes')
 return A,{'path':str(p.resolve()),'dtype':'BF16_RAW_UINT16','shape':[5120,5120],
  'layout':'input_output','sha256':sha(p),'bytes':p.stat().st_size,'provenance':provenance}

def make_loader(entries,require_compute):
 def load(expert,layer,device):
  require_compute()
  import torch,numpy as np
  e=entries[f'{expert}/{layer}'];p=Path(e['path']);before=p.stat()
  if e['dtype']!='BF16_RAW_UINT16' or e['shape']!=[5120,5120] or e['layout']!='input_output' or before.st_size!=5120*5120*2 or sha(p)!=e['sha256']:raise ValueError('raw matrix identity')
  a=np.fromfile(p,dtype='<u2').reshape(5120,5120);after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('matrix changed')
  result=torch.from_numpy(a).view(torch.bfloat16).to(device)
  if not bool(torch.isfinite(result).all()):raise ValueError('nonfinite matrix')
  return result
 return load
