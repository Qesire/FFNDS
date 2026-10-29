"""Byte-only V3 generation lineage bridge for isolated metric interpreters."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class NoNumericalImports:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy'):raise RuntimeError('Numerical import forbidden in byte verifier')
if sys.argv[1:]==['--verify-child']:
 if {'torch','numpy'}&set(sys.modules):raise RuntimeError('Numerical import before child guard')
 sys.meta_path.insert(0,NoNumericalImports())
spec=importlib.util.spec_from_file_location('_teal_v2_original_lineage',ROOT/'tools/e0_teal_wan_quality_lineage.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
native=original.native;PARENT=original.PARENT;ORDINALS=original.ORDINALS;MODES=original.MODES
source_equal_vectors=original.source_equal_vectors;raft_source_equal=original.raft_source_equal
parent_data=original.parent_data
CPU_PROTOCOL='config/e0_teal_wan_all160_cpu_local_20261029_pending_review_v1.json'
SELF='tools/e0_teal_wan_quality_lineage_v2.py'

def contract():
 env=native.read(ROOT/CPU_PROTOCOL)['environment'];py=ROOT/'.venv/bin/python'
 native.need(env['python_path']==str(py) and native.sha(py)==env['python_sha256'],'Generation verifier Python differs')
 return {'python_path':str(py),'python_sha256':env['python_sha256'],'source':SELF,
         'argv':[str(py),'-I','-B',str(ROOT/SELF),'--verify-child'],
         'max_child_processes_per_parent_check':1,'child_timeout_seconds':300,
         'child_CPUs':'inherited_from_metric_worker','CUDA_VISIBLE_DEVICES':'',
         'numerical_imports':'torch_and_numpy_forbidden','model_calls':0,'GPU_calls':0}

def failed_v1_preflights(hf):
 proofs=[]
 for i in (0,1):
  protocol=ROOT/f'config/e0_teal_wan_raft_part{i}_20261029_pending_review_v1.json'
  plan=native.read(protocol);worker=Path(hf)/plan['output_HF_relative'];worker=worker.parent.parent/'cpu_preflight/worker'
  proof={'protocol':native.binding(protocol,ROOT),'RUN':native.binding(worker/'results/RUN.json',hf),'exit':native.binding(worker/'EXIT_RECEIPT.json',hf)}
  run=native.read(native.bound(hf,proof['RUN']));receipt=native.read(native.bound(hf,proof['exit']))
  native.need(run.get('id')==plan['id']==receipt.get('id') and run.get('protocol_sha256')==receipt.get('protocol_sha256')==proof['protocol']['sha256']
   and run.get('status')=='FAILED_CPU_PREFLIGHT_PRESERVED_NO_RETRY' and run.get('error_type')=='PackageNotFoundError'
   and run.get('error')=='No package metadata was found for torch' and receipt.get('run_sha256')==proof['RUN']['sha256']
   and receipt.get('child_returncode')==1 and receipt.get('process_exited') is True and receipt.get('timeout') is False
   and receipt.get('automatic_retry') is False,'Actual V1 failure proof differs')
  proofs.append(proof)
 return proofs

class ChildLedger:
 def __init__(self,record,persist,phase):
  self.record=record;self.persist=persist
  record.update(phase=phase,predebited=0,completed=0,launch_state='NOT_STARTED',error=None)
 def call(self,fn):
  native.need(self.record['predebited']==0,'Only one parent verifier child permitted')
  self.record.update(predebited=1,launch_state='IN_FLIGHT_OR_LAUNCH_UNKNOWN');self.persist()
  try:
   value=fn();self.record.update(launch_state='EXITED_ZERO_VALIDATED',completed=1);self.persist();return value
  except BaseException as e:
   state='TIMED_OUT_STARTED' if isinstance(e,subprocess.TimeoutExpired) else 'LAUNCH_OR_IO_FAILED_UNKNOWN' if isinstance(e,OSError) else 'FAILED_OR_INTERRUPTED_LAUNCH_UNKNOWN'
   self.record.update(launch_state=state,error={'type':type(e).__name__,'message':str(e)});self.persist();raise

def closed_rows(proof,hf,part,ledger=None):
 native.need(proof['protocol']['path']==PARENT.format(part),'Wrong native parent protocol')
 c=contract();env={k:v for k,v in os.environ.items() if k in ('HOME','USER','XDG_CONFIG_HOME','FFNDS_WORKER_CONFIG')}
 env.update(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 request={'proof':proof,'hf':str(Path(hf).resolve()),'part':part}
 if ledger is None:ledger=ChildLedger({},lambda:None,'read_only_parent_check')
 def run_checked():
  result=subprocess.run(c['argv'],input=json.dumps(request),text=True,capture_output=True,env=env,timeout=c['child_timeout_seconds'],check=False)
  native.need(result.returncode==0,'Generation byte verifier failed: '+result.stderr[-2000:])
  value=json.loads(result.stdout);native.need(value.get('status')=='PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU' and value.get('request')==request and value.get('child_cpu_affinity')==sorted(os.sched_getaffinity(0)),'Generation verifier result differs')
  return value['rows']
 return ledger.call(run_checked)

def child():
 native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','numpy'}&set(sys.modules),'Child CUDA/import environment differs')
 request=json.load(sys.stdin);rows=original.closed_rows(request['proof'],Path(request['hf']),request['part'])
 native.need(not {'torch','numpy'}&set(sys.modules),'Numerical module loaded')
 print(json.dumps({'status':'PASS_BYTE_ONLY_NO_TORCH_NUMPY_GPU','request':request,'child_cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows}))

if __name__=='__main__':
 native.need(sys.argv==[str(ROOT/SELF),'--verify-child'],'Fixed child invocation required');child()
