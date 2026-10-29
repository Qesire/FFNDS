"""Bounded snapshot and soft sampled output accounting tolerant of NFS unlink races."""
import importlib.util,os,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_teal_v3_immutable_snapshot',ROOT/'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class SnapshotBudget(old.SnapshotBudget):
 def __init__(self,*args):
  super().__init__(*args);self.vanished_mutable_entries=0;self.scan_count=0
 def observation(self):
  return {'scan_count':self.scan_count,'vanished_mutable_entries':self.vanished_mutable_entries,
          'enforcement':'SOFT_SAMPLED_NOT_FILESYSTEM_QUOTA'}
 def scan(self,exclude_snapshot):
  self.scan_count+=1;total=0
  def missing(err):
   p=Path(err.filename) if err.filename else None
   if isinstance(err,FileNotFoundError) and p is not None and p!=self.out and not (p==self.snapshot or self.snapshot in p.parents):
    self.vanished_mutable_entries+=1;return
   raise err
  for directory,dirs,names in os.walk(self.out,topdown=True,followlinks=False,onerror=missing):
   root=Path(directory)
   if exclude_snapshot and root==self.out:dirs[:]=[d for d in dirs if d!='EXECUTED_SOURCES']
   for name in names:
    path=root/name
    try:st=path.lstat()
    except FileNotFoundError as err:missing(err);continue
    if stat.S_ISREG(st.st_mode):total+=st.st_size
  return total
 def total_before_snapshot(self):
  total=self.scan(False);old.need(total<=self.limit,'Output byte budget exceeded');return total
 def total(self):
  old.need(self.initial is not None,'Snapshot not initialized')
  mutable=self.scan(True);total=mutable+self.initial['bytes'];old.need(total<=self.limit,'Output byte budget exceeded')
  return {'snapshot_bytes':self.initial['bytes'],'mutable_bytes':mutable,'total_bytes':total,'limit_bytes':self.limit,**self.observation()}

def prior_evidence(hf,kind,shard):
 base=old.base
 if kind=='raft':
  old.need(shard==1,'Only V2 failed RAFT shard eligible');protocol=ROOT/'config/e0_teal_wan_raft_v2_part1_20261029_pending_review_v1.json';suffix='worker';expected='FAILED_SM120_RAFT_PRESERVED_NO_RETRY';rc=1
 else:
  old.need(kind=='quality' and shard in range(4),'Invalid quality shard');protocol=ROOT/f'config/e0_teal_wan_quality_v2_part{shard}_20261029_pending_review_v1.json';suffix='cpu_preflight/worker';expected='PASS_TEAL_WAN_EIGHT_AXIS_BYTE_PREFLIGHT_NO_NUMPY_TORCH_GPU_MODEL';rc=0
 p=base.read(protocol);worker=(Path(hf)/p['output_HF_relative']).parent.parent/suffix
 proof={'protocol':base.binding(protocol,ROOT),'RUN':base.binding(worker/'results/RUN.json',hf),'exit':base.binding(worker/'EXIT_RECEIPT.json',hf)}
 run=base.read_bound(hf,proof['RUN']);receipt=base.read_bound(hf,proof['exit'])
 old.need(run.get('id')==p['id']==receipt.get('id') and run.get('protocol_sha256')==receipt.get('protocol_sha256')==proof['protocol']['sha256']
  and receipt.get('run_sha256')==proof['RUN']['sha256'] and receipt.get('child_returncode')==rc and receipt.get('process_exited') is True
  and receipt.get('timeout') is False and receipt.get('automatic_retry') is False and run.get('status')==expected,'V2 prior actual receipt differs')
 if kind=='raft':old.need(run.get('error_type')=='FileNotFoundError' and '.nfs' in run.get('error','') and run['completed']['RAFT_forwards']==83 and run['predebited']['RAFT_forwards']==84,'V2 NFS failure identity differs')
 return proof
