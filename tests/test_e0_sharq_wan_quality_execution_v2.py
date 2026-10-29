import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('test_teal_exec',ROOT/'tools/e0_sharq_wan_quality_execution_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_affinity_disjoint_and_mode(self):
  rows=[m.affinity('visual',p)+m.affinity('quality',p) for p in range(4)]
  self.assertEqual(len(set(sum(rows,[]))),16)
  self.assertEqual([m.affinity('raft',p) for p in range(4)],[[2],[6],[10],[14]])
  with self.assertRaises(ValueError):m.affinity('bad',0)
  self.assertEqual([m.physical_gpu(p) for p in (0,1)],[1,2])
  with self.assertRaises(ValueError):m.physical_gpu(True)
 def test_visual_isolated_import_in_actual_subprocess(self):
  import subprocess
  python=ROOT/'.venv/bin/python'
  code="import sys;sys.path.insert(0,"+repr(str(ROOT/'tools'))+");import e0_sharq_wan_visual_cpu;assert sys.flags.isolated"
  subprocess.run([str(python),'-I','-B','-c',code],check=True,timeout=20)
 def test_nonisolated_visual_job_is_rejected(self):
  import os,tempfile
  from unittest.mock import patch
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as td:
   f=Path(td)/'x';f.write_text('x');plan={'part':0,'id':'x','execution_private_worker_config_sha256':m.PROFILE_SHA};a=SimpleNamespace(job_spec=f,protocol=f,hf_root=Path(td));argv=['env','FFNDS_WORKER_CONFIG='+str(f),'placeholder'];j={'id':'x','protocol_sha256':'sha','protocol_path':str(f),'cwd':str(ROOT),'hf_root':td,'argv':argv};base=SimpleNamespace(sha=lambda p:m.PROFILE_SHA if p==f and False else 'sha',read=lambda p:j)
   with patch.object(m.os,'sched_getaffinity',return_value={0,1}),patch.object(m,'PROFILE_SHA','sha'),patch.object(m.sys,'flags',SimpleNamespace(isolated=0)):
    plan['execution_private_worker_config_sha256']='sha'
    with self.assertRaisesRegex(ValueError,'isolation'):m.verify_job(a,plan,{'launch_spec_sha256':'sha'},base,'visual','run','source.py',{})
 def test_no_unfrozen_fake_job(self):
  from types import SimpleNamespace
  with self.assertRaises(ValueError):m.verify_job(SimpleNamespace(job_spec=None),{},None,None,'raft','run','',{})
if __name__=='__main__':unittest.main()

class ExactJobTests(unittest.TestCase):
 def test_metric_exact_argv_and_uuid_change_rejected(self):
  import tempfile,os
  from types import SimpleNamespace
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   hf=Path(t);python=hf/'metric-python';python.write_bytes(b'p');profile=hf/'profile';profile.write_bytes(b'f');protocol=hf/'plan';protocol.write_bytes(b'p');spec=hf/'job';spec.write_bytes(b'j')
   plan={'id':'x','shard':0,'output_HF_relative':'out/results','execution_private_worker_config_sha256':m.PROFILE_SHA,
         'import_environment':{'python_HF_relative':'metric-python','python_sha256':'pythonsha'},
         'execution_device':{'physical_gpu_index':1},'execution_monitor_binary':{'path':m.MONITOR,'bytes':1,'sha256':'monsha'}}
   uuid='GPU-abc';argv=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin',
     'bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),'1',uuid,'taskset','-c','2',str(python),'-I','-S','-B',str(ROOT/'source.py'),'run',
     '--protocol',str(protocol),'--hf-root',str(hf),'--job-spec',str(spec)]
   job={'id':'x','protocol_sha256':'plan_sha','protocol_path':str(protocol),'cwd':str(ROOT),'hf_root':str(hf),'argv':argv,
        'output_dir':str(hf/'out'),'run_json':str(hf/'out/results/RUN.json'),'external_timeout_seconds':3900}
   def sha(path):
    path=Path(path)
    return {profile:m.PROFILE_SHA,protocol:'plan_sha',spec:'jobsha',python:'pythonsha'}.get(path,'monsha')
   base=SimpleNamespace(sha=sha,read=lambda _:job);scope={'gpu_launch_spec_sha256':'jobsha'};args=SimpleNamespace(job_spec=spec,protocol=protocol,hf_root=hf)
   with patch.object(m.os,'sched_getaffinity',return_value={2}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':uuid,'FFNDS_PHYSICAL_GPU':'1'}),patch.object(m.shutil,'which',return_value=m.MONITOR),patch.object(m,'monitor_binding',return_value=plan['execution_monitor_binary']):
    m.verify_job(args,plan,scope,base,'raft','run','source.py',{'outer_seconds':3900})
    job['argv'][8]='0'
    with self.assertRaises(ValueError):m.verify_job(args,plan,scope,base,'raft','run','source.py',{'outer_seconds':3900})
