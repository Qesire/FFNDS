import csv
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_monitor_v2 as m

class MonitorTest(unittest.TestCase):
 def monitor(self):
  with patch.dict(os.environ,{'FFNDS_PHYSICAL_GPU':'0','CUDA_VISIBLE_DEVICES':'GPU-test'}),patch.object(m.startup,'gpu_uuid',return_value='GPU-test'):
   return m.Monitor()
 def test_query_success_and_timeout(self):
  x=self.monitor()
  with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'42, 3\n','')) as call:
   self.assertEqual(x.query(['--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']),[['42',' 3']]); self.assertEqual(call.call_args.kwargs['timeout'],30)
  with patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',30)):
   with self.assertRaises(subprocess.TimeoutExpired):x.query(['--query-gpu=memory.used','--format=csv,noheader,nounits'])
  self.assertEqual([r['error_type'] for r in x.query_diagnostics],[None,'TimeoutExpired'])
 def test_foreign_fail_stop(self):
  x=self.monitor()
  with patch.object(x,'query',side_effect=[[[str(os.getpid()+100000),'1']],[['gpu']],[['1']]]):x.sample()
  with self.assertRaises(RuntimeError):x.check()
 def test_audit_matches_exact_code(self):
  import e0_ub1_consumer_official_quality as common
  import shutil
  with patch.object(shutil,'which',return_value='/usr/bin/nvidia-smi'):
   audit=common.audit_scope(Path('/tmp'),m.Monitor.query.__code__,Path('/usr/bin/nvidia-smi').resolve(),'0')
   with self.assertRaises(RuntimeError):audit('subprocess.Popen',('nvidia-smi',['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']))
 def test_real_query_is_allowed_by_audit(self):
  import e0_ub1_consumer_official_quality as common
  import shutil
  x=self.monitor()
  with patch.object(shutil,'which',return_value='/usr/bin/nvidia-smi'):
   audit=common.audit_scope(Path('/tmp'),m.Monitor.query.__code__,Path('/usr/bin/nvidia-smi').resolve(),'0')
   def fake(argv,**kwargs):
    audit('subprocess.Popen',('nvidia-smi',argv))
    return subprocess.CompletedProcess(argv,0,'3\n','')
   with patch.object(m.subprocess,'run',side_effect=fake):self.assertEqual(x.query(['--query-gpu=memory.used','--format=csv,noheader,nounits']),[['3']])
if __name__=='__main__':unittest.main()
