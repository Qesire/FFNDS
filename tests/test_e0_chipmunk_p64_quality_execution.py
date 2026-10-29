import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_quality_execution as ex
class Tests(unittest.TestCase):
 def test_gpu_and_cpu_partitions(self):
  self.assertEqual([ex.physical_gpu(k,p) for k in ('quality','raft') for p in (0,1)],[0,1,2,3])
  groups=[set(ex.affinity(k,p)) for k in ('quality','raft','visual') for p in (0,1)]
  self.assertTrue(all(a.isdisjoint(b) for i,a in enumerate(groups) for b in groups[i+1:]))
  self.assertEqual([len(x) for x in groups],[2,2,1,1,2,2])
 def test_private_profile_and_rejected_assignments(self):
  self.assertEqual(ex.PROFILE_SHA,'5a4e5bf320c525deb8f952e22972fd2f8bc8c35fe8f66dcea1db1e64ef8a90d2')
  for args in (('visual',0),('quality',2),('x',0)):
   with self.assertRaises(ValueError):ex.physical_gpu(*args)
if __name__=='__main__':unittest.main()
class JobTests(unittest.TestCase):
 def test_four_gpu_jobs_bind_the_correct_index_and_no_cpu_overlap(self):
  from unittest.mock import patch
  import e0_chipmunk_p64_metric_jobs as jobs
  from pathlib import Path
  cpus=[]
  for kind in ('quality','raft'):
   for part in (0,1):
    plan={'status':'FROZEN_PENDING_REGISTRATION','assets_finalized':True,'execution_private_worker_config_sha256':ex.PROFILE_SHA,'part':part,'execution_device':{'physical_gpu_index':ex.physical_gpu(kind,part)},'output_HF_relative':f'test/part{part}/results','resources':{'outer_seconds':10},'CPU_preflight_resources':{'outer_seconds':5},'id':'test','import_environment':{'python_HF_relative':'test-python'}}
    paths={'cpu':Path('/tmp/cpu'),'gpu':Path('/tmp/gpu'),'verify':Path('/tmp/verify')}
    with patch.object(jobs,'sha',return_value=ex.PROFILE_SHA):made=jobs.jobs(kind,plan,Path('/tmp/protocol'),Path('/tmp/profile'),'GPU-TEST',paths)
    argv=made['gpu']['argv'];i=argv.index('GPU-TEST');self.assertEqual(argv[i-1],str(ex.physical_gpu(kind,part)));cpus.append(set(ex.affinity(kind,part)))
  self.assertTrue(all(a.isdisjoint(b) for i,a in enumerate(cpus) for b in cpus[i+1:]))
