import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('author_bench',ROOT/'tools/e0_teal_author_benchmark_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Event:
    def record(self): pass
    def elapsed_time(self,other): return 5.0
class Cache:
    def __init__(self,log):self.log=log
    def zero_(self):self.log.append('flush')
class Tensor:
    def __init__(self,vals):self.vals=vals
    def tolist(self):return self.vals
class FakeTorch:
    int='int';float='float'
    def __init__(self,log):
        self.log=log
        self.cuda=type('Cuda',(),{'Event':lambda *a,**k:Event(),'synchronize':lambda *a:None})()
    def empty(self,n,dtype,device):
        assert n==64000000 and dtype=='int' and device=='cuda'
        return Cache(self.log)
    def tensor(self,v,dtype):return Tensor(v)
    def quantile(self,a,b):return Tensor([1.0,0.9,1.1])

class TestBenchmark(unittest.TestCase):
    def test_bound_sources_and_v1_v2(self):
        p=m.draft();m.validate(p)
        self.assertEqual(p['design']['input_shape'],[1,1,4096]);self.assertEqual(p['design']['weight_shape'],[14336,4096])
        self.assertEqual(p['v2_closed']['files'][1]['sha256'],'fd83d1ebb9d8d5e82068ea8a3149558a3fcecbac55fca628bdc99ad90873280a')
        other=copy.deepcopy(p);other['v2_closed']['identity']='changed'
        self.assertNotEqual(m.fingerprint(p),m.fingerprint(other))
        with self.assertRaises(ValueError):m.validate(other)
    def test_exact_guarded_loop_accounting_and_flush(self):
        log=[];saved=[];row={'predebited_python_calls':6,'completed_python_calls':0}
        m.guarded_do_bench(FakeTorch(log),lambda:log.append('fn'),row,lambda:saved.append(copy.deepcopy(row)),100)
        self.assertEqual(row['warmup_calls'],25);self.assertEqual(row['repeat_calls'],100)
        self.assertEqual(row['predebited_python_calls'],131);self.assertEqual(row['completed_python_calls'],131)
        self.assertEqual(log.count('fn'),131);self.assertEqual(log.count('flush'),105)
        self.assertEqual(row['quantiles_ms'],[1.0,0.9,1.1]);self.assertEqual(saved[1]['completed_python_calls'],6)
    def test_budget_rejects_before_warmup(self):
        log=[];row={'predebited_python_calls':6,'completed_python_calls':0}
        with self.assertRaises(ValueError):m.guarded_do_bench(FakeTorch(log),lambda:log.append('fn'),row,lambda:None,100000)
        self.assertEqual(log.count('fn'),6);self.assertEqual(row['completed_python_calls'],6)
    def test_correctness_predebit_survives_failure_and_sync_marks_completion(self):
        row={};saved=[]
        def fail():raise RuntimeError('synthetic failure')
        with self.assertRaises(RuntimeError):m.correctness_call(row,'sparse_call',fail,lambda:None,lambda:saved.append(copy.deepcopy(row)))
        self.assertEqual(saved[-1]['sparse_call'],{'predebited':1,'completed':'UNKNOWN'})
        self.assertEqual(m.correctness_call(row,'dense_call',lambda:42,lambda:None,lambda:saved.append(copy.deepcopy(row))),42)
        self.assertEqual(saved[-1]['dense_call'],{'predebited':1,'completed':1})
    def test_actual_author_script_identity(self):
        script=(ROOT/m.SOURCE/'scripts/benchmark_gemv.py').read_text()
        self.assertIn('triton.testing.do_bench(lambda: our_sparse_gemv(x, W, sparsity_level), quantiles=quantiles, rep=1000)',script)
        self.assertIn('triton.testing.do_bench(lambda: dense_gemv(x, W_T), quantiles=quantiles)',script)
        self.assertNotIn('torch.compile',script);self.assertNotIn('do_bench_cudagraph',script)
        self.assertEqual(m.DESIGN['quantiles'],[0.5,0.2,0.8])
    def test_unregistered_scope_is_denied(self):
        p=m.draft();p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
        p['environment'].update(worker_config_sha256='a'*64,prepared_environment={},driver_version='550',monitor_binary={})
        with self.assertRaises(ValueError):m.authorize(p,'d')

if __name__=='__main__':unittest.main()
