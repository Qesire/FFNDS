import importlib.util, unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_activation24_spmm_probe.py'
s=importlib.util.spec_from_file_location('probe24',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class ProbeTests(unittest.TestCase):
    def test_k_axis_and_ties(self):
        import torch
        a=torch.tensor([[1.,-4.,3.,2.,9.,9.,9.,9.],[0.,0.,0.,0.,-1.,2.,-3.,4.]],dtype=torch.bfloat16)
        actual=m.select_top2(a)
        expected=torch.tensor([[0.,-4.,3.,0.,9.,9.,0.,0.],[0.,0.,0.,0.,0.,0.,-3.,4.]],dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(bool(((actual.reshape(2,2,4)!=0).sum(-1)<=2).all()))
        with self.assertRaises(ValueError):m.select_top2(a[:,:7])
    def test_padding(self):
        self.assertEqual(m.padded_rows(31668),31680)
        self.assertEqual(m.padded_rows(16),16)
        with self.assertRaises(ValueError):m.padded_rows(0)
    def test_ledger_precharge_and_failure(self):
        counts={'compress':m.LIMITS['compress']-1}
        m.charge(counts,'compress')
        with self.assertRaises(RuntimeError):m.charge(counts,'compress')
        self.assertEqual(counts['compress'],m.LIMITS['compress'])
        calls={}
        def failed():
            m.charge(calls,'sparse_gemm')
            raise ValueError('backend failure')
        with self.assertRaises(ValueError):failed()
        self.assertEqual(calls['sparse_gemm'],1)
    def test_frozen_scope(self):
        plan={'id':m.ID,'limits':m.LIMITS.copy(),'shapes':[list(x) for x in m.SHAPES],'warmups':2,'repetitions':5,'wall_seconds':300,'assets_finalized':True}
        active={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'hash','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
        m.authorize(plan,'hash',active)
        with self.assertRaises(ValueError):m.authorize(plan,'wrong',active)
        plan['shapes'][0][0]=1
        with self.assertRaises(ValueError):m.authorize(plan,'hash',active)

class LoadedBackendTests(unittest.TestCase):
    def test_exact_loaded_backend_bindings(self):
        plan={'bindings':[{'path':'libtorch_cuda.so','sha256':'torch-sha'},{'path':'libcusparseLt.so.0','sha256':'lt-sha'}]}
        t=(m.ROOT/'libtorch_cuda.so').resolve();c=(m.ROOT/'libcusparseLt.so.0').resolve()
        maps='100-200 r-xp 0 00:00 0 '+str(t)+'\n200-300 r-xp 0 00:00 0 '+str(c)+'\n'
        result=m.verify_loaded_backends(plan,maps)
        self.assertEqual(result['torch_cuda']['sha256'],'torch-sha')
        self.assertEqual(result['cusparselt']['binding_path'],'libcusparseLt.so.0')
    def test_wrong_path_or_missing_backend_fails(self):
        plan={'bindings':[{'path':'libtorch_cuda.so','sha256':'t'},{'path':'libcusparseLt.so.0','sha256':'c'}]}
        t=(m.ROOT/'libtorch_cuda.so').resolve();wrong=Path('/tmp/libcusparseLt.so.0')
        with self.assertRaises(ValueError):m.verify_loaded_backends(plan,'100-200 r-xp 0 00:00 0 '+str(t)+'\n')
        with self.assertRaises(ValueError):m.verify_loaded_backends(plan,'100-200 r-xp 0 00:00 0 '+str(t)+'\n200-300 r-xp 0 00:00 0 '+str(wrong)+'\n')
