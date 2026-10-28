import subprocess
import sys
import unittest
import copy
from pathlib import Path
from collections import Counter
from contextlib import contextmanager

import torch
from tools import raw_bf16_output_projection as core
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_original_bf16_low39_four_corner as driver


class FourCornerControls(unittest.TestCase):
    def test_weight_basis_and_forward(self):
        torch.manual_seed(71)
        p={**core.P,'D':8,'H':12,'head_outputs':1,'head_columns':2,'rank':4}
        w=torch.randn(12,8).bfloat16()
        heads={s:torch.randn(8,1,dtype=torch.float64) for s in (2,3)}
        gates={s:torch.randn(8).bfloat16() for s in (2,3)}
        counts=Counter()
        @contextmanager
        def work(name):
            counts[name]+=1
            yield
        pack, evidence=core.fit_weight_basis(torch,w,heads,gates,work,p)
        self.assertEqual(counts['FP64_stat_matrix_products'],8)
        self.assertLess(evidence['FP64_head_span_squared_leak'],1e-18)
        self.assertLess(evidence['FP64_orthogonality_max_abs'],1e-9)
        self.assertEqual(pack['B'].shape,(12,4)); self.assertEqual(pack['Phi'].shape,(8,4))
        pack['b2']=torch.randn(8).bfloat16(); hidden=torch.randn(3,12).bfloat16()
        actual=core.projected_W2(torch,hidden,pack,work,p)
        expected=torch.addmm(pack['b2'],torch.mm(hidden,pack['B']),pack['Phi'].T)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(core.projected_W2(torch,torch.zeros_like(hidden),pack,work,p),pack['b2'].expand(3,-1)))
        self.assertEqual(counts['candidate_coefficient_GEMMs'],2)
        self.assertEqual(counts['candidate_output_GEMMs'],2)
        self.assertEqual(actual.dtype,torch.bfloat16)

    def test_head_signed_decomposition(self):
        target=torch.tensor([[2.,-1.]],dtype=torch.bfloat16)
        actual=torch.tensor([[1.5,-.5]],dtype=torch.bfloat16)
        stats={'q':torch.tensor([[1.,-2.]],dtype=torch.float64),'variance':torch.tensor([4.],dtype=torch.float64)}
        truth={'q':torch.tensor([[2.,-1.]],dtype=torch.float64),'variance':torch.tensor([1.],dtype=torch.float64)}
        t=core.head_terms(torch,actual,target,stats,truth)
        self.assertAlmostEqual(t['continuous_relative_SSE'],t['direct_relative_SSE']+t['radial_relative_SSE']+t['signed_cross_relative_SSE'])
        self.assertAlmostEqual(t['head_relative_SSE'],t['continuous_relative_SSE']+t['native_remainder_relative_SSE']+t['native_signed_remainder_cross'])
        self.assertGreater(t['native_remainder_relative_SSE'],0)

    def test_scheduler_separates_float_arithmetic_and_cast(self):
        before=torch.tensor([1.01,-2.34],dtype=torch.bfloat16); flow=torch.tensor([.31,1.17],dtype=torch.float32)
        pre=before.float()-.9375*flow+.625*flow; after=pre.bfloat16()
        t=core.scheduler_residual(torch,before,flow,after,.9375,.625)
        ideal=before.double()-(.9375-.625)*flow.double()
        torch.testing.assert_close(t['arithmetic']+t['cast'],after.double()-ideal)
        self.assertGreater(t['BF16_cast_squared'],0)
        with self.assertRaises(ValueError): core.scheduler_residual(torch,before,flow,torch.zeros_like(after),.9375,.625)

    def test_four_corner_negative_cross_and_distinct_changes(self):
        dd=torch.tensor([[[[10.]],[[10.]]]])
        sd=dd+torch.tensor([[[[2.]],[[1.]]]])
        ds=dd+torch.tensor([[[[-1.]],[[2.]]]])
        ss=ds+(sd-dd)+torch.tensor([[[[-2.]], [[-1.]]]])
        r=core.four_corner(torch,dd,sd,ds,ss); t=r['terms']
        self.assertEqual(t['teacher_state_local'],5); self.assertEqual(t['Dense_state_propagation'],5)
        self.assertEqual(t['approximation_state_change'],5); self.assertEqual(t['local_change_cross'],-10)
        self.assertEqual(t['state_change_cross'],0); self.assertEqual(t['local_state_cross'],0)
        self.assertEqual(r['total_squared'],5); self.assertAlmostEqual(r['closure_residual'],0)
        self.assertEqual(len(r['time_slots']),2)
        for slot in r['time_slots']:
            self.assertEqual(len(slot['terms']),6)
            self.assertAlmostEqual(slot['total_squared'],sum(slot['terms'].values()))
            self.assertGreater(slot['Dense_target_energy'],0)

    def test_core_import_does_not_import_torch(self):
        cmd='import sys; import tools.raw_bf16_output_projection; assert "torch" not in sys.modules'
        subprocess.run([sys.executable,'-c',cmd],check=True)

    def test_wrapper_fresh_import_has_no_numeric_imports(self):
        cmd='import sys;sys.path.insert(0,"tools");import e0_original_bf16_low39_four_corner;assert "torch" not in sys.modules and "numpy" not in sys.modules'
        subprocess.run([sys.executable,'-c',cmd],check=True)

    def test_failed_call_and_zero_model_extensions_keep_actual_debits(self):
        report={};events=[];ledger=driver.Ledger(report,events.append)
        def fail():raise RuntimeError('actual call failed')
        with self.assertRaisesRegex(RuntimeError,'actual call failed'):
            ledger.call('setup',{'scheduler_prepare':1},fail)
        self.assertEqual(report['predebited']['scheduler_prepare'],1)
        self.assertEqual(report['observed'],{})
        called=[]
        for key in ('video_generations','activation_QDQ','training_updates','official_quality_calls'):
            with self.assertRaisesRegex(ValueError,'budget exceeded'):
                ledger.call('setup',{key:1},lambda:called.append(key))
        self.assertEqual(called,[])

    def test_each_real_library_decomposition_also_predebits_generic_total(self):
        report={};ledger=driver.Ledger(report)
        for i,key in enumerate(('head_span_QR','weight_covariance_eigh','final_basis_QR'),1):
            with ledger.work(key):self.assertEqual(report['predebited']['decompositions'],i)
        self.assertEqual(report['observed']['decompositions'],3)
        self.assertEqual(driver.limits()['decompositions'],3)
        with self.assertRaisesRegex(ValueError,'budget exceeded'):
            with ledger.work('head_span_QR'):pass

    def test_twelve_flow_budget_preserves_original_W1_and_pays_prefetch(self):
        b=driver.limits();self.assertEqual(len(set(driver.RUN_IDS)),12)
        self.assertEqual(b['native_FFN_calls'],12*40)
        self.assertEqual(b['native_FFN_linears']+b['candidate_overridden_W2_dispatches'],2*12*40)
        self.assertEqual(b['candidate_coefficient_GEMMs'],6);self.assertEqual(b['candidate_output_GEMMs'],6)
        self.assertEqual(b['original_checkpoint_block_buffer_loads'],12*41)
        self.assertEqual(b['offload_buffer_resets'],12)
        self.assertEqual(b['scheduler_prepare'],2);self.assertEqual(b['FP32_initial_noise_draws'],2)

    def test_native_post_identity_keeps_arithmetic_and_cast_cross(self):
        groups=[]
        for source,flow in (([1.,-2.],[.31,.7]),([1.1,-1.8],[.30,.8])):
            before=torch.tensor(source,dtype=torch.bfloat16);f=torch.tensor(flow,dtype=torch.float32)
            after=(before.float()-.625*f).bfloat16()
            groups.append({'input':before,'flow':f,'post':after,
                'rounding':core.scheduler_residual(torch,before,f,after,.625,None)})
        report={};ledger=driver.Ledger(report)
        result=driver.post_identity(torch,[groups[0],None,None,groups[1]],ledger)
        self.assertLess(result['post_identity_residual_squared'],1e-25)
        self.assertGreater(result['native_rounding_difference_squared'],0)
        self.assertEqual(report['predebited']['post_error_identity_cases'],1)

if __name__=='__main__': unittest.main()
