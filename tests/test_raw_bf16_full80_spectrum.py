"""Tiny, synthetic CPU controls for the BF16 spectral math; no real assets."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full80_spectrum',ROOT/'tools/raw_bf16_full80_spectrum.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)


class SpectrumControls(unittest.TestCase):
    def test_source_holdout_principal_gram_excludes_extreme_source(self):
        a=torch.tensor([[1.,0.,0.,0.],[2.,0.,0.,0.],[0.,1.,0.,0.],[0.,2.,0.,0.],[99.,99.,0.,0.],[100.,100.,0.,0.]],dtype=torch.float64)
        gram=m.sample_gram(a);train,gtrain=m.source_subset(a,gram,[0,1],3,2)
        torch.testing.assert_close(gtrain,m.sample_gram(train))
        basis,record=m.fit_basis(train,gtrain,2)
        self.assertEqual(record['active_rank'],2)
        torch.testing.assert_close(basis@basis.T,torch.diag(torch.tensor([1.,1.,0.,0.],dtype=torch.float64)),atol=1e-10,rtol=0)
        changed=a.clone();changed[4:]*=1000
        train2,gtrain2=m.source_subset(changed,m.sample_gram(changed),[0,1],3,2)
        torch.testing.assert_close(train,train2);torch.testing.assert_close(gtrain,gtrain2)

    def test_bias_anchoring_and_distinct_gates(self):
        bias=torch.tensor([1.,2.,-1.,3.],dtype=torch.float64);g0=torch.tensor([1.,0.,1.,0.]);g1=torch.tensor([0.,2.,0.,1.])
        cases={(s,t):{'Y':(bias+torch.tensor([[1.,2.,0.,1.],[0.,1.,1.,2.]],dtype=torch.float64)), 'gate':(g0 if t==0 else g1)} for s in range(3) for t in range(2)}
        ms=m.metrics([g0,g1]);a=m.anchored_rows(cases,bias,ms['shared'],steps=[0,1],source_indices=[0,1,2])
        self.assertEqual(tuple(a.shape),(12,4));self.assertTrue(torch.isfinite(a).all())
        a0=m.anchored_rows(cases,bias,ms['step0'],steps=[0],source_indices=[0,1,2]);a1=m.anchored_rows(cases,bias,ms['step1'],steps=[1],source_indices=[0,1,2])
        v0,_=m.fit_basis(a0,m.sample_gram(a0),1);v1,_=m.fit_basis(a1,m.sample_gram(a1),1)
        self.assertLess(float((v0.T@v1).abs().max()),1e-10)
        self.assertFalse(torch.allclose(ms['step0'],ms['step1']))
        constant={(s,t):{'Y':bias.repeat(2,1),'gate':g0 if t==0 else g1} for s in range(3) for t in range(2)}
        z=m.anchored_rows(constant,bias,ms['shared'],steps=[0,1],source_indices=[0,1,2]);self.assertEqual(float(z.abs().max()),0.)

    def test_rank_deficient_zero_gate_and_bf16_two_stage(self):
        a=torch.tensor([[1.,0.,0.,0.],[2.,0.,0.,0.]],dtype=torch.float64)
        basis,info=m.fit_basis(a,m.sample_gram(a),2);self.assertEqual(info['active_rank'],1)
        gate=torch.tensor([1.,0.,2.,0.],dtype=torch.float64)
        w=torch.tensor([[1.,2.,3.,4.],[2.,0.,1.,0.],[-1.,3.,0.,2.],[0.,0.,0.,0.],[4.,1.,2.,3.],[2.,2.,2.,2.]],dtype=torch.float64)
        bias=torch.tensor([1.,-2.,3.,4.],dtype=torch.bfloat16)
        factors=m.compile_factors(w*gate,bias,gate,basis)
        self.assertTrue(all(torch.isfinite(x).all() for x in factors.values()))
        self.assertEqual(tuple(factors['left.weight'].shape),(2,6));self.assertEqual(tuple(factors['right.weight'].shape),(4,2))
        h=torch.tensor([[1.,2.,-1.,0.,.5,1.]],dtype=torch.bfloat16)
        actual=m.factor_forward(h,factors)
        middle=torch.mm(h,factors['left.weight'].T)
        expected=torch.addmm(bias,middle,factors['right.weight'].T)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(float(actual[0,1]),float(bias[1]));self.assertEqual(float(actual[0,3]),float(bias[3]))

    def test_postwrite_rounding_and_signed_closure(self):
        r=torch.tensor([[2.,-1.,3.,-2.],[1.,2.,-3.,4.]],dtype=torch.bfloat16)
        y=torch.tensor([[.5,-1.,2.,1.],[2.,1.,1.,-1.]],dtype=torch.bfloat16);g=torch.tensor([.5,-1.,0.,2.],dtype=torch.bfloat16)
        u=m.postwrite(r,y,g);manual=r.clone();manual.add_(y*g);self.assertTrue(torch.equal(u,manual))
        yh=(y*.75).to(torch.bfloat16);uh=m.postwrite(r,yh,g)
        result=m.row_statistics(r,y,u,g,yh,uh)
        self.assertLess(abs(result['normalized_closure_error']),1e-12)
        self.assertAlmostEqual(result['normalized_u_relative_SSE'],result['normalized_direct']+result['normalized_radial']+result['normalized_cross_signed'],places=12)

    def test_commutator_matches_explicit_hidden_moments(self):
        a=torch.tensor([[1.,2.,0.],[0.,1.,1.]],dtype=torch.float64)
        b=torch.tensor([[1.,0.,1.],[0.,2.,0.]],dtype=torch.float64)
        c0,c1=a.T@a,b.T@b
        expected=float(((c0@c1-c1@c0)**2).sum()/(c0.square().sum()*c1.square().sum()))
        self.assertGreater(expected,0.)
        self.assertAlmostEqual(m.hidden_moment_commutator(a,b)['normalized_Frobenius_squared'],expected,places=12)
        self.assertAlmostEqual(m.hidden_moment_commutator(a,a)['normalized_Frobenius_squared'],0.,places=12)

    def test_rounding_triangle_and_zero_clamp(self):
        case={'Y':torch.tensor([[1.,0.,0.,0.]]),'gate':torch.ones(4),
              'continuous_rounding_error':torch.tensor([[.25,0.,0.,0.]])}
        cases={(0,0):case}; metric=torch.ones(4)
        out=m.rounding_bridge(cases,metric,steps=[0],source_indices=[0],continuous_tail=1.)
        self.assertAlmostEqual(out['teacher_addmm_rounding_relative_RMS_squared'],.0625)
        self.assertAlmostEqual(out['continuous_hW_rank_lower_relaxation_after_rounding_triangle'],.5625)
        out=m.rounding_bridge(cases,metric,steps=[0],source_indices=[0],continuous_tail=.01)
        self.assertEqual(out['continuous_hW_rank_lower_relaxation_after_rounding_triangle'],0.)

    def test_metric_uses_rounded_effective_gain(self):
        r=torch.tensor([[1.,-1.,2.,-2.]],dtype=torch.bfloat16)
        y=torch.tensor([[1.,2.,-1.,.5]],dtype=torch.bfloat16)
        yh=torch.tensor([[.5,1.,-2.,1.]],dtype=torch.bfloat16)
        g=torch.tensor([.7,.3,-.8,1.1],dtype=torch.bfloat16)
        u=m.postwrite(r,y,g); uh,effective=m.postwrite(r,yh,g,1.3,return_gate=True)
        self.assertFalse(torch.equal(effective.float(),g.float()*m.fp32(1.3)))
        result=m.row_statistics(r,y,u,g,yh,uh,effective_gate=effective)
        ref=y.double()*g.double()
        expected=float(((yh.double()*effective.double()-ref)**2).sum()/ref.square().sum())
        self.assertAlmostEqual(result['gated_Y_relative_SSE'],expected,places=12)
        self.assertNotAlmostEqual(result['gated_Y_relative_SSE'],result['unscaled_factor_gated_Y_relative_SSE'],places=5)

    def test_bounded_fixed_gains_and_zero_case(self):
        self.assertAlmostEqual(m.bounded_quartic_minimum([0.,0.,1.,-2.,1.])['beta_fit_FP64'],1.,places=8)
        self.assertAlmostEqual(m.bounded_quartic_minimum([9.,-6.,1.,0.,0.])['beta_fit_FP64'],2.,places=8)
        r=torch.tensor([[1.,-1.,2.,-2.]],dtype=torch.bfloat16);y=torch.tensor([[2.,-2.,1.,-1.]],dtype=torch.bfloat16);g=torch.ones(4,dtype=torch.bfloat16);u=m.postwrite(r,y,g)
        case={'r':r,'Y':y,'u':u,'gate':g};result=m.fit_gains([case,case],[y*.5,y*.5])
        self.assertAlmostEqual(result['gated_LS']['beta_fit_FP64'],2.)
        self.assertTrue(0.<=result['variance']['beta_fit_FP64']<=2.)
        zero={'r':r,'Y':y*0,'u':r,'gate':g*0};z=m.fit_gains([zero],[y*0]);self.assertEqual(z['gated_LS']['beta_fit_FP64'],1.)


if __name__=='__main__':unittest.main()
