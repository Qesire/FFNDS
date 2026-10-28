"""Synthetic tiny CPU controls; no checkpoint, training, full model or GPU."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from tools import e0_original_bf16_low39_fixed_norm as driver
from tools import raw_bf16_fixed_norm as core


class BudgetTests(unittest.TestCase):
    def test_eight_operations_have_five_projections_and_three_nonidentity_gains(self):
        self.assertEqual(len(driver.OPERATIONS),8)
        projected=sum(driver.mode_for(op)=='S' for op,_ in driver.OPERATIONS)
        gains=sum(driver.kappa_for(op,s)!=1 for op,s in driver.OPERATIONS)
        self.assertEqual((projected,gains),(5,3))
        limits=driver.limits()
        self.assertEqual(limits['native_FFN_calls'],8*40)
        self.assertEqual(limits['native_FFN_linears'],2*8*40-projected)
        self.assertEqual(limits['candidate_coefficient_GEMMs'],5)
        self.assertEqual(limits['candidate_output_GEMMs'],5)
        self.assertEqual(limits['teacher_full_Y_references_retained'],3)
        self.assertEqual(limits['original_checkpoint_block_buffer_loads'],8*41)
        self.assertEqual(limits['conditioning_normalization_gain_calls'],3)
        for key in ('decompositions','training_updates','video_generations','raw_W2_H2D','weight_covariance_eigh','projection_export_writes'):
            self.assertEqual(limits[key],0,key)

    def test_kappa_direction_frozen_and_no_runtime_fit(self):
        self.assertEqual(driver.KAPPAS,{'2':1.039908414775213,'3':1.020973593727097})
        self.assertTrue(0.95<core.gain_scalar(driver.KAPPAS['2'])<1)
        for op,s in driver.OPERATIONS:
            self.assertEqual(driver.kappa_for(op,s),driver.KAPPAS[str(s)] if op.startswith('C') else 1)
        for bad in (0,-1,float('nan'),float('inf'),0.9):
            with self.subTest(bad=bad),self.assertRaises(ValueError):core.gain_scalar(bad)

    def test_ledger_failure_is_predebited_and_no_retry(self):
        report={};events=[];ledger=driver.Ledger(report,events.append)
        with self.assertRaisesRegex(RuntimeError,'failed'):
            with ledger.work('norm_gain_FP32_multiplies',rid=driver.RUN_IDS[4]):raise RuntimeError('failed')
        self.assertEqual(report['predebited']['norm_gain_FP32_multiplies'],1)
        self.assertEqual(report['observed'].get('norm_gain_FP32_multiplies',0),0)
        self.assertEqual(events[0]['kind'],'reserved_before_call')
        with self.assertRaises(ValueError):
            with ledger.work('weight_covariance_eigh'):pass
        with self.assertRaises(ValueError):
            with ledger.work('conditioning_normalization_gain_calls',rid='old_run'):pass
        with self.assertRaises(ValueError):ledger.finish()

    def test_saved_pack_byte_checker_rejects_payload_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            hf=Path(temp);p=hf/'tiny.safetensors';payload=struct.pack('<4f',1,2,3,4)
            header=json.dumps({'flow':{'dtype':'F32','shape':[2,2],'data_offsets':[0,16]}}).encode()
            p.write_bytes(struct.pack('<Q',len(header))+header+payload)
            tensors={'flow':{'shape':[2,2],'dtype':'float32','raw_tensor_sha256':hashlib.sha256(payload).hexdigest()}}
            driver.check_pack(hf,driver.native.binding(p,hf),tensors)
            p.write_bytes(p.read_bytes()[:-1]+b'x')
            with self.assertRaises(ValueError):driver.check_pack(hf,driver.native.binding(p,hf),tensors)

    def test_import_does_not_load_torch_numpy_or_mutate_prior_globals(self):
        code="import sys; sys.path.insert(0,'tools'); import e0_original_bf16_low39_four_corner as p; a=(p.ID,p.RUN_IDS,p.limits()); import e0_original_bf16_low39_fixed_norm; assert a==(p.ID,p.RUN_IDS,p.limits()); assert not {'torch','numpy'}&set(sys.modules)"
        completed=subprocess.run([sys.executable,'-c',code],cwd=driver.ROOT,text=True,capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)


class TinyNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    def work(self):
        events=[]
        @contextmanager
        def counted(key):
            events.append(('pre',key));yield;events.append(('return',key))
        return events,counted

    def test_gain_changes_fresh_norm_only_and_has_explicit_fp32_semantics(self):
        t=self.torch;raw=t.tensor([[1,3,-2,4],[-1,7,-4,2]],dtype=t.bfloat16)
        snapshot=raw.clone();normalized=t.nn.functional.layer_norm(raw,(4,),eps=core.EPS)
        old=normalized.clone();events,work=self.work();kappa=driver.KAPPAS['2']
        expected=(old.float()*core.gain_scalar(kappa)).bfloat16()
        returned=core.normalized_gain(t,normalized,raw,kappa,work)
        self.assertIs(returned,normalized);self.assertTrue(t.equal(returned.view(t.uint8),expected.view(t.uint8)))
        self.assertTrue(t.equal(raw.view(t.uint8),snapshot.view(t.uint8)))
        self.assertEqual([key for kind,key in events if kind=='pre'],['normalization_storage_identity_checks','conditioning_normalization_gain_calls','norm_gain_BF16_to_FP32_casts','norm_gain_FP32_multiplies','norm_gain_FP32_to_BF16_copies'])

    def test_identity_wrapper_is_byte_neutral_and_storage_alias_rejected_before_write(self):
        t=self.torch;raw=t.tensor([[1,3,-2,4]],dtype=t.bfloat16);normalized=t.nn.functional.layer_norm(raw,(4,),eps=core.EPS)
        old=normalized.clone();events,work=self.work()
        self.assertIs(core.normalized_gain(t,normalized,raw,1.,work),normalized)
        self.assertTrue(t.equal(old.view(t.uint8),normalized.view(t.uint8)))
        self.assertNotIn(('pre','conditioning_normalization_gain_calls'),events)
        snapshot=raw.clone();events,work=self.work()
        with self.assertRaises(ValueError):core.normalized_gain(t,raw.view_as(raw),raw,driver.KAPPAS['2'],work)
        self.assertTrue(t.equal(raw.view(t.uint8),snapshot.view(t.uint8)))
        self.assertEqual(events,[('pre','normalization_storage_identity_checks')])

    def test_effective_sigma_uses_deployed_alpha_and_keeps_raw_variance_unchanged(self):
        t=self.torch;stats={'q':t.tensor([[1.,2.]],dtype=t.float64),'variance':t.tensor([2.],dtype=t.float64)}
        saved=stats['variance'].clone();events,work=self.work();kappa=driver.KAPPAS['3']
        eff,diagnostic=core.effective_statistics(t,stats,kappa,work)
        self.assertIs(eff['q'],stats['q']);self.assertTrue(t.equal(stats['variance'],saved))
        ratio=float(t.sqrt((eff['variance']+core.EPS)/(stats['variance']+core.EPS)))
        self.assertAlmostEqual(ratio,1/core.gain_scalar(kappa),places=14)
        self.assertGreater(diagnostic['effective_readout_sigma_mean'],diagnostic['raw_u_sigma_mean'])
        self.assertEqual(diagnostic['deployed_FP32_reciprocal_alpha'],core.gain_scalar(kappa))

    def test_constant_shift_and_bias_survive_and_signed_identity_closes(self):
        t=self.torch;u=t.tensor([[1.,-2.,3.,0.]],dtype=t.float64);K=t.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],dtype=t.float64)
        q=u@K;variance=(u-u.mean(1,keepdim=True)).square().mean(1)
        raw={'q':q,'variance':variance};kappa=driver.KAPPAS['2'];eff,_=core.effective_statistics(t,raw,kappa)
        constant=t.tensor([[7.,-3.]],dtype=t.float64)
        old=q/t.sqrt(variance[:,None]+core.EPS)+constant
        changed=q/t.sqrt(eff['variance'][:,None]+core.EPS)+constant
        alpha=core.gain_scalar(kappa)
        self.assertTrue(t.allclose(changed,(old-constant)*alpha+constant,atol=1e-14,rtol=1e-14))
        self.assertFalse(t.allclose(changed,old*alpha,atol=1e-6,rtol=1e-6))
        diag=driver.geometry_core.head_terms(t,changed,old,eff,raw)
        self.assertEqual(diag['direct_relative_SSE'],0)
        self.assertLess(abs(diag['native_remainder_relative_SSE']),1e-28)
        self.assertLess(abs(diag['closure_residual']),1e-12)


if __name__=='__main__':unittest.main()
