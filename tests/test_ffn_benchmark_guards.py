"""Acceptance must reject omitted online work, policy swaps and duplicate scopes."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_ffn_benchmark import validate_candidate_runtime


class FFNBenchmarkGuards(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        request=json.loads((root/'reports/worker_20261020/resident_RR1_smoke_REQUEST.json').read_text())
        self.packed=request['hif4_runtime']['packed_reference']
        # Fixture uses a real complete160 warmup record. Turning on this field
        # tests validation only; this fixture never becomes timing evidence.
        self.packed['residual_correction']['timing_includes_auxiliary_cost']=True
        self.sha=self.packed['policy_sha256']

    def affine(self):
        packed=copy.deepcopy(self.packed)
        packed['policy'].update(kind='OUTPUT_VARIANCE_CALIBRATED_REFERENCE',kernel_source_sha256='fixed-kernel')
        packed['output_affine']={'identity_control':False,'actual_positions':160,'timing_includes_affine_cost':True,
            'kernel_source_sha256':'fixed-kernel','coverage':[{'step':s,'block':b,'pointwise_kernel_calls':1,
                'coefficient_H2D_bytes':0} for s in range(4) for b in range(40)]}
        return packed

    def test_real_full160_control_and_affine_shapes_are_accepted(self):
        validate_candidate_runtime(self.packed,self.sha)
        validate_candidate_runtime(self.affine(),self.sha)

    def test_policy_swap_and_false_half_width_claim_are_rejected(self):
        with self.assertRaises(RuntimeError):validate_candidate_runtime(self.packed,'another-policy')
        value=copy.deepcopy(self.packed)
        value['coverage']['step0_block0_ffn0']['actual_weight_shape']=[13824,5120]
        with self.assertRaises(RuntimeError):validate_candidate_runtime(value,self.sha)

    def test_affine_missing_charge_identity_and_duplicate_scope_are_rejected(self):
        for mutation in ['missing_charge','identity','duplicate_scope','kernel_swap']:
            value=self.affine();affine=value['output_affine']
            if mutation=='missing_charge':affine['timing_includes_affine_cost']=False
            if mutation=='identity':affine['identity_control']=True
            if mutation=='duplicate_scope':affine['coverage'][-1]=dict(affine['coverage'][0])
            if mutation=='kernel_swap':affine['kernel_source_sha256']='changed-kernel'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                validate_candidate_runtime(value,self.sha)


if __name__=='__main__':unittest.main()
