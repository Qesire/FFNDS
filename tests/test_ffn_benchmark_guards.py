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

    def budget(self):
        packed=copy.deepcopy(self.packed)
        counts=[54,108,108,162]
        packed['policy'].update(kind='STEP_BUDGET_PACKED_REFERENCE',identity_control=False,groups_by_step=counts)
        for key,row in packed['coverage'].items():
            step=int(key.split('_')[0][4:]);width=counts[step]*64
            row['actual_weight_shape']=[width,5120] if key.endswith('ffn0') else [5120,width]
            row['main_GEMM_density']=width/13824
        packed['actual_global_main_MAC_density']=.5
        return packed

    def test_true_global_budget_is_accepted_but_wrong_shape_is_rejected(self):
        value=self.budget();validate_candidate_runtime(value,self.sha)
        value['coverage']['step3_block39_ffn2']['actual_weight_shape']=[5120,6912]
        with self.assertRaises(RuntimeError):validate_candidate_runtime(value,self.sha)

    def test_unweighted_average_is_not_accepted_as_actual_compute(self):
        value=self.budget()
        for key,row in value['coverage'].items():
            if key.startswith('step0_'):row['rows']*=2
        for key,row in value['residual_correction']['coverage'].items():
            if key.startswith('step0_'):row['rows']*=2
        with self.assertRaises(RuntimeError):validate_candidate_runtime(value,self.sha)

    def test_auxiliary_call_count_cannot_hide_wrong_matrix_shapes(self):
        value=self.budget()
        value['residual_correction']['coverage']['step0_block0']['auxiliary_GEMM_shapes'][0]=[128,5120]
        with self.assertRaises(RuntimeError):validate_candidate_runtime(value,self.sha)

    def test_unprofiled_quality_coverage_never_becomes_timing_evidence(self):
        value=self.budget()
        value['residual_correction']['timing_includes_auxiliary_cost']=False
        validate_candidate_runtime(value,self.sha,require_timing=False)
        with self.assertRaises(RuntimeError):validate_candidate_runtime(value,self.sha)
        value['coverage']['step3_block39_ffn2']['actual_weight_shape']=[5120,6912]
        with self.assertRaises(RuntimeError):
            validate_candidate_runtime(value,self.sha,require_timing=False)


if __name__=='__main__':unittest.main()
