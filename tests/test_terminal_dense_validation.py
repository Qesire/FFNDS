import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from terminal_dense_validation import validate_runtime


class TerminalDenseValidationTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[1]
        # This is a structural test fixture, never timing or execution evidence.
        packed = json.loads((root/'reports/worker_20261020/resident_RR1_smoke_REQUEST.json').read_text())['hif4_runtime']['packed_reference']
        packed['policy'].update(kind='TERMINAL_DENSE_STEP_REFERENCE',identity_control=False,
                                groups_by_step=[54,54,108,216],residual_rank=256,terminal_step=3)
        for key,row in packed['coverage'].items():
            step = int(key.split('_')[0][4:]);width = [54,54,108,216][step]*64
            row.update(actual_weight_shape=[width,5120] if key.endswith('ffn0') else [5120,width],main_GEMM_density=width/13824)
        aux = packed['residual_correction']
        aux['coverage'] = {k:v for k,v in aux['coverage'].items() if not k.startswith('step3_')}
        aux.update(actual_positions=120,expected_positions=120,timing_includes_auxiliary_cost=True)
        for row in aux['coverage'].values():row['auxiliary_parameter_H2D_bytes']=0
        packed['actual_global_main_MAC_density']=.5
        packed['terminal_dense']={'active':True,'step':3,'identity_control':False,'full_main_positions':40,
            'expected_residual_positions':120,'expected_auxiliary_GEMMs':240,'nominal_auxiliary_MAC_density':120*256/(160*13824)}
        return packed

    def test_full_terminal_and_only120_residuals_are_accepted(self):
        packed=self.fixture()
        validate_runtime(packed,packed['policy_sha256'])
        self.assertEqual(len(packed['coverage']),320)
        self.assertEqual(len(packed['residual_correction']['coverage']),120)

    def test_hidden_terminal_auxiliary_work_or_wrong_dense_shape_is_rejected(self):
        for mutation in ['terminal_aux','dense_shape','quarter_shape','transfer','identity']:
            p=self.fixture();sha=p['policy_sha256']
            if mutation=='terminal_aux':p['residual_correction']['coverage']['step3_block0']=copy.deepcopy(p['residual_correction']['coverage']['step2_block0'])
            if mutation=='dense_shape':p['coverage']['step3_block0_ffn2']['actual_weight_shape']=[5120,10368]
            if mutation=='quarter_shape':p['coverage']['step1_block0_ffn0']['actual_weight_shape']=[6912,5120]
            if mutation=='transfer':p['residual_correction']['coverage']['step0_block0']['auxiliary_parameter_H2D_bytes']=1
            if mutation=='identity':p['policy']['identity_control']=True
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):validate_runtime(p,sha)

    def test_unequal_dense_step_rows_cannot_hide_extra_main_compute(self):
        p=self.fixture()
        for name,row in p['coverage'].items():
            if name.startswith('step3_'):row['rows']*=2
        with self.assertRaises(RuntimeError):validate_runtime(p,p['policy_sha256'])


if __name__=='__main__':
    unittest.main()
