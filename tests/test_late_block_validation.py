import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from late_block_validation import validate_runtime
from late_block_rank_runtime import expected_rank


class LateBlockValidationTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[1]
        packed = json.loads((root/'reports/worker_20261020/resident_RR1_smoke_REQUEST.json').read_text())['hif4_runtime']['packed_reference']
        packed['policy'].update(kind='LATE_BLOCK_RANK_RESIDUAL_REFERENCE',identity_control=False,
            main_groups_by_step=[54,108,108,162],changed_positions=[[3,b] for b in range(30,40)],default_rank=256,changed_rank=512)
        for key,row in packed['coverage'].items():
            step = int(key.split('_')[0][4:]);width = [54,108,108,162][step]*64
            row.update(actual_weight_shape=[width,5120] if key.endswith('ffn0') else [5120,width],main_GEMM_density=width/13824)
        for step in range(4):
            for block in range(40):
                rank = expected_rank(step,block)
                row = packed['residual_correction']['coverage'][f'step{step}_block{block}']
                row.update(auxiliary_GEMM_shapes=[[rank,5120],[5120,rank]],auxiliary_MACs_over_original_dense_FFN=rank/13824,auxiliary_parameter_H2D_bytes=0)
        packed['actual_global_main_MAC_density'] = .5
        packed['residual_correction']['timing_includes_auxiliary_cost'] = True
        packed['late_block_rank'] = {'identity_control':False,'changed_positions':[{'step':3,'block':b} for b in range(30,40)],
            'mean_rank':272,'actual_row_weighted_auxiliary_MAC_density':272/13824}
        return packed

    def test_mixed_rank_trace_and_identity_boundary(self):
        packed = self.fixture()
        validate_runtime(packed,packed['policy_sha256'])
        self.assertEqual(sum(expected_rank(s,b) for s in range(4) for b in range(40))/160,272)
        packed['policy']['identity_control'] = True
        with self.assertRaises(RuntimeError):
            validate_runtime(packed,packed['policy_sha256'])

    def test_underreported_auxiliary_shapes_cost_and_policy_are_rejected(self):
        for change in ['shape','cost','main','policy','timing']:
            packed = self.fixture()
            row = packed['residual_correction']['coverage']['step3_block39']
            if change == 'shape': row['auxiliary_GEMM_shapes'] = [[256,5120],[5120,256]]
            if change == 'cost': row['auxiliary_MACs_over_original_dense_FFN'] = 256/13824
            if change == 'main': packed['coverage']['step0_block0_ffn0']['actual_weight_shape'] = [6912,5120]
            if change == 'policy': packed['policy_sha256'] = 'different'
            if change == 'timing': packed['residual_correction']['timing_includes_auxiliary_cost'] = False
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                validate_runtime(packed,self.fixture()['policy_sha256'])


if __name__ == '__main__':
    unittest.main()
