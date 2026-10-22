"""Protect scientific scope, negative evidence and non-GEMM cost accounting."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_r15_artifacts import validate_cpu,validate_backend
from check_r15_rollout import validate as validate_rollout


class R15Records(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'research/20261023'
        cls.cpu=json.loads((base/'R15_mean_contrast_CPU_results.json').read_text())
        cls.cp=json.loads((base/'R15_mean_contrast_CPU_protocol.json').read_text())

    def test_original_CPU_data_and_quantizer_records(self):
        self.assertEqual(validate_cpu(self.cpu,self.cp),[])

    def test_cannot_hide_intrinsically_bad_smooth_groups(self):
        bad=copy.deepcopy(self.cpu)
        bad['saved_scopes'][0]['groups_smooth_floor_above_native_UB1']=0
        self.assertTrue(validate_cpu(bad,self.cp))

    def test_cannot_promote_smooth_bound_through_original_quantizer(self):
        bad=copy.deepcopy(self.cpu)
        bad['quantizer_counterexample']['rows'][-1]['original_HiF4_mean_absolute_error']=0.
        self.assertTrue(validate_cpu(bad,self.cp))

    def test_cannot_omit_all_four_hidden_nonlinear_rows(self):
        bad=copy.deepcopy(self.cpu)
        bad['MAC_ledger']['GELU_Q2_rows']=11875
        self.assertTrue(validate_cpu(bad,self.cp))

    def test_CPU_ledger_cannot_authorize_original_model(self):
        bad=copy.deepcopy(self.cpu)
        bad['native_registration']='new-name'
        self.assertTrue(validate_cpu(bad,self.cp))


class PairedRolloutEvidence(unittest.TestCase):
    def setUp(self):
        base=ROOT/'research/20261023'
        self.records=[json.loads((base/name).read_text()) for name in (
            'R15V0_native_rollout_results.json','R15V0_budget_debit.json',
            'R15V0_video_inspection.json','R15V0_completed_resources.json')]

    def test_actual_pair_reproduces(self):
        self.assertEqual(validate_rollout(*self.records),[])

    def test_different_initial_noise_cannot_be_paired(self):
        self.records[0]['requests'][1]['initial_noise_sha256']='0'*64
        self.assertIn('Paired initial noise differs',validate_rollout(*self.records))

    def test_dense_resets_do_not_count_as_free_video(self):
        self.records[0]['intermediate_latent_reset_to_DQ']=True
        self.assertTrue(validate_rollout(*self.records))

    def test_video_without_actual_candidate_path_is_insufficient(self):
        self.records[0]['requests'][1]['candidate_events']=[]
        self.assertTrue(validate_rollout(*self.records))

    def test_cold_warm_wall_times_cannot_become_native_speed(self):
        self.records[0]['requests'][1]['performance_benchmark']=True
        self.assertTrue(validate_rollout(*self.records))


if __name__=='__main__':unittest.main()
