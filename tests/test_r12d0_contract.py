"""Fixed D0 budgets and scientific stopping rules, without model libraries."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from r12d0_contract import ACTIONS, INDICES, decide, groups, jobs, require_repeats, residual_count


class D0ContractTests(unittest.TestCase):
    def test_plan_is_exactly24_disjoint_complete_cases(self):
        schedule = [jobs('controls', a) for a in ('DQ', 'UB1')]
        schedule += [jobs(p, shard=s) for p in ('baselines', 'interventions') for s in (0, 1)]
        pairs = [v for _, values in schedule for v in values]
        self.assertEqual(len(pairs), 24)
        self.assertEqual(set(pairs), {(i, a) for i in INDICES for a in ('DQ', 'UB1', 'N0', 'N0_repeat', 'NA', 'NB')})
        with self.assertRaises(ValueError):
            jobs('extended_search')

    def test_every_raw_action_has_identical50pct_main_MAC(self):
        for arm in ACTIONS:
            self.assertEqual(sum(groups(arm, s, b) for s in range(4) for b in range(40)), 17280)
            self.assertEqual(groups(arm, 3, 30) + groups(arm, 3, 31), 324)
            self.assertEqual(residual_count(arm), 158)
        self.assertEqual(residual_count('UB1'), 160)
        self.assertEqual(residual_count('DQ'), 0)

    def test_beating_N0_does_not_imply_conditional_headroom(self):
        matrix = {i: {'N0': 100., 'NA': 90., 'NB': 120.} for i in INDICES}
        result = decide(matrix, dict.fromkeys(INDICES, 95.), dict.fromkeys(INDICES, 100.))
        self.assertEqual(result['best_fixed_action'], 'NA')
        self.assertEqual(result['oracle_reduction_over_best_fixed'], 0.)
        self.assertFalse(result['diagnostic_gate_pass'])

    def test_conditional_gain_must_also_beat_frozen_UB1(self):
        matrix = {i: {'N0': 100., 'NA': 90. if j % 2 else 150., 'NB': 150. if j % 2 else 90.}
                  for j, i in enumerate(INDICES)}
        result = decide(matrix, dict.fromkeys(INDICES, 80.), dict.fromkeys(INDICES, 100.))
        self.assertTrue(result['gates']['oracle_beats_best_fixed_1pct'])
        self.assertFalse(result['gates']['oracle_beats_UB1_1pct'])
        self.assertFalse(result['diagnostic_gate_pass'])
        self.assertTrue(decide(matrix, dict.fromkeys(INDICES, 100.), dict.fromkeys(INDICES, 100.))['diagnostic_gate_pass'])

    def test_noise_is_charged_and_missing_actions_fail(self):
        matrix = {i: {'N0': 100., 'NA': 90. if j % 2 else 150., 'NB': 150. if j % 2 else 90.}
                  for j, i in enumerate(INDICES)}
        result = decide(matrix, dict.fromkeys(INDICES, 100.), dict.fromkeys(INDICES, 102.))
        self.assertFalse(result['gates']['headroom_exceeds_10eta'])
        del matrix[0]['NB']
        with self.assertRaises(ValueError):
            decide(matrix, dict.fromkeys(INDICES, 100.), dict.fromkeys(INDICES, 100.))

    def test_repeat_verification_ignores_only_file_locations(self):
        cases = {}
        for i in INDICES:
            cases[i, 'N0'] = {'steps': [{'flow_sha256': 'a', 'path': 'original', 'sha256': 'b'}], 'scopes': [{'Q1': 'c'}]}
            cases[i, 'N0_repeat'] = copy.deepcopy(cases[i, 'N0'])
            cases[i, 'N0_repeat']['steps'][0]['path'] = 'repeat'
        require_repeats(cases)
        cases[0, 'N0_repeat']['steps'][0]['flow_sha256'] = 'changed'
        with self.assertRaises(RuntimeError):
            require_repeats(cases)


if __name__ == '__main__':
    unittest.main()
