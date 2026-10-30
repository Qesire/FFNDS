"""Synthetic replay recipe fixtures; fake quantizer is never native evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('a5_replay', ROOT / 'tools/a5_hwg_e0_native_replay_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NativeRecipeFixtures(unittest.TestCase):
    def test_explicit_native_boundaries_and_bias_once(self):
        v1 = torch.tensor([[1., -2.], [3., 1.]], dtype=torch.bfloat16)
        w1 = torch.tensor([[1., 2.], [2., -1.]], dtype=torch.bfloat16)
        w2 = torch.eye(2, dtype=torch.bfloat16)
        b1 = torch.tensor([.5, -.5], dtype=torch.bfloat16)
        b2 = torch.tensor([2., 3.], dtype=torch.bfloat16)
        calls = []
        measure = lambda name, fn: (calls.append(name), fn())[1]
        qdq = lambda x, force_fp32: x.round().to(x.dtype)
        result = m.native_replay(v1, w1, b1, w2, b2, torch.tensor([2., 4.]), qdq, measure)
        self.assertEqual(calls, ['native_W1', 'native_GELU', 'native_smooth2', 'native_Q2', 'native_W2'])
        self.assertTrue(torch.equal(result['V2'], result['Q2_input'].round()))
        self.assertTrue(torch.equal(result['Y'], result['V2'] + b2))
        sampled = {k: v[1:].clone() for k, v in result.items()}
        self.assertTrue(all(m.verify_actual_samples(result, sampled, torch.tensor([1])).values()))
        sampled['V2'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'native boundary mismatch'):
            m.verify_actual_samples(result, sampled, torch.tensor([1]))

    def test_full_factorial_counts_and_diagnostic_identity(self):
        arms = m.arm_specs()
        self.assertEqual(len(arms), 58)
        self.assertEqual(sum(a['ranker'] != 'oldY' for a in arms), 57)
        self.assertEqual(sum(a['shuffled'] for a in arms), 12)
        self.assertEqual(sum(a['hidden_group'] in (16, 32) for a in arms), 16)
        self.assertEqual(sum(a['row_group'] == 1 and a['hidden_group'] == 1 for a in arms), 4)

    def test_mean_input_cancellation_and_unpadded_tail(self):
        old = torch.zeros(65, 64, dtype=torch.bfloat16)
        new = old.clone(); new[:32, 0] = 1; new[32:64, 0] = -1; new[64, 0] = 2
        weights = torch.eye(64, dtype=torch.bfloat16)
        score = m.mean_m64_score(old, new, weights, torch.zeros(64, dtype=torch.bfloat16),
                                torch.ones(64), torch.ones(64), lambda name, fn: fn())
        self.assertTrue(torch.equal(score[:64], torch.zeros(64, 64)))
        self.assertGreater(float(score[-1, 0]), 1.)
        self.assertTrue((new[:64, 0].abs() > 0).all())


if __name__ == '__main__': unittest.main()
