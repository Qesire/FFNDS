"""A controlled diagnostic must never silently round the injected state."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_teacher_forced_flow import error_stats, exact_native_latent


class TeacherForcedStateTests(unittest.TestCase):
    def test_native_values_including_signed_zero_restore_exactly(self):
        native = torch.tensor([0., -0., .125, -1.5, 128.], dtype=torch.bfloat16)
        restored = exact_native_latent(native.float(), torch.empty_like(native))
        self.assertTrue(torch.equal(restored.view(torch.int16), native.view(torch.int16)))

    def test_unrepresentable_state_is_rejected_without_tolerance(self):
        source = torch.tensor([1.001], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'exactly'):
            exact_native_latent(source, torch.zeros(1, dtype=torch.bfloat16))
        self.assertEqual(error_stats(torch.tensor([2.]), torch.tensor([1.]))['relative_RMSE'], 1.)


if __name__ == '__main__':
    unittest.main()
