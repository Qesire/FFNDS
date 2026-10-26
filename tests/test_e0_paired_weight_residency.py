"""Synthetic CPU weight lifetimes; no native assets or CUDA initialization."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_paired_weight_residency import PairedWeightResidency


class ResidencyTest(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(_weight=lambda **kw: None, counters={'weight_transfers': 0})
        self.old = self.runtime._weight
        self.source = {'weight_qdq': torch.ones(2, 4).bfloat16()}

    def fetch(self, block, name):
        return self.runtime._weight(expert='high', block_index=block, module_name=f'blocks.{block}.{name}',
                                    record=self.source, device=torch.device('cpu'), dtype=torch.bfloat16)

    def test_non_ffn_retained_but_each_ffn_evicted_only_after_fence(self):
        with PairedWeightResidency(self.runtime, max_non_ffn_bytes=32, max_block_ffn_bytes=32, backend=torch) as policy:
            non_ffn = self.fetch(0, 'attn.q')
            for block in [0, 1, 0]:
                policy.begin_block('high', block)
                value = self.fetch(block, 'ffn.0')
                self.assertIs(value, self.fetch(block, 'ffn.0'))
                self.fetch(block, 'ffn.2')
                fenced = []
                policy.end_block(lambda: fenced.append(policy.snapshot()['retained_ffn_tensors']))
                self.assertEqual(fenced, [2])
                self.assertEqual(policy.snapshot()['retained_ffn_tensors'], 0)
                self.assertIs(non_ffn, self.fetch(0, 'attn.q'))
            self.assertEqual(policy.snapshot()['peak_ffn_bytes'], 32)
            self.assertEqual(policy.snapshot()['ffn_transfers'], 6)
            self.assertEqual(policy.snapshot()['non_ffn_transfers'], 1)
        self.assertIs(self.runtime._weight, self.old)

    def test_wrong_block_and_unfenced_lifetime_are_rejected(self):
        with PairedWeightResidency(self.runtime, max_non_ffn_bytes=32, max_block_ffn_bytes=32, backend=torch) as policy:
            with self.assertRaises(RuntimeError):
                self.fetch(0, 'ffn.0')
            policy.begin_block('high', 0)
            self.fetch(0, 'ffn.0')
            with self.assertRaises(RuntimeError):
                self.fetch(1, 'ffn.0')
            with self.assertRaises(RuntimeError):
                policy.begin_block('high', 1)
            with self.assertRaises(RuntimeError):
                policy.end_block(None)
            policy.end_block(lambda: None)

    def test_budget_failure_restores_original_weight_function(self):
        with self.assertRaises(RuntimeError):
            with PairedWeightResidency(self.runtime, max_non_ffn_bytes=1, max_block_ffn_bytes=1, backend=torch):
                self.fetch(0, 'attn.q')
        self.assertIs(self.runtime._weight, self.old)


if __name__ == '__main__':
    unittest.main()
