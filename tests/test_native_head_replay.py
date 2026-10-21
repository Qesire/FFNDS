"""Native patch order, BF16 rounding boundaries and deployed parameter updates."""
import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from native_head_replay import normalized_update, patchify_flow, replay, unpatchify_flow
from verify_native_head_bytes import same_bytes


class NativeHeadReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.data = {k: torch.randn(12, 8).bfloat16() for k in ('u', 'base', 'main')}
        self.data.update({k: torch.randn(8).bfloat16() for k in ('gate', 'scale', 'shift')})
        self.data.update(head_weight=torch.randn(8, 4).bfloat16(), head_bias=torch.randn(4).bfloat16())
        self.params = {k: torch.randn(*shape).requires_grad_() for k, shape in
                       [('first', (3, 8)), ('bias_first', (3,)), ('second', (8, 3)), ('bias_second', (8,))]}

    def test_patch_order_matches_original_einsum(self):
        grid = (2, 3, 2)
        tokens = torch.arange(12 * 16).reshape(12, 16)
        native = torch.einsum('fhwpqrc->cfphqwr', tokens.view(*grid, 1, 2, 2, 4)).reshape(4, 2, 6, 4)
        self.assertTrue(torch.equal(unpatchify_flow(tokens, grid), native))
        self.assertTrue(torch.equal(patchify_flow(native, grid), tokens))
        with self.assertRaises(ValueError):
            unpatchify_flow(tokens[:-1], grid)

    def test_separate_native_inplace_rounding_matches_replay(self):
        d, p = self.data, self.params
        with torch.no_grad():
            hidden = F.linear(d['u'], p['first'].bfloat16(), p['bias_first'].bfloat16())
            y = d['main'] + F.linear(hidden, p['second'].bfloat16(), p['bias_second'].bfloat16())
            x = d['base'].clone()
            x.add_(y * d['gate'])
            x = F.layer_norm(x, (8,), eps=1e-6)
            x.mul_(1 + d['scale']).add_(d['shift'])
            native = torch.addmm(d['head_bias'], x, d['head_weight'], out=torch.empty(12, 4, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(replay(d, p), native))

    def test_connected_gradients_and_normalized_deployed_update(self):
        replay(self.data, self.params).float().square().mean().backward()
        for p in self.params.values():
            self.assertTrue(bool(torch.isfinite(p.grad).all()))
            self.assertGreater(int(torch.count_nonzero(p.grad)), 0)
        updated, report = normalized_update(self.params, .001)
        delta = sum((updated[k].double() - v.detach().double()).square().sum() for k, v in self.params.items()).sqrt()
        self.assertAlmostEqual(float(delta) / report['parameter_L2'], .001, places=7)
        self.assertGreater(sum(int(torch.count_nonzero(updated[k].bfloat16() != v.detach().bfloat16())) for k, v in self.params.items()), 0)

    def test_missing_or_nonfinite_gradient_fails_closed(self):
        with self.assertRaises(ValueError):
            normalized_update(self.params, .001)
        for p in self.params.values():
            p.grad = torch.full_like(p, float('nan'))
        with self.assertRaises(ValueError):
            normalized_update(self.params, .001)

    def test_strict_identity_detects_signed_zero(self):
        positive = torch.tensor([0.], dtype=torch.bfloat16)
        negative = torch.tensor([-0.], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(same_bytes(positive, negative))
        self.assertTrue(same_bytes(negative, negative.clone()))


if __name__ == '__main__':
    unittest.main()
