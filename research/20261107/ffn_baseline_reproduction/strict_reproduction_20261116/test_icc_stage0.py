"""Non-experiment unit checks for ICC state and source statistics."""
import unittest
import torch
import torch.nn.functional as F
from icc_stage0 import Factors, FourState, SourceStatistics

class ICCStage0Test(unittest.TestCase):
    def test_four_state_anchor_and_identity(self):
        torch.manual_seed(5)
        w1 = torch.randn(5, 3, dtype=torch.float64)
        w2 = torch.randn(3, 5, dtype=torch.float64)
        b1, b2 = torch.randn(5, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
        f1 = Factors(w1, torch.eye(3, dtype=torch.float64))
        f2 = Factors(w2, torch.eye(5, dtype=torch.float64))
        u = torch.randn(1, 4, 3, dtype=torch.float64)
        z = F.linear(u, w1, b1); h = F.gelu(z, approximate='tanh'); y = F.linear(h, w2, b2)
        state = FourState(f1, f2); key = ('request', 'H', 19)
        state.anchor(key, u, z, h, y)
        torch.testing.assert_close(state.hit(key, u), y, rtol=0, atol=0)
        for delta in (0.1, -0.2):
            v = u + delta
            expected = F.linear(F.gelu(F.linear(v, w1, b1), approximate='tanh'), w2, b2)
            torch.testing.assert_close(state.hit(key, v), expected, rtol=1e-12, atol=1e-12)
        with self.assertRaises(RuntimeError): state.hit(('request', 'L', 19), u)
        state.reset()
        with self.assertRaises(RuntimeError): state.hit(key, u)

    def test_source_statistics(self):
        s = SourceStatistics('CA')
        _, a = s.add('a', torch.tensor([[[2., 3., 4.], [2., 3., 4.]]]))
        _, b = s.add('b', torch.ones(1, 1, 3))
        self.assertTrue(torch.equal(b, torch.ones_like(b)))
        mean, root, cap = s.scales()
        torch.testing.assert_close(mean, (a + b) / 2)
        self.assertTrue(bool(torch.all(cap >= root.max() / 1000)))
        with self.assertRaises(ValueError): s.add('a', torch.ones(1, 1, 3))
        d = SourceStatistics('CD')
        _, zero = d.add('a', torch.ones(1, 2, 3), torch.ones(1, 2, 3))
        self.assertTrue(torch.equal(zero, torch.ones_like(zero)))

if __name__ == '__main__': unittest.main()
