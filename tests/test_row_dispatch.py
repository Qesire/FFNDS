"""CPU computation tests; native original HiF4 controls live in the scoped job."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
try:
    import torch
    import torch.nn.functional as F
    from ub1_row_dispatch import RowDispatch, row_partition
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch CPU tests run in the existing worker environment')
class RowDispatchTests(unittest.TestCase):
    def test_partition_budget_ties_and_actual_input_dependence(self):
        scores = torch.arange(16, dtype=torch.float32)
        selected, rest = row_partition(scores, 'dynamic_residual', 16)
        self.assertEqual(selected.tolist(), [12, 13, 14, 15])
        reversed_selected, _ = row_partition(-scores, 'dynamic_residual', 16)
        self.assertEqual(reversed_selected.tolist(), [0, 1, 2, 3])
        tied, _ = row_partition(torch.zeros(16), 'dynamic_residual', 16)
        self.assertEqual(tied.tolist(), [0, 1, 2, 3])
        for arm in ('fixed_stride', 'fixed_hash', 'dynamic_residual'):
            dense, sparse = row_partition(scores, arm, 16)
            self.assertEqual(len(dense), 4)
            self.assertEqual(sorted(dense.tolist() + sparse.tolist()), list(range(16)))

    def test_native_pointwise_gelu_compaction_and_row_scatter(self):
        torch.manual_seed(71023)
        rows, dim, hidden, rank = 16, 8, 16, 2
        rand = lambda shape: (torch.randn(shape) * .2).bfloat16()
        x, w0, b0, w2, b2 = rand((rows, dim)), rand((hidden, dim)), rand((hidden,)), rand((dim, hidden)), rand((dim,))
        smooth0, smooth2 = torch.linspace(.8, 1.2, dim), torch.linspace(.7, 1.3, hidden)
        channels = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
        residual = {'first': rand((rank, dim)), 'second': rand((dim, rank)),
                    'bias_first': rand((rank,)), 'bias_second': rand((dim,))}
        p = {'channels': channels, 'b2': rand((dim,)), 'alpha': .8, 'residual': residual}
        # A row/group-local deterministic CPU QDQ surrogate isolates buffer
        # layout and cast semantics. It is NOT a native HiF4 accuracy test.
        quant = lambda value, force_fp32: (value.float() * 4).round().div(4).bfloat16()
        rec0 = {'weight_qdq': w0, 'smooth_scale': smooth0, 'activation_qdq': True, 'force_fp32': True}
        rec2 = {'weight_qdq': w2, 'smooth_scale': smooth2, 'activation_qdq': True, 'force_fp32': True}
        runtime = SimpleNamespace(records={'high_noise_model': {'blocks.0.ffn.0': rec0, 'blocks.0.ffn.2': rec2}},
                                  counters={'linear_forward_calls': 0, 'activation_qdq_calls': 0},
                                  _weight=lambda **kwargs: kwargs['record']['weight_qdq'])
        v1 = quant((x.float() / smooth0).bfloat16(), force_fp32=True)
        r = F.linear(F.linear(v1, residual['first'], residual['bias_first']), residual['second'], residual['bias_second'])
        for arm in ('UB1', 'fixed_stride', 'fixed_hash'):
            hook = RowDispatch(runtime, {(0, 0): p}, arm, expected_shape=(rows, dim, hidden))
            hook.adapter = SimpleNamespace(native_hif4_activation_qdq=quant)
            hook.gate = torch.ones(dim)
            z = hook.apply(SimpleNamespace(_get_actual_bias=lambda: b0), x, 'high_noise_model', 0, 0, 'blocks.0.ffn.0')
            actual = hook.apply(SimpleNamespace(_get_actual_bias=lambda: b2), F.gelu(z, approximate='tanh'),
                                'high_noise_model', 0, 0, 'blocks.0.ffn.2')
            dense, sparse = row_partition(torch.zeros(rows), arm, rows)
            expected = torch.empty_like(x)
            zs = F.linear(v1[sparse], w0[channels], b0[channels])
            vs = quant((F.gelu(zs, approximate='tanh').float() / smooth2[channels]).bfloat16(), force_fp32=True)
            expected[sparse] = F.linear((vs.float() * .8).bfloat16(), w2[:, channels].contiguous(), p['b2']) + r[sparse]
            if len(dense):
                zd = F.linear(v1[dense], w0, b0)
                vd = quant((F.gelu(zd, approximate='tanh').float() / smooth2).bfloat16(), force_fp32=True)
                expected[dense] = F.linear(vd, w2, b2)
            self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)), arm)
            self.assertFalse(hook.pending)


if __name__ == '__main__':
    unittest.main()
