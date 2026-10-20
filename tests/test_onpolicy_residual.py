import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from onpolicy_residual import native_sse, replay_rows, same_bits
from reduced_rank_residual import predict_y


class NativeGeometryReplayTest(unittest.TestCase):
    def test_selected_rows_keep_two_and_three_dimensional_strides(self):
        g = torch.Generator().manual_seed(82)
        parameters = {k: torch.randn(shape, generator=g).bfloat16() for k, shape in
            [('first', (3, 5)), ('second', (5, 3)), ('bias_first', (3,)), ('bias_second', (5,))]}
        for shape, stride in [((11, 5), (8, 1)), ((1, 11, 5), (88, 8, 1))]:
            full = torch.empty_strided(shape, stride, dtype=torch.bfloat16)
            full.copy_(torch.randn(shape, generator=g))
            ids = torch.tensor([0, 3, 10])
            x = full.reshape(-1, 5).index_select(0, ids)
            scope = {'native_V1_shape': shape, 'native_V1_stride': stride, 'full_rows': 11, 'selected_rows': ids.tolist()}
            actual = replay_rows(x, scope, parameters)
            expected = predict_y(full, parameters).reshape(-1, 5).index_select(0, ids)
            self.assertTrue(same_bits(actual, expected))
            with self.assertRaises(RuntimeError):
                replay_rows(x, {**scope, 'selected_rows': [0, 0, 10]}, parameters)

    def test_native_risk_keeps_bf16_gate_rounding_and_signed_zero(self):
        target = torch.tensor([[2., -3.]], dtype=torch.bfloat16)
        value = torch.tensor([[1., -1.]], dtype=torch.bfloat16)
        gate = torch.tensor([[.25, -2.]], dtype=torch.bfloat16)
        self.assertEqual(native_sse(value, target, gate), .0625 + 16.)
        self.assertFalse(same_bits(torch.tensor([0.], dtype=torch.bfloat16), torch.tensor([-0.], dtype=torch.bfloat16)))


if __name__ == '__main__': unittest.main()
