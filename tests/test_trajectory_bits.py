from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_dense_trajectory import same_tensor_bits


class TrajectoryBitIdentity(unittest.TestCase):
    def test_signed_zero_requires_byte_comparison_and_dtype_identity(self):
        positive, negative = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(same_tensor_bits(positive, negative))
        self.assertTrue(same_tensor_bits(negative, negative.clone()))
        self.assertFalse(same_tensor_bits(positive, positive.to(torch.int32)))


if __name__ == "__main__":
    unittest.main()
