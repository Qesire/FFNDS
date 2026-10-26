import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_e0_flow_spectrum import spectral_panels


class SpectrumTests(unittest.TestCase):
    def test_non_cpu_and_complex_operands_are_rejected(self):
        import torch
        p = torch.ones(1, 2, 3, 4)
        with self.assertRaises(ValueError):
            spectral_panels(p, torch.empty_like(p, device='meta'))
        with self.assertRaises(ValueError):
            spectral_panels(p.to(torch.complex128), p.to(torch.complex128))

    def test_full_band_keeps_signed_cancellation_and_parseval(self):
        import torch
        p = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4)
        result = spectral_panels(p, -.5*p, temporal=(1,), spatial=(2,))
        band = result['bands'][0]
        self.assertAlmostEqual(band['next_SSE'], float((.5*p).square().sum()), places=9)
        self.assertLess(band['signed_cross'], 0)
        self.assertEqual(band['real_subspace_dimension'], 24)

    def test_constant_and_single_spatial_mode_separate(self):
        import torch
        x = torch.arange(8, dtype=torch.float64)
        p = torch.ones(1, 3, 4, 8, dtype=torch.float64)
        n = torch.cos(2*torch.pi*x/8).expand_as(p)
        result = spectral_panels(p, n, temporal=(0,), spatial=(0, 1))
        self.assertAlmostEqual(result['bands'][0]['N_SSE'], 0, places=9)
        self.assertAlmostEqual(result['bands'][1]['N_SSE'], float(n.square().sum()), places=9)
        self.assertEqual(result['bands'][0]['real_subspace_dimension'], 1)


if __name__ == '__main__':
    unittest.main()
