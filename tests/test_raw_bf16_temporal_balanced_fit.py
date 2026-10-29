"""Toy row-space objective identities; no real models, weights or GPU."""
import unittest
import torch

from tools import raw_bf16_temporal_balanced_fit as m


class TemporalCalibrationControls(unittest.TestCase):
    def test_identity_preserves_ordinary_target_and_conditioning_time(self):
        generator = torch.Generator().manual_seed(713)
        target = torch.randn(21, 8, 4, generator=generator, dtype=torch.float64)
        reference = target+3
        result, record = m.temporal_transform(target, reference, torch.ones(4).double(), balance=False)
        torch.testing.assert_close(result, target, rtol=1e-14, atol=1e-14)
        self.assertTrue(torch.equal(result[0], target[0]))
        self.assertEqual(record['DC_weight'], 1)
        self.assertEqual(result.reshape(168, 4).shape, (168, 4))

    def test_reference_DC_AC_have_equal_mass_without_extra_rows(self):
        t = (torch.arange(21).double()-10)[:, None, None]
        ref = (100+t).expand(21, 8, 2).clone()
        out, weights = m.temporal_transform(ref, ref, torch.ones(2).double(), balance=True)
        rest = out[1:]
        dc = 20*rest.mean(0).square().sum()
        ac = (rest-rest.mean(0)).square().sum()
        torch.testing.assert_close(dc, ac, rtol=1e-12, atol=1e-9)
        self.assertTrue(torch.equal(out[0], ref[0]))
        self.assertGreater(weights['AC_weight'], weights['DC_weight'])

    def test_transformed_covariance_is_exact_weighted_orthogonal_objective(self):
        target = torch.arange(21*8*3).double().reshape(21, 8, 3)/100
        ref = target+2
        z, w = m.temporal_transform(target, ref, torch.ones(3).double(), balance=True)
        rest = target[1:]
        mu = rest.mean(0)
        ac = (rest-mu).reshape(-1, 3)
        covariance = target[0].T@target[0]+w['DC_weight']*20*(mu.T@mu)+w['AC_weight']*(ac.T@ac)
        torch.testing.assert_close(z.reshape(-1, 3).T@z.reshape(-1, 3), covariance, rtol=1e-12, atol=1e-9)

    def test_bias_is_not_removed_from_reference_normalization(self):
        ref = torch.ones(21, 8, 2).double()*100
        target = ref-99
        z, w = m.temporal_transform(target, ref, torch.ones(2).double(), balance=True)
        self.assertEqual(w['reference_DC_energy_tau1_to20'], 20*8*2*10000)
        self.assertEqual(w['reference_AC_energy_tau1_to20'], 0)
        self.assertEqual(w['DC_weight'], 1)
        self.assertTrue(torch.isfinite(z).all())

    def test_source_selection_and_zero_reference_are_well_defined(self):
        zero = torch.zeros(21, 8, 2).double()
        z, weights = m.temporal_transform(zero, zero, torch.ones(2).double(), balance=True)
        self.assertEqual(weights['active_components'], 0)
        self.assertTrue(torch.equal(z, zero))
        with self.assertRaises(ValueError):
            m.temporal_transform(zero[:20], zero[:20], torch.ones(2).double(), balance=True)

    def test_temporal_output_metrics_detect_wrong_phase_at_equal_energy(self):
        t = (torch.arange(21).double()-10)[:, None, None]
        reference = (100+t).expand(21, 8, 2).reshape(168, 2)
        prediction = (100-t).expand(21, 8, 2).reshape(168, 2)
        v = m.temporal_metrics(reference, prediction, torch.ones(2).double())
        self.assertAlmostEqual(v['tau1_to20_AC_energy_ratio'], 1)
        self.assertAlmostEqual(v['tau1_to20_AC_cosine'], -1)
        self.assertAlmostEqual(v['tau1_to20_AC_relative_SSE'], 4)
        self.assertAlmostEqual(v['time_error_closure_signed'], 0)


if __name__ == '__main__':
    unittest.main()
