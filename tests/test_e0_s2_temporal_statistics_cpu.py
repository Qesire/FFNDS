"""Producer boundary controls, with synthetic trajectories and no saved assets."""
import unittest
import numpy as np
from tools import e0_s2_temporal_statistics_cpu as producer


class ProducerTests(unittest.TestCase):
    def test_zero_ac_keeps_missing_statistics_missing(self):
        values = producer.extract_scalars(producer.stats.summarize(np.ones((21, 8, 2)), np))
        observed = dict(zip(producer.KEYS, values))
        self.assertEqual(observed['ac_energy_fraction'], 0)
        self.assertIsNone(observed['lag1_centered'])
        self.assertIsNone(observed['rank95'])
        self.assertIsNone(observed['high_band'])

    def test_frequency_and_lag_labels_survive_producer_projection(self):
        t = np.arange(21, dtype=float)
        x = np.broadcast_to(np.sin(2*np.pi*9*t/21)[:, None, None], (21, 8, 2))
        observed = dict(zip(producer.KEYS, producer.extract_scalars(producer.stats.summarize(x, np))))
        self.assertAlmostEqual(observed['high_band'], 1)
        self.assertLess(observed['lag1_centered'], 0)
        self.assertEqual(observed['rank95'], 1)


if __name__ == '__main__':
    unittest.main()
