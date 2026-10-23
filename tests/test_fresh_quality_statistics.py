from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fresh_quality_statistics import paired_sources, verify_official_aggregate


class FreshQualityStatisticsTests(unittest.TestCase):
    def test_musiq_raw_details_are_not_official_aggregate_units(self):
        raw = [.6, [{'video_path': 'a', 'video_results': 50},
                    {'video_path': 'b', 'video_results': 70}]]
        self.assertAlmostEqual(verify_official_aggregate('imaging_quality', raw, {}), .6)
        with self.assertRaises(ValueError):
            verify_official_aggregate('imaging_quality', [60, raw[-1]], {})

    def test_consistency_is_frame_pair_weighted_before_source_statistics(self):
        raw = [.25, [{'video_path': 'short', 'video_results': 1},
                     {'video_path': 'long', 'video_results': 0}]]
        self.assertEqual(verify_official_aggregate('subject_consistency', raw,
                                                  {'short': 2, 'long': 4}), .25)
        with self.assertRaises(ValueError):
            verify_official_aggregate('subject_consistency', [.5, raw[-1]],
                                     {'short': 2, 'long': 4})

    def test_identical_paired_sources_have_zero_interval(self):
        data = {0: .3, 1: .8, 2: .5}
        result = paired_sources(data, data, draws=100)
        self.assertEqual(result['bootstrap']['absolute_difference_percentile95'], [0, 0])
        self.assertEqual(result['bootstrap']['relative_change_percentile95'], [0, 0])
        self.assertEqual(result['independent_source_count'], 3)

    def test_zero_denominator_is_not_implicitly_dropped_or_epsilon_smoothed(self):
        result = paired_sources({0: 0, 1: 1}, {0: 1, 1: 1}, draws=100)
        self.assertEqual(result['relative_change_of_means'], 1)
        self.assertGreater(result['bootstrap']['undefined_zero_denominator_resamples'], 0)
        self.assertIsNone(result['bootstrap']['relative_change_percentile95'])
        self.assertIsNone(paired_sources({0: 0}, {0: 1}, draws=10)['relative_change_of_means'])

    def test_source_pairing_and_ratio_of_means_cannot_be_substituted(self):
        result = paired_sources({0: 1, 1: 100}, {0: 2, 1: 90}, draws=50)
        self.assertAlmostEqual(result['relative_change_of_means'], 92/101 - 1)
        with self.assertRaises(ValueError):
            paired_sources({0: 1}, {1: 1}, draws=50)


if __name__ == '__main__':
    unittest.main()
