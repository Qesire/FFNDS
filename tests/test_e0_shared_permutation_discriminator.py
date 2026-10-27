import unittest

from tools.e0_shared_permutation_discriminator import (
    CASES, collapse, histogram_prefix_support, linear_check, nonlinear_checks, support,
)


class SharedPermutationTests(unittest.TestCase):
    def test_duplicate_signed_function_has_zero_range(self):
        r = linear_check(CASES[0])
        self.assertEqual(r['marginal_output_box'], [-4, 4])
        self.assertEqual(r['joint_linear_exact_range'], [0, 0])

    def test_signed_and_grouped_exact_oracles(self):
        for case, count in zip(CASES, (2, 2, 24, 576), strict=True):
            with self.subTest(case=case['name']):
                r = linear_check(case)
                self.assertEqual(r['input_permutations'], count)
                self.assertEqual(r['joint_linear_exact_range'], r['sorted_support_range'])

    def test_histogram_prefix_query_equals_sorted_matching(self):
        x = [-2, -2, 0, 1, 1, 2]
        v = [-3, -2, 0, 2, 4, 7]
        hist = [(-2, 2), (0, 1), (1, 2), (2, 1)]
        lo, hi = support(x, v, [[0, 6]])
        self.assertEqual(histogram_prefix_support(hist, v), hi)
        self.assertEqual(histogram_prefix_support(hist, v[::-1]), lo)
        with self.assertRaises(ValueError):
            histogram_prefix_support([(-2, 1)], v)

    def test_linear_support_is_not_a_nonlinear_certificate(self):
        r = nonlinear_checks()
        self.assertEqual(r['row_histogram_nonidentifiability']['output_ranges_with_c_ones'],
                         [(-4, 4), (0, 0)])
        self.assertGreater(min(r['antipodal_gelu']['discrete_nonlinear_values']), 1)
        self.assertEqual(r['negative_tail_extreme_reversal']['nonlinear_maximizer'], -1)


if __name__ == '__main__':
    unittest.main()
