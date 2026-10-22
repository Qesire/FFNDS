"""Test the identities that distinguish copying from a shared mean response."""
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_spatial_sharing_limits import common_output_summary
from audit_anchored_affine_remainder import stripe_reference_error
from probe_pooled_nonlinear_moments import gaussian_gelu_mean, projection_cost
from prepare_complete_stripe_operand_draft import select_stripes, stripe_rows


class SpatialSharingLimitTests(unittest.TestCase):
    def test_arbitrary_common_output_has_irreducible_contrast(self):
        a, b, c = [1., -4., 3.], [-2., 2., 5.], [7., 1., -3.]
        left = sum((x-z)**2 + (y-z)**2 for x, y, z in zip(a, b, c))
        right = sum(.5*(x-y)**2 + 2*(z-(x+y)/2)**2 for x, y, z in zip(a, b, c))
        self.assertEqual(left, right)
        # Even a nonrepresentable continuous mean is a valid lower bound.
        self.assertEqual(sum((x-(x+y)/2)**2 + (y-(x+y)/2)**2 for x, y in zip(a, b)),
                         sum(.5*(x-y)**2 for x, y in zip(a, b)))

    def test_anchor_copy_is_twice_best_common_output_pair_error(self):
        a, b = [1., 2.], [-3., 6.]
        anchor = sum((x-y)**2 for x, y in zip(a, b))
        mean = sum((x-(x+y)/2)**2 + (y-(x+y)/2)**2 for x, y in zip(a, b))
        self.assertEqual(anchor, 2*mean)

    def test_random_anchor_stripe_identity_and_pair_inclusion(self):
        values = [1., 3., -2., 4.]
        truth = sum((x-sum(values)/4)**2 for x in values)
        random_anchor = sum(sum((x-a)**2 for x in values) for a in values)/4
        self.assertEqual(random_anchor, 2*truth)
        # Exhaust every size-two sample. This detects incorrect factors from
        # using directed pairs, independent sampling, or all-pair risk.
        estimates = []
        for ids in itertools.combinations(range(4), 2):
            directed_sum = sum((values[i]-values[j])**2 for i in ids for j in ids if i != j)
            estimates.append(stripe_reference_error(directed_sum, 2, 4))
        self.assertAlmostEqual(sum(estimates)/len(estimates), random_anchor)
        with self.assertRaises(ValueError):
            stripe_reference_error(1., 1, 4)

    def test_finite_conditional_bound_uses_two_row_baseline_once(self):
        result = common_output_summary([
            {'hidden_pruning_pair_SSE': 10., 'row_sharing_pair_SSE': 4.},
            {'hidden_pruning_pair_SSE': 10., 'row_sharing_pair_SSE': 60.}])
        self.assertEqual(result['arbitrary_common_output_lower_SSE'], 32.)
        self.assertAlmostEqual(result['ideal_pairwise_choice_upper_gain_over_UB1'], .4)

    def test_gaussian_mean_reduces_to_point_response_and_known_zero_mean(self):
        for mean in (-3., -1., 0., 2.):
            self.assertAlmostEqual(gaussian_gelu_mean(mean, 0), mean*.5*(1+math.erf(mean/math.sqrt(2))))
        self.assertAlmostEqual(gaussian_gelu_mean(0, 1), 1/math.sqrt(4*math.pi))
        with self.assertRaises(ValueError):
            gaussian_gelu_mean(0., -1.)

    def test_sign_contrast_preserves_joint_covariance(self):
        points = [(1., 4.), (3., -2.), (-1., 2.), (-3., -4.)]
        original_cross = sum(x*y for x, y in points)/4
        sketches = [(sum(s[i]*points[i][0] for i in range(4))/2,
                     sum(s[i]*points[i][1] for i in range(4))/2)
                    for s in itertools.product((-1., 1.), repeat=4)]
        self.assertEqual(sum(x*y for x, y in sketches)/16, original_cross)

    def test_budget_charges_extra_first_projection_and_odd_count(self):
        cost = projection_cost()
        self.assertLess(cost['total'], .5)
        self.assertAlmostEqual(cost['total'], 0.497677290780739)
        self.assertAlmostEqual(projection_cost(two_point=7917)['total'], 0.5601851851851851)
        self.assertGreater(projection_cost(two_point=7917)['total'], .5)
        self.assertAlmostEqual(projection_cost(two_point=0)['total'], .4351851851851852)

    def test_complete_stripe_sampling_really_contains_four_native_neighbours(self):
        groups, rows = select_stripes(0)
        self.assertEqual(len(groups), 64)
        self.assertEqual(len(rows), 256)
        self.assertEqual(len(set(rows)), 256)
        for group in groups:
            native = stripe_rows(group)
            self.assertTrue(set(native).issubset(rows))
            self.assertEqual(len({row//52 for row in native}), 1)
            self.assertEqual(native[-1]-native[0], 3)
        self.assertEqual(select_stripes(0), select_stripes(0))
        self.assertNotEqual(select_stripes(0), select_stripes(1))

    def test_complete_stripes_reject_implicit_boundary_padding(self):
        self.assertEqual(stripe_rows(7916), [31664,31665,31666,31667])
        with self.assertRaises(ValueError):
            stripe_rows(7917)
        with self.assertRaises(ValueError):
            stripe_rows(0, (21,29,51))

    def test_three_directions_recover_four_centered_affine_outputs(self):
        points = [(1.,3.), (4.,-2.), (-3.,2.), (2.,1.)]
        signs = ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))
        directions = [tuple(sum(signs[i][j]*points[i][d] for i in range(4))/2 for d in range(2)) for j in range(3)]
        # Non-diagonal affine map: its bias must disappear after centering.
        linear = lambda x: (2*x[0]+3*x[1], -x[0]+4*x[1])
        affine = [tuple(z+17 for z in linear(x)) for x in points]
        projected = [linear(x) for x in directions]
        mean = tuple(sum(row[d] for row in affine)/4 for d in range(2))
        for i in range(4):
            reconstructed = tuple(sum(signs[i][j]*projected[j][d] for j in range(3))/2 for d in range(2))
            self.assertEqual(reconstructed, tuple(affine[i][d]-mean[d] for d in range(2)))


if __name__ == '__main__':
    unittest.main()
