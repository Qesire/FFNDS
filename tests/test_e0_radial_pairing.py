import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_e0_radial_pairing import expected_trace, pairing_statistics
from analyze_e0_radial_template import fit_template


class PairingControls(unittest.TestCase):
    def test_template_excludes_held_source(self):
        means = {i: np.array([[float(i), 1., -2.]]) for i in range(6)}
        original = fit_template(means, 2)
        means[2] = np.full((1, 3), 1e30)
        np.testing.assert_array_equal(original, fit_template(means, 2))

    def test_exact_expectation_matches_enumeration_with_strata(self):
        matrix = np.arange(36, dtype=float).reshape(6, 6)**2
        labels = np.array([0, 0, 1, 1, 1, 2])
        permutations = [p for p in itertools.permutations(range(6)) if
                        all(labels[i] == labels[p[i]] and
                            (i != p[i] or i == 5) for i in range(6))]
        expected = np.mean([sum(matrix[i, p[i]] for i in range(6)) for p in permutations])
        actual, moved = expected_trace(matrix, labels)
        self.assertEqual(moved, 5)
        self.assertAlmostEqual(actual, expected)

    def test_common_direction_survives_derangement_but_pair_specific_does_not(self):
        # Constant shared directions have large alignment without token identity.
        x = np.tile([1., -1., 0., 0.], (4, 1))
        d = np.arange(1., 5.)[:, None] * x
        labels = np.zeros(4)
        common = pairing_statistics(x, d, labels, labels, eps=0)
        self.assertAlmostEqual(common["paired_radial_fraction"], 1.)
        self.assertAlmostEqual(common["deranged_radial_fraction"], 1.)
        # Orthogonal zero-channel-mean directions bind to the correct token.
        x = np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]])
        specific = pairing_statistics(x, x, np.zeros(2), np.zeros(2), eps=0)
        self.assertAlmostEqual(specific["paired_radial_fraction"], 1.)
        self.assertAlmostEqual(specific["deranged_radial_fraction"], 0.)

    def test_radial_matrix_matches_independent_epsilon_tangent(self):
        x = np.array([[2., 3., -1., 0.], [5., -3., 1., 2.], [1., 2., 4., -2.]])
        d = np.array([[.3, -.2, .1, .4], [.1, .2, .1, -.1], [.1, -.3, .4, .2]])
        eps = .2
        px, pd = x-x.mean(axis=1, keepdims=True), d-d.mean(axis=1, keepdims=True)
        tangent = pd-px*np.sum(px*pd, axis=1, keepdims=True)/(
            np.sum(px*px, axis=1, keepdims=True)+x.shape[1]*eps)
        expected = (np.sum(pd*pd)-np.sum(tangent*tangent))/np.sum(d*d)
        actual = pairing_statistics(x, d, np.zeros(3), np.zeros(3), eps)
        self.assertAlmostEqual(actual["paired_radial_fraction"], expected, places=14)
        self.assertAlmostEqual(actual["all_permutations_radial_fraction"],
                               actual["paired_radial_fraction"]/3 +
                               2*actual["deranged_radial_fraction"]/3, places=14)

    def test_opposing_common_mode_hides_positive_pair_specific_relation(self):
        # Three position-centered directions form a regular simplex. A large
        # opposite common mode reverses the raw squared-alignment comparison.
        root = np.sqrt(3.)/2
        variation = np.array([[1., -1., 0., 0., 0., 0.],
                              [-.5, .5, root, -root, 0., 0.],
                              [-.5, .5, -root, root, 0., 0.]])
        common = np.array([0., 0., 0., 0., 3., -3.])
        labels = np.zeros(3)
        raw = pairing_statistics(common+variation, -common+variation,
                                 labels, labels, eps=0)
        centered = pairing_statistics(variation, variation, labels, labels, eps=0)
        self.assertLess(raw["paired_minus_deranged_radial_fraction"], 0.)
        self.assertGreater(centered["paired_minus_deranged_radial_fraction"], 0.)
        self.assertAlmostEqual(centered["paired_radial_fraction"], 1.)
        self.assertAlmostEqual(centered["deranged_radial_fraction"], .25)


if __name__ == "__main__":
    unittest.main()
