"""Discriminate direction cancellation, nonlinear interaction and repeat noise."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from cross_layer_transport import transport_accounting


class TransportAccountingTests(unittest.TestCase):
    def test_linear_opposite_directions_give_positive_interaction(self):
        f = {k: torch.tensor(v, dtype=torch.float32) for k, v in {
            'none': [2., 1.], 'A': [1., 1.], 'B': [3., 1.], 'AB': [2., 1.], 'none_repeat': [2., 1.]}.items()}
        r = transport_accounting(f, torch.zeros(2))
        self.assertEqual(r['interaction_advantage'], 2.)
        self.assertEqual(r['linear_direction_cross_term'], 2.)
        self.assertEqual(r['nonlinear_transport_term'], 0.)
        self.assertEqual(r['none_repeat_SSE'], 0.)

    def test_nonlinear_term_needed_for_interaction_identity(self):
        f = {k: torch.tensor(v, dtype=torch.float32) for k, v in {
            'none': [2., 1.], 'A': [1., 1.], 'B': [2., 0.], 'AB': [0., 0.]}.items()}
        r = transport_accounting(f, torch.zeros(2))
        self.assertEqual(r['interaction_advantage'], 1.)
        self.assertEqual(r['linear_direction_cross_term'], 0.)
        self.assertEqual(r['nonlinear_transport_term'], 1.)
        self.assertEqual(r['curvature_norm2'], 1.)

    def test_exact_repeat_noise_is_measured_separately(self):
        f = {k: torch.tensor([2.]) for k in ('none', 'A', 'B', 'AB')}
        f['none_repeat'] = torch.tensor([2.5])
        r = transport_accounting(f, torch.zeros(1))
        self.assertEqual(r['none_repeat_SSE'], .25)
        self.assertEqual(r['interaction_advantage'], 0.)

    def test_incomplete_or_nonfinite_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            transport_accounting({'none': torch.ones(2)}, torch.zeros(2))
        f = {k: torch.ones(2) for k in ('none', 'A', 'B', 'AB')}
        f['AB'][0] = float('nan')
        with self.assertRaises(ValueError):
            transport_accounting(f, torch.zeros(2))
