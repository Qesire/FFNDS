import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from e0_intrinsic_math import (group_values, iid_nonempty_group_probability,
                               source_fixed_score, support_tile_statistics)
from e0_intrinsic_temporal_support import innovation_support


class IntrinsicMechanismChecks(unittest.TestCase):
    def test_full_group_layout_reconstructs_dense_projection_with_signed_gate(self):
        generator=torch.Generator().manual_seed(26)
        q=torch.randn((3,128),generator=generator,dtype=torch.float64)
        w=torch.randn((4,128),generator=generator,dtype=torch.float64)
        gate=torch.tensor([[1.,-2.,.3,0.]],dtype=torch.float64)
        groups=group_values(q,w,gate)
        torch.testing.assert_close(groups.sum(1),(q@w.T)*gate,atol=1e-13,rtol=1e-13)

    def test_signed_values_can_cancel_despite_large_coefficients(self):
        q=torch.tensor([[1.,1.]],dtype=torch.float64)
        w=torch.tensor([[2.,-2.],[-1.,1.]],dtype=torch.float64)
        gate=torch.ones((1,2),dtype=torch.float64)
        c=group_values(q,w,gate,width=1)
        self.assertGreater(float((c*c).sum()),0.)
        torch.testing.assert_close(c.sum(1),torch.zeros((1,2),dtype=torch.float64))

    def test_tile_occupancy_distinguishes_concentration_from_scattering(self):
        concentrated=torch.zeros((3,128));concentrated[:,:64]=1.
        scattered=concentrated.clone();scattered[1]=torch.roll(scattered[1],64)
        a=support_tile_statistics(concentrated,[.5],row_tiles=(1,16))['fractions']['0.5']
        b=support_tile_statistics(scattered,[.5],row_tiles=(1,16))['fractions']['0.5']
        self.assertEqual(a['retained_nonzero_element_fraction'],b['retained_nonzero_element_fraction'])
        self.assertEqual(a['row_tiles']['16']['retained_dense_tile_work_fraction'],.5)
        self.assertEqual(b['row_tiles']['16']['retained_dense_tile_work_fraction'],1.)

    def test_whole_source_prior_cannot_read_held_labels(self):
        x={i:torch.tensor([[float(i+1),1.]]) for i in range(6)}
        before=source_fixed_score(x,2)
        x[2]=torch.tensor([[1e9,0.]])
        torch.testing.assert_close(before,source_fixed_score(x,2))

    def test_fixed_count_comparator_has_correct_small_case(self):
        # One of four channels, group width two: one row is cold with p=1/2.
        self.assertAlmostEqual(iid_nonempty_group_probability(4,1,2,3),.875)

    def test_streamed_innovation_matches_full_difference_with_ragged_row_tile(self):
        generator=torch.Generator().manual_seed(25)
        a=torch.randn(131,128,generator=generator).bfloat16()
        b=torch.randn(131,128,generator=generator).bfloat16()
        expected=support_tile_statistics(a.float()-b.float(),[.25,.5],row_tiles=(1,16,64,128))
        actual=innovation_support(a,b,[.25,.5],row_tiles=(1,16,64,128))
        self.assertEqual(actual['fractions'],expected['fractions'])

    def test_gelu_antipodal_identity_does_not_imply_rotation_equivariance(self):
        from torch.nn.functional import gelu
        x=torch.linspace(-6.,6.,257,dtype=torch.float64)
        for approximation in ['none','tanh']:
            positive=gelu(x,approximate=approximation)
            negative=gelu(-x,approximate=approximation)
            torch.testing.assert_close(positive-negative,x,atol=1e-14,rtol=1e-14)
            # Even the orthogonal transformation R=-I does not commute.
            self.assertFalse(torch.allclose(negative,-positive))


if __name__=='__main__':unittest.main()
