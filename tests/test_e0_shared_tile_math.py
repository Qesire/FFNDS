import random
import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_contiguous_capture import contiguous_plan
from e0_shared_tile_math import block_means, shared_mask, selector_scores
from run_e0_shared_closure import closure_fraction


class SharedTileTests(unittest.TestCase):
    def test_original_group_closure_changes_with_layout(self):
        mask=torch.zeros((2,128),dtype=torch.bool)
        mask[0,0]=True;mask[0,64]=True;mask[1,:64]=True
        self.assertTrue(torch.equal(closure_fraction(mask,64),
                                   torch.tensor([1.,.5],dtype=torch.float64)))

    def test_native_rows_preserve_control_and_rng(self):
        from ffnds_observation_common import row_plan
        state=random.getstate();p=contiguous_plan([21,29,52],'x')
        self.assertEqual(state,random.getstate())
        self.assertEqual(p['uniform_rows'],row_plan([21,29,52],'x')['uniform_rows'])
        self.assertEqual(len(set(sum(p['contiguous_tiles'],[]))),384)
        for t in p['contiguous_tiles']:
            self.assertEqual(t,list(range(t[0],t[0]+128)))
            self.assertEqual(t[0]%128,0)

    def test_shared_budget_and_within_tile_identity(self):
        score=torch.arange(15.).reshape(3,5)
        mask=shared_mask(score,4,2)
        self.assertTrue(torch.equal(mask.sum(-1),torch.full((12,),2)))
        self.assertTrue(torch.equal(mask,mask.reshape(3,4,5)[:,0].repeat_interleave(4,0)))

    def test_mean_blind_to_changed_nonlinear_spread(self):
        previous=torch.tensor([[-1.],[1.]],dtype=torch.float64)
        current=2*previous
        self.assertEqual(float((block_means(current,2)-block_means(previous,2)).abs().sum()),0.)
        self.assertGreater(float((block_means(F.gelu(current,approximate='tanh'),2)-
                                  block_means(F.gelu(previous,approximate='tanh'),2)).abs().sum()),.5)

    def test_preW1_selectors_independent_of_current_hidden_labels(self):
        a=torch.arange(32.,dtype=torch.float64).reshape(8,4)/20
        args=(a,a/2,torch.ones(4),torch.arange(1.,5.),4,2)
        x=selector_scores(torch.zeros(16,4),*args)
        y=selector_scores(torch.ones(16,4)*31,*args)
        for k in x:
            if k.startswith('preW1_'):
                self.assertTrue(torch.equal(x[k],y[k]),k)


if __name__=='__main__':unittest.main()
