import unittest
import numpy as np
from tools import b_terminal_head_math as m

class TestHeadMath(unittest.TestCase):
 def test_unit_l2_gives_requested_row_rms(self):
  u=np.arange(1.,9.); d=np.ones(8)/np.sqrt(8)
  for a in (1/16,1/4):
   delta=m.nominal_delta(u,d,a)
   self.assertAlmostEqual(np.linalg.norm(delta)/np.linalg.norm(u),a)
   self.assertAlmostEqual(np.sqrt(np.mean(delta**2))/np.sqrt(np.mean(u**2)),a)
 def test_finite_matches_direct_and_jacobian(self):
  rng=np.random.default_rng(4);u=rng.normal(size=9);t=rng.normal(size=(3,9));d=rng.normal(size=9)*.2
  got=m.finite_and_linear(u,d,t)
  def head(x):
   z=x-x.mean();return t@z/np.sqrt(np.mean(z*z)+1e-6)
  np.testing.assert_allclose(got['finite'],head(u+d)-head(u),atol=1e-12)
  tiny=d*1e-5
  np.testing.assert_allclose(m.finite_and_linear(u,tiny,t)['linear'],head(u+tiny)-head(u),atol=1e-10)
 def test_null_is_first_order_invisible_but_finite_not_zero(self):
  rng=np.random.default_rng(9);w=rng.normal(size=(2,10));u=rng.normal(size=10);geo=m.head_geometry(w,np.zeros(10));dirs=m.directions(u,geo)
  delta=m.nominal_delta(u,dirs['null'],1/4);out=m.finite_and_linear(u,delta,geo['T'])
  self.assertLess(np.linalg.norm(out['linear']),1e-12);self.assertGreater(np.linalg.norm(out['finite']),1e-6)
 def test_degenerate_head_and_visible_rejected(self):
  with self.assertRaises(ValueError):m.head_geometry(np.zeros((2,8)),np.zeros(8))
  rng=np.random.default_rng(12);w=rng.normal(size=(2,8));g=m.head_geometry(w,np.zeros(8));u=np.ones(8)
  self.assertIsNone(m.directions(u,g)['norm'])
  with self.assertRaises(ValueError):m.nominal_delta(u,np.ones(8),1/4)
if __name__=='__main__':unittest.main()
