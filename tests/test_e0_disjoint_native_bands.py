import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_disjoint_native_bands import describe,masks


class NativeBandIdentities(unittest.TestCase):
    def test_native_dimensions_and_exhaustive_disjointness(self):
        m=masks((16,21,58,104),np)
        self.assertEqual([int(v.sum()*16) for v in m.values()],[16,4608,92480,1929648])

    def test_signed_cancellation_is_preserved_and_parseval_closes(self):
        shape=(2,3,6,8); rng=np.random.default_rng(20261026)
        zd=rng.normal(size=shape); zs=zd+rng.normal(size=shape)
        ad=rng.normal(size=shape); p=rng.normal(size=shape); n=-.5*p
        r=describe(dict(z_dense=zd,z_student=zs,z_dense_after=ad,
            z_student_after=ad+p+n,z_dense_at_student_after=ad+p),np,spatial_cutoff=1)
        self.assertTrue(all(b['signed_cross']<0 for b in r['bands']))
        self.assertAlmostEqual(r['whole']['next_delta_SSE'],.25*r['whole']['P_SSE'])

    def test_neutral_state_input_has_zero_input_and_dense_response(self):
        z=np.ones((1,3,6,8)); n=np.arange(z.size).reshape(z.shape)/z.size
        r=describe(dict(z_dense=z,z_student=z,z_dense_after=z,
            z_student_after=z+n,z_dense_at_student_after=z),np,spatial_cutoff=1)
        self.assertEqual(r['whole']['P_SSE'],0.)
        self.assertEqual(r['whole']['input_delta_SSE'],0.)
        self.assertAlmostEqual(r['whole']['next_delta_SSE'],r['whole']['N_SSE'])

    def test_large_energy_near_zero_cross_uses_natural_dot_product_scale(self):
        # Orthogonal P/N have a nearly zero global cross despite large,
        # oppositely signed band crosses. FFT roundoff must be judged against
        # ||P||*||N|| rather than the small cancelled dot product itself.
        rng=np.random.default_rng(31); shape=(2,3,6,8)
        p=rng.normal(size=shape); n=rng.normal(size=shape)
        n-=float((p*n).sum()/(p*p).sum())*p
        p*=1e4; n*=1e4; zero=np.zeros(shape)
        r=describe(dict(z_dense=zero,z_student=p,z_dense_after=zero,
            z_student_after=p+n,z_dense_at_student_after=p),np,spatial_cutoff=1)
        natural=2*np.sqrt(r['whole']['P_SSE']*r['whole']['N_SSE'])
        self.assertGreater(natural,1e9)
        self.assertLess(abs(r['whole']['signed_cross']),1e-12*natural)
        self.assertTrue(any(b['signed_cross']>0 for b in r['bands']))
        self.assertTrue(any(b['signed_cross']<0 for b in r['bands']))
        check=r['Parseval_closure']['signed_cross']
        self.assertAlmostEqual(check['natural_energy_scale'],natural)
        self.assertAlmostEqual(check['absolute_tolerance'],1e-10*natural)
        self.assertAlmostEqual(check['band_minus_time_domain'],
            sum(b['signed_cross'] for b in r['bands'])-r['whole']['signed_cross'])
        self.assertLessEqual(abs(check['band_minus_time_domain']),check['absolute_tolerance'])


if __name__=='__main__':unittest.main()
