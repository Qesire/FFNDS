"""Synthetic signed-energy counterexamples for the JSON evidence join."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from summarize_e0_multisource_directional import BANDS,DIMENSIONS,bands_join


class SignedJoinTests(unittest.TestCase):
    def fixture(self):
        # The second band has beneficial N+cross despite positive N energy.
        p,n,c=[4.,9.,1.,2.],[1.,4.,1.,1.],[2.,-10.,0.,0.]
        rows=[dict(name=name,real_dimensions=d,P_SSE=a,N_SSE=b,signed_cross=x,
                   next_delta_SSE=a+b+x,input_delta_SSE=1.)
              for name,d,a,b,x in zip(BANDS.values(),DIMENSIONS,p,n,c)]
        native=dict(dense_map_response_SSE=sum(p),candidate_effect_SSE=sum(n),
                    dense_map_candidate_effect_dot=sum(c)/2,actual_next_delta_SSE=sum(p)+sum(n)+sum(c))
        return rows,native

    def test_beneficial_candidate_term_is_not_mistaken_for_total_next(self):
        rows,n=self.fixture();out=bands_join(rows,n)
        self.assertEqual(out[1]['N_SSE']+out[1]['signed_cross'],-6.)
        self.assertEqual(sum(r['next_delta_SSE'] for r in out),15.)
        self.assertEqual(n['candidate_effect_SSE']+2*n['dense_map_candidate_effect_dot'],-1.)

    def test_old_names_join_without_inventing_input_band_energy(self):
        rows,n=self.fixture()
        old=[dict(name=name,dimension=r['real_dimensions'],P_SSE=r['P_SSE'],N_SSE=r['N_SSE'],
                  signed_cross=r['signed_cross'],next_SSE=r['next_delta_SSE'])
             for name,r in zip(BANDS,rows)]
        out=bands_join(old,n,True)
        self.assertEqual([r['name'] for r in out],list(BANDS.values()))
        self.assertTrue(all(r['input_delta_SSE'] is None for r in out))

    def test_overlapping_or_mislabelled_band_inventory_is_rejected(self):
        rows,n=self.fixture();rows[1]['name']=rows[0]['name']
        with self.assertRaises(ValueError):bands_join(rows,n)

    def test_global_signed_identity_drift_is_rejected(self):
        rows,n=self.fixture();n['dense_map_candidate_effect_dot']+=.01
        with self.assertRaises(ValueError):bands_join(rows,n)


if __name__=='__main__':unittest.main()
