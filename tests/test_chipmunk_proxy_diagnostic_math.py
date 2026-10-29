import unittest
import numpy as np
from tools.chipmunk_proxy_diagnostic_math import diagnose, stable_topk

class DiagnosticTests(unittest.TestCase):
    def test_cross_terms_and_oracles(self):
        h=np.array([[1.,1.,0.],[1.,-1.,3.]])
        d=diagnose(h,[[1.,3.,1.]],[1.],[2])
        self.assertEqual(d['arms']['proxy']['omitted_projected_energy'],20)
        self.assertEqual(d['arms']['proxy']['omitted_diagonal_energy'],20)
        self.assertEqual(d['arms']['activation_oracle']['selected'],[2])
        self.assertEqual(d['arms']['consumer_oracle']['selected'],[1])
    def test_signed_cross_and_stable_ties(self):
        d=diagnose([[1.,1.]],[[1.,1.]],[1.],[])
        self.assertEqual(d['arms']['proxy']['signed_cross_energy'],2)
        self.assertEqual(stable_topk([2.,2.,1.],1).tolist(),[0])
        with self.assertRaises(ValueError): diagnose([[1.,2.]],[[1.,1.]],[1.],[0,0])
if __name__=='__main__': unittest.main()

class ShapeTests(unittest.TestCase):
    def test_invalid_operand_and_nonfinite(self):
        with self.assertRaises(ValueError): diagnose([[1.,2.]], [[1.]], [1.], [0])
        with self.assertRaises(ValueError): diagnose([[float('nan')]], [[1.]], [1.], [0])
    def test_opposing_contributions_cancel(self):
        d=diagnose([[1.,1.]],[[1.,-1.]],[1.],[])
        self.assertEqual(d['arms']['proxy']['omitted_projected_energy'],0)
        self.assertEqual(d['arms']['proxy']['signed_cross_energy'],-2)
