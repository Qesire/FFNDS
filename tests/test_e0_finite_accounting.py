import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_e0_finite_group_diagnostic import four_arm, squared_response


class NativeAccountingIdentities(unittest.TestCase):
    def test_four_arm_cross_and_curvature_are_separate(self):
        dense = np.zeros((3,4))
        a,b,ab = [dense.copy() for _ in range(3)]
        a[:,0]=1
        b[:,1]=2
        ab=a+b
        r=four_arm(dense,a,b,ab)
        self.assertEqual(r["linear_cross_term"],0)
        self.assertEqual(r["nonlinear_term"],0)
        self.assertEqual(r["AB_over_A_plus_B_SSE"],1)
        ab[:,2]=3
        r=four_arm(dense,a,b,ab)
        self.assertEqual(r["curvature_norm2"],27)
        self.assertEqual(r["nonlinear_term"],27)
        self.assertEqual(r["identity_residual"],0)

    def test_response_uses_reference_energy_and_preserves_dimensions(self):
        reference=np.ones((2,3,4))
        r=squared_response(2*reference,reference)
        self.assertEqual(r["SSE"],24)
        self.assertEqual(r["reference_energy"],24)
        self.assertEqual(r["relative_SSE"],1)


if __name__=="__main__":
    unittest.main()
