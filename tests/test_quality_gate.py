import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_vbench3 import metric_gate


class QualityGateTest(unittest.TestCase):
    def test_boundary_and_adjacent_failure_do_not_get_rounded_to_same_result(self):
        self.assertTrue(metric_gate(.63,.98*.63)['pass'])
        self.assertFalse(metric_gate(.63,math.nextafter(.98*.63,0.))['pass'])
        self.assertTrue(metric_gate(.63,.64)['pass'])

    def test_invalid_baseline_and_nonfinite_score_are_rejected(self):
        for dense,candidate in [(0.,.6),(-1.,.6),(.6,float('nan')),(float('inf'),.6)]:
            with self.assertRaises(ValueError):metric_gate(dense,candidate)


if __name__=='__main__':unittest.main()
