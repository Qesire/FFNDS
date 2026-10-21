"""Known-vector controls for scientific interpretations of saved flow errors."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_research_failures import describe_flow_error


class SavedFlowGeometryTests(unittest.TestCase):
    def test_pure_gain_error_is_removable_by_oracle(self):
        d=np.arange(1,17,dtype=np.float64).reshape(2,2,2,2)
        r=describe_flow_error(d,2*d)
        self.assertAlmostEqual(r['hindsight_optimal_scalar'],.5)
        self.assertAlmostEqual(r['hindsight_optimal_scalar_remaining_SSE_fraction'],0.)
        self.assertAlmostEqual(r['teacher_parallel_error_energy_fraction'],1.)

    def test_teacher_orthogonality_is_not_a_bound_on_student_rescaling(self):
        c,t,y,x=np.indices((2,2,2,2))
        d=(-1.)**(t+y+x)
        e=(-1.)**t
        r=describe_flow_error(d,d+e)
        self.assertAlmostEqual(r['teacher_parallel_error_energy_fraction'],0.)
        # Even with e perpendicular to d, shrinking d+e trades signal for error.
        self.assertAlmostEqual(r['hindsight_optimal_scalar'],.5)
        self.assertAlmostEqual(r['hindsight_optimal_scalar_remaining_SSE_fraction'],.5)

    def test_channel_constant_error_is_fully_in_bias_subspace(self):
        c,t,y,x=np.indices((2,2,2,2))
        d=(-1.)**(t+y+x)
        r=describe_flow_error(d,d+(c+1))
        self.assertAlmostEqual(r['per_channel_constant_bias_error_energy_fraction'],1.)
        for axis in ['time','height','width']:
            self.assertAlmostEqual(r['axis_differences'][axis]['error_difference_energy'],0.)


if __name__=='__main__':unittest.main()
