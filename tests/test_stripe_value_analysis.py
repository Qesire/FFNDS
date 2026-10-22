"""Matched physical action counts and a fixed mask cannot use evaluation labels."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_stripe_native_values import analyze


class StripeValueAnalysisTests(unittest.TestCase):
    def cases(self):
        return [{'index':i,'one_point_SSE':[10.]*64,'two_point_SSE':[2. if g<32 else 12. for g in range(64)],
                 'UB1_SSE':[8.]*64,'pre_W1_contrast_norm2':[float(64-g) for g in range(64)]} for i in (0,1,10,11)]

    def test_best_training_static_count_and_oracle_are_matched(self):
        result=analyze(self.cases())
        self.assertEqual(result['fixed_two_point_stripes'],list(range(32)))
        for row in result['per_source']:
            self.assertEqual(len(row['dynamic_two_point_stripes']),32)
            self.assertEqual(len(row['oracle_two_point_stripes']),32)
            self.assertEqual(row['oracle_SSE'],384.)

    def test_evaluation_labels_cannot_change_fixed_policy(self):
        cases=self.cases();prior=analyze(cases)
        for case in cases:
            if case['index'] in (0,10):case['two_point_SSE']=list(reversed(case['two_point_SSE']))
        changed=analyze(cases)
        self.assertEqual(prior['fixed_two_point_stripes'],changed['fixed_two_point_stripes'])
        self.assertLess(changed['evaluation_dynamic_gain_over_fixed']+1,2)
        self.assertGreater(changed['evaluation_totals']['dynamic_SSE'],changed['evaluation_totals']['oracle_SSE'])

    def test_nonfinite_missing_or_changed_scope_rejected(self):
        cases=self.cases();cases[0]['one_point_SSE'][0]=float('nan')
        with self.assertRaises(ValueError):analyze(cases)
        with self.assertRaises(ValueError):analyze(self.cases()[:3])
        with self.assertRaises(ValueError):analyze(self.cases(),count=33)
        with self.assertRaises(ValueError):analyze(self.cases()+[self.cases()[0]])


if __name__=='__main__':
    unittest.main()
