import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('teal_cpu_test',ROOT/'tools/e0_teal_wan_all160_cpu_local.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_scope_exact(self):
        self.assertEqual(m.LIMITS['packet_reads'],8*160)
        self.assertEqual(m.LIMITS['W2_tensor_loads'],80)
        self.assertEqual(m.LIMITS['W2_FP32_dense_matrix_calls'],160*2)
        self.assertEqual(m.LIMITS['row_method_evaluations'],2*160*256*2*3)
        self.assertIn('<=t',m.DESIGN['threshold'])
        self.assertEqual(len(set(m.SOURCES+m.CAL)),8)
    def test_histogram_inclusive_ties(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        # Small artificial BF16 bit histogram; patch dimensional constants only.
        with patch.object(m,'CAL',(1,)),patch.object(m,'D',1),patch.object(m,'RATES',(0.25,0.5)):
            hist=torch.zeros(32768,dtype=torch.int64);hist[0]=64;hist[16256]=192
            rows=m.thresholds_from_hist(hist,torch)
            self.assertEqual(rows[0]['threshold'],0.0);self.assertEqual(rows[0]['equal_count'],64)
            self.assertEqual(rows[1]['threshold'],1.0);self.assertEqual(rows[1]['equal_count'],192)
    def test_seed_unique_for_all_real_coordinates(self):
        seeds={m.random_seed(o,s,b,r) for o in m.SOURCES for s in range(4) for b in range(40) for r in range(256)}
        self.assertEqual(len(seeds),2*4*40*256)
        self.assertNotEqual(m.random_seed(894,1,0,0),m.random_seed(894,0,39,16))

    def test_arm_actual_zero_tie_and_nonzero(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        counts=m.deletion_counts(torch.tensor([0.,-1.,1.,3.]),1.,torch)
        self.assertEqual(counts,{'total':4,'zero':1,'nonzero':3,'threshold_tie':2,'finite':4})

    def test_approval_fingerprint_changes_with_scope(self):
        self.assertNotEqual(m.fingerprint({'id':'x','resources':{'wall_seconds':3600}}),
                            m.fingerprint({'id':'x','resources':{'wall_seconds':3601}}))
        self.assertEqual(m.fingerprint({'id':'x','status':'DRAFT','review':None}),
                         m.fingerprint({'id':'x','status':'FROZEN_REGISTERED','review':None}))

    def test_absent_parent_fails(self):
        with patch.object(m,'parent',side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):m.draft(Path('/tmp'))
    def test_equal_weight_means(self):
        records=[]
        for source in m.SOURCES:
            for step in range(4):
                for block in range(40):
                    rates={str(rate):{'arms':{name:{'mean_row_relative_sse':float(source)} for name in m.DESIGN['arms']}} for rate in m.RATES}
                    records.append({'source_ordinal':source,'step':step,'block':block,'rates':rates})
        result=m.source_summary(records)
        self.assertEqual(result['equal_source_mean_site_relative_sse']['0.25:layer_CDF_abs_threshold'],575)
        records.pop()
        with self.assertRaises(ValueError):m.source_summary(records)
if __name__=='__main__':unittest.main()
