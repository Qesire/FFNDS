import ast,copy,importlib.util,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('chipmunk_probe',ROOT/'tools/e0_chipmunk_a100_delta_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_index_safety_and_selection_contract(self):
        self.assertEqual(m.validate_indices([]),[])
        self.assertEqual(m.validate_indices([2,0,1],3),[2,0,1])
        for v in ([1,1],[3],[-1],[True],[1.0]):
            with self.assertRaises(ValueError):m.validate_indices(v,3)
        self.assertEqual(len(m.case_indices('fixed_first_4096')),4096)
        self.assertEqual(len(m.case_indices('full')),13824)
    def test_program_grid_and_tails(self):
        x=m.tile_counts(17,33,71,35);self.assertEqual(x['A_programs'],4);self.assertEqual(x['B_programs'],4)
        self.assertEqual(x['padded_selected'],64);self.assertEqual(x['A_k_tiles'],2);self.assertEqual(x['B_k_tiles'],2)
        self.assertEqual(m.tile_counts(17,33,71,0)['A_programs'],0)
    def test_ledger_predebit_and_uncertain_interrupted_call(self):
        report={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[]
        ledger=m.Ledger(report,lambda:saved.append(copy.deepcopy(report)))
        def fail():raise RuntimeError('synthetic')
        with self.assertRaises(RuntimeError):ledger.run('kernel_a',fail)
        self.assertEqual(saved[-1]['predebited']['kernel_a'],1);self.assertEqual(saved[-1]['completed']['kernel_a'],0)
        self.assertEqual(saved[-1]['inflight'],'kernel_a')
    def test_file_hash_persists_actual_reads_and_rejects_stat_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'input';f.write_bytes(b'abc');report={};saved=[]
            identity=m.stable_file_stat(f)
            with patch.object(m,'stable_file_stat',side_effect=[identity,identity,(*identity[:2],4,*identity[3:])]):
                with self.assertRaises(ValueError):m.charged_file_hash(f,report,lambda:saved.append(copy.deepcopy(report)),'input')
            self.assertEqual(saved[-1]['input']['logical_bytes_read'],3)
            self.assertFalse(saved[-1]['input']['completed']);self.assertEqual(saved[-1]['input']['unaccounted_inflight_read_bytes_max'],8<<20)
    def test_resource_violation_rejects_host_and_output_overflow(self):
        self.assertEqual(m.resource_violations(0,0),[])
        self.assertIn('host_peak_RSS',m.resource_violations(m.RESOURCES['host_peak_RSS_bytes']+1,0))
        self.assertIn('output_bytes_with_receipt_reserve',m.resource_violations(0,m.RESOURCES['new_HF_bytes']))
    def test_closed_case_rejects_unselected_mutation_and_bad_reference(self):
        z={'relative_l2':0.0};ref={'unselected_h_bitwise_equal':True,'delta_bitwise_equal':True,
          'selected_h_vs_M128_dense':z,'delta_vs_independent_M128_reference':z,
          'Y_vs_actual_delta_FP32_reference':z,'Y_vs_independent_delta_reference':z,
          'gated_Y_vs_independent_delta_reference':z}
        case={'status':'COMPLETE','selected_count':1,'operator_reference':ref};self.assertTrue(m.case_closed(case))
        bad=copy.deepcopy(case);bad['operator_reference']['unselected_h_bitwise_equal']=False;self.assertFalse(m.case_closed(bad))
        bad=copy.deepcopy(case);bad['operator_reference']['Y_vs_independent_delta_reference']['relative_l2']=0.2;self.assertFalse(m.case_closed(bad))
    def test_kernel_source_safety_and_two_distinct_launches(self):
        src=(ROOT/m.KERNEL).read_text();ast.parse(src)
        self.assertIn('mask=(rows[:,None]<M)&(cols[None,:]<S)',src)
        self.assertIn('mask=(out[None,:]<D)&(k[:,None]<S)',src)
        self.assertIn('(acc+bias[None,:]).to(tl.bfloat16).to(tl.float32)',src)
        self.assertIn('(new.to(tl.float32)-old.to(tl.float32)).to(tl.bfloat16)',src)
        self.assertIn('(old+acc).to(t.bfloat16)'.replace('t.bfloat16','tl.bfloat16'),src)
        self.assertNotIn('B2',src)
    def test_two_conceptual_references_and_no_proxy_claim(self):
        self.assertEqual(m.DESIGN['initializer'],['same_M128_dense','captured_fullM_native'])
        self.assertIn('NO_AUTHOR_PROXY',m.DESIGN['selection'])
        self.assertEqual(m.LIMITS['kernel_a'],4);self.assertEqual(m.LIMITS['kernel_b'],4)
    def test_scalar_delta_algebra_preserves_first_bias_only(self):
        old_h=[1.0,2.0,3.0];new_h=[1.5,-1.0,4.0];weight=[2.0,-3.0,0.5];bias=7.0
        old_y=sum(h*w for h,w in zip(old_h,weight))+bias
        for indices in ([],[0,2],[0,1,2]):
            actual=old_y+sum((new_h[j]-old_h[j])*weight[j] for j in indices)
            mixed=[new_h[j] if j in indices else old_h[j] for j in range(3)]
            self.assertAlmostEqual(actual,sum(h*w for h,w in zip(mixed,weight))+bias)
        self.assertNotEqual(old_y+(new_h[0]-old_h[0])*weight[0]+bias,
                            sum(h*w for h,w in zip([new_h[0],old_h[1],old_h[2]],weight))+bias)
    def test_unregistered_draft_is_inert(self):
        p=m.draft();m.validate(p)
        with self.assertRaises(ValueError):m.authorize(p,'fake')
        self.assertNotIn('torch',__import__('sys').modules)

if __name__=='__main__':unittest.main()
