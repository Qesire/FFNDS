import copy,importlib.util,math,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import chipmunk_paid_proxy_selection as sel
spec=importlib.util.spec_from_file_location('paid_chip',ROOT/'tools/e0_chipmunk_paid_proxy_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_sample_cutoff_ties_rng_and_padding_are_explicit(self):
        scores=list(range(1024))+[0]*1024
        x=sel.selector(scores,seed=9,random_rate=0,align=256)
        self.assertEqual(x['cutoff'],716);self.assertEqual(x['base_count'],308)
        self.assertEqual(x['K'],512);self.assertEqual(x['padding_count'],204)
        self.assertEqual(x['indices'],list(range(204))+list(range(716,1024)))
        self.assertEqual(len(set(x['indices'])),x['K'])
        tie=sel.selector([1.0]*1024,random_rate=0);self.assertEqual(tie['K'],1024);self.assertEqual(tie['threshold_tie_count'],1024)
        changed=scores[:1024]+[999999]*1024
        self.assertEqual(sel.selector(changed,random_rate=0)['cutoff'],716)
        a=sel.selector(scores,seed=9);self.assertEqual(a,sel.selector(scores,seed=9))
        self.assertEqual(a['rng'],'PYTHON_MT19937_FIXED_SEED_NOT_AUTHOR_CURAND')
    def test_degenerate_zero_and_full_no_seed_change(self):
        zeros=sel.selector([0.0]*1024,quantile=1);full=sel.selector([0.0]*1024,quantile=0)
        self.assertEqual(zeros['K'],0);self.assertEqual(full['K'],1024)
        for k in (0,1024):
            z=sel.masks(k,1024)
            self.assertTrue(all(len(z['random_'+str(seed)])==k for seed in sel.RANDOM_SEEDS))
            self.assertEqual(len(z['fixed']),k)
    def test_nan_duplicate_and_oob_rejected(self):
        with self.assertRaises(ValueError):sel.selector([float('nan')]+[1]*1023)
        for x in ([1,1],[-1],[1024],[True]):
            with self.assertRaises(ValueError):sel.validate_indices(x,1024)
    def test_exact_equal_k_masks_and_hashes(self):
        k=256;sets=sel.masks(k,m.H);sets['proxy']=list(range(k));records={name:{'count':len(v),'indices':v,'sha256':sel.digest(v)} for name,v in sets.items()}
        self.assertTrue(m.masks_valid(records,k))
        bad=copy.deepcopy(records);bad['random_101']['indices'][0]=bad['random_101']['indices'][1]
        self.assertFalse(m.masks_valid(bad,k))
        bad=copy.deepcopy(records);bad['fixed']['indices'][0]=7;self.assertFalse(m.masks_valid(bad,k))
        for k in (0,m.H):
            sets=sel.masks(k,m.H);sets['proxy']=list(range(k));r={n:{'count':len(v),'indices':v,'sha256':sel.digest(v)} for n,v in sets.items()}
            self.assertTrue(m.masks_valid(r,k))
    def test_partial_mathematical_counterexample(self):
        # Opposite token changes have zero group mean but each changes separately.
        delta=[2.0,-2.0];self.assertEqual(sum(delta)/len(delta),0)
        self.assertGreater(sum(x*x for x in delta),0)
        # Dropping both signed output contributions can cancel while dropping one cannot.
        contrib=[5.0,-5.0];self.assertEqual(sum(contrib),0);self.assertNotEqual(contrib[0],0)
    def test_predebit_failure_and_input_stat_mutation(self):
        report={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[]
        ledger=m.Ledger(report,lambda:saved.append(copy.deepcopy(report)))
        def fail():raise RuntimeError('synthetic')
        with self.assertRaises(RuntimeError):ledger.run('safe_selector',fail)
        self.assertEqual(saved[-1]['predebited']['safe_selector'],1);self.assertEqual(saved[-1]['inflight'],'safe_selector')
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'x';f.write_bytes(b'abc');before=m.stable_file_stat(f);f.write_bytes(b'abcd');self.assertNotEqual(before,m.stable_file_stat(f))
    def test_hash_hard_kill_bound_covers_unpersisted_checkpoints(self):
        durable=0;volatile=31*(8<<20)
        self.assertLessEqual(volatile-durable,m.durable_hash_loss_bound())
        self.assertEqual(m.durable_hash_loss_bound(),264<<20)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'blob';path.write_bytes(b'abc');identity=m.stable_file_stat(path)
            r={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[]
            persist=lambda:saved.append(copy.deepcopy(r));ledger=m.Ledger(r,persist)
            with patch.object(m,'stable_file_stat',side_effect=[identity,identity,(*identity[:2],4,*identity[3:])]):
                with self.assertRaises(ValueError):ledger.run('checkpoint_hash',lambda:m.charged_file_hash(path,r,persist,'hash'))
            self.assertEqual(saved[-1]['inflight'],'checkpoint_hash');self.assertEqual(saved[-1]['hash']['logical_bytes_read'],3)
            self.assertEqual(saved[-1]['hash']['durable_unrecorded_read_bytes_max'],264<<20)
    def test_closed_reference_handles_zero_and_rejects_undefined(self):
        z={'relative_l2':0.0};ref={'unselected_h_elementwise_equal':True,'delta_elementwise_equal':True,
          'selected_h_vs_M128_dense':None,'delta_vs_independent_M128_reference':None,
          'Y_vs_actual_delta_FP32_reference':z,'Y_vs_independent_delta_reference':z,
          'gated_Y_vs_independent_delta_reference':z}
        c={'status':'COMPLETE','selected_count':0,'operator_reference':ref};self.assertTrue(m.case_closed(c))
        bad=copy.deepcopy(c);bad['operator_reference']['Y_vs_independent_delta_reference']['relative_l2']=None
        self.assertFalse(m.case_closed(bad))
    def test_source_and_active_gate(self):
        p=m.draft();m.validate(p);self.assertEqual(len(p['selector_sources']),2)
        with self.assertRaises(ValueError):m.authorize(p,'fake')
        self.assertNotIn('torch',sys.modules)
    def test_resource_and_tail_invariants(self):
        self.assertEqual(m.tile_counts(17,33,71,35)['padded_selected'],64)
        self.assertEqual(m.tile_counts(17,33,71,0)['A_programs'],0)
        self.assertTrue(m.resource_violations(m.RESOURCES['host_peak_RSS_bytes']+1,0))
        self.assertEqual(m.DESIGN['cases'],['proxy','random_101','random_202','random_303','fixed','full','empty'])
        source=(ROOT/m.SELF).read_text();self.assertLess(source.index('(sel,scores)=timed('),source.index('current_mid,current_h,current_y='))

if __name__=='__main__':unittest.main()
