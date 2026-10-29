import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('teal_video_test',ROOT/'tools/e0_teal_wan_all160_video.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_finite_scope(self):
        self.assertEqual(len(m.ORDINALS),4);self.assertEqual(len(set(m.ORDINALS)),4)
        self.assertEqual(m.limits()['mask_calls'],320);self.assertEqual(m.limits()['state_tensor_D2H'],50)
        self.assertEqual(m.DESIGN['evidence'],'QUALITY_DIAGNOSTIC_NO_PHYSICAL_SAVING')
        self.assertFalse(m.DESIGN['physical_sparse_W2'])
    def test_mask_bf16_inclusive_and_zero_counts(self):
        try:import torch
        except ImportError:self.skipTest('CPU torch unavailable')
        h=torch.tensor([-2,-1,0,1,2],dtype=torch.bfloat16);actual,counts=m.mask_values(torch,h,1)
        self.assertEqual(actual.tolist(),[-2,0,0,0,2]);self.assertEqual(counts,{'total':5,'selected':3,'preexisting_zero':1,'selected_nonzero':2,'threshold_ties':2})
        h[0]=float('nan')
        with self.assertRaises(ValueError):m.mask_values(torch,h,1)
    def test_threshold_export_rejects_wrong_source_or_missing_site(self):
        rows=[]
        for step in range(4):
            for block in range(40):
                vals={str(rate):{'threshold_calibration':{'target_rate':rate,'threshold':rate,'count':6*256*13824}} for rate in m.RATES}
                first={'source_ordinal':894,'step':step,'block':block,'expert':'high' if step<2 else 'low','rates':vals}
                second=copy.deepcopy(first);second['source_ordinal']=256;rows.append({'records':[first,second]})
        self.assertEqual(len(m.threshold_rows(rows)),160)
        bad=copy.deepcopy(rows);bad[0]['records'][1]['rates']['0.25']['threshold_calibration']['threshold']=0.3
        with self.assertRaises(ValueError):m.threshold_rows(bad)
        with self.assertRaises(ValueError):m.threshold_rows(rows[:-1])
    def test_real_nested_wrapper_bypass_mask_and_duplicate(self):
        events=[]
        class Ledger:
            def __init__(self):self.pre=0;self.done=0
            def call(self,rid,cost,fn,*args):
                self.pre+=cost['mask_calls'];result=fn(*args);self.done+=cost['mask_calls'];return result
        ledger=Ledger();mm=SimpleNamespace();mm.apply=lambda x:(events.append(('original_guard_dense_W2_with_bias',x)) or 'result')
        active=SimpleNamespace(block=(0,3),step=lambda e:1)
        runtime=m.TealDenseRuntime.__new__(m.TealDenseRuntime);runtime.objects={id(mm):(mm,0)};runtime.saved=[]
        runtime.guard=SimpleNamespace(active=active,torch=SimpleNamespace(bfloat16='bf16'));runtime.mask_mode='dense';runtime.rid='x'
        runtime.events=[];runtime.sites=set();runtime.thresholds={(1,3):[0.25,0.5]};runtime.ledger=ledger
        value=SimpleNamespace(dtype='bf16',shape=(31668,13824))
        with patch.object(m.FactorW2Runtime,'__enter__',return_value=None), patch.object(m.FactorW2Runtime,'__exit__',return_value=None):
            runtime.__enter__()
            self.assertEqual(mm.apply(value),'result');self.assertEqual(events,[('original_guard_dense_W2_with_bias',value)])
            runtime.mask_mode='threshold50'
            counts={'total':31668*13824,'selected':2,'preexisting_zero':1,'selected_nonzero':1,'threshold_ties':0}
            with patch.object(m,'mask_values',return_value=('masked',counts)) as mask:
                self.assertEqual(mm.apply(value),'result');mask.assert_called_once_with(runtime.guard.torch,value,0.5)
                self.assertEqual(events[-1],('original_guard_dense_W2_with_bias','masked'))
                self.assertEqual((ledger.pre,ledger.done),(1,1));self.assertEqual(runtime.events[0]['step'],1)
                with self.assertRaises(ValueError):mm.apply(value)
                self.assertEqual((ledger.pre,ledger.done),(1,1))
            active.block=(1,3)
            with self.assertRaises(ValueError):mm.apply(value)
            active.block=(0,4);runtime.thresholds[(1,4)]=[0.25,0.5]
            with patch.object(m,'mask_values',side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError):mm.apply(value)
                self.assertEqual((ledger.pre,ledger.done),(2,1))
                with self.assertRaises(ValueError):mm.apply(value)

    def test_missing_or_duplicate_site_rejects_closed_mask(self):
        events=[{'step':step,'block':block} for step in range(4) for block in range(40)]
        m.validate_action_coverage('threshold25',events);m.validate_action_coverage('dense',[])
        with self.assertRaises(ValueError):m.validate_action_coverage('threshold25',events[:-1])
        with self.assertRaises(ValueError):m.validate_action_coverage('threshold25',events[:-1]+events[:1])
        with self.assertRaises(ValueError):m.validate_action_coverage('dense',events)

    def test_real_cpu_parent_is_required(self):
        with patch.object(m.native,'binding',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):m.draft(0,Path('/tmp'))
    def test_distinct_run_ids(self):
        p={'neutral_case':{'native_source_ordinal':1}};c={'native_source_ordinal':1040}
        names=[m.rid(m.control(p,t),'dense') for t in ('old','new')]+[m.rid(c,x) for x in m.MODES]
        self.assertEqual(len(names),len(set(names)))
if __name__=='__main__':unittest.main()
