from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ffnds_weight_residency import DenseWeightResidency


class MockNativeRuntime:
    def __init__(self):
        self._resident_key=None;self._resident_weights={};self.counters={'weight_transfers':0}
    def _weight(self,*,expert,block_index,module_name,record,device,dtype):
        key=(expert,int(block_index),str(device),str(dtype))
        if key!=self._resident_key:self._resident_weights.clear();self._resident_key=key
        if module_name not in self._resident_weights:
            self._resident_weights[module_name]=record['weight_qdq'].to(device=device,dtype=dtype).contiguous()
            self.counters['weight_transfers']+=1
        return self._resident_weights[module_name]


class ResidencyTests(unittest.TestCase):
    def test_exact_values_expert_isolation_and_only_FFN_retention(self):
        def replay(mode):
            rt=MockNativeRuntime();outs=[]
            with DenseWeightResidency(rt,mode) as state:
                for expert,block,module,value in [('high',0,'blocks.0.ffn.0',1),('high',1,'blocks.1.attn.q',2),
                                                  ('low',0,'blocks.0.ffn.0',3),('high',0,'blocks.0.ffn.0',1),
                                                  ('high',1,'blocks.1.attn.q',2)]:
                    outs.append(rt._weight(expert=expert,block_index=block,module_name=module,
                        record={'weight_qdq':torch.full((3,2),float(value))},device=torch.device('cpu'),dtype=torch.bfloat16))
                stats=state.snapshot()
            return outs,stats,rt
        original,_,_=replay('original')
        for mode in ['block_no_flush','ffn_resident_no_flush']:
            actual,stats,rt=replay(mode)
            for x,y in zip(original,actual):self.assertTrue(torch.equal(x,y))
            if mode=='ffn_resident_no_flush':
                self.assertIs(actual[0],actual[3]);self.assertEqual(stats['FFN_retained_hits'],1)
                self.assertEqual(stats['retained_bytes'],24)
                self.assertEqual(rt.counters['weight_transfers'],4)
            self.assertEqual(rt._weight.__func__,MockNativeRuntime._weight)

    def test_retention_budget_failure_restores_native_method(self):
        rt=MockNativeRuntime()
        with self.assertRaisesRegex(RuntimeError,'budget'):
            with DenseWeightResidency(rt,'ffn_resident_no_flush',max_retained_bytes=1):
                rt._weight(expert='high',block_index=0,module_name='blocks.0.ffn.0',
                    record={'weight_qdq':torch.ones(3,2)},device=torch.device('cpu'),dtype=torch.bfloat16)
        self.assertEqual(rt._weight.__func__,MockNativeRuntime._weight)


if __name__=='__main__':unittest.main()
