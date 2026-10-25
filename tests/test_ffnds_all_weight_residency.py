from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ffnds_all_weight_residency import AllWeightResidency


class Runtime:
    def __init__(self):self.counters={'weight_transfers':0}
    def _weight(self,**kw):raise AssertionError('Full residency should use the exact CPU source directly')


class AllResidencyTests(unittest.TestCase):
    def test_attention_and_FFN_share_correct_expert_aware_cache(self):
        rt=Runtime();original=rt._weight;values=[]
        with AllWeightResidency(rt,'all_resident_no_flush',1024) as policy:
            for expert,name,value in [('hi','blocks.0.attn.q',1),('hi','blocks.0.ffn.0',2),
                                      ('lo','blocks.0.ffn.0',3),('hi','blocks.0.attn.q',1),
                                      ('hi','blocks.0.ffn.0',2)]:
                tensor=rt._weight(expert=expert,block_index=0,module_name=name,
                    record={'weight_qdq':torch.full((3,2),float(value))},device=torch.device('cpu'),dtype=torch.bfloat16)
                self.assertTrue(torch.equal(tensor,torch.full((3,2),float(value),dtype=torch.bfloat16)))
                values.append(tensor)
            self.assertIs(values[0],values[3]);self.assertIs(values[1],values[4])
            self.assertEqual(rt.counters['weight_transfers'],3)
            self.assertEqual(policy.stats['all_retained_hits'],2)
            self.assertEqual(policy.stats['FFN_retained_hits'],1)
            self.assertEqual(policy.stats['retained_bytes'],36)
        self.assertEqual(rt._weight,original);self.assertEqual(policy.stats['retained_bytes'],0)

    def test_budget_checked_before_new_weight_transfer(self):
        rt=Runtime()
        with self.assertRaisesRegex(RuntimeError,'budget'):
            with AllWeightResidency(rt,'all_resident_no_flush',1):
                rt._weight(expert='hi',block_index=0,module_name='blocks.0.ffn.0',
                    record={'weight_qdq':torch.ones(3,2)},device=torch.device('cpu'),dtype=torch.bfloat16)
        self.assertEqual(rt.counters['weight_transfers'],0)


if __name__=='__main__':unittest.main()
