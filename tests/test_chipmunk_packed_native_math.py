import sys,unittest,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_packed_native_math import packed_action,mask_check
class Tests(unittest.TestCase):
    def test_packed_action_preserves_inputs_and_bf16_rounding(self):
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);w1=torch.eye(2,dtype=torch.bfloat16);w2=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);h=torch.zeros(2,2,dtype=torch.bfloat16);y=torch.ones(2,2,dtype=torch.bfloat16);g=torch.ones(2,dtype=torch.bfloat16);mask=torch.tensor([0])
        z=packed_action(x,w1,b,w2,h,y,g,mask,mm_fp32=lambda a,b:a.float()@b.float());self.assertTrue(torch.equal(h,torch.zeros_like(h)));self.assertTrue(torch.equal(y,torch.ones_like(y)));self.assertEqual(z.shape,(2,2));self.assertTrue(torch.equal(z[:,1],y[:,1]))
    def test_wrong_mask_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'before packed action'):mask_check(torch.tensor([1],dtype=torch.int64),'0'*64)
if __name__=='__main__':unittest.main()

class OrderTests(unittest.TestCase):
    def test_hash_guard_fails_before_action(self):
        from unittest.mock import patch
        import chipmunk_packed_native_math as m
        x=torch.zeros((64,2),dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);g=torch.ones(2,dtype=torch.bfloat16);rows=torch.arange(64);idx=torch.tensor([0]);calls=[]
        def measured(n,counts,fn):calls.append(n);return fn()
        with patch.object(m,'packed_action') as action:
            with self.assertRaisesRegex(ValueError,'before packed action'):
                m.execute_pair(x,x,w,b,w,b,g,rows,1,idx,{'mean':'0'*64},measured=measured)
            action.assert_not_called()
        self.assertEqual(calls,['old_cache','cold_mean'])

class LedgerTests(unittest.TestCase):
    def test_four_matched_online_and_dense_repeats(self):
        import collections,hashlib
        from unittest.mock import patch
        import chipmunk_packed_native_math as m
        x=torch.zeros((64,2),dtype=torch.bfloat16); w=torch.eye(2,dtype=torch.bfloat16); b=torch.zeros(2,dtype=torch.bfloat16); g=torch.ones(2,dtype=torch.bfloat16); rows=torch.arange(64); idx=torch.tensor([0]); digest=hashlib.sha256(idx.numpy().tobytes()).hexdigest(); ledger=collections.Counter(); names=[]
        def measured(name,counts,fn): names.append(name);ledger.update(counts);return fn()
        with patch.object(m,'online_mask',return_value=idx),patch.object(m,'oracle_mask',return_value=idx):
            m.execute_pair(x,x,w,b,w,b,g,rows,1,idx,dict.fromkeys(('mean','p4','p64','random','oracle'),digest),measured=measured,mm_fp32=lambda a,b:a.float()@b.float())
        self.assertEqual(sum(n.startswith('dense_reference_') for n in names),4)
        self.assertEqual(sum(n.startswith('online_') for n in names),12)
        self.assertEqual((ledger['packed_action'],ledger['mask_guard'],ledger['selector'],ledger['dense_w1'],ledger['gate_product']),(17,17,16,5,21))
