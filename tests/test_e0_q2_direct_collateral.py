import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

P = Path(__file__).resolve().parents[1]/'tools/e0_q2_direct_collateral.py'
spec = importlib.util.spec_from_file_location('c02',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class C02Test(unittest.TestCase):
    def test_tie_and_smoothing_change_selection(self):
        h=torch.zeros((1,13824),dtype=torch.bfloat16); h[0,6912:6976]=2; h[0,6912]=8; h[0,6913]=6
        smooth=torch.ones((1,13824)); smooth[0,6912]=4
        _, masks=m.masks(h,smooth)
        self.assertEqual(masks[0].nonzero().tolist(),[[0,6913]])
        self.assertEqual(masks[1].nonzero().tolist(),[[0,6912]])
        self.assertEqual(int(masks[2].sum()),64)

    def test_direct_and_collateral_and_closure(self):
        a=torch.ones((1,13824),dtype=torch.bfloat16); q=a.clone(); mask=torch.zeros_like(a,dtype=torch.bool); mask[:,6912]=True
        q[:,6912]=0; q[:,6913]=2
        d,c,t=m.terms(a,q,mask)
        self.assertTrue(torch.equal(d+c,t)); self.assertEqual(c[0,6913].item(),1)
        whole=torch.zeros_like(mask);whole[:,6912:6976]=True; q=a.masked_fill(whole,0)
        m.terms(a,q,whole,True)
        q[:,0]=2
        with self.assertRaisesRegex(ValueError,'closure'):m.terms(a,q,whole,True)
        q[:,6912]=1
        with self.assertRaisesRegex(ValueError,'target'):m.terms(a,q,whole,True)

    def test_authorization_and_budget(self):
        plan={'id':m.ID,'limits':dict(m.LIMITS),'cases':[{'source':s,'step':t,'block':b} for s in (0,12) for t in (0,2) for b in (0,39)],'shards':[{'expert':e,'block':b} for e in ('high_noise_model','low_noise_model') for b in (0,39)]}
        active={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'abc','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
        m.check_authorization(plan,'abc',active)
        with self.assertRaises(ValueError):m.check_authorization(plan,'wrong',active)
        plan['limits']['qdq']=33
        with self.assertRaises(ValueError):m.check_authorization(plan,'abc',active)

    def test_bf16_difference_promotes_before_subtraction(self):
        a=torch.tensor([256.0],dtype=torch.bfloat16); b=torch.tensor([-1.0],dtype=torch.bfloat16)
        self.assertEqual((a.float()-b.float()).item(),257.0)
        self.assertNotEqual((a-b).float().item(),257.0)

    def test_asset_sha_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'panel').write_bytes(b'actual')
            plan={'cases':[{'path':'panel','sha256':hashlib.sha256(b'other').hexdigest()}],'shards':[],'sources':[]}
            with self.assertRaisesRegex(ValueError,'Asset SHA'):m.verify_assets(plan,root)

if __name__=='__main__':unittest.main()
