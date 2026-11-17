"""Small CPU engineering checks; fixtures are not research results."""
import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from icc_local_replay import (DEV11,SITES,FAMILIES,RANKS,MODES,FactorPair,canonical_endpoints,
                              icc_prediction,gated_risk,error_decomposition,select_family)
from q1_cal11_reproduction import Q1NativeCache
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'reports/20261114/g20_two_track_core/BWS/v1/source'))
from ridge_fit_core import native_ffn,bf16_replay
from icc_ffn import ICCFFN,Factors

class PrimitiveTests(unittest.TestCase):
    def test_icc_and_q1(self):
        torch.manual_seed(121)
        d,m,r,n = 3,5,2,4
        u0,u1 = torch.randn(n,d).bfloat16(),torch.randn(n,d).bfloat16()
        w1,b1,w2,b2 = torch.randn(m,d).bfloat16(),torch.randn(m).bfloat16(),torch.randn(d,m).bfloat16(),torch.randn(d).bfloat16()
        c=canonical_endpoints(native_ffn,u0,u1,w1,b1,w2,b2,rows=n,d=d,m=m)
        f1=FactorPair(torch.randn(m,r).bfloat16(),torch.randn(r,d).bfloat16())
        f2=FactorPair(torch.randn(d,r).bfloat16(),torch.randn(r,m).bfloat16())
        for mode,old in [('4S','author_four_state'),('2R','ffnds_two_residual')]:
            ref=ICCFFN(Factors(f1.left,f1.right),Factors(f2.left,f2.right),old)
            ref.anchor(u0,lambda x:F.linear(x,w1,b1),lambda x:F.linear(x,w2,b2))
            h,y=icc_prediction(c,f1,f2,r,mode)
            torch.testing.assert_close(y,ref.hit(u1),rtol=0,atol=0)
            risk=gated_risk(y,c['y1'],torch.ones(d))
            self.assertGreaterEqual(risk,0)
            if mode=='4S':
                dec=error_decomposition(c,h,y,f2,torch.ones(d))
                self.assertLess(dec['closure_max_abs'],1e-12)
        a=torch.randn(d,d).float(); gate=torch.ones(d)
        ref,_,_=bf16_replay(u0,u1,c['y0'],c['y1'],gate,a.bfloat16())
        q=Q1NativeCache(a); key=('request','high',19); q.anchor(key,u0,c['y0'])
        torch.testing.assert_close(q.hit(key,u1),ref,rtol=0,atol=0)
        with self.assertRaises(RuntimeError): q.hit(('request','low',19),u1)
        q.clear()
        with self.assertRaises(RuntimeError): q.hit(key,u1)

    def test_complete_grid_and_selection(self):
        rows=[{'source':s,'expert':e,'layer':l,'family':f,'rank':r,'mode':m,'risk':float(FAMILIES.index(f)+1)}
              for s in DEV11 for e,l in SITES for f in FAMILIES for r in RANKS for m in MODES]
        self.assertEqual(len(rows),792)
        self.assertEqual(select_family(rows)['family'],'SVD')
        with self.assertRaises(ValueError): select_family(rows[:-1])
        with self.assertRaises(ValueError): select_family(rows+[rows[0]])
        for row in rows: row['risk']=1.0
        self.assertEqual(select_family(rows)['family'],'SVD')

if __name__=='__main__': unittest.main()
