import importlib.util
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

p=Path(__file__).resolve().parents[1]/'tools/e0_w2_shape_control.py';spec=importlib.util.spec_from_file_location('w2shape',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ShapeControlTests(unittest.TestCase):
    def test_row_independence_at_padding_positions(self):
        x=torch.tensor([[1.,2.,-1.],[4.,-2.,3.]])
        w=torch.tensor([[2.,1.,-3.],[-1.,2.,1.]])
        bias=torch.tensor([1.,-2.]);ids=[1,5]
        y=F.linear(x,w,bias);pad=torch.zeros(7,3);pad[ids]=x
        self.assertTrue(torch.equal(y,F.linear(pad,w,bias)[ids]))
        self.assertEqual(m.compare(y,F.linear(pad,w,bias)[ids])['unequal_elements'],0)
        self.assertEqual(m.compare(y,y)['bitwise_equal_elements'],y.numel())
    def test_rows_reject_duplicates_out_of_bounds_shape(self):
        m.validate_rows([1,5],7,2)
        for rows in ([1,1],[1,7],[-1,1],[1], [True,1]):
            with self.assertRaises(ValueError):m.validate_rows(rows,7,2)
    def test_comparison_shape_mismatch(self):
        with self.assertRaises(ValueError):m.compare(torch.zeros(2,2),torch.zeros(2,3))
    def test_authorization_and_budget(self):
        plan={'id':m.ID,'limits':dict(m.LIMITS),'full_rows':31668,'saved_rows':160}
        active={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'ok','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
        m.authorize(plan,'ok',active)
        with self.assertRaises(ValueError):m.authorize(plan,'wrong',active)
        plan['limits']['bf16_linear']=17
        with self.assertRaises(ValueError):m.authorize(plan,'ok',active)

if __name__=='__main__':unittest.main()
