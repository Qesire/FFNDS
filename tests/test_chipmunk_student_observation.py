import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_student_observation_coordinates import coordinates
from chipmunk_student_observation_math import prepare_omission,omission_energy,paired_errors,partition_sums
from chipmunk_student_observation_contract import WORK
class Tests(unittest.TestCase):
 def test_coordinates_disjoint_and_raster(self):
  sel=list(range(64));rows=coordinates(sel);self.assertEqual(len(rows),256);self.assertEqual(len({x['row'] for x in rows}),256);self.assertTrue(set(sel).isdisjoint(x['row'] for x in rows));self.assertTrue(all(x['row']==x['tau']*29*52+x['y']*52+x['x'] and 0<x['inclusion_probability']<=1 for x in rows));self.assertEqual(rows,coordinates(sel))
 def test_signed_cross_identity(self):
  d=torch.tensor([[1.,1.,3.],[2.,-1.,0.]])
  w=torch.tensor([[1.,-1.,2.],[2.,1.,-1.]])
  gate=torch.tensor([1.,.5])
  r=omission_energy(torch,d,prepare_omission(torch,w,torch.tensor([2]),gate));self.assertTrue(torch.allclose(r['total'],r['diagonal']+r['cross']));self.assertTrue(torch.allclose(r['projection'],(d[:,:2]@w[:,:2].T)*gate));self.assertGreater(abs(r['cross'][0].item()),0)
 def test_write_boundary(self):
  a=torch.tensor([[1.,2.]]);b=torch.tensor([[.5,1.]]);g=torch.tensor([[2.,.5]]);r=paired_errors(torch,a,b,g,lambda x:(x*g+1).to(torch.bfloat16));self.assertTrue(torch.equal(r['actual_write_error'],(a*g+1).to(torch.bfloat16).float()-(b*g+1).to(torch.bfloat16).float()))
 def test_expensive_chunk_accounting(self):self.assertEqual(WORK['diagonal_gemv'],40*124);self.assertEqual(WORK['full_shadow_W1'],40)
 def test_full_row_partition_additivity(self):
  v=torch.tensor([[1.,2.],[3.,4.]]);ideal={'projection':torch.zeros_like(v),'total':torch.tensor([4.,9.]),'diagonal':torch.tensor([3.,6.]),'cross':torch.tensor([1.,3.])};p=partition_sums(torch,0,2,v,v,v,ideal,v,v,v);self.assertEqual(int(p['count'].sum()),2);self.assertEqual(float(p['pre_squared'].sum()),30);self.assertEqual(float(p['ideal_cross'].sum()),4)
if __name__=='__main__':unittest.main()

class SignTest(unittest.TestCase):
 def test_exact_omission_has_zero_difference(self):
  d=torch.tensor([[1.,2.]]);w=torch.tensor([[1.,3.],[2.,-1.]]);g=torch.tensor([.5,2.]);ideal=omission_energy(torch,d,prepare_omission(torch,w,torch.tensor([0]),g));actual=-ideal['projection'];stats=partition_sums(torch,0,1,actual,actual,actual,ideal,actual,actual,actual);self.assertEqual(float(stats['ideal_vs_actual_gated_squared'].sum()),0.0);self.assertGreater(float(ideal['total'].sum()),0)

class WriteTest(unittest.TestCase):
 def test_bf16_write_and_live_residual_unchanged(self):
  from chipmunk_student_observation_write import native_bf16_write
  y=torch.tensor([[.3,.7]],dtype=torch.bfloat16);g=torch.tensor([[[.5,.75]]],dtype=torch.bfloat16);r=torch.tensor([[2.,3.]],dtype=torch.bfloat16);before=r.clone();out=native_bf16_write(torch,y,g,0,1,r,None);self.assertTrue(torch.equal(out,before.clone().add_(y*g.squeeze())));self.assertTrue(torch.equal(r,before))
 def test_perrow_gate_rejected(self):
  from chipmunk_student_observation_math import prepare_omission
  with self.assertRaises(ValueError):prepare_omission(torch,torch.ones(2,3),torch.tensor([0]),torch.ones(4,2))

class TransferBudgetTest(unittest.TestCase):
 def test_selector_and_metadata_byte_counts(self):
  from chipmunk_student_observation_contract import STUDENT_WORK,DENSE_WORK
  for k in ('selector_identity_D2H_bytes','metadata_selector_D2H_bytes','sample_selector_D2H_bytes'):self.assertEqual(STUDENT_WORK[k],40*64*8)
  self.assertEqual(STUDENT_WORK['mask_index_D2H_bytes'],40*5120*8);self.assertEqual(STUDENT_WORK['selector_score_D2H_bytes'],40*13824*4)
  self.assertEqual(STUDENT_WORK['gate_D2H_bytes'],40*5120*4);self.assertEqual(DENSE_WORK['dense_gate_D2H_bytes'],40*5120*4)
  self.assertEqual(STUDENT_WORK['student_rows_H2D_bytes'],40*320*8);self.assertEqual(DENSE_WORK['dense_rows_H2D_bytes'],40*320*8)
