import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file

p=Path(__file__).resolve().parents[1]/'tools/b_head_asset_export.py';spec=importlib.util.spec_from_file_location('head',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_export_layout_and_identity(self):
  with tempfile.TemporaryDirectory() as d:
   values={'weight':torch.arange(64*5120,dtype=torch.float32).reshape(64,5120).to(torch.bfloat16),'bias':torch.zeros(64,dtype=torch.bfloat16),'modulation':torch.zeros((1,2,5120),dtype=torch.bfloat16)}
   names={'weight':'head.head.weight','bias':'head.head.bias','modulation':'head.modulation'};p=Path(d)/'view.safetensors';save_file({'high_noise_model.'+names[k]:v for k,v in values.items()},str(p))
   expected={k+'_raw_sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in values.items()}
   out=m.export(p,m.sha(p),'high_noise_model',expected);w=out['head_weight_runtime_KN'];self.assertEqual(tuple(w.shape),(5120,64));self.assertEqual(w.stride(),(64,1));self.assertTrue(torch.equal(w.t().contiguous().view(torch.uint8),values['weight'].view(torch.uint8)))
   expected['bias_raw_sha256']='0'*64
   with self.assertRaises(ValueError):m.export(p,m.sha(p),'high_noise_model',expected)
if __name__=='__main__':unittest.main()
