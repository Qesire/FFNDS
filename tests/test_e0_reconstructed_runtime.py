import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_reconstructed_runtime import ReconstructedRuntime


class ReconstructedRuntimeTest(unittest.TestCase):
    def test_real_nonlinear_path_and_folded_input_use_distinct_states(self):
        gen = torch.Generator().manual_seed(17)
        def sample(*shape): return torch.randn(*shape, generator=gen).bfloat16()
        pack = dict(w1=sample(64,5120), v=sample(5120,64), matrix=sample(5120,5120),
                    b1=sample(64), bias=sample(5120), s1=torch.full((1,5120),2.), s2=torch.full((1,64),3.))
        x = sample(2,5120)
        fallback = lambda *a, **kw: 'untouched'
        runtime = SimpleNamespace(apply=fallback)
        adapter = SimpleNamespace(native_hif4_activation_qdq=lambda x, **kw: x)
        kwargs = dict(expert='high_noise_model', step_index=0, block_index=0)
        q1 = (x.float()/2).bfloat16()
        z = torch.nn.functional.linear(q1,pack['w1'],pack['b1'])
        h = torch.nn.functional.gelu(z,approximate='tanh')
        expected = torch.nn.functional.linear((h.float()/3).bfloat16(),pack['v'],pack['bias']) + torch.nn.functional.linear(q1,pack['matrix'])
        with ReconstructedRuntime(runtime,{('high_noise_model',0):pack},adapter) as hook:
            self.assertEqual(runtime.apply(None,x,module_name='blocks.0.self_attn.q',**kwargs),'untouched')
            actual_z=runtime.apply(None,x,module_name='blocks.0.ffn.0',**kwargs)
            actual=runtime.apply(None,torch.nn.functional.gelu(actual_z,approximate='tanh'),module_name='blocks.0.ffn.2',**kwargs)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(hook.counts,dict(up=1,down=1,Q1=1,Q2=1,affine=1))
        self.assertIs(runtime.apply,fallback)


if __name__ == '__main__': unittest.main()
