"""CPU engineering checks only; not scientific calibration or author acceptance."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from icc_four_boundary_collector import FourBoundaryCollector, FourBoundaryTap

class ModelOwner:
    @staticmethod
    def infer_ffn(model, phase, x, attn, shift, scale, gate):
        return phase.ffn_2.apply(torch.nn.functional.gelu(phase.ffn_0.apply(x), approximate='tanh'))

class TestCollector(unittest.TestCase):
    def test_ca_cd_same_live_boundaries(self):
        torch.manual_seed(4)
        w1 = torch.randn(5,3).bfloat16(); w2 = torch.randn(3,5).bfloat16()
        phase = SimpleNamespace(ffn_0=SimpleNamespace(apply=lambda x: x @ w1.T),
                                ffn_2=SimpleNamespace(apply=lambda x: x @ w2.T))
        model = SimpleNamespace(scheduler=SimpleNamespace(step_index=0), block_idx=0)
        u0 = torch.randn(7,3).bfloat16(); u1 = (u0.float()+.25).bfloat16()
        native = ModelOwner.infer_ffn
        with tempfile.TemporaryDirectory() as tmp:
            ca = FourBoundaryCollector(torch, Path(tmp)/'ca', 's', 'high', 'CA', tokens=7,d=3,m=5,layers=1,chunk_rows=3,identity={'test':'fixture'})
            tap = FourBoundaryTap(ca, 'CA', 0); tap.dispatch = 'high'; tap.install(ModelOwner)
            result = ModelOwner.infer_ffn(model, phase, u0, None, None, None, torch.ones(1))
            self.assertTrue(torch.equal(result, native(model, phase, u0, None, None, None, None)))
            tap.uninstall(); ca.finish()
            self.assertIs(ModelOwner.infer_ffn, native)
            data = torch.load(Path(tmp)/'ca/stats_00.pt', weights_only=True)
            torch.testing.assert_close(data['u']['magnitude'], u0.float().abs().mean(0), rtol=1e-6, atol=1e-6)
            cd = FourBoundaryCollector(torch, Path(tmp)/'cd', 's', 'high', 'CD', tokens=7,d=3,m=5,layers=1,chunk_rows=3,identity={'test':'fixture'})
            for name, value, step in [('anchor', u0, 0), ('hit', u1, 1)]:
                model.scheduler.step_index = step
                tap = FourBoundaryTap(cd, name, step); tap.dispatch = 'high'; tap.install(ModelOwner)
                try: ModelOwner.infer_ffn(model, phase, value, None, None, None, torch.ones(1))
                finally: tap.uninstall()
            cd.finish()
            data = torch.load(Path(tmp)/'cd/stats_00.pt', weights_only=True)
            torch.testing.assert_close(data['u']['magnitude'], (u1.float()-u0.float()).abs().mean(0), rtol=1e-6, atol=1e-6)
            self.assertFalse(list((Path(tmp)/'cd').glob('anchor_*.pt')))
    def test_incomplete_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = FourBoundaryCollector(torch, Path(tmp)/'x', 's','high','CA', tokens=1,d=2,m=3,layers=1,identity={'x':1})
            c.begin_site('CA',0)
            with self.assertRaises(RuntimeError): c.observe('z', torch.ones(1,3,dtype=torch.bfloat16))
            with self.assertRaises(RuntimeError): c.finish()

if __name__ == '__main__': unittest.main()
