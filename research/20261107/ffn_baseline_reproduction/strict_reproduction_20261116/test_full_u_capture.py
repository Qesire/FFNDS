"""Small non-experimental observer/sink engineering tests; CPU only."""
import tempfile
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import full_u_capture as m

class FakeV2:
    def __init__(self, sink): self.sink, self.context = sink, ('vbi_1013', 42)
    def install(self, owner):
        self.owner, self.original = owner, owner.infer_ffn
        sink = self.sink
        def checked(model, phase, x, attn, shift, scale, gate):
            old = phase.ffn_0.apply
            def wrapped(u):
                y = old(u)
                ids = torch.arange(512) % 3
                sha = hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()
                sink.write_panel({'source':'vbi_1013', 'step':model.scheduler.step_index,
                                  'layer':model.block_idx, 'token_id_sha256_le_int64': sha},
                                 {'U':u.index_select(0, ids), 'Y_pre_gate':torch.zeros(512, 2),
                                  'token_indices': ids, 'token_thw':torch.zeros(512, 3)})
                return y
            phase.ffn_0.apply = wrapped
            try: return self.original(model, phase, x, attn, shift, scale, gate)
            finally: phase.ffn_0.apply = old
        owner.infer_ffn = checked
    def uninstall(self): self.owner.infer_ffn = self.original

class Owner:
    @staticmethod
    def infer_ffn(model, phase, x, attn, shift, scale, gate): return phase.ffn_0.apply(x)

class TestCapture(unittest.TestCase):
    def test_wrapper_and_readback(self):
        old_tokens, old_dim = m.TOKENS, m.DIM
        m.TOKENS, m.DIM = 3, 2
        try:
            with tempfile.TemporaryDirectory() as d:
                sink = m.FullUSink(torch, Path(d) / 'sink')
                sink.record_weights = lambda *args: None
                v2 = FakeV2(sink); tap = m.FullUTap(torch, v2, sink)
                original = Owner.infer_ffn; tap.install(Owner)
                phase = SimpleNamespace(ffn_0=SimpleNamespace(apply=lambda x: x + 1))
                model = SimpleNamespace(scheduler=SimpleNamespace(step_index=0), block_idx=4)
                u = torch.arange(6, dtype=torch.bfloat16).reshape(3,2)
                out = Owner.infer_ffn(model, phase, u, None, None, None, None)
                self.assertTrue(torch.equal(out, u+1))
                self.assertTrue(torch.equal(torch.load(sink.root / 'vbi_1013__s00__b04__U.pt', weights_only=True), u))
                with self.assertRaises(RuntimeError): Owner.infer_ffn(model, phase, u, None, None, None, None)
                self.assertTrue(torch.equal(phase.ffn_0.apply(u), u+1))
                tap.uninstall()
                self.assertIs(Owner.infer_ffn, original)
        finally:
            m.TOKENS, m.DIM = old_tokens, old_dim

if __name__ == '__main__': unittest.main()




