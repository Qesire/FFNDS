"""CPU-only synthetic wiring checks; no Wan/HiF4 import, CUDA or model assets."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_paired_ffn_observer import PairedFFNObserver, raster_strata, uniform_rows


class RuntimeFixture:
    def __init__(self):
        generator = torch.Generator().manual_seed(125)
        self.expert = 'high_noise_model'
        self.step = 1
        self.records = {self.expert: {}}
        for suffix, shape in [(0, (8, 4)), (2, (4, 8))]:
            self.records[self.expert][f'blocks.0.ffn.{suffix}'] = {
                'weight_qdq': torch.randn(shape, generator=generator).bfloat16() / 8,
                'smooth_scale': torch.ones(shape[1]), 'activation_qdq': True, 'force_fp32': True}
        self.activation_qdq_override = None
        self.counters = dict.fromkeys(['linear_forward_calls', 'activation_qdq_calls', 'weight_transfers'], 0)
        self.adapter = SimpleNamespace(native_hif4_activation_qdq=lambda x, **kw: torch.round(x.float() * 8) / 8)
        self.pack = {'w1': self.records[self.expert]['blocks.0.ffn.0']['weight_qdq'][:4],
                     'v': self.records[self.expert]['blocks.0.ffn.2']['weight_qdq'][:, :4],
                     'b1': torch.zeros(4).bfloat16(), 'bias': torch.ones(4).bfloat16() / 32,
                     'matrix': torch.eye(4).bfloat16() / 64, 's1': torch.ones(1, 4), 's2': torch.ones(1, 4)}

    def _record(self, expert, module_name):
        return self.records[expert][module_name]

    def _weight(self, **kw):
        return kw['record']['weight_qdq']

    def apply(self, mm, x, **kw):
        if not kw['module_name'].endswith(('.ffn.0', '.ffn.2')):
            return x + 100
        rec = self._record(kw['expert'], kw['module_name'])
        q = self.adapter.native_hif4_activation_qdq((x.float() / rec['smooth_scale']).to(x.dtype)).to(x.dtype)
        return F.linear(q, rec['weight_qdq'], mm._get_actual_bias())


class InferFixture:
    def __init__(self, runtime):
        self.runtime = runtime
        modules = []
        for suffix, n in [(0, 8), (2, 4)]:
            mm = SimpleNamespace(_get_actual_bias=lambda n=n: torch.zeros(n).bfloat16())
            mm.apply = lambda x, mm=mm, suffix=suffix: runtime.apply(mm, x, expert=runtime.expert,
                step_index=runtime.step, block_index=0, module_name=f'blocks.0.ffn.{suffix}')
            modules.append(mm)
        self.phase = SimpleNamespace(ffn_0=modules[0], ffn_2=modules[1])

    def infer_ffn(self, phase, x, attention, shift, scale, gate=None):
        x.add_(attention)
        up = phase.ffn_0.apply(x + shift)
        return phase.ffn_2.apply(F.gelu(up, approximate='tanh'))

    def post_process(self, x, y, gate, pre=None):
        x.add_(y * gate.squeeze())
        return x

    def run(self, x):
        gate = torch.ones(1, 4).bfloat16() / 2
        y = self.infer_ffn(self.phase, x, torch.ones_like(x) / 16, 0, 0, gate)
        return self.post_process(x, y, gate)


class RecordingTrace:
    def __init__(self):
        self.sites = {}

    def add_site(self, step, block, expert, **kw):
        self.sites[step, block] = kw


class NativeWiringSoftwareTest(unittest.TestCase):
    def test_strata_partition_full_raster_without_overlap(self):
        rows = [i for b in raster_strata((21, 29, 52)) for i in b['rows']]
        self.assertEqual(sorted(rows), list(range(31668)))
        self.assertEqual(uniform_rows(3, 32), [0, 1, 2])

    def test_same_raw_ffn_difference_can_have_different_native_writeback_error(self):
        # A finite-precision counterexample, not a Wan observation.
        dense = torch.tensor(0, dtype=torch.bfloat16)
        candidate = torch.tensor(2**-9, dtype=torch.bfloat16)
        high_base = torch.tensor(1, dtype=torch.bfloat16)
        low_base = torch.tensor(0, dtype=torch.bfloat16)
        self.assertEqual(float((high_base + candidate) - (high_base + dense)), 0)
        self.assertEqual(float((low_base + candidate) - (low_base + dense)), 2**-9)

    def test_four_arms_preserve_unfused_dense_and_candidate_references(self):
        runtime = RuntimeFixture(); infer = InferFixture(runtime)
        x = torch.linspace(-1, 1, 108).reshape(27, 4).bfloat16()
        expected_dense = infer.run(x.clone())
        pack = runtime.pack
        residual = x + .25 + torch.ones_like(x) / 16
        q1 = runtime.adapter.native_hif4_activation_qdq(residual).bfloat16()
        hidden = F.gelu(F.linear(q1, pack['w1'], pack['b1']), approximate='tanh')
        q2 = runtime.adapter.native_hif4_activation_qdq(hidden).bfloat16()
        expected_candidate = residual + (F.linear(q2, pack['v'], pack['bias']) + F.linear(q1, pack['matrix'])) / 2
        old_apply, old_ffn, old_post = runtime.apply, infer.infer_ffn, infer.post_process
        trace = RecordingTrace()
        with PairedFFNObserver(runtime, {(runtime.expert, 0): runtime.pack}, runtime.adapter, trace,
                               {(runtime.expert, 0): 'SYNTHETIC'}, numerical_rows=3) as observer:
            results = []
            for name, kind, value in [('teacher', 'dense', x), ('student', 'candidate', x + .25)]:
                with observer.branch(infer, branch=name, kind=kind, expert=runtime.expert, step=runtime.step, block=0):
                    results.append(infer.run(value.clone()))
            self.assertTrue(torch.equal(results[0], expected_dense))
            self.assertTrue(torch.equal(results[1], expected_candidate))
            observer.finish_site(runtime.step, 0, runtime.expert, *results, grid=(3, 3, 3))
            self.assertEqual(observer.counts['native_QDQ'], 8)
            self.assertEqual(observer.counts['dense_main_GEMM'] + observer.counts['dense_shadow_GEMM'], 4)
            self.assertEqual(observer.counts['candidate_main_GEMM'] + observer.counts['candidate_shadow_GEMM'], 6)
            self.assertEqual(observer.counts['continuous_GEMM'], 6)
            self.assertIsNone(observer.pending)
        self.assertEqual(runtime.apply, old_apply)
        self.assertEqual(infer.infer_ffn, old_ffn)
        self.assertEqual(infer.post_process, old_post)
        self.assertEqual(len(trace.sites[runtime.step, 0]['spatiotemporal_strata']), 9)

    def test_exception_restores_all_hooks_and_missing_context_is_rejected(self):
        runtime = RuntimeFixture(); infer = InferFixture(runtime)
        originals = runtime.apply, infer.infer_ffn, infer.post_process
        with self.assertRaises(RuntimeError):
            with PairedFFNObserver(runtime, {(runtime.expert, 0): runtime.pack}, runtime.adapter,
                                   RecordingTrace(), {(runtime.expert, 0): 'SYNTHETIC'}) as observer:
                with self.assertRaises(RuntimeError):
                    infer.run(torch.zeros(27, 4).bfloat16())
                with observer.branch(infer, branch='teacher', kind='dense', expert=runtime.expert, step=runtime.step, block=0):
                    raise RuntimeError('synthetic interruption')
        self.assertEqual((runtime.apply, infer.infer_ffn, infer.post_process), originals)

    def test_unequal_initial_native_prefix_cannot_be_treated_as_operator_drift(self):
        runtime = RuntimeFixture(); runtime.step = 0; infer = InferFixture(runtime)
        with self.assertRaisesRegex(RuntimeError, 'First native prefix differs'):
            with PairedFFNObserver(runtime, {(runtime.expert, 0): runtime.pack}, runtime.adapter,
                                   RecordingTrace(), {(runtime.expert, 0): 'SYNTHETIC'}) as observer:
                values = []
                for branch, delta in [('teacher', 0), ('student', .25)]:
                    with observer.branch(infer, branch=branch, kind='dense', expert=runtime.expert, step=0, block=0):
                        values.append(infer.run(torch.full((27, 4), delta).bfloat16()))
                observer.finish_site(0, 0, runtime.expert, *values, grid=(3, 3, 3))


if __name__ == '__main__':
    unittest.main()
