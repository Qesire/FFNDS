"""Matched-calibration controls; no real source bytes, models, or GPU."""
import io
import subprocess
import sys
import unittest
import torch

from tools import e0_raw_bf16_temporal_balanced_fit as m


class MatchedFitControls(unittest.TestCase):
    def test_ordinary_same_rows_is_frozen_ordinary_objective(self):
        cases = {}
        for source in range(2):
            for step in range(2):
                y = (torch.arange(168*3).double().reshape(168, 3)/100+source+step).bfloat16()
                cases[source, step] = {'Y': y, 'gate': torch.tensor([.5, 1., 2.]).bfloat16()}
        bias = torch.tensor([1., 2., 3.]).bfloat16()
        metric = torch.tensor([.5, 1., 2.]).double()
        expected = m.factor.anchored_rows(cases, bias, metric, steps=(0, 1), source_indices=(0, 1))
        actual, records = m.target.anchored_rows(cases, bias, metric, steps=(0, 1), source_indices=(0, 1), balance=False)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        self.assertEqual(len(records), 4)
        # A held diagnostic can never alter target construction or its scale.
        cases[99, 0] = {'Y': torch.full((168, 3), float('nan')), 'gate': torch.zeros(3)}
        held_poison, _ = m.target.anchored_rows(cases, bias, metric, steps=(0, 1), source_indices=(0, 1), balance=False)
        self.assertTrue(torch.equal(actual, held_poison))

    def test_per_expert_exact_work_covers_both_families_and_old_diagnostic(self):
        train = {'cases': [{'packets': [{'step': s, 'payload': {'bytes': 100}} for s in range(4) for _ in range(40)]}
                           for _ in range(10)]}
        old = {i: {'packets': [{'step': s, 'payload': {'bytes': 200}} for s in range(4) for _ in range(40)]}
               for i in range(8)}
        for part in (0, 1):
            b = m.limits(part, train, old)
            self.assertEqual(b['new_packet_reads'], 800)
            self.assertEqual(b['old_packet_reads'], 640)
            self.assertEqual(b['FP64_eigh'], 80)
            self.assertEqual(b['BF16_export_files'], 80)
            self.assertEqual(b['BF16_factor_GEMM'], 5760)
            self.assertEqual(b['teacher_postwrite_byte_neutrals'], 1440)
            self.assertEqual(b['fixed_gate_gain_calls'], b['BF16_postwrite_calls'])
        with self.assertRaises(ValueError):
            m.scope_id(True)

    def test_meter_adapter_supports_export_work_and_two_GEMM_calls(self):
        class ToyMeter:
            def __init__(self):
                self.calls = []
            def work(self, costs, fn, *a, **kw):
                self.calls.append(costs)
                return fn(*a, **kw)
        base = ToyMeter()
        meter = m.NumericalMeter(base)
        self.assertEqual(meter.call('BF16_factor_GEMM', lambda: 1), 1)
        self.assertEqual(meter.work({'factor_export_D2H_tensors': 3}, lambda: 2), 2)
        self.assertEqual(base.calls, [{'BF16_factor_GEMM': 1}, {'factor_export_D2H_tensors': 3}])

    def test_incremental_journal_adapter_preserves_real_lines(self):
        stream = io.StringIO()
        writer = m.journal_adapter(stream)
        writer.append('operations.jsonl', b'{"synthetic":true}\n')
        self.assertEqual(stream.getvalue(), '{"synthetic":true}\n')
        with self.assertRaises(ValueError):
            writer.append('unknown', b'x')

    def test_import_is_no_model_or_numerical_execution(self):
        p = subprocess.run([sys.executable, '-c', "import sys;sys.path.insert(0,'tools');import e0_raw_bf16_temporal_balanced_fit;assert not {'torch','numpy'}&set(sys.modules)"],
                           cwd=m.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == '__main__':
    unittest.main()
