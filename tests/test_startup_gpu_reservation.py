"""CPU-only startup ordering, ownership and unchanged scientific-loop checks."""
import ast
import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import startup_gpu_reservation as lease
from c2_reserved_contract import Budget, METHOD
from c2_contract import CAPS, sha
from export_c2_native_results import complete_pair_available


class StartupReservation(unittest.TestCase):
    def fake_torch(self, events):
        cuda = SimpleNamespace(device_count=lambda: 1,
            init=lambda: events.append('CUDA_init'), synchronize=lambda: events.append('synchronize'))
        def empty(*args, **kwargs):
            self.assertEqual(args, (1,)); self.assertEqual(kwargs['device'], 'cuda')
            events.append('tiny_allocation'); return object()
        return SimpleNamespace(cuda=cuda, uint8='synthetic_uint8', empty=empty)

    def environment(self):
        return patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-synthetic-2', 'FFNDS_PHYSICAL_GPU':'2'})

    def test_model_context_visible_before_any_assets_are_loaded(self):
        events = []
        def query(monitor, args):
            if args[0].startswith('--query-compute-apps'):
                events.append('NVML_owns_model_PID'); return [[str(os.getpid()), '1']]
            if args[0].startswith('--query-gpu=memory.used'): return [['2']]
            return [['synthetic_A100']]
        with self.environment(), patch.object(lease, 'gpu_uuid', return_value='GPU-synthetic-2'), \
             patch.object(lease.UUIDAssignedGPUResources, 'query', query):
            tensor, monitor, receipt = lease.reserve_before_assets(self.fake_torch(events))
            try:
                events.append('load_assets')
                self.assertTrue(receipt['same_model_process'])
                self.assertTrue(receipt['before_CPU_asset_loading'])
                self.assertEqual(receipt['new_FFN_norm_selector_or_generation_calls'], 0)
                self.assertEqual(receipt['own_NVML_process_memory_MiB_at_claim'], 1)
                self.assertEqual(monitor.report()['peak_physical_used_memory_MiB'], 2)
                self.assertLess(events.index('tiny_allocation'), events.index('NVML_owns_model_PID'))
                self.assertLess(events.index('NVML_owns_model_PID'), events.index('load_assets'))
            finally: monitor.close()

    def test_foreign_process_or_invisible_ownership_stops_before_assets(self):
        for rows, message in [([[str(os.getpid()+1000000), '1']], 'Foreign compute'),
                              ([], 'not visible')]:
            with self.subTest(rows=rows), self.environment(), \
                 patch.object(lease, 'gpu_uuid', return_value='GPU-synthetic-2'), \
                 patch.object(lease.UUIDAssignedGPUResources, 'query', return_value=rows):
                with self.assertRaisesRegex(RuntimeError, message):
                    lease.reserve_before_assets(self.fake_torch([]), timeout_seconds=0)

    def test_wrong_UUID_is_rejected_before_CUDA_initialization(self):
        events = []
        with self.environment(), patch.object(lease, 'gpu_uuid', return_value='GPU-other'):
            with self.assertRaisesRegex(RuntimeError, 'UUID and locked physical GPU differ'):
                lease.reserve_before_assets(self.fake_torch(events))
        self.assertEqual(events, [])

    def test_closed_original_budget_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)); self.assertEqual(b.state['id'], METHOD)
            self.assertEqual(b.state['used'], {k:0 for k in CAPS}); b.close()
            with self.assertRaisesRegex(RuntimeError, 'no retry'): Budget(Path(d))

    def test_scientific_generation_loop_matches_frozen_original(self):
        original = ROOT / 'tools/run_c2_norm_scaled_smoke.py'
        protocol = json.loads((ROOT / 'research/20261023/C2N0_registered_protocol_v1.json').read_text())
        self.assertEqual(sha(original), protocol['source_sha256']['tools/run_c2_norm_scaled_smoke.py'])
        def loop(path):
            nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.For)
                     and isinstance(n.iter,ast.Name) and n.iter.id=='ARMS']
            self.assertEqual(len(nodes), 1)
            return ast.dump(nodes[0], include_attributes=False)
        self.assertEqual(loop(original), loop(ROOT / 'tools/run_c2_reserved_smoke.py'))
        source = (ROOT / 'tools/run_c2_reserved_smoke.py').read_text()
        self.assertLess(source.index('reservation, monitor, lease_receipt = reserve_before_assets(torch)'),
                        source.index('history = HistoricalSources(); kwargs, cache, assets = verified_restored_assets(history)'))

    def test_missing_authority_prevents_even_Torch_import(self):
        import run_c2_reserved_smoke as runner
        old_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'torch': raise AssertionError('Torch imported before authority')
            return old_import(name, *args, **kwargs)
        with patch.object(runner, 'require_registration', side_effect=RuntimeError('unapproved scope')), \
             patch.object(sys, 'argv', ['runner', '--source-view-receipt', 'unused']), \
             patch('builtins.__import__', side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, 'unapproved scope'): runner.main()

    def test_zero_video_failure_has_no_human_review_obligation(self):
        actual = json.loads((ROOT / 'research/20261023/C2N0_native_results.json').read_text())
        self.assertEqual(actual['requests'], [])
        self.assertFalse(complete_pair_available(actual))
        self.assertFalse(complete_pair_available({'status':'STOPPED_C2_FIXED_SCOPE_NO_RETRY','requests':[{}]}))
        self.assertTrue(complete_pair_available({'status':'COMPLETE_C2N1_NATIVE_VIDEOS_PENDING_FULL_REVIEW','requests':[{}]*6}))


if __name__ == '__main__': unittest.main()
