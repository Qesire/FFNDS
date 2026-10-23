"""Synthetic event API fixtures; these do not provide hardware timing data."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class NativeEventFixtureTests(unittest.TestCase):
    def setUp(self):
        class Event:
            tick = 0
            def __init__(self, enable_timing):
                self.ready = True
            def record(self):
                Event.tick += 1
                self.tick = Event.tick
            def query(self):
                return self.ready
            def elapsed_time(self, other):
                return float(other.tick - self.tick)
        class Infer:
            def __init__(self):
                self.scheduler = SimpleNamespace(step_index=0)
                self.block_idx = 0
            def pre_process(self):
                return 'modulation'
            def infer_ffn(self):
                return 'actual FFN result fixture'
            def post_process(self):
                return 'writeback fixture'
        self.Infer = Infer
        # No synchronize method exists in this fixture: per-block sync fails.
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
        native_module = SimpleNamespace(WanTransformerInfer=Infer)
        patch = mock.patch.dict(sys.modules, {'torch': fake_torch,
            'lightx2v.models.networks.wan.infer.transformer_infer': native_module})
        patch.start()
        self.addCleanup(patch.stop)
        spec = importlib.util.spec_from_file_location('ffnds_event_fixture', ROOT / 'tools/native_ffn_events.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_actual160_region_pairing_and_restore_without_sync(self):
        original = self.Infer.infer_ffn
        infer = self.Infer()
        with self.module.NativeFFNEvents() as events:
            for step in range(4):
                infer.scheduler.step_index = step
                for block in range(40):
                    infer.block_idx = block
                    self.assertEqual(infer.pre_process(), 'modulation')
                    self.assertEqual(infer.infer_ffn(), 'actual FFN result fixture')
                    self.assertEqual(infer.post_process(), 'writeback fixture')
        self.assertIs(self.Infer.infer_ffn, original)
        report = events.report_after_request_synchronized()
        self.assertEqual(len(report['rows']), 320)
        self.assertFalse(report['instrumentation_per_block_device_synchronization'])
        self.assertFalse(report['operator_internal_synchronization_excluded_from_cost'])

    def test_missing_region_and_premature_read_fail(self):
        infer = self.Infer()
        with self.module.NativeFFNEvents() as events:
            infer.pre_process()
            infer.infer_ffn()
            infer.post_process()
        with self.assertRaises(RuntimeError):
            events.report_after_request_synchronized()
        events.rows[0][-1].ready = False
        with self.assertRaises(RuntimeError):
            events.report_after_request_synchronized()

    def test_unfinished_ffn_restores_hook_and_refuses_success(self):
        original = self.Infer.infer_ffn
        infer = self.Infer()
        with self.assertRaises(RuntimeError):
            with self.module.NativeFFNEvents():
                infer.infer_ffn()
        self.assertIs(self.Infer.infer_ffn, original)


if __name__ == '__main__':
    unittest.main()
