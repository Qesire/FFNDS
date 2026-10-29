"""Controls for new actual packet reader and bounded native replay accounting."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from tools import e0_s2_spatial_prefix_replay as m


class ReplayControls(unittest.TestCase):
    def test_new_reader_does_not_inherit_old_32gib_budget(self):
        payload = b'real bound bytes'
        with tempfile.TemporaryDirectory() as directory:
            hf = Path(directory); (hf/'packet').write_bytes(payload)
            binding = {'path': 'packet', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            audit = {'packet_bytes_read': (32 << 30)+1}
            self.assertEqual(m.stream_packet(hf, binding, audit), payload)
            self.assertEqual(audit['packet_bytes_read'], (32 << 30)+1+len(payload))
            bad = dict(binding, sha256='0'*64)
            with self.assertRaises(ValueError): m.stream_packet(hf, bad, {'packet_bytes_read': 0})
            with self.assertRaises(ValueError): m.stream_packet(hf, binding, {'packet_bytes_read': 40 << 30})

    def test_confined_reader_rejects_external_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); hf = root/'hf'; hf.mkdir(); (root/'outside').write_bytes(b'x')
            (hf/'escape').symlink_to(root/'outside')
            with self.assertRaises(ValueError):
                m.stream_packet(hf, {'path': 'escape', 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}, {'packet_bytes_read': 0})

    def test_native_counts_match_three_raw_and_four_two_gemm_outputs(self):
        limits = m.limits()
        self.assertEqual(limits['packet_reads'], 480)
        self.assertEqual(limits['original_u_byte_neutrals'], 480)
        self.assertEqual(limits['BF16_original_W2_GEMM'], 480)
        self.assertEqual(limits['BF16_factor_GEMM'], 1280)
        self.assertEqual(limits['predicted_Y_D2H_tensors'], 1120)
        self.assertEqual(limits['rFFT_calls'], 5120)
        for key in ('W1_calls', 'full_model_calls', 'fits', 'new_videos', 'official_quality_calls'):
            self.assertEqual(limits[key], 0)

    def test_predebit_is_retained_when_actual_native_call_fails(self):
        report = {}; meter = m.Meter(report, lambda _: None)
        def failure(): raise RuntimeError('intentional failure')
        with self.assertRaises(RuntimeError): meter.call('BF16_original_W2_GEMM', failure)
        self.assertEqual(report['predebited'], {'BF16_original_W2_GEMM': 1})
        self.assertEqual(report['observed'], {})
        with self.assertRaises(ValueError): meter.call('new_videos', lambda: None)

    def test_collector_dependency_identity_is_not_mutated_by_replay_import(self):
        from tools import e0_s2_spatial_student_collect as original
        self.assertNotEqual(original.BASE_ID, m.core.BASE_ID)
        self.assertEqual(m.core.BASE_ID, 'E0_S2_SPATIAL_TEACHER_STUDENT_FULL160_CAPTURE_PORTABLE_20261029_V2')


if __name__ == '__main__': unittest.main()
