"""Software-only tests of the draft boundary and exact raw-payload arithmetic."""
import sys
import hashlib
import json
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_chipmunk_student_observation_owner as owner
import e0_chipmunk_student_observation_jobs as jobs

def make_draft(part=0):
    return {'id': owner.sid(part), 'part': part, 'status': 'DRAFT_REVIEW_REQUESTED',
            'assets_finalized': False, 'review': None, 'authors': owner.AUTHORS,
            'resources': owner.RES, 'CPU_preflight_resources': owner.CPU,
            'verification_resources': owner.VERIFY, 'payload_budget': owner.payload_budget(),
            'run_ids': owner.run_ids(part), 'output_HF_relative': owner.outpath(part),
            'limits': owner.limits(part),
            'design': owner.design(part), 'native_group_id': owner.GROUP, 'automatic_retry': False,
            'source_parent': {'parent_protocol': {'sha256': 'x'}},
            'missing_source_files': [], 'limits_status': 'INCOMPLETE_PENDING_DRIVER_SINK_LEDGER_INTERFACE'}

class DraftTests(unittest.TestCase):
    def test_raw_payload_arithmetic(self):
        budget = owner.payload_budget()
        self.assertEqual(budget['sample_and_full_raw_payload_bytes'], 25421840384)
        self.assertFalse(budget['includes_file_headers_metadata_states_videos_journals'])
        self.assertLess(budget['sample_and_full_raw_payload_bytes'], owner.RES['new_HF_bytes'])

    def test_limits_cover_both_observers(self):
        limits = owner.limits(1)
        self.assertEqual(limits['postprocess_actual_bitwise_checks'], 80)
        self.assertEqual(limits['dense_h_Y_bitwise_checks'], 40)
        self.assertEqual(limits['full_shadow_W1'], 40)

    def test_mutated_resource_or_status_is_rejected(self):
        p = make_draft()
        p['resources'] = dict(owner.RES, GPU=2)
        with self.assertRaises(ValueError):
            owner.validate_draft(p)
        p = make_draft()
        p['status'] = 'FROZEN_PENDING_REGISTRATION'
        with self.assertRaises(ValueError):
            owner.validate_draft(p)

    def test_source_and_parent_phase_mapping(self):
        self.assertEqual(owner.PARENT_PARTS, (0, 3))
        self.assertEqual(owner.SOURCES, (898, 979))
        self.assertEqual(owner.MODES, ('h_cache_dense', 'high_h_readonly_p64'))
        with self.assertRaises(ValueError):
            owner.sid(True)

    def test_multiple_tensor_payload_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.safetensors'
            header = {'x': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
                      'y': {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]}}
            encoded = json.dumps(header, separators=(',', ':')).encode()
            x, y = b'1234', b'5678'
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + x + y)
            expected = {'x': {'dtype': 'BF16', 'shape': [2], 'raw_sha256': hashlib.sha256(x).hexdigest()},
                        'y': {'dtype': 'F32', 'shape': [1], 'raw_sha256': hashlib.sha256(y).hexdigest()}}
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(owner.inspect_typed_payload(path, sha, expected)['tensor_count'], 2)
            expected['y']['raw_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                owner.inspect_typed_payload(path, sha, expected)

    def test_reference_checks_require_all_actual_comparisons(self):
        names = {'raw_noise_FP32', 'initial_noise', *(f'flow_step{i}' for i in range(4)),
                 *(f'latent_step{i}' for i in range(4))}
        state = {'sha256': 'x', 'bytes': 1, 'raw_tensor_sha256': 'y', 'shape': [1], 'dtype': 'bfloat16'}
        reference = {'states': {name: dict(state) for name in names}, 'video': {'sha256': 'z', 'bytes': 2}}
        entry = {'dense': reference, 'student': reference}
        phases = [{'mode': mode, **reference} for mode in owner.MODES]
        states = [{'mode': mode, 'state': name, 'equal': True} for mode in owner.MODES for name in names]
        videos = [{'mode': mode, 'equal': True} for mode in owner.MODES]
        self.assertTrue(owner.verify_reference_metadata(entry, phases, states, videos))
        states[-1]['equal'] = False
        with self.assertRaises(ValueError):
            owner.verify_reference_metadata(entry, phases, states, videos)

    def test_sink_with_counted_io_uses_pre_guard_serializer(self):
        import torch
        import safetensors.torch as safe
        from chipmunk_student_observation_sink import Sink
        raw_save = safe.save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counter = owner.io.WrittenBytes(root, 4 << 20)
            with owner.io.counted_io(counter):
                self.assertIsNot(safe.save_file, raw_save)
                sink = Sink(torch=torch, counter=counter, out=root/'observer', hf_root=root,
                            save_file=raw_save, max_bytes=4 << 20, host_guard=lambda n: None,
                            max_host_copy_bytes=1 << 20)
                path=root/'observer'/'sample.safetensors'
                result=sink._tensor_file(path, {'x':torch.tensor([1, 2],dtype=torch.bfloat16)})
                self.assertEqual(result['file']['bytes'], path.stat().st_size)
                self.assertEqual(counter.summary()['unconsumed_reserved_bytes'], 0)
                self.assertEqual(counter.external_file_observations, 1)
                self.assertFalse(path.with_suffix(path.suffix+'.tmp').exists())

    def test_jobs_have_exact_phase_commands_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); profile=root/'profile'; profile.write_bytes(b'private')
            protocol=root/'protocol'; protocol.write_bytes(b'bound protocol')
            p=make_draft(); p.update(status='FROZEN_PENDING_REGISTRATION', assets_finalized=True,
                                     cpu_affinity=[1,2,3,4], execution_private_worker_config_sha256=hashlib.sha256(b'private').hexdigest())
            paths={kind:root/(kind+'.json') for kind in ('cpu','gpu','verify')}
            with patch.object(owner,'validate',return_value=True):
                specs=jobs.jobs(p,protocol,profile,root,1,'GPU-fixture',paths)
            self.assertEqual(specs['cpu']['external_timeout_seconds'],1020)
            self.assertEqual(specs['gpu']['external_timeout_seconds'],10920)
            self.assertEqual(specs['verify']['external_timeout_seconds'],7320)
            self.assertIn('preflight',specs['cpu']['argv'])
            self.assertIn('run',specs['gpu']['argv'])
            self.assertIn('verify-closed',specs['verify']['argv'])

    def test_observation_manifest_rejects_missing_sites(self):
        phase={'mode':owner.MODES[0], 'observer_manifest':{'sites':40, 'records':{}, 'quality_acceptance':False}}
        with self.assertRaises(ValueError):
            owner.observation_assets([phase])

    def test_video_size_rejected_before_parent_byte_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new.mp4'; path.write_bytes(b'longer')
            with patch.object(owner.parent,'compare_reference',return_value=True) as reader:
                with self.assertRaises(ValueError):
                    owner.compare_video_reference('old.mp4',path,3,'expected','actual')
                reader.assert_not_called()
                self.assertTrue(owner.compare_video_reference('old.mp4',path,6,'expected','actual'))
                reader.assert_called_once_with('old.mp4',path,6,'expected','actual')

    def test_video_holds_match_single_file_limit(self):
        self.assertEqual(owner.RES['video_bytes_max'], 1 << 30)
        self.assertEqual(owner.RES['video_bytes_max'], owner.RES['single_file_bytes_max'])
        self.assertEqual(owner.limits(0)['video_file_hash_reserved_bytes'], 2 << 30)

    def test_partial_video_always_predebits_its_own_hash(self):
        from unittest.mock import Mock
        counter=Mock(); counter.binding.return_value={'bytes':7,'path':'fixture','sha256':'a'*64}
        path=Path('/tmp/partial-video-fixture.mp4')
        report={}; persisted=[]
        owner.observe_failed_video(counter,'second',path,{'second'},report,lambda:persisted.append(dict(report['partial_video_observations'][0])))
        self.assertEqual(persisted[0]['predebited_hash_bytes'],1<<30)
        self.assertEqual(persisted[0]['completed_hash_bytes'],'UNKNOWN')
        self.assertEqual(persisted[-1]['completed_hash_bytes'],7)
        counter.observe_external.assert_called_once_with(path,hold=str(path))
        counter.reset_mock()
        report={}
        owner.observe_failed_video(counter,'first',path,set(),report,lambda:None)
        self.assertFalse(report['partial_video_observations'][0]['inspection_hash_started'])
        self.assertEqual(report['partial_video_observations'][0]['predebited_hash_bytes'],1<<30)
        counter.observe_external.assert_called_once_with(path,hold=str(path))

if __name__ == '__main__':
    unittest.main()
