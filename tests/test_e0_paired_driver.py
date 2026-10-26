"""Finite ledger and failed-artifact software checks; no model/GPU calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_e0_paired_full_depth import (close_attempt, collect_failed_video_evidence,
                                    execute, historical_branch, persist_observation, preflight,
                                    reserve_attempt, sha256, write_json)


class PairedDriverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pair = dict(id='dd', modes=['dense', 'dense'], seed=42, source_id='SYNTHETIC')

    def test_no_authority_or_changed_output_root_cannot_start_a_model(self):
        with self.assertRaisesRegex(RuntimeError, 'separate explicit authorization'):
            execute(None, None, None, {'model_scope_authorized': False})
        with self.assertRaisesRegex(ValueError, 'canonical study ledger'):
            execute(SimpleNamespace(study_root=self.root/'other'),
                    {'canonical_study_root': str(self.root/'canonical')}, [], {'model_scope_authorized': True})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_source_maps_do_not_certify_complete_binding(self):
        protocol = self.root/'plan.json'
        protocol.write_text(json.dumps({'driver_sources_sha256': {}, 'native_source_sha256': {}}))
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            preflight(protocol, self.root/'missing-bank')

    def test_failed_or_running_attempt_cannot_be_retried_or_followed(self):
        plan = {'id': 'SYNTHETIC', 'pairs': [{'id': k} for k in ['dd', 'dc0', 'dc12']]}
        reserve_attempt(self.root, plan, 'fixture', {'id': 'dd'})
        with self.assertRaises(RuntimeError):
            reserve_attempt(self.root, plan, 'fixture', {'id': 'dc0'})
        receipt = self.root/'RUN.json'; receipt.write_text('{}')
        close_attempt(self.root, 'dd', 'PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY', receipt)
        reserve_attempt(self.root, plan, 'fixture', {'id': 'dc0'})
        close_attempt(self.root, 'dc0', 'FAILED_PRESERVED_NO_RETRY', receipt)
        for pair in ['dd', 'dc0', 'dc12']:
            with self.assertRaises(RuntimeError):
                reserve_attempt(self.root, plan, 'fixture', {'id': pair})
        state = json.loads((self.root/'STUDY.json').read_text())
        self.assertEqual(len(state['attempts']), 2)

    def test_generated_failure_clip_is_evaluated_and_missing_clip_explicit(self):
        (self.root/'student.mp4').write_bytes(b'SYNTHETIC_NOT_A_VIDEO')
        report = {'metrics': {'artifact': {'decoded_frames': 81}}, 'quality_accepted': False}
        with patch('ffnds_video_metrics.evaluate', return_value=report) as evaluate, \
                patch('ffnds_video_metrics.validate', return_value=['cost unavailable after interruption']):
            result = collect_failed_video_evidence(self.root, {'status': 'FAILED'}, self.pair, {'prompt': 'fixture'}, 'fixture')
        self.assertEqual(evaluate.call_count, 1)
        evidence = json.loads((self.root/result['path']).read_text())
        self.assertEqual([v['status'] for v in evidence['videos']],
                         ['MISSING_NOT_A_GENERATED_VIDEO', 'FULL_FRAME_DIAGNOSTICS_OF_FAILED_NATIVE'])
        self.assertFalse(evidence['core_gate_pass'])
        failed_report = json.loads((self.root/'student_failed_native_metrics.json').read_text())
        self.assertEqual(failed_report['metadata_failures'], ['cost unavailable after interruption'])

    def test_corrupt_failure_clip_keeps_hashed_decode_failure_record(self):
        (self.root/'teacher.mp4').write_bytes(b'SYNTHETIC_BROKEN')
        with patch('ffnds_video_metrics.evaluate', side_effect=ValueError('truncated fixture')):
            collect_failed_video_evidence(self.root, {'status': 'FAILED'}, self.pair, {}, 'fixture')
        evidence = json.loads((self.root/'FAILED_VIDEO_EVIDENCE.json').read_text())
        self.assertEqual(evidence['videos'][0]['status'], 'DECODE_OR_EVALUATION_FAILED')
        self.assertEqual(len(evidence['videos'][0]['report_sha256']), 64)
        self.assertFalse(evidence['quality_accepted'])

    def test_reference_uses_derived_root_and_checks_real_video_and_seed(self):
        out = self.root/'ffnds-derived'/'old'; out.mkdir(parents=True)
        (out/'video.mp4').write_bytes(b'SYNTHETIC_VIDEO')
        raw = {'phases': [{'source_index': 0, 'seed': 42, 'video': 'video.mp4',
                           'video_sha256': sha256(out/'video.mp4')}]}
        write_json(out/'RUN.json', raw)
        previous = {'runs': [{'mode': 'dense', 'raw_binding': {
            'HF_relative': 'old/RUN.json', 'sha256': sha256(out/'RUN.json')}}]}
        write_json(self.root/'previous.json', previous)
        with patch('run_e0_paired_full_depth.ROOT', self.root):
            plan = {'previous_result': 'previous.json'}
            pair = {'source_index': 0, 'seed': 42}
            path, _, phase = historical_branch(plan, self.root, pair, 'dense')
            self.assertEqual(path, out/'RUN.json'); self.assertEqual(phase['seed'], 42)
            with self.assertRaisesRegex(ValueError, 'source/seed'):
                historical_branch(plan, self.root, {**pair, 'seed': 54}, 'dense')
            (out/'video.mp4').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(ValueError, 'video changed'):
                historical_branch(plan, self.root, pair, 'dense')

    def test_reference_paths_cannot_escape_derived_root(self):
        write_json(self.root/'previous.json', {'runs': [{'mode': 'dense', 'raw_binding': {
            'HF_relative': '../outside.json', 'sha256': 'unused'}}]})
        with patch('run_e0_paired_full_depth.ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'escaped derived root'):
                historical_branch({'previous_result': 'previous.json'}, self.root, {}, 'dense')

    def test_partial_trace_survives_failure_without_completing_tail_site(self):
        trace = SimpleNamespace(source=0, seed=42, provenance={'fixture': True},
                                sites={(0, 0): {'step': 0, 'block': 0}, (0, 1): {'step': 0, 'block': 1}}, steps={})
        controller = SimpleNamespace(trace=trace, completed_trace=None,
                                     trajectory={'teacher': [], 'student': []})
        result = persist_observation(self.root, controller, SimpleNamespace(counts={'completed_sites': 1}))
        saved = json.loads((self.root/'TRACE.json').read_text())
        observed = json.loads((self.root/result['path']).read_text())
        self.assertEqual(saved['incomplete_or_unverified_tail_sites'], [[0, 1]])
        self.assertEqual(observed['recorded_site_geometries'], 2)
        self.assertEqual(observed['fully_completed_sites'], 1)
        self.assertEqual(result['sha256'], sha256(self.root/'OBSERVATION.json'))
        controller.completed_trace = {'sites': list(trace.sites.values()), 'steps': []}
        persist_observation(self.root, controller)
        self.assertEqual(json.loads((self.root/'OBSERVATION.json').read_text())['fully_completed_sites'], 2)


if __name__ == '__main__':
    unittest.main()
