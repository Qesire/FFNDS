"""Independent receipt identity/inventory checks beyond the unchanged v1 gate."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ffnds_paired_video_strict_gate import inspect, model_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrictGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = {'assets': {'checkpoint_sha256': {'high': 'fixture'}, 'cache_manifest_sha256': 'cache'},
                    'bank_RUN_sha256': 'bank', 'source_id': 'fixture-source', 'seed': 42,
                    'pair': {'modes': ['dense', 'candidate']}, 'videos': {}}
        for branch in ['teacher', 'student']:
            path = self.root/(branch+'.mp4'); path.write_bytes(('SYNTHETIC_'+branch).encode())
            self.raw['videos'][branch] = {'path': path.name, 'sha256': sha(path)}
        self.receipt = self.root/'NATIVE_RUN.json'; self.write(self.receipt, self.raw)
        self.bundle = {'schema': 'ffnds.video.bundle.v1', 'expected_video_count': 2, 'videos': []}
        for branch, kind in zip(['teacher', 'student'], ['dense', 'candidate']):
            report = {'identity': {'source_id': 'fixture-source', 'seed': 42, 'model_digest': model_digest(self.raw, kind)},
                      'reference': {'sha256': self.raw['videos']['teacher']['sha256']},
                      'reference_identity': {'source_id': 'fixture-source', 'seed': 42, 'model_digest': model_digest(self.raw, 'dense')}}
            path = self.root/(branch+'_metrics.json'); self.write(path, report)
            entry = {'video_path': branch+'.mp4', 'video_sha256': self.raw['videos'][branch]['sha256'],
                     'run_receipt_path': self.receipt.name, 'run_sha256': sha(self.receipt), 'metric_report_path': path.name}
            entry.update(reference_video_path='teacher.mp4', reference_video_sha256=self.raw['videos']['teacher']['sha256'],
                         reference_run_receipt_path=self.receipt.name)
            self.bundle['videos'].append(entry)
        self.core = self.root/'BUNDLE.json'; self.strict = self.root/'STRICT_BUNDLE.json'
        self.refresh()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def refresh(self):
        self.write(self.core, self.bundle)
        self.write(self.strict, {'schema': 'ffnds.paired.video.strict.v1', 'core_bundle_path': self.core.name,
                                'core_bundle_sha256': sha(self.core),
                                'native_generation_receipts': [{'path': self.receipt.name, 'sha256': sha(self.receipt)}]})

    def audit(self):
        # Assume the old core gate accepts: these tests target the two extra provenance obligations.
        with patch('ffnds_paired_video_strict_gate.core_inspect', return_value=[]):
            return inspect(self.strict)

    def test_complete_inventory_and_real_dense_identity_pass(self):
        self.assertEqual(self.audit(), [])

    def test_inheriting_candidate_digest_cannot_prove_dense_identity(self):
        path = self.root/'student_metrics.json'; report = json.loads(path.read_text())
        report['reference_identity']['model_digest'] = model_digest(self.raw, 'candidate')
        self.write(path, report)
        self.assertTrue(any('Dense reference/report identity' in e for e in self.audit()))

    def test_lowering_expected_count_cannot_hide_a_generated_clip(self):
        self.bundle['videos'].pop(); self.bundle['expected_video_count'] = 1; self.refresh()
        self.assertTrue(any('omits' in e for e in self.audit()))

    def test_unregistered_actual_clip_is_not_hidden_by_the_manifest(self):
        (self.root/'omitted.mp4').write_bytes(b'SYNTHETIC_EXTRA')
        self.assertTrue(any('unregistered' in e for e in self.audit()))

    def test_missing_authoritative_inventory_is_rejected(self):
        strict = json.loads(self.strict.read_text()); strict['native_generation_receipts'] = []
        self.write(self.strict, strict)
        self.assertTrue(any('independent native generation inventory' in e for e in self.audit()))

    def test_dropping_reference_cannot_bypass_the_paired_identity_obligation(self):
        path = self.root/'student_metrics.json'; report = json.loads(path.read_text())
        report.update(reference=None, reference_identity=None); self.write(path, report)
        self.assertTrue(any('requires an actual dense reference' in e for e in self.audit()))

    def test_native_receipt_identity_and_artifact_hash_are_binding(self):
        path = self.root/'student_metrics.json'; report = json.loads(path.read_text())
        report['identity']['model_digest'] = 'incorrect'; self.write(path, report)
        self.assertTrue(any('Candidate/report identity' in e for e in self.audit()))
        (self.root/'student.mp4').write_bytes(b'CHANGED')
        self.assertTrue(any('artifact digest mismatch' in e for e in self.audit()))

    def test_original_dense_reference_identity_is_derived_from_its_own_assets(self):
        directory = self.root/'historical'; directory.mkdir()
        video = directory/'source00.mp4'; video.write_bytes(b'SYNTHETIC_ORIGINAL_DENSE')
        old = {'mode': 'dense', 'assets': {'checkpoint_sha256': {'high': 'original'},
               'cache_manifest_sha256': 'original-cache'},
               'phases': [{'video': video.name, 'video_sha256': sha(video), 'seed': 42}]}
        receipt = directory/'RUN.json'; self.write(receipt, old)
        path = self.root/'student_metrics.json'; report = json.loads(path.read_text())
        report['reference_identity']['model_digest'] = model_digest(old, 'dense'); self.write(path, report)
        self.bundle['videos'][1].update(reference_video_path=str(video), reference_video_sha256=sha(video),
                                       reference_run_receipt_path=str(receipt))
        self.refresh(); self.assertEqual(self.audit(), [])
        report['reference_identity']['model_digest'] = model_digest(self.raw, 'dense'); self.write(path, report)
        self.assertTrue(any('Dense reference/report identity' in e for e in self.audit()))


if __name__ == '__main__':
    unittest.main()
