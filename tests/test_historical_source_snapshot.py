import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import historical_source_snapshot as history


class HistoricalSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'public'
        self.snapshot = Path(self.directory.name) / 'private'
        (self.root / 'tools').mkdir(parents=True)
        (self.snapshot / 'worktree/tools').mkdir(parents=True)
        self.original = b'LABEL="PRIVATE-UNIT-HOST_RUN"\ndef calculate(x):\n    return x * 2\n'
        self.sha = hashlib.sha256(self.original).hexdigest()
        (self.snapshot / 'worktree/tools/producer.py').write_bytes(self.original)
        (self.snapshot / 'manifest.json').write_text(json.dumps([{'path':'tools/producer.py','sha256':self.sha}]))
        self.current = self.root / 'tools/producer.py'
        self.current.write_text('LABEL="WORKER_RUN"\ndef calculate(x):\n    return x * 2\n')
        config = Path(self.directory.name) / 'worker.json'
        config.write_text(json.dumps({'expected_hostname':'private-unit-host','source_snapshot':str(self.snapshot)}))
        for context in [patch.object(history, 'ROOT', self.root), patch.object(history, 'private_config_path', return_value=config)]:
            context.start()
            self.addCleanup(context.stop)

    def test_identifier_only_change_keeps_explicit_original_binding(self):
        resolver = history.HistoricalSources()
        original = resolver.original(self.current, self.sha)
        self.assertEqual(original.read_bytes(), self.original)
        self.assertEqual(resolver.receipts[0]['original_sha256'], self.sha)
        self.assertNotEqual(resolver.receipts[0]['current_sha256'], self.sha)

    def test_math_change_is_rejected(self):
        self.current.write_text(self.current.read_text().replace('x * 2', 'x * 3'))
        with self.assertRaises(RuntimeError):
            history.HistoricalSources().original(self.current, self.sha)

    def test_private_snapshot_tampering_is_rejected(self):
        (self.snapshot / 'worktree/tools/producer.py').write_bytes(self.original + b'# altered\n')
        with self.assertRaises(RuntimeError):
            history.HistoricalSources().original(self.current, self.sha)

    def test_unregistered_expected_hash_is_rejected(self):
        with self.assertRaises(RuntimeError):
            history.HistoricalSources().original(self.current, '0' * 64)
