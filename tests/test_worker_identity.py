"""An explicit executor change preserves historical identity without widening hosts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import worker_identity as identity
import check_public_metadata as metadata


class WorkerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / 'worker.json'
        self.config.write_text(json.dumps({'expected_hostname': 'origin-worker'}))
        patch = mock.patch.object(identity, 'private_config_path', return_value=self.config)
        patch.start()
        self.addCleanup(patch.stop)

    def test_default_still_requires_original_executor(self):
        with mock.patch.object(identity.socket, 'gethostname', return_value='origin-worker'):
            identity.require_worker()
        with mock.patch.object(identity.socket, 'gethostname', return_value='another-worker'):
            with self.assertRaises(RuntimeError):
                identity.require_worker()

    def test_explicit_migration_separates_origin_from_execution(self):
        self.config.write_text(json.dumps({'expected_hostname': 'origin-worker', 'execution_hostname': 'target-worker'}))
        self.assertEqual(identity.expected_hostname(), 'origin-worker')
        with mock.patch.object(identity.socket, 'gethostname', return_value='target-worker.example'):
            identity.require_worker()
            self.assertTrue(identity.matches_worker('origin-worker'))
            self.assertTrue(identity.matches_worker('target-worker'))
            self.assertTrue(identity.matches_worker('worker'))
            self.assertFalse(identity.matches_worker('unapproved-worker'))
        with mock.patch.object(identity.socket, 'gethostname', return_value='origin-worker'):
            with self.assertRaises(RuntimeError):
                identity.require_worker()

    def test_socket_cache_distinguishes_actual_hosts(self):
        self.assertIn('ControlPath=/tmp/ffnds-worker-%h-%r-%p', identity.ssh_arguments())

    def test_public_scan_covers_both_private_workers(self):
        self.config.write_text(json.dumps({'expected_hostname': 'origin-worker', 'execution_hostname': 'target-worker'}))
        with mock.patch.object(metadata, 'private_config_path', return_value=self.config):
            identifiers = metadata.private_identifiers()
        self.assertIn(b'origin-worker', identifiers)
        self.assertIn(b'target-worker', identifiers)


if __name__ == '__main__':
    unittest.main()
