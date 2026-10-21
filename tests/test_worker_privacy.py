import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_public_metadata import inspect_entry, private_identifiers
from worker_identity import matches_worker, require_worker, ssh_arguments


class WorkerPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / 'worker.json'
        self.config.write_text(json.dumps({'expected_hostname': 'private-unit-host'}))
        self.environment = patch.dict(os.environ, {'FFNDS_WORKER_CONFIG': str(self.config),
            'FFNDS_SSH_CONFIG': str(self.config.with_name('ssh_config'))})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_wrong_host_is_rejected(self):
        with patch('worker_identity.socket.gethostname', return_value='another-host'):
            with self.assertRaises(RuntimeError):
                require_worker()

    def test_expected_host_and_private_manifest_remain_accepted(self):
        with patch('worker_identity.socket.gethostname', return_value='private-unit-host.example'):
            require_worker()
            self.assertTrue(matches_worker('private-unit-host'))
            self.assertTrue(matches_worker('worker'))
            self.assertFalse(matches_worker('another-host'))

    def test_public_alias_does_not_bypass_actual_host_guard(self):
        with patch('worker_identity.socket.gethostname', return_value='another-host'):
            with self.assertRaises(RuntimeError):
                matches_worker('worker')

    def test_missing_private_config_fails_closed(self):
        self.config.unlink()
        with self.assertRaises(FileNotFoundError):
            require_worker()

    def test_scan_checks_case_insensitive_contents_and_paths(self):
        forbidden = private_identifiers()
        self.assertEqual(inspect_entry('report.json', b'PRIVATE-UNIT-HOST', forbidden), 1)
        self.assertEqual(inspect_entry('private-unit-host/report.json', b'{}', forbidden), 1)
        self.assertEqual(inspect_entry('worker/report.json', b'{"host":"worker"}', forbidden), 0)

    def test_private_ssh_destination_is_also_checked(self):
        self.config.with_name('ssh_config').write_text('Host ffnds-worker\n    HostName 192.0.2.77\n')
        self.assertIn(b'192.0.2.77', private_identifiers())

    def test_ssh_uses_only_neutral_alias_and_private_config(self):
        argv = ssh_arguments()
        self.assertEqual(argv[-1], 'ffnds-worker')
        self.assertEqual(argv[1:3], ['-F', str(self.config.with_name('ssh_config'))])
        self.assertNotIn('private-unit-host', argv)
