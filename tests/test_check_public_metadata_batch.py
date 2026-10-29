import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'tools/check_public_metadata.py'
NEW = ROOT / 'tools/check_public_metadata_batch.py'


class BatchIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.config = self.root / 'worker.json'
        self.ssh = self.root / 'ssh_config'
        self.config.write_text(json.dumps({'expected_hostname':'redwood.invalid',
            'execution_hostname':'cedar.invalid','additional_private_hostnames':['willow.invalid']}))
        self.ssh.write_text('Host one\n  HostName 192.0.2.17\nHost two\n  HostName 192.0.2.18\n')
        self.env = dict(os.environ, FFNDS_WORKER_CONFIG=str(self.config), FFNDS_SSH_CONFIG=str(self.ssh),
                        GIT_CONFIG_NOSYSTEM='1', HOME=str(self.root), LC_ALL='C.UTF-8')
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')

    def git(self, *args, input=None, check=True):
        return subprocess.run(['git', *args], cwd=self.repo, env=self.env, input=input,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def put(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def scan(self, script):
        return subprocess.run([os.sys.executable, str(script), '--staged'], cwd=self.repo,
                              env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def compare(self):
        old, new = self.scan(OLD), self.scan(NEW)
        self.assertEqual(old.returncode, new.returncode, (old.stderr, new.stderr))
        self.assertEqual(json.loads(old.stdout), json.loads(new.stdout))
        result = json.loads(new.stdout)
        self.assertEqual(result['private_rules'], 8)
        for forbidden in (b'redwood', b'cedar', b'willow', b'192.0.2.17', b'192.0.2.18'):
            self.assertNotIn(forbidden, new.stdout.lower())
            self.assertNotIn(forbidden, new.stderr.lower())
        return result

    def test_all_index_worktree_staging_binary_unicode_and_large_blob(self):
        self.put('tracked.txt', b'normal')
        self.put('empty', b'')
        self.put('unicode-\u96ea', b'\x00\xffBINARY')
        self.put('line\nbreak', b'public')
        self.put('tab\tname', b'public')
        with open(os.fsencode(self.repo) + b'/invalid-\xff', 'wb') as stream: stream.write(b'public')
        self.put('large', b'X' * (3 << 20))
        self.git('add', '--all')
        self.assertEqual(self.compare()['files_checked'], 7)
        self.git('commit', '-qm', 'baseline')
        self.put('tracked.txt', b'redwood.invalid only in unstaged worktree')
        self.put('added', b'clean staged')
        self.git('add', '--', 'added')
        result = self.compare()
        self.assertEqual(result['files_checked'], 8)
        self.assertEqual(result['status'], 'PASS_PUBLIC_METADATA_SCAN')
        self.put('added', b'192.0.2.17 staged leak')
        self.git('add', '--', 'added')
        self.assertEqual(self.compare()['status'], 'REJECTED_PRIVATE_METADATA')
        self.git('rm', '-q', '-f', '--', 'added')
        self.git('rm', '-q', '--', 'line\nbreak')
        self.assertEqual(self.compare()['files_checked'], 6)

    def test_private_path_is_redacted(self):
        self.put('directory/cedar.invalid/name', b'clean')
        self.git('add', '--all')
        result = self.compare()
        self.assertEqual(result['status'], 'REJECTED_PRIVATE_METADATA')
        self.assertEqual(len(result['violations']), 1)

    def test_intent_to_add_matches_legacy_index(self):
        self.put('intent', b'not yet staged')
        self.git('add', '-N', '--', 'intent')
        self.assertEqual(self.compare()['files_checked'], 1)

    def test_symlink_scans_staged_link_text_not_target_contents(self):
        self.put('target', b'clean')
        (self.repo/'link').symlink_to('redwood.invalid')
        self.git('add', '--all')
        self.assertEqual(self.compare()['status'], 'REJECTED_PRIVATE_METADATA')

    def test_gitlink_commit_uses_legacy_rendering(self):
        self.put('normal', b'clean')
        self.git('add', '--all')
        self.git('commit', '-qm', 'baseline')
        oid = self.git('rev-parse', 'HEAD').stdout.strip().decode()
        self.git('update-index', '--add', '--cacheinfo', f'160000,{oid},linked')
        self.assertEqual(self.compare()['files_checked'], 2)

    def test_unmerged_index_fails_closed(self):
        self.put('conflict', b'base\n'); self.git('add', '--all'); self.git('commit', '-qm', 'base')
        self.git('checkout', '-qb', 'other'); self.put('conflict', b'other\n'); self.git('commit', '-qam', 'other')
        self.git('checkout', '-q', '-'); self.put('conflict', b'main\n'); self.git('commit', '-qam', 'main')
        self.assertNotEqual(self.git('merge', 'other', check=False).returncode, 0)
        old, new = self.scan(OLD), self.scan(NEW)
        self.assertNotEqual(old.returncode, 0); self.assertNotEqual(new.returncode, 0)
        self.assertNotIn(b'conflict', new.stdout.lower())
        self.assertNotIn(b'conflict', new.stderr.lower())


if __name__ == '__main__': unittest.main()
