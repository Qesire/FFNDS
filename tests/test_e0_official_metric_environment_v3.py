"""Software guard/lock tests only; no Torch/metric imports or model execution."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import subprocess
import contextlib
import errno
import threading

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('metric_environment', ROOT / 'tools/e0_official_metric_environment_v3.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MetricEnvironmentGuards(unittest.TestCase):
    def test_scope_cache_directories_exist_before_child_and_all_temp_variables_are_local(self):
        with tempfile.TemporaryDirectory() as temp:
            hf = Path(temp)
            scope = hf / 'scope'
            scope.mkdir()
            env, paths = module.scope_environment(scope, hf, {'HF_relative_root': 'original_assets'})
            required = ['TMPDIR', 'TEMP', 'TMP', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_COMPILE_DEBUG_DIR']
            for key in required:
                self.assertTrue(paths[key].is_dir())
                self.assertTrue(Path(env[key]).is_relative_to(scope))
            self.assertEqual(env['TMPDIR'], env['TEMP'])
            self.assertEqual(env['TMPDIR'], env['TMP'])
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)

    def test_real_isolated_child_tempfile_probe_succeeds_under_the_write_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            hf = Path(temp)
            scope = hf / 'scope'
            scope.mkdir()
            env, paths = module.scope_environment(scope, hf, {'HF_relative_root': 'original_assets'})
            program = '''
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('guarded_environment',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
scope=Path(sys.argv[2]);result={'denied_attempts':[]}
sys.addaudithook(m.import_audit_guard(scope,result)[1])
import tempfile
with tempfile.NamedTemporaryFile() as handle:
    name=handle.name
print(json.dumps({'tempdir':tempfile.gettempdir(),'created':name,'denied':result['denied_attempts']}))
'''
            completed = subprocess.run([sys.executable, '-I', '-S', '-c', program,
                str(ROOT / 'tools/e0_official_metric_environment_v3.py'), str(scope)],
                env=env, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout)
            self.assertEqual(Path(result['tempdir']), paths['TMPDIR'])
            self.assertTrue(Path(result['created']).is_relative_to(paths['TMPDIR']))
            self.assertEqual(result['denied'], [])

    def test_reused_site_write_stays_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            hf = Path(temp)
            scope = hf / 'v3'
            scope.mkdir()
            old_site = hf / 'v2/site'
            old_site.mkdir(parents=True)
            _, audit = module.import_audit_guard(scope, {'denied_attempts': []})
            with self.assertRaisesRegex(RuntimeError, 'WRITE_OUTSIDE_SCOPE'):
                audit('open', (str(old_site / 'module.py'), 'w', os.O_WRONLY))
            self.assertFalse((old_site / 'module.py').exists())

    def _real_removal_race(self, remove_before_child_scan):
        """A real thread removes real files; no OSError is injected by a mock."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'stable').write_bytes(b'stable-data')
            volatile = root / '.uv-temporary'
            volatile.mkdir()
            (volatile / 'payload').write_bytes(b'volatile-data')
            reached = threading.Event()
            removed = threading.Event()
            errors = []

            def remover():
                try:
                    if not reached.wait(3):
                        raise AssertionError('Accounting did not reach the race point')
                    (volatile / 'payload').unlink()
                    volatile.rmdir()
                except BaseException as error:
                    errors.append(error)
                finally:
                    removed.set()

            real_scandir = os.scandir

            @contextlib.contextmanager
            def coordinated_scandir(path):
                if remove_before_child_scan and Path(path) == volatile:
                    reached.set()
                    if not removed.wait(3):
                        raise AssertionError('Real remover did not finish')
                    # This call raises real kernel ENOENT after directory removal.
                with real_scandir(path) as entries:
                    def coordinate_entries():
                        for entry in entries:
                            if not remove_before_child_scan and entry.name == volatile.name:
                                reached.set()
                                if not removed.wait(3):
                                    raise AssertionError('Real remover did not finish')
                                # The subsequent real DirEntry.stat raises ENOENT.
                            yield entry
                    yield coordinate_entries()

            thread = threading.Thread(target=remover)
            thread.start()
            missing = []
            try:
                with mock.patch.object(module.os, 'scandir', side_effect=coordinated_scandir):
                    total = module.tree_bytes(root, missing)
            finally:
                thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(total, len(b'stable-data'))
            self.assertTrue(missing, 'The accounting traversal must observe a real ENOENT')
            self.assertTrue(any(volatile.name in name for name in missing))

    def test_real_directory_removal_between_queue_and_child_scandir(self):
        self._real_removal_race(remove_before_child_scan=True)

    def test_real_entry_removal_between_scandir_and_lstat(self):
        self._real_removal_race(remove_before_child_scan=False)

    def test_permission_and_other_accounting_errors_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for code, error_type in [(errno.EACCES, PermissionError),
                                     (errno.EIO, OSError), (errno.ENOTDIR, NotADirectoryError)]:
                with self.subTest(errno=code):
                    error = error_type(code, 'preserve non-ENOENT error', str(root))
                    with mock.patch.object(module.os, 'scandir', side_effect=error):
                        with self.assertRaises(OSError) as observed:
                            module.tree_bytes(root)
                    self.assertEqual(observed.exception.errno, code)

    @unittest.skipIf(os.geteuid() == 0, 'Root bypasses filesystem permission bits; injected EACCES is tested above')
    def test_actual_unreadable_directory_permission_error_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blocked = root / 'blocked'
            blocked.mkdir()
            (blocked / 'payload').write_bytes(b'x')
            blocked.chmod(0)
            try:
                with self.assertRaises(PermissionError):
                    module.tree_bytes(root)
            finally:
                blocked.chmod(0o700)

    def test_shared_symlinks_are_excluded_from_new_byte_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'new'
            root.mkdir()
            shared = Path(temp) / 'shared'
            shared.mkdir()
            (shared / 'large').write_bytes(b'x' * 4096)
            (root / 'own').write_bytes(b'abc')
            (root / 'runtime').symlink_to(shared, target_is_directory=True)
            (root / 'borrowed').symlink_to(shared / 'large')
            self.assertEqual(module.tree_bytes(root), 3)


    def test_verified_helpers_load_after_runtime_paths_and_restore_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper_dir = root / 'tools'
            helper_dir.mkdir()
            (helper_dir / 'rollout_contract.py').write_text("marker='verified helper'\n")
            (helper_dir / 'offline_i2v_metric_loaders.py').write_text(
                'import sys\nobserved_path=list(sys.path)\nfrom rollout_contract import marker\n')
            plan = {'helper_sha256': {relative: module.sha(root / relative) for relative in module.BOUND_HELPERS}}
            before = list(sys.path)
            with mock.patch.object(module, 'ROOT', root), mock.patch.dict(sys.modules):
                for relative in module.BOUND_HELPERS:
                    sys.modules.pop(Path(relative).stem, None)
                helper = module.load_bound_helper(plan)
                self.assertEqual(helper.marker, 'verified helper')
                self.assertEqual(helper.observed_path[:-1], before)
                self.assertEqual(helper.observed_path[-1], str(helper_dir))
            self.assertEqual(sys.path, before)

    def test_changed_or_missing_helper_binding_fails_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'tools').mkdir()
            for relative in module.BOUND_HELPERS:
                (root / relative).write_text('raise RuntimeError("must not import")\n')
            plan = {'helper_sha256': {relative: module.sha(root / relative) for relative in module.BOUND_HELPERS}}
            before = list(sys.path)
            with mock.patch.object(module, 'ROOT', root):
                broken = {'helper_sha256': dict(plan['helper_sha256'])}
                broken['helper_sha256'].pop(module.BOUND_HELPERS[0])
                with self.assertRaisesRegex(RuntimeError, 'Exact repository helper bindings'):
                    module.load_bound_helper(broken)
                (root / module.BOUND_HELPERS[1]).write_text('changed=True\n')
                with self.assertRaisesRegex(RuntimeError, 'Bound repository helper changed'):
                    module.load_bound_helper(plan)
            self.assertEqual(sys.path, before)

    def test_shadowing_helper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ('tools', 'unbound'):
                (root / directory).mkdir()
            for relative in module.BOUND_HELPERS:
                (root / relative).write_text('marker=True\n')
            (root / 'unbound/offline_i2v_metric_loaders.py').write_text('raise AssertionError("must not execute")\n')
            plan = {'helper_sha256': {relative: module.sha(root / relative) for relative in module.BOUND_HELPERS}}
            original = list(sys.path)
            try:
                sys.path.insert(0, str(root / 'unbound'))
                before = list(sys.path)
                with mock.patch.object(module, 'ROOT', root), mock.patch.dict(sys.modules):
                    for relative in module.BOUND_HELPERS:
                        sys.modules.pop(Path(relative).stem, None)
                    with self.assertRaisesRegex(RuntimeError, 'Bound helper import origin differs'):
                        module.load_bound_helper(plan)
                self.assertEqual(sys.path, before)
            finally:
                sys.path[:] = original

    def test_unregistered_run_creates_no_cache_or_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            hf = Path(temp) / 'hf'
            with mock.patch.object(module, 'authorized', side_effect=RuntimeError('not registered')):
                with self.assertRaisesRegex(RuntimeError, 'not registered'):
                    module.run(argparse.Namespace(protocol='unregistered.json', hf_root=hf))
            self.assertFalse(hf.exists())

    def test_guard_resolves_symlink_before_write_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            allowed = base / 'scope'
            allowed.mkdir()
            shared = base / 'project_runtime'
            shared.mkdir()
            (allowed / 'torch').symlink_to(shared, target_is_directory=True)
            result = {'denied_attempts': []}
            _, audit = module.import_audit_guard(allowed, result)
            with self.assertRaisesRegex(RuntimeError, 'WRITE_OUTSIDE_SCOPE'):
                audit('open', (str(allowed / 'torch/version.py'), 'w', os.O_WRONLY))
            self.assertEqual(result['denied_attempts'], ['WRITE_OUTSIDE_SCOPE'])
            self.assertEqual(list(shared.iterdir()), [])

    def test_guard_blocks_checkpoint_read_network_and_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = Path(temp)
            for event, args, message in [
                ('open', (str(scope / 'weights.safetensors'), 'rb', 0), 'CHECKPOINT_FILE_READ'),
                ('open', (str(scope / 'weights.pt'), None, os.O_RDONLY), 'CHECKPOINT_FILE_READ'),
                ('socket.__new__', (), 'NETWORK_OR_SUBPROCESS'),
                ('subprocess.Popen', ('python', [], None, None), 'NETWORK_OR_SUBPROCESS'),
                ('os.mkdir', ('/outside-scope', 0o755, -1), 'FILESYSTEM_MUTATION_OUTSIDE_SCOPE'),
            ]:
                with self.subTest(event=event, args=args):
                    _, audit = module.import_audit_guard(scope, {'denied_attempts': []})
                    with self.assertRaisesRegex(RuntimeError, message):
                        audit(event, args)

    def test_guard_allows_source_read_and_scope_receipt_write(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = Path(temp)
            result = {'denied_attempts': []}
            _, audit = module.import_audit_guard(scope, result)
            audit('open', (str(ROOT / 'tools/e0_official_metric_environment_v3.py'), 'r', 0))
            audit('open', (str(scope / 'RUN.json'), 'w', os.O_WRONLY | os.O_CREAT))
            self.assertEqual(result['denied_attempts'], [])



    def test_full_lock_closure_and_artifact_hashes_are_present(self):
        lock = json.loads((ROOT / 'config/e0_official_metric_environment_lock_20261026_v1.json').read_text())
        module.validate_closure(lock['artifacts'], lock['shared_runtime'])
        self.assertEqual(len(lock['artifacts']), 91)
        self.assertEqual(sum(row['kind'] == 'source' for row in lock['artifacts']), 7)
        for row in lock['artifacts']:
            self.assertRegex(row['sha256'], r'^[a-f0-9]{64}$')
            self.assertTrue(row['url'].startswith('https://files.pythonhosted.org/'))
            if row['name'] == 'vbench':
                self.assertEqual(row['kind'], 'wheel')
                self.assertEqual(row['sha256'], '13266ed4715d91a7669ab58484b9e010bc91bb96e129d414fb512d4937890023')

    def test_missing_dependency_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'Unclosed runtime dependency'):
            module.validate_closure([{'name': 'metric', 'version': '1', 'requires_dist': ['absent>=1']}],
                                    {'distributions': []})


if __name__ == '__main__':
    unittest.main()
