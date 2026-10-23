"""Regression for the native import path cleared by the previous launcher."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelEnvironment(unittest.TestCase):
    def run_wrapper(self, wrapper, visibility):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('scripts', 'config', 'tools', '.venv/bin', 'native_sources'):
                (root / name).mkdir(parents=True)
            shutil.copy(ROOT / 'scripts' / wrapper, root / 'scripts' / wrapper)
            fake = root / '.venv/bin/python'
            fake.write_text('#!/bin/sh\nexit 0\n'); fake.chmod(0o755)
            (root / 'config/worker.env').write_text(
                'export PYTHONPATH="$FFNDS_ROOT/native_sources"\n'
                'export CUDA_VISIBLE_DEVICES="must-not-overwrite"\n'
                'export SKIP_PLATFORM_CHECK=True\n')
            (root / 'native_sources/native_probe.py').write_text('value=1\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visibility, PYTHONPATH='old-poison')
            env.pop('PYTHONHOME', None)
            child = ('import importlib.util,json,os;'
                     'print(json.dumps({"visible":os.environ.get("CUDA_VISIBLE_DEVICES"),'
                     '"native":importlib.util.find_spec("native_probe") is not None,'
                     '"skip":os.environ.get("SKIP_PLATFORM_CHECK")}))')
            output = subprocess.check_output(['bash', str(root / 'scripts' / wrapper), sys.executable, '-c', child], env=env, text=True)
            return json.loads(output)

    def test_uuid_preserved_with_real_native_import_path(self):
        self.assertEqual(self.run_wrapper('worker_c2_model_env.sh', 'GPU-CPU-TEST'),
                         {'visible': 'GPU-CPU-TEST', 'native': True, 'skip': None})

    def test_hidden_gpu_stays_hidden_in_cpu_proof(self):
        self.assertEqual(self.run_wrapper('worker_c2_model_env.sh', ''),
                         {'visible': '', 'native': True, 'skip': None})

    def test_previous_wrapper_reproduces_missing_project_module(self):
        self.assertFalse(self.run_wrapper('worker_eval_env.sh', '')['native'])

    def test_closed_scope_rejected_before_torch_import(self):
        code = ('import sys;sys.path.insert(0,"tools");'
                'from c2_environment_contract import require_registration\n'
                'try: require_registration()\n'
                'except RuntimeError: assert "torch" not in sys.modules\n'
                'else: raise AssertionError("unregistered retry accepted")')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
