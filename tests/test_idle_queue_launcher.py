"""Exercise idle waiting and inherited lock with synthetic GPU inventory."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IdleQueueLauncher(unittest.TestCase):
    def run_fixture(self, child_exit, uuid_mode=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('scripts', 'tools', 'config', '.venv/bin', 'bin', 'locks'):
                (root / name).mkdir(parents=True)
            name = 'worker_wait_uuid_idle_exec.sh' if uuid_mode else 'worker_wait_idle_exec.sh'
            source = (ROOT / 'scripts' / name).read_text()
            # Isolate only the production lock path; keep launch logic exact.
            self.assertEqual(source.count('/tmp/ffnds-'), 1)
            script = root / 'scripts' / name
            script.write_text(source.replace('/tmp/ffnds-', str(root / 'locks/ffnds-')))
            (root / '.venv/bin/python').symlink_to(sys.executable)
            (root / 'tools/worker_identity.py').write_text('raise SystemExit(0)\n')
            (root / 'config/worker.env').write_text('export PATH=' + shlex.quote(str(root / 'bin') + ':/usr/bin:/bin') + '\n')
            commands = {
                'nvidia-smi': '''import os,sys
from pathlib import Path
r=Path(os.environ['FFNDS_QUEUE_FIXTURE_ROOT'])
if '--query-gpu=index' in sys.argv: print('0\\n1')
elif '--query-gpu=uuid' in sys.argv: print('GPU-synthetic-'+sys.argv[sys.argv.index('-i')+1])
elif '-i' in sys.argv:
 gpu=sys.argv[sys.argv.index('-i')+1]
 if gpu=='0' or not (r/'released').exists(): print('123')
else: raise SystemExit(2)
''',
                'sleep': '''import json,os
from pathlib import Path
r=Path(os.environ['FFNDS_QUEUE_FIXTURE_ROOT'])
assert json.loads((r/'QUEUE.json').read_text())['status']=='WAITING_FOR_IDLE_GPU_NO_MODEL_STARTED'
assert not (r/'child_count').exists()
(r/'released').write_text('one formerly occupied GPU released')
'''}
            for name, body in commands.items():
                p = root / 'bin' / name; p.write_text('#!/usr/bin/env python3\n' + body); p.chmod(0o755)
            child = '''import fcntl,json,os,sys
from pathlib import Path
r=Path(os.environ['FFNDS_QUEUE_FIXTURE_ROOT'])
uuid_mode=os.environ['FFNDS_QUEUE_UUID']=='1'
assert os.environ['CUDA_VISIBLE_DEVICES']==('GPU-synthetic-1' if uuid_mode else '1')
if uuid_mode: assert os.environ['FFNDS_PHYSICAL_GPU']=='1'
assert (r/'released').exists()
assert json.loads((r/'QUEUE.json').read_text())['selected_GPU']==1
os.fstat(9)
p=r/'locks'/('ffnds-'+os.environ['USER']+'-gpu1.lock')
with p.open('a') as lock:
 try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: pass
 else: raise AssertionError('GPU lock not retained through exec')
assert not (r/'child_count').exists()
(r/'child_count').write_text('1')
raise SystemExit(int(sys.argv[1]))
'''
            env = dict(os.environ, FFNDS_QUEUE_FIXTURE_ROOT=str(root), FFNDS_QUEUE_UUID=str(int(uuid_mode)))
            result = subprocess.run(['bash', str(script), str(root / 'QUEUE.json'), sys.executable,
                                     '-c', child, str(child_exit)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, child_exit, result.stderr)
            self.assertEqual((root / 'child_count').read_text(), '1')
            expected = 'IDLE_GPU_UUID_LOCKED_STARTING_SINGLE_ATTEMPT' if uuid_mode else 'IDLE_GPU_LOCKED_STARTING_SINGLE_ATTEMPT'
            self.assertEqual(json.loads((root / 'QUEUE.json').read_text())['status'], expected)

    def test_wait_for_unoccupied_card_and_retain_lock(self):
        self.run_fixture(0)

    def test_child_failure_is_not_retried(self):
        self.run_fixture(7)

    def test_UUID_binding_waits_and_keeps_the_same_physical_lock(self):
        self.run_fixture(0, uuid_mode=True)

    def test_UUID_child_failure_is_not_retried(self):
        self.run_fixture(7, uuid_mode=True)


if __name__ == '__main__': unittest.main()
