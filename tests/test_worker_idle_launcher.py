"""Run launcher guards with fake worker/GPU commands; never initialize CUDA."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class IdleLauncher(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ffnds-launcher-')
        self.root=Path(self.temp.name)
        for directory in ('scripts','config','.venv/bin','mockbin'):
            (self.root/directory).mkdir(parents=True)
        shutil.copyfile(ROOT/'scripts/worker_idle_exec.sh',self.root/'scripts/worker_idle_exec.sh')
        self.marker=self.root/'executed.txt'
        self.worker_marker=self.root/'worker_checked.txt'
        self.command('.venv/bin/python','#!/bin/sh\ntouch '+shlex.quote(str(self.worker_marker))+'\n')
        self.command('mockbin/nvidia-smi','#!/bin/sh\nif [ "${FFNDS_TEST_QUERY_FAIL:-0}" = 1 ]; then exit 7; fi\nprintf "%s" "${FFNDS_TEST_OCCUPANTS:-}"\n')
        # Isolate the test from real per-GPU flock holders on the test host.
        self.command('mockbin/flock','#!/bin/sh\nexit "${FFNDS_TEST_LOCK_FAIL:-0}"\n')
        (self.root/'config/worker.env').write_text('export PATH='+shlex.quote(str(self.root/'mockbin')+':/usr/bin:/bin')+'\n')

    def tearDown(self):
        self.temp.cleanup()

    def command(self,path,contents):
        f=self.root/path;f.write_text(contents);f.chmod(0o755)

    def run_guard(self,index,extra=None):
        env=os.environ.copy();env.update(extra or {})
        # Command only records visibility; no actual CUDA command is present.
        body='printf "%s" "$CUDA_VISIBLE_DEVICES" > '+shlex.quote(str(self.marker))
        return subprocess.run(['bash',str(self.root/'scripts/worker_idle_exec.sh'),str(index),'bash','-c',body],
                              env=env,capture_output=True,text=True)

    def test_idle_physical_two_is_one_visible_device(self):
        r=self.run_guard(2)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(self.marker.read_text(),'2')
        self.assertTrue(self.worker_marker.exists())

    def test_occupied_device_never_starts_job(self):
        r=self.run_guard(2,{'FFNDS_TEST_OCCUPANTS':'12345'})
        self.assertEqual(r.returncode,3)
        self.assertIn('already has compute processes',r.stderr)
        self.assertFalse(self.marker.exists())

    def test_query_failure_and_busy_lock_fail_closed(self):
        for overrides in ({'FFNDS_TEST_QUERY_FAIL':'1'},{'FFNDS_TEST_LOCK_FAIL':'1'}):
            with self.subTest(overrides=overrides):
                r=self.run_guard(2,overrides)
                self.assertNotEqual(r.returncode,0)
                self.assertFalse(self.marker.exists())

    def test_invalid_physical_index_fails_before_worker_or_job(self):
        r=self.run_guard(4)
        self.assertEqual(r.returncode,2)
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.worker_marker.exists())


if __name__=='__main__':unittest.main()
