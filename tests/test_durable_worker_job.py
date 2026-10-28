import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import durable_worker_job as job


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'research').mkdir();self.protocol=self.root/'protocol.json';self.protocol.write_text('{}')
        self.proto_sha=job.digest(self.protocol);self.out=self.root/'job';self.spec=self.root/'spec.json'
        self.active=self.root/'research/ACTIVE_RESEARCH.json'
        self.active.write_text(json.dumps({'user_continuation_authorization_20261027':{'status':'ACTIVE_CURRENT_USER_DIRECTED_SESSION'},'experiment_authority':{'current_user_authorized_scopes':[{'id':'TEST_CPU','protocol_sha256':self.proto_sha,'execution_status':'AUTHORIZED_NOT_STARTED'}]}}))
        self.data={'id':'TEST_CPU','protocol_path':str(self.protocol),'protocol_sha256':self.proto_sha,'cwd':str(self.root),'argv':[sys.executable,'-c','import time; print("hello", flush=True); time.sleep(.02)'],'output_dir':str(self.out),'hf_root':str(self.root),'run_json':str(self.out/'RUN.json'),'external_timeout_seconds':2}
        self.save()

    def save(self): self.spec.write_text(json.dumps(self.data));self.sha=job.digest(self.spec)
    def claim(self):
        self.out.mkdir();job.exclusive_json(self.out/'START_CLAIM.json',{'id':'TEST_CPU','spec_sha256':self.sha,'protocol_sha256':self.proto_sha})
    def supervise(self): return job.supervise(self.spec,self.sha,worker_check=lambda:None,active_path=self.active,kill_grace=.1)

    def test_negative_binding(self):
        with self.assertRaises(ValueError):job.validate(self.spec,'0'*64)
        self.protocol.write_text('changed')
        with self.assertRaises(ValueError):job.validate(self.spec,self.sha)

    def test_missing_authorization(self):
        self.active.write_text('{}')
        with self.assertRaises(ValueError):job.validate(self.spec,self.sha)

    def test_duplicate_start(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):job.start(self.spec,self.sha)

    def test_exitcode_logs_and_receipt(self):
        self.data['argv']=[sys.executable,'-c','print("hello"); raise SystemExit(7)'];self.save();self.claim();r=self.supervise()
        self.assertEqual(r['child_returncode'],7);self.assertTrue(r['process_exited']);self.assertFalse(r['timeout'])
        self.assertEqual((self.out/'child.stdout.log').read_text().strip(),'hello')
        self.assertEqual(json.loads((self.out/'EXIT_RECEIPT.json').read_text()),r)
        with self.assertRaises(FileExistsError):self.supervise()

    def test_timeout(self):
        self.data['argv']=[sys.executable,'-c','import time; time.sleep(10)'];self.data['external_timeout_seconds']=.05;self.save();self.claim();r=self.supervise()
        self.assertTrue(r['timeout']);self.assertTrue(r['process_exited']);self.assertLess(r['child_returncode'],0)

    def test_run_hash(self):
        self.data['argv']=[sys.executable,'-c',f'from pathlib import Path; Path({str(self.out/"RUN.json")!r}).write_text("done")'];self.save();self.claim();r=self.supervise()
        self.assertEqual(r['run_sha256'],hashlib.sha256(b'done').hexdigest())

    def test_inspect_pid_reuse(self):
        self.claim();job.exclusive_json(self.out/'START_RECEIPT.json',{'id':'TEST_CPU','spec_sha256':self.sha,'supervisor_pid':os.getpid(),'supervisor_startticks':job.ticks(os.getpid())})
        self.assertTrue(job.inspect(self.spec,self.sha)['supervisor_alive'])
        with patch.object(job,'ticks',return_value=0):self.assertFalse(job.inspect(self.spec,self.sha)['supervisor_alive'])

    def test_worker_failure_records_no_exit(self):
        self.claim()
        def fail():raise RuntimeError('not worker')
        r=job.supervise(self.spec,self.sha,worker_check=fail,active_path=self.active,kill_grace=.1)
        self.assertFalse(r['process_exited']);self.assertIsNone(r['child_returncode']);self.assertEqual(r['error_type'],'RuntimeError')


if __name__=='__main__':unittest.main()
