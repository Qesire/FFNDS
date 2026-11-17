"""CPU-only checks for prerequisite receipt binding; no research execution."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).with_name('chip_true_m_job.py')
spec = importlib.util.spec_from_file_location('guard_under_test', path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ReceiptTests(unittest.TestCase):
    def test_receipts_must_match_hash_job_and_disposition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def write(name, payload):
                file = root/name
                file.write_text(json.dumps(payload), encoding='utf8')
                return {'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
            run = write('run.json', {'slurm_job_id':'168713','wan_nfe':0,'videos':0})
            end = write('end.json', {'slurm_job_id':'168713','exit_code':0})
            self.assertEqual(guard.prior(run,end,'168713')['slurm_job_id'],'168713')
            with self.assertRaises(RuntimeError): guard.prior(run,end,'168787')
            Path(end['path']).write_text('{"slurm_job_id":"168713","exit_code":1}', encoding='utf8')
            with self.assertRaises(RuntimeError): guard.prior(run,end,'168713')

if __name__ == '__main__': unittest.main()
