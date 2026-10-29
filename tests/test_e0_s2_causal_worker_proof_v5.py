"""Actual small-file hashes and consumer checks for the measured-I/O budget."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('causal_v5_test', ROOT/'tools/e0_s2_causal_worker_proof_v5.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def models(root):
    rows = []
    for kind in ('high', 'low', 'T5', 'VAE'):
        p = root/kind; data = ('small synthetic ' + kind).encode(); p.write_bytes(data)
        rows.append({'kind': kind, 'path': str(p), 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest(), 'header_sha256': 'header'})
    return rows


class MeasuredBudget(unittest.TestCase):
    def test_actual_hashes_and_consumer_allow_real_wall_over_300(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = models(root); out = root/'out'; out.mkdir()
            plan = {'models': rows, 'models_sha256': m.native.canonical(rows),
                    'private_worker_config_sha256': '1'*64, 'execution_private_worker_config_sha256': '1'*64}
            header = {'header_sha256': 'header', 'header_bytes': 6, 'tensor_count': 1095}
            with mock.patch.object(m, 'TOTAL', sum(r['bytes'] for r in rows)), \
                 mock.patch.object(m.native, 'header', return_value=header), \
                 mock.patch.object(m.native, 'raw_header_check', side_effect=lambda h: h), \
                 mock.patch.object(m.time, 'monotonic', return_value=1400):
                run = m.run_proof(plan, out, '2'*64, started=1000)
                self.assertEqual(run['wall_seconds'], 400)
                m.verify_report(plan, {'sha256': '2'*64}, run)
                run['wall_seconds'] = m.RESOURCES['wall_seconds'] + 1
                with self.assertRaises(ValueError): m.verify_report(plan, {'sha256': '2'*64}, run)
            self.assertEqual(len(run['model_payloads']), 4)
            self.assertEqual(run['actual_full_payload_read_bytes'], sum(r['bytes'] for r in rows))
            self.assertFalse(run['automatic_retry'])

    def test_partial_or_wrong_hash_preserved_and_cannot_be_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = models(root); rows[0]['sha256'] = '0'*64
            out = root/'out'; out.mkdir()
            plan = {'models': rows, 'models_sha256': m.native.canonical(rows),
                    'private_worker_config_sha256': '1'*64, 'execution_private_worker_config_sha256': '1'*64}
            with mock.patch.object(m, 'TOTAL', sum(r['bytes'] for r in rows)), self.assertRaises(ValueError):
                m.run_proof(plan, out, '2'*64)
            run = json.loads((out/'RUN.json').read_text())
            self.assertEqual(run['status'], 'FAILED_MIRROR_PROOF_PRESERVED_NO_RETRY')
            self.assertTrue(any(r['status'] == 'FAILED_OR_STOPPED_READ_PRESERVED_NO_RETRY' for r in run['model_payloads']))
            with self.assertRaises(ValueError): m.verify_report(plan, {'sha256': '2'*64}, run)

    def test_changed_file_after_success_fails_consumer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = models(root); out = root/'out'; out.mkdir()
            plan = {'models': rows, 'models_sha256': m.native.canonical(rows),
                    'private_worker_config_sha256': '1'*64, 'execution_private_worker_config_sha256': '1'*64}
            header = {'header_sha256': 'header', 'header_bytes': 6, 'tensor_count': 1095}
            with mock.patch.object(m, 'TOTAL', sum(r['bytes'] for r in rows)), \
                 mock.patch.object(m.native, 'header', return_value=header), \
                 mock.patch.object(m.native, 'raw_header_check', side_effect=lambda h: h):
                run = m.run_proof(plan, out, '2'*64)
                Path(rows[0]['path']).write_bytes(b'changed')
                with self.assertRaises(ValueError): m.verify_report(plan, {'sha256': '2'*64}, run)


if __name__ == '__main__': unittest.main()
