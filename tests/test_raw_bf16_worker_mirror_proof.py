"""Small synthetic master controls; no real checkpoint/model/GPU access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('mirror_proof',ROOT/'tools/raw_bf16_worker_mirror_proof.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fake_models(directory):
    result=[]
    for kind in ('high','low','T5','VAE'):
        data=(kind+' synthetic').encode(); path=Path(directory)/kind;path.write_bytes(data)
        result.append({'kind':kind,'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'header_sha256':'h'})
    return result


class ProofControls(unittest.TestCase):
    def test_import_without_numeric_libraries(self):
        subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'tools');import raw_bf16_worker_mirror_proof;assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],cwd=ROOT,check=True)

    def test_native_full_hash_actual_stat_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            item=fake_models(directory)[0]; result=m.native.full_hash(item)
            self.assertEqual(result['status'],'PASS_FRESH_FULL_SHA')
            self.assertEqual(result['actual_read_bytes'],result['predebited_bytes'])
            self.assertEqual(result['stat'],m.native.stat_record(item['path']))
            item['sha256']='0'*64; failed=m.native.full_hash(item)
            self.assertEqual(failed['status'],'FAILED_OR_STOPPED_READ_PRESERVED_NO_RETRY')
            self.assertEqual(failed['actual_read_bytes'],item['bytes'])

    def test_two_thread_synthetic_full_success(self):
        with tempfile.TemporaryDirectory() as directory:
            models=fake_models(directory); out=Path(directory)/'out';out.mkdir()
            plan={'models':models,'models_sha256':'synthetic','private_worker_config_sha256':'1'*64}
            header={'header_sha256':'h','header_bytes':3,'tensor_count':1095}
            with mock.patch.object(m,'TOTAL',sum(x['bytes'] for x in models)), mock.patch.object(m.native,'header',return_value=header), mock.patch.object(m.native,'raw_header_check',side_effect=lambda x:x):
                report=m.run_proof(plan,out,'x'*64)
            self.assertEqual(report['status'],'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH')
            self.assertEqual(len(report['model_payloads']),4)
            self.assertEqual(report['actual_full_payload_read_bytes'],sum(x['bytes'] for x in models))
            self.assertEqual(set(report['native_headers']),{'high','low'})
            self.assertEqual(json.loads((out/'RUN.json').read_text())['status'],report['status'])

    def test_failure_retains_actual_predebit_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            models=fake_models(directory);models[0]['sha256']='0'*64
            out=Path(directory)/'out';out.mkdir();plan={'models':models,'models_sha256':'synthetic','private_worker_config_sha256':'1'*64}
            with mock.patch.object(m,'TOTAL',sum(x['bytes'] for x in models)), self.assertRaises(ValueError): m.run_proof(plan,out,'x'*64)
            report=json.loads((out/'RUN.json').read_text())
            self.assertEqual(report['status'],'FAILED_MIRROR_PROOF_PRESERVED_NO_RETRY')
            self.assertGreaterEqual(report['predebited_full_payload_bytes'],sum(x['bytes'] for x in models[:2]))
            self.assertFalse(report['automatic_retry'])
            self.assertTrue(any(x['status']=='FAILED_OR_STOPPED_READ_PRESERVED_NO_RETRY' for x in report['model_payloads']))


if __name__=='__main__': unittest.main()
