import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sm120_proof_test', ROOT/'tools/e0_s2_sm120_bf16_worker_proof.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class SM120Proof(unittest.TestCase):
    def test_scope_copy_accounting_distinguishes_hash_stage(self):
        row = {'copied_bytes_scope':'SHA_STAGE_ONLY', 'copied_bytes':0, 'sha_stage_copied_bytes':0,
               'environment_copied_package_bytes':42053803, 'scope_copied_package_bytes':42053803}
        m.validate_copy_accounting(row)
        for key in row:
            broken = dict(row); broken[key] = None
            with self.assertRaises(ValueError): m.validate_copy_accounting(broken)

    def test_view_link_resolved_containment_with_real_native_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); hf_view = root/'hf/view'; hf_view.mkdir(parents=True)
            (hf_view/'config.json').write_bytes(b'original')
            view = root/'local/view'; m.prepare_view(view, hf_view)
            item = {'path':'config.json', 'bytes':8, 'sha256':hashlib.sha256(b'original').hexdigest()}
            self.assertEqual(m.native.bound(view, item), (hf_view/'config.json').resolve())
            with self.assertRaises(ValueError): m.prepare_view(view, hf_view)
            external = root/'external'; external.write_bytes(b'original')
            (hf_view/'bad').symlink_to(external)
            with self.assertRaises(ValueError): m.native.bound(view, {**item, 'path':'bad'})

    def test_strict_inventory_rejects_extra_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'a').write_bytes(b'a')
            rows = [{'path':'a','bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}]
            m.strict_package(root, rows)
            (root/'extra').write_bytes(b'x')
            with self.assertRaises(ValueError): m.strict_package(root, rows)
            (root/'extra').unlink(); (root/'extra').symlink_to(root/'a')
            with self.assertRaises(ValueError): m.strict_package(root, rows)

    def test_private_v5_hash_and_consumer_stat(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root/'out'; out.mkdir(); rows=[]
            for kind in ('high','low','T5','VAE'):
                p=root/kind; data=kind.encode(); p.write_bytes(data)
                rows.append({'path':str(p),'kind':kind,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'header_sha256':'header'})
            plan={'models':rows,'models_sha256':m.native.canonical(rows), 'private_worker_config_sha256':'1'*64,
                  'execution_private_worker_config_sha256':'1'*64}
            header={'header_sha256':'header','header_bytes':4,'tensor_count':1095}
            with mock.patch.object(m.v5, 'TOTAL', sum(r['bytes'] for r in rows)), \
                 mock.patch.object(m.native, 'header', return_value=header), \
                 mock.patch.object(m.native, 'raw_header_check', side_effect=lambda x:x):
                result=m.v5.run_proof(plan, out, '2'*64)
                m.v5.verify_report(plan, {'sha256':'2'*64}, result)
                Path(rows[0]['path']).write_bytes(b'changed')
                with self.assertRaises(ValueError): m.v5.verify_report(plan, {'sha256':'2'*64}, result)

if __name__ == '__main__': unittest.main()
