import importlib.util
from pathlib import Path
import tempfile
import unittest

R=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('proof_v4',R/'tools/e0_s2_causal_worker_proof_v4.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class ContainedMetadata(unittest.TestCase):
    def test_owned_alias_becomes_contained_regular_file_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);view=root/'local';view.mkdir();source=root/'HF';source.mkdir();(source/'configuration.json').write_bytes(b'{}')
            dst=view/'configuration.json';dst.symlink_to(source/'configuration.json')
            row={'filename':'configuration.json','bytes':2,'sha256':m.native.sha(source/'configuration.json')}
            with self.assertRaises(ValueError):m.native.bound(view,{'path':row['filename'],'bytes':2,'sha256':row['sha256']})
            self.assertEqual(m.materialize_owned_metadata(view,source,[row]),1)
            self.assertFalse(dst.is_symlink());self.assertEqual(m.native.bound(view,{'path':row['filename'],'bytes':2,'sha256':row['sha256']}),dst)
            self.assertEqual((source/'configuration.json').read_bytes(),b'{}')

    def test_regular_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);view=root/'local';view.mkdir();source=root/'HF';source.mkdir();(source/'a').write_bytes(b'a');(view/'a').write_bytes(b'a')
            with self.assertRaises(ValueError):m.materialize_owned_metadata(view,source,[{'filename':'a','bytes':1,'sha256':m.native.sha(source/'a')}])
            self.assertEqual((view/'a').read_bytes(),b'a')


if __name__=='__main__':unittest.main()
