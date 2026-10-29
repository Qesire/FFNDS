import importlib.util
from pathlib import Path
import tempfile
import unittest

R=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('proof_v3',R/'tools/e0_s2_causal_worker_proof_v3.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class PartialViewRepair(unittest.TestCase):
    def test_metadata_leaves_preserve_parent_and_existing_weight_link(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'shared';cache.mkdir();view=root/'local';(view/'high_noise_model').mkdir(parents=True)
            (cache/'high_noise_model').mkdir();(cache/'high_noise_model/weights').write_bytes(b'unchanged')
            weight=view/'high_noise_model/weights';weight.symlink_to(cache/'high_noise_model/weights')
            metadata=[]
            for name in ('configuration.json','high_noise_model/config.json','google/tokenizer.json'):
                p=cache/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'{}')
                metadata.append({'filename':name,'bytes':2,'sha256':m.native.sha(p)})
            result=m.link_metadata(view,cache,metadata)
            self.assertEqual(len(result),3);self.assertTrue(view.is_dir());self.assertFalse(view.is_symlink())
            self.assertTrue(weight.is_symlink());self.assertEqual(weight.read_bytes(),b'unchanged')
            self.assertEqual((view/'high_noise_model/config.json').read_bytes(),b'{}')

    def test_preflight_all_metadata_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);view=root/'local';view.mkdir();cache=root/'shared';cache.mkdir()
            (cache/'a').write_bytes(b'a');rows=[{'filename':'a','bytes':1,'sha256':m.native.sha(cache/'a')},
                {'filename':'missing','bytes':1,'sha256':'0'*64}]
            with self.assertRaises(ValueError):m.link_metadata(view,cache,rows)
            self.assertEqual(list(view.iterdir()),[])


if __name__=='__main__':unittest.main()
