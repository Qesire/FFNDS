import importlib.util
from pathlib import Path
import tempfile
import unittest

R=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('proof_v2',R/'tools/e0_s2_causal_worker_proof_v2.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class MissingEntriesOnly(unittest.TestCase):
    def test_link_inside_existing_cache_preserves_other_entries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'oldcache';cache.mkdir();(cache/'other').write_bytes(b'unchanged')
            target=root/'shared';target.write_bytes(b'weight');dest=cache/'new'/'weight'
            m.link_missing(dest,target);self.assertEqual(dest.read_bytes(),b'weight')
            self.assertEqual((cache/'other').read_bytes(),b'unchanged')
            with self.assertRaises(ValueError):m.link_missing(cache/'other',target)
            self.assertEqual((cache/'other').read_bytes(),b'unchanged')


if __name__=='__main__':unittest.main()
