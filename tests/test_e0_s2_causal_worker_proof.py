import importlib.util
from pathlib import Path
import tempfile
import unittest

R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('new_worker',R/'tools/e0_s2_causal_worker_proof.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class LocalPreparation(unittest.TestCase):
    def test_alias_preserves_existing_target_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'HF';target.mkdir();(target/'sentinel').write_bytes(b'existing')
            alias=root/'local'/'huggingface';m.cache_alias(alias,target)
            self.assertEqual((alias/'sentinel').read_bytes(),b'existing')
            with self.assertRaises(ValueError):m.cache_alias(alias,target)
            self.assertEqual((target/'sentinel').read_bytes(),b'existing')

    def test_package_members_cannot_escape_through_parent_or_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'package';root.mkdir();(root/'escape').symlink_to(Path(d),target_is_directory=True)
            for name in ('../evil','/absolute','escape/evil'):
                with self.assertRaises(ValueError):m.safe_member(root,name)


if __name__=='__main__':unittest.main()
