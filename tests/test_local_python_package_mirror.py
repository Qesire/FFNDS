import importlib.util
import tempfile
import unittest
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mirror', ROOT / 'tools/local_python_package_mirror.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class MirrorTest(unittest.TestCase):
    def test_archive_extract_and_bad_members(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root / 'src/transformers'; src.mkdir(parents=True)
            (src / '__init__.py').write_text('x=1\n'); (src / 'models').mkdir(); (src / 'models/a.py').write_text('a=2\n')
            rows = m.inventory(src); identity = m.canonical_sha(rows); archive = root / 'safe.tar'
            with patch.object(m.importlib.metadata, 'version', return_value='5.6.2'):
                m.archive(src, archive, identity, '5.6.2')
                dst = root / 'out/transformers'; m.extract(archive, dst, rows, identity, '5.6.2')
                self.assertEqual(m.verify_origin(dst / '__init__.py', dst, rows), '__init__.py')
                for name, contents in [('duplicate', [('__init__.py', b'x=1\n'), ('__init__.py', b'x=1\n')]),
                                       ('traversal', [('../evil', b'x')]), ('tamper', [('__init__.py', b'x=2\n')])]:
                    bad = root / (name + '.tar')
                    with tarfile.open(bad, 'w') as tar:
                        for member, data in contents:
                            info = tarfile.TarInfo(member); info.size = len(data); tar.addfile(info, io.BytesIO(data))
                    with self.assertRaises(ValueError): m.extract(bad, root / name / 'transformers', rows, identity, '5.6.2')
                with self.assertRaises(ValueError): m.extract(archive, root / 'small/transformers', rows, identity, '5.6.2', max_bytes=1)
            with patch.object(m.importlib.metadata, 'version', return_value='wrong'):
                with self.assertRaises(ValueError): m.extract(archive, root / 'wrong/transformers', rows, identity, '5.6.2')

    def test_copy_origin_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root / 'source/transformers'; src.mkdir(parents=True)
            (src / '__init__.py').write_text('value = 1\n'); (src / 'models').mkdir()
            (src / 'models/a.py').write_text('a = 2\n'); (src / '__pycache__').mkdir()
            (src / '__pycache__/x.pyc').write_bytes(b'ignored')
            rows = m.inventory(src); identity = m.canonical_sha(rows); dst = root / 'local/transformers'
            with patch.object(m.importlib.metadata, 'version', return_value='5.6.2'):
                result = m.build(src, dst, identity, '5.6.2')
                self.assertEqual(result['files'], rows)
                self.assertEqual(m.verify_origin(dst / 'models/a.py', dst, rows), 'models/a.py')
                self.assertFalse((dst / '__pycache__').exists())
                with self.assertRaises(ValueError): m.verify_origin(src / '__init__.py', dst, rows)
                (dst / 'models/a.py').write_text('changed')
                with self.assertRaises(ValueError): m.verify_origin(dst / 'models/a.py', dst, rows)
                with self.assertRaises(ValueError): m.build(src, root / 'other/transformers', '0' * 64, '5.6.2')
                with self.assertRaises(ValueError): m.build(src, dst, identity, '5.6.2')

    def test_symlink_and_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root / 'transformers'; src.mkdir(); (src / 'a.py').write_text('a')
            (src / 'link').symlink_to(src / 'a.py')
            with self.assertRaises(ValueError): m.inventory(src)
            (src / 'link').unlink(); (src / 'dirlink').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): m.inventory(src)
            (src / 'dirlink').unlink()
            with patch.object(m.importlib.metadata, 'version', return_value='wrong'):
                with self.assertRaises(ValueError): m.build(src, root / 'out/transformers', m.canonical_sha(m.inventory(src)), '5.6.2')

if __name__ == '__main__': unittest.main()
