import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.export_sm120_kernel_overlay import archive_files

class OverlayTest(unittest.TestCase):
    def test_atime_change_allowed_inode_and_content_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=p/'common_ops.abi3.so';data=b'\x7fELFtiny';x.write_bytes(data)
            expected={x.name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
            real=x.lstat()
            def snapshot(**changes):
                attrs={k:getattr(real,k) for k in ('st_mode','st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_atime_ns')}
                attrs.update(changes);return SimpleNamespace(**attrs)
            with patch('tools.export_sm120_kernel_overlay.NAMES',(x.name,)):
                with patch.object(Path,'lstat',side_effect=[snapshot(),snapshot(st_atime_ns=real.st_atime_ns+1)]):
                    archive_files(p,expected)
                with patch.object(Path,'lstat',side_effect=[snapshot(),snapshot(st_ino=real.st_ino+1)]):
                    with self.assertRaises(ValueError): archive_files(p,expected)
                x.write_bytes(b'\x7fELFfake')  # Same length; SHA, rather than size, rejects it.
                with self.assertRaises(ValueError): archive_files(p,expected)

    def test_regular_elf_and_reject_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=p/'common_ops.abi3.so';data=b'\x7fELFtiny';x.write_bytes(data);expected={x.name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
            with patch('tools.export_sm120_kernel_overlay.NAMES',(x.name,)):
                a,_=archive_files(p,expected);b,_=archive_files(p,expected);self.assertEqual(a,b)
                x.unlink();target=p/'target';target.write_bytes(data);x.symlink_to(target)
                with self.assertRaises(ValueError):archive_files(p,expected)
                x.unlink();bad=b'BAD!tiny';x.write_bytes(bad);expected[x.name]['sha256']=hashlib.sha256(bad).hexdigest()
                with self.assertRaises(ValueError):archive_files(p,expected)

if __name__=='__main__':unittest.main()
