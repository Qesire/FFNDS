"""Synthetic archive and metadata-only extraction controls."""
import io
import tarfile
import unittest
from tools import raw_bf16_fresh_source_extract as m


def archive(items):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as t:
        for name,kind,data in items:
            x=tarfile.TarInfo(name)
            if kind=='link':x.type=tarfile.SYMTYPE;x.linkname='elsewhere';t.addfile(x)
            else:x.size=len(data);t.addfile(x,io.BytesIO(data))
    return io.BytesIO(out.getvalue())


class Controls(unittest.TestCase):
    def test_exact_stream_and_duplicate_link_traversal_size(self):
        name='data/crop/16-9/a.jpg';self.assertEqual(m.extract_members(archive([(name,'file',b'abc')]),{name},10),{name:b'abc'})
        for items,wanted,limit in [([(name,'file',b'a'),(name,'file',b'b')],{name},10),
                ([(name,'link',b'')],{name},10),([('../escape','file',b'x')],{'../escape'},10),
                ([(name,'file',b'abc')],{name},2), ([],{name},10)]:
            with self.assertRaises(ValueError):m.extract_members(archive(items),wanted,limit)

    def test_hashreader_actual_bytes_and_exact_input(self):
        report={'actual_archive_bytes':0};r=m.HashReader(io.BytesIO(b'abcdef'),report)
        self.assertEqual(r.read(2),b'ab');self.assertEqual(r.read(8),b'cdef');self.assertEqual(report['actual_archive_bytes'],6)
        self.assertEqual(r.hash.hexdigest(),m.sha(b'abcdef'))

    def test_provisional_fixed_identity(self):
        self.assertEqual(len(m.ORDINALS),10);self.assertEqual(len(set(m.ORDINALS)),10)
        self.assertEqual(m.STRATA.count('subject'),5);self.assertEqual(m.STRATA.count('camera'),3)
        self.assertEqual(m.RESOURCES['GPU'],0);self.assertFalse(m.RESOURCES['automatic_retry'])

if __name__=='__main__':unittest.main()
