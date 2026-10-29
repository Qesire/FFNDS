import io
import tarfile
import unittest
from unittest.mock import patch
from tools import raw_bf16_s2_source18_tar_header_audit as m


def tar_stream(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as tar:
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            if kind == 'hardlink':
                info.type = tarfile.LNKTYPE; info.linkname = 'prior/target'; tar.addfile(info)
            elif kind == 'symlink':
                info.type = tarfile.SYMTYPE; info.linkname = '../escape'; tar.addfile(info)
            else:
                info.size = len(data); tar.addfile(info, io.BytesIO(data))
    return io.BytesIO(out.getvalue())


class HeaderAuditControls(unittest.TestCase):
    def test_hardlink_duplicate_oversize_unsafe_missing_and_non_target(self):
        entries = [('a', 'file', b'a'), ('a', 'file', b'b'), ('b', 'hardlink', b''),
                   ('c', 'file', b'xyz'), ('../unsafe', 'file', b'x'), ('z', 'file', b'ignored')]
        with patch.object(tarfile.TarFile, 'extractfile', side_effect=AssertionError('must not extract')):
            result = m.audit_headers(tar_stream(entries), ['a', 'b', 'c', '../unsafe', 'missing'], max_image_bytes=2)
        self.assertEqual(result['total_tar_headers'], 6)
        targets = {x['target_name']: x for x in result['targets']}
        self.assertEqual(len(targets['a']['occurrences']), 2)
        self.assertTrue(all('duplicate' in x['rejection_reasons'] for x in targets['a']['occurrences']))
        self.assertEqual(targets['b']['occurrences'][0]['linkname'], 'prior/target')
        self.assertIn('nonregular', targets['b']['occurrences'][0]['rejection_reasons'])
        self.assertIn('size_gt_4MiB', targets['c']['occurrences'][0]['rejection_reasons'])
        self.assertIn('unsafe', targets['../unsafe']['occurrences'][0]['rejection_reasons'])
        self.assertTrue(targets['missing']['missing'])
        self.assertTrue(all(not x['original_extractor_would_accept_unique_header'] for x in targets.values()))

    def test_valid_member_and_nonpositive(self):
        result = m.audit_headers(tar_stream([('good', 'file', b'xyz'), ('empty', 'file', b'')]), ['good', 'empty'])
        self.assertTrue(result['targets'][0]['original_extractor_would_accept_unique_header'])
        self.assertIn('size_nonpositive', result['targets'][1]['occurrences'][0]['rejection_reasons'])

    def test_reject_duplicate_target_definitions(self):
        with self.assertRaises(ValueError): m.audit_headers(tar_stream([]), ['a', 'a'])

    def test_fixed_cpu_only_scope(self):
        self.assertEqual(m.LIMITS['target_members'], 18)
        self.assertEqual(m.RESOURCES['archive_stream_bytes_max'], 3 << 30)
        self.assertEqual(m.RESOURCES['new_HF_bytes'], 8 << 20)
        self.assertEqual(m.LIMITS['exported_images'], 0)
        self.assertEqual(m.LIMITS['image_decode_calls'], 0)
        self.assertFalse(m.RESOURCES['automatic_retry'])


if __name__ == '__main__': unittest.main()
