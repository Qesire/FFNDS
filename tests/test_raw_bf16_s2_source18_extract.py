import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from tools import raw_bf16_s2_source18_extract as m


class Source18Controls(unittest.TestCase):
    def test_nested_image_sha_and_other_identity_does_not_absorb_checkpoint_sha(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'metadata.json'
            path.write_text(json.dumps({'a': [{'image': {'sha256': 'a'*64}}, {'image_sha256': 'b'*64}],
                                        'checkpoint': {'sha256': 'c'*64}, 'native_source_ordinal': 910,
                                        'image_name': 'one.jpg', 'img_paths': ['/old/source/two.jpg']}))
            hashes, names, ordinals = m.history_index([path])
            self.assertEqual(set(hashes), {'a'*64, 'b'*64})
            self.assertIn('one.jpg', names); self.assertIn('two.jpg', names); self.assertIn(910, ordinals)

    def test_conflicts_retained_and_duplicate_excludes_both_no_replacement(self):
        candidates = [{'native_source_ordinal': 1, 'archive_member': 'a', 'image_name': 'x'},
                      {'native_source_ordinal': 2, 'archive_member': 'b', 'image_name': 'y'},
                      {'native_source_ordinal': 3, 'archive_member': 'c', 'image_name': 'z'}]
        payload = {'a': b'same', 'b': b'same', 'c': b'other'}
        info = lambda x: {'width': 1, 'height': 1, 'dhash64': '0000000000000001'}
        rows = m.candidate_results(candidates, payload, {m.base.sha(b'other'): {'old'}}, {}, {}, info)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(x['exclusion'] for x in rows))
        self.assertEqual(rows[0]['candidate_exact_duplicate_ordinals'], [2])
        self.assertEqual(rows[1]['candidate_exact_duplicate_ordinals'], [1])
        self.assertEqual(rows[2]['historical_exact_sha_hits'], ['old'])
        self.assertEqual(rows[0]['historical_near_duplicate'], 'UNKNOWN_NO_HISTORICAL_IMAGES_OPENED')
        self.assertEqual(len(rows[0]['candidate_dhash_distances']), 2)

    def test_original_tar_primitive_rejects_symlink_and_duplicate(self):
        for duplicate in (False, True):
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode='w:gz') as tar:
                t = tarfile.TarInfo('data/crop/16-9/x.jpg')
                if not duplicate:
                    t.type = tarfile.SYMTYPE; t.linkname = 'other'; tar.addfile(t)
                else:
                    t.size = 1; tar.addfile(t, io.BytesIO(b'x')); tar.addfile(t, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                m.base.extract_members(io.BytesIO(raw.getvalue()), {'data/crop/16-9/x.jpg'}, 1024)

    def test_fixed_scope_and_required_cpu_budget(self):
        self.assertEqual(len(set(m.ORDINALS)), 18)
        self.assertTrue(all(m.TYPES.count(x) == 3 for x in set(m.TYPES)))
        self.assertEqual(m.RESOURCES['archive_stream_bytes_max'], 3 << 30)
        self.assertEqual(m.RESOURCES['new_HF_bytes'], 128 << 20)
        self.assertEqual(m.LIMITS['GPU_calls'], 0)
        self.assertFalse(m.RESOURCES['automatic_retry'])


if __name__ == '__main__': unittest.main()
