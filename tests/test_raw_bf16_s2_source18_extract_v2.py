import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from tools import raw_bf16_s2_source18_extract as original
from tools import raw_bf16_s2_source18_extract_v2 as v2


class Source18V2Controls(unittest.TestCase):
    def test_only_two_parent_protocols_excluded_from_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'config').mkdir()
            old, audit = [root/p for p in v2.EXCLUDED]
            other = root/'config/other_draft.json'
            for path in (old, audit, other):
                path.write_text(json.dumps({'image_name': 'example.jpg', 'image': {'sha256': 'a'*64}}))
            retained = v2.filter_history([old, audit, other], root=root)
            self.assertEqual(retained, [other])
            hashes, names, _ = v2.original_history_index(retained)
            self.assertEqual(hashes['a'*64], {str(other)})
            self.assertEqual(names['example.jpg'], {str(other)})
            self.assertEqual(len([old, audit, other]), 3)  # Frozen bindings are retained separately.

    def test_only_size_limit_changes_and_old_namespace_is_unmodified(self):
        self.assertNotEqual(v2.core.ID, original.ID)
        self.assertNotEqual(v2.core.OUTPUT, original.OUTPUT)
        self.assertEqual(original.RESOURCES['single_image_bytes_max'], 4 << 20)
        self.assertEqual(v2.core.RESOURCES['single_image_bytes_max'], 8 << 20)
        self.assertEqual({k:v for k,v in v2.core.RESOURCES.items() if k != 'single_image_bytes_max'},
                         {k:v for k,v in original.RESOURCES.items() if k != 'single_image_bytes_max'})
        self.assertEqual(v2.core.ORDINALS, original.ORDINALS)
        self.assertEqual(v2.core.LIMITS, original.LIMITS)
        self.assertNotIn('tools/raw_bf16_s2_source18_extract_v2.py', original.REPO_SOURCES)

    def test_oversize_rejected_in_old_and_accepted_in_new_guard(self):
        raw = io.BytesIO(); name = 'data/crop/16-9/test.jpg'; data = b'a' * ((4 << 20) + 1)
        with tarfile.open(fileobj=raw, mode='w:gz') as tar:
            item = tarfile.TarInfo(name); item.size = len(data); tar.addfile(item, io.BytesIO(data))
        with self.assertRaises(ValueError):
            v2.base.extract_members(io.BytesIO(raw.getvalue()), {name}, original.RESOURCES['single_image_bytes_max'])
        actual = v2.base.extract_members(io.BytesIO(raw.getvalue()), {name}, v2.core.RESOURCES['single_image_bytes_max'])
        self.assertEqual(actual[name], data)


if __name__ == '__main__': unittest.main()
