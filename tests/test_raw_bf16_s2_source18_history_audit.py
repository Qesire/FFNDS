import tempfile
import unittest
from pathlib import Path
from tools import raw_bf16_s2_source18_history_audit as m


class HistoryScreenControls(unittest.TestCase):
    def test_reserved_is_filtered_before_any_path_stat_or_open(self):
        inventory = {'cached_images': [
            {'path': 'missing_reserved.jpg', 'bytes': 1, 'sha256': 'r', 'known_exposure_banks': ['calibration24']},
            {'path': 'missing_other_reserved.jpg', 'bytes': 1, 'sha256': 's', 'known_exposure_banks': ['diverse_confirmation20']},
            {'path': 'existing_later.jpg', 'bytes': 3, 'sha256': 'd', 'known_exposure_banks': ['formal20']}]}
        result = m.eligible_history(inventory, {'sources': []}, {'r'}, Path('/none'))
        self.assertEqual([x['sha256'] for x in result], ['d'])

    def test_reserved_collision_in_fitted_bank_fails_before_stat(self):
        with self.assertRaises(ValueError):
            m.eligible_history({'cached_images': []}, {'sources': [{'image_id': 'r', 'image_path': '/does/not/exist'}]}, {'r'}, Path('/none'))

    def test_duplicate_exposed_sha_is_one_image(self):
        with tempfile.TemporaryDirectory() as td:
            hf = Path(td); (hf/'image.jpg').write_bytes(b'abc')
            inv = {'cached_images': [{'path': 'image.jpg', 'bytes': 3, 'sha256': 'd', 'known_exposure_banks': ['formal20']}]}
            bank = {'sources': [{'image_id': 'd', 'image_path': str(hf/'image.jpg')}]}
            self.assertEqual(len(m.eligible_history(inv, bank, set(), hf)), 1)

    def test_reserved_hash_field_semantics(self):
        obj = {'image': {'sha256': 'a'*64}, 'irrelevant': {'sha256': 'b'*64}, 'image_id': 'c'*64}
        self.assertEqual(m.image_hashes(obj), {'a'*64, 'c'*64})


if __name__ == '__main__': unittest.main()
