"""A new source cohort must never silently evaluate an empty old prompt index."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from vbench_input_index import DIMS, build_full_info, validate_video_name_coverage


class VBenchInputIndexTests(unittest.TestCase):
    def fixtures(self, prefix):
        samples, historical = [], []
        for dimension, kind in [('i2v_subject', 'animal'), ('i2v_background', 'architecture')]:
            for index in range(10):
                prompt = f'{prefix} {dimension} {index}'
                samples.append({'source_group': f'{dimension}:{index}', 'dimension': dimension,
                                'prompt': prompt, 'image_name': prompt + '.jpg'})
                historical.append({'prompt_en': prompt, 'dimension': [dimension, *DIMS],
                                   'image_type': kind, 'image_name': prompt + '.jpg'})
        return samples, historical

    def test_reconstruction_preserves_original_schema_and_dimension_order(self):
        samples, original = self.fixtures('old')
        self.assertEqual(build_full_info(samples, original), original)

    def test_new_names_cover_every_metric_and_reject_the_old_empty_match(self):
        _, old = self.fixtures('old')
        samples, _ = self.fixtures('new')
        names = [f'{v["prompt"]}-{copy}.mp4' for v in samples for copy in range(5)]
        index = build_full_info(samples, old)
        self.assertEqual(validate_video_name_coverage(index, names), {v: 100 for v in DIMS})
        with self.assertRaisesRegex(ValueError, 'all100'):
            validate_video_name_coverage(old, names)

    def test_missing_motion_tag_cannot_silently_drop_background_sources(self):
        samples, old = self.fixtures('old')
        for value in old[10:]: value['dimension'].remove('motion_smoothness')
        with self.assertRaisesRegex(ValueError, 'all three'):
            build_full_info(samples, old)


if __name__ == '__main__':
    unittest.main()
