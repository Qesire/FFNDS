import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('visual_scope', ROOT/'tools/e0_s2_consecutive_visual_cpu.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class GenerationIdentity(unittest.TestCase):
    def test_candidate_cannot_be_bound_as_dense(self):
        source = {'source': 898, 'source_id': 's', 'image_sha256': 'i', 'prompt_sha256': 'p'}
        good = {'mode': 'dense', 'numeric_path': 'BF16_ORIGINAL', 'native_source_ordinal': 898,
                'source_id': 's', 'image_sha256': 'i', 'prompt_sha256': 'p', 'seed': 42,
                'status': 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE',
                'all160_FFN_covered': True, 'completed_native_flow_steps': 4, 'scheduler_generator_reset': True}
        m.check_generation(source, 'dense', good)
        for key, value in [('mode', 'value10'), ('seed', 43), ('native_source_ordinal', 96),
                           ('numeric_path', 'HIF4'), ('all160_FFN_covered', False), ('completed_native_flow_steps', 3)]:
            wrong = copy.deepcopy(good); wrong[key] = value
            with self.assertRaises(ValueError): m.check_generation(source, 'dense', wrong)


if __name__ == '__main__': unittest.main()
