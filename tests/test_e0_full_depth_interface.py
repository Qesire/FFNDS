import copy
import hashlib
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_e0_full_depth_interface import NATIVE_ROOT, REQUIRED_APIS, REQUIRED_CONFIG, validate_design


class StaticInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        classes = defaultdict(lambda: defaultdict(list))
        for relative, cls, method, arguments in REQUIRED_APIS:
            classes[relative][cls].append('    def ' + method + '(' + ','.join(arguments) + '): pass\n')
        hashes = {}
        for relative, definitions in classes.items():
            path = self.root / NATIVE_ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join('class ' + cls + ':\n' + ''.join(methods)
                                    for cls, methods in definitions.items()))
            hashes[NATIVE_ROOT + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.design = dict(status='INTERFACE_DESIGN_ONLY_NOT_A_MODEL_PROTOCOL',
                           model_execution_authority='NONE', new_generations=0,
                           required_effective_config=copy.deepcopy(REQUIRED_CONFIG), source_sha256=hashes,
                           residency_policy='NON_FFN_RETAIN_CURRENT_BLOCK_FFN_ONLY_UNIMPLEMENTED',
                           offload_swap_policy='ONCE_AFTER_BOTH_TRAJECTORIES_AND_SHADOWS',
                           scheduler_boundaries=['before_step_pre', 'after_step_pre', 'after_step_post'],
                           unimplemented_native_components=['synthetic API fixture, no native producer'])

    def test_static_pass_explicitly_preserves_native_unknowns(self):
        result = validate_design(self.design, self.root)
        self.assertEqual(result['peak_memory_fit'], 'UNKNOWN')
        self.assertEqual(result['model_execution_authority'], 'NONE')
        self.assertFalse(result['quality_accepted'])

    def test_incompatible_actual_execution_modes_are_rejected(self):
        for key, value in [('feature_caching', 'Tea'), ('seq_parallel', True), ('attention', 'other'),
                           ('cpu_offload', False), ('infer_steps', 8), ('enable_cfg', 0)]:
            design = copy.deepcopy(self.design)
            design['required_effective_config'][key] = value
            with self.assertRaises(ValueError):
                validate_design(design, self.root)

    def test_source_drift_and_rebound_wrong_api_are_rejected(self):
        relative = NATIVE_ROOT + REQUIRED_APIS[0][0]
        path = self.root / relative
        path.write_text(path.read_text().replace('infer_block(', 'wrong_api('))
        with self.assertRaises(ValueError):
            validate_design(self.design, self.root)
        self.design['source_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            validate_design(self.design, self.root)

    def test_source_pass_cannot_grant_gpu_permission_or_unbounded_residency(self):
        for key, value in [('new_generations', 2), ('model_execution_authority', 'YES'),
                           ('residency_policy', 'ALL_WEIGHTS_RETAIN_FOREVER')]:
            design = copy.deepcopy(self.design)
            design[key] = value
            with self.assertRaises(ValueError):
                validate_design(design, self.root)


if __name__ == '__main__':
    unittest.main()
