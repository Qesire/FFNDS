"""Scalar software fixtures only; no saved operands, torch or native imports."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'tools/e0_vector_curve_cpu_feasibility.py'
spec = importlib.util.spec_from_file_location('curve_cpu_fixture', SOURCE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class CurveScalarFixtures(unittest.TestCase):
    def test_original_shard_resolves_only_from_exact_manifest_record(self):
        plan = {'weight_shard_filename': module.SHARD_FILENAME, 'weight_shard_sha256': 'digest'}
        record = {'expert': 'high_noise_model', 'block': 10,
                  'path': '/private-runtime/shards/' + module.SHARD_FILENAME, 'sha256': 'digest'}
        self.assertEqual(module.resolve_original_shard(plan, {'shards': [record]}).name, module.SHARD_FILENAME)
        with self.assertRaises(ValueError):
            module.resolve_original_shard(plan, {'shards': [dict(record, sha256='different')]})
        with self.assertRaises(ValueError):
            module.resolve_original_shard(plan, {'shards': [dict(record, path='/private-runtime/other.pt')]})

    def test_absolute_cosine_tie_uses_index_and_retains_signed_alpha(self):
        clusters = module.rank_neighbors([[1., -.9, .9, .2]], [1., 1., 1., 1.], anchors=[0], count=2)
        self.assertEqual(clusters[0]['ids'], [0, 1, 2])
        self.assertEqual(clusters[0]['alpha'], [1., -.9, .9])

    def test_other_representatives_are_not_used_as_neighbors(self):
        clusters = module.rank_neighbors([[1., .99, .8, .4], [.99, 1., .4, .8]],
                                         [1., 1., 1., 1.], anchors=[0, 1], count=1)
        self.assertEqual([c['ids'] for c in clusters], [[0, 2], [1, 3]])

    def test_signed_cancellation_counterexample_keeps_nonzero_function(self):
        example = module.cancellation_example()
        self.assertAlmostEqual(example['actual_signed_pair_output'], example['expected_output'], places=15)
        self.assertEqual(example['centroid_merge_output'], 0.)
        self.assertEqual(example['relative_output_SSE_lost_by_merge'], 1.)


if __name__ == '__main__':
    unittest.main()
