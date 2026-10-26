"""Synthetic numerical software fixtures; no real operands/torch/model imports."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import e0_fixed_coordinate_basis_oracle as module


class BasisSoftwareFixtures(unittest.TestCase):
    def test_module_import_does_not_import_numpy_torch_or_native_models(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys;sys.path.insert(0,sys.argv[1]);import e0_fixed_coordinate_basis_oracle;'
            'assert "torch" not in sys.modules and "numpy" not in sys.modules', str(TOOLS)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_equal_columns_and_joint_interaction_is_a_distinct_space(self):
        t = np.random.Generator(np.random.PCG64(11)).normal(size=(128, 4))
        joint, jdef = module.features(t, 'joint_quadratic', np)
        separable, sdef = module.features(t, 'separable15', np)
        self.assertEqual(joint.shape, (128, 15))
        self.assertEqual(separable.shape, (128, 15))
        self.assertEqual(sdef['column_names'][-2:], ['T0^4', 'T1^4'])
        w = module.Work(np)
        jq, _, _ = module.span(joint, w, np)
        sq, _, _ = module.span(separable, w, np)
        target = joint[:, jdef['column_names'].index('T0*T1')]
        self.assertLess(np.linalg.norm(target - jq @ (jq.T @ target)), 1e-12)
        self.assertGreater(np.linalg.norm(target - sq @ (sq.T @ target)), 1.)

    def test_small_decisive_direction_is_retained_and_rank_disclosed(self):
        rng = np.random.Generator(np.random.PCG64(9))
        q, _ = np.linalg.qr(np.column_stack([np.ones(128), rng.normal(size=(128, 2))]))
        # The second feature has a tiny independent component but can explain
        # an order-one target with a large readout coefficient.
        matrix = np.column_stack([np.ones(128), q[:, 1], q[:, 1] + 1e-10 * q[:, 2]])
        u, s, info = module.span(matrix, module.Work(np), np)
        self.assertEqual(info['retained_rank'], 3)
        target = q[:, 2]
        self.assertLess(np.linalg.norm(target - u @ (u.T @ target)), 1e-5)
        self.assertGreater(info['condition_number_positive_spectrum'], 1e9)
        self.assertEqual(len(info['centered_singular_values']), 2)
        self.assertEqual(info['standard_eps_rank'], 3)
        self.assertFalse(info['eligible_fixed_family_comparison'])

    def test_duplicate_anchor_roundoff_modes_do_not_become_primary_family(self):
        rng = np.random.Generator(np.random.PCG64(3))
        t = rng.normal(size=128)
        coordinates = np.column_stack([t, t, t, t])
        for family, expected_rank in [('joint_quadratic', 3), ('separable15', 5)]:
            matrix, _ = module.features(coordinates, family, np)
            expanded_q, _, info = module.span(matrix, module.Work(np), np)
            self.assertEqual(info['retained_rank'], expected_rank)
            self.assertFalse(info['eligible_fixed_family_comparison'])
            self.assertFalse(info['rank_stable'])
            self.assertGreater(info['expanded_numerical_retained_width'], expected_rank)
            # An added roundoff direction is orthogonal to the primary true
            # polynomial span. The expanded numerical span can fit it anyway.
            target = expanded_q[:, expected_rank]
            primary_q = expanded_q[:, :expected_rank]
            self.assertAlmostEqual(np.linalg.norm(target-primary_q@(primary_q.T@target))**2, 1.)
            self.assertLess(np.linalg.norm(target-expanded_q@(expanded_q.T@target))**2, 1e-25)

    def test_random_rank_control_contains_constant_and_is_nested_reproducible(self):
        q3, _ = module.random_span(128, 3, module.Work(np), np)
        q7, _ = module.random_span(128, 7, module.Work(np), np)
        q3again, _ = module.random_span(128, 3, module.Work(np), np)
        np.testing.assert_array_equal(q3, q3again)
        np.testing.assert_array_equal(q3, q7[:, :3])
        np.testing.assert_allclose(q7.T @ q7, np.eye(7), atol=1e-14)
        np.testing.assert_allclose(q3[:, 0], np.ones(128)/np.sqrt(128))

    def test_signed_gram_scores_match_direct_output_including_cancellation(self):
        rng = np.random.Generator(np.random.PCG64(4))
        weights = rng.normal(size=(7, 16))
        weights[:, 1] = -weights[:, 0]
        residual = rng.normal(size=(128, 16))
        gram = weights.T @ weights
        scored = module.energy(residual, gram, module.Work(np), np)
        direct = float(((residual @ weights.T) ** 2).sum())
        self.assertAlmostEqual(scored, direct, places=10)
        cancel = np.zeros((2, 16)); cancel[:, :2] = 1
        self.assertLess(module.energy(cancel, gram, module.Work(np), np), 1e-12)

    def test_common_raw_projection_gate_score_is_not_gated_optimum(self):
        # A single shared intercept is raw-fitted to4 unequal case responses.
        # Per-case gates reweight cases; their optimum intercept changes.
        c = np.repeat(np.array([0., 0., 0., 4.]), 32).reshape(128, 1)
        raw_prediction = np.full_like(c, c.mean())
        gate = np.repeat(np.array([1., 1., 1., 10.]), 32).reshape(128, 1)
        raw_gated_error = float((((c - raw_prediction) * gate) ** 2).sum())
        gated_mean = float((c * gate**2).sum() / (gate**2).sum())
        gated_optimum = float((((c - gated_mean) * gate) ** 2).sum())
        self.assertGreater(raw_gated_error, gated_optimum)

    def test_joint_oracle_kernel_uses_raw_shared_space_and_rank_matched_control(self):
        rng = np.random.Generator(np.random.PCG64(21))
        t = rng.normal(size=(128, 4))
        c = np.zeros((128, 16)); c[:, 0] = t[:, 0] * t[:, 1]
        targets = dict(zip(module.TARGETS, [c.copy(), c.copy(), c.copy()]))
        weights = rng.normal(size=(5, 16))
        gates = [np.ones((1, 5)) * value for value in [1., 2., 3., 4.]]
        work = module.Work(np)
        result = module.analyze(t, targets, weights, gates, work, np)
        joint = result['comparisons']['joint_quadratic']['continuous']
        separate = result['comparisons']['separable15']['continuous']
        self.assertLess(joint['pooled_raw_relative_SSE'], 1e-25)
        self.assertGreater(separate['pooled_raw_relative_SSE'], .1)
        for family in ['joint_quadratic', 'separable15']:
            basis = result['basis_spaces'][family]
            control = result['basis_spaces'][basis['dimension_matched_control']]
            self.assertEqual(basis['retained_rank'], control['retained_rank'])
        self.assertEqual(joint['gated_objective'],
                         'SCORE_COMMON_RAW_ORACLE_RESIDUAL; NOT_COMMON_GATED_OPTIMUM')
        self.assertEqual(len(joint['truncation_sensitivity']), 4)
        self.assertIn('constant_only', result['basis_spaces'])
        self.assertIn('anchor_affine', result['basis_spaces'])
        constant = result['comparisons']['constant_only']['continuous']
        self.assertAlmostEqual(constant['pooled_raw_SSE_over_centered_innovation_energy'], 1.)

    def test_large_mean_does_not_hide_centered_innovation_error(self):
        rng = np.random.Generator(np.random.PCG64(42))
        t = rng.normal(size=(128, 4))
        c = np.zeros((128, 16)); c[:, 0] = 10000 + rng.normal(size=128)
        targets = dict(zip(module.TARGETS, [c.copy(), c.copy(), c.copy()]))
        result = module.analyze(t, targets, np.eye(16), [np.ones((1, 16))] * 4, module.Work(np), np)
        constant = result['comparisons']['constant_only']['continuous']
        self.assertLess(constant['pooled_raw_relative_SSE'], 1e-6)
        self.assertAlmostEqual(constant['pooled_raw_SSE_over_centered_innovation_energy'], 1.)

    def test_budget_rejects_before_product(self):
        work = module.Work(np, maximum=3)
        with self.assertRaisesRegex(RuntimeError, 'would be exceeded'):
            work.dot(np.ones((2, 2)), np.ones((2, 2)))
        self.assertEqual(work.mac, 0)

    def test_frozen_worst_case_matmul_bound_stays_below20M(self):
        # 4 replay products,5 actual W2 Grams,2 basis SVD reconstruction/
        # orthogonality products, at most2 different-rank random controls,
        # target references and4 total scored spaces with mode diagnostics.
        replay = 128 * 5120 * 16
        grams = 5 * 5120 * 16 * 16
        references = 3 * 3 * 128 * 16 * 16
        basis_diagnostics = 2 * (128 * 14 * 14 + 15 * 128 * 15)
        random_diagnostics = 2 * 15 * 128 * 15
        projections_and_scoring = 4 * 3 * (2 * 128 * 15 * 16 + 2 * 128 * 16 * 16)
        mode_diagnostics = 2 * 3 * 15 * 16 * 16
        centered_references = 3 * 2 * 128 * 16 * 16
        baseline_diagnostics = 128 + (128 * 4 * 4 + 5 * 128 * 5)
        baseline_projection_and_scores = 3 * (2*128*1*16 + 2*128*5*16 + 4*128*16*16)
        baseline_mode_diagnostics = 3 * (1+5) * 16 * 16
        bound = (replay + grams + references + basis_diagnostics + random_diagnostics
                 + projections_and_scoring + mode_diagnostics + centered_references
                 + baseline_diagnostics + baseline_projection_and_scores + baseline_mode_diagnostics)
        self.assertEqual(bound, 19719936)
        self.assertLess(bound, module.MAX_MAC)


if __name__ == '__main__':
    unittest.main()
