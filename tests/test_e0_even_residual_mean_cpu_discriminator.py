"""Synthetic mechanism/cost/failure checks; zero saved numerical/model/GPU work."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_even_residual_mean_cpu_discriminator as study


class EvenResidualSoftwareTests(unittest.TestCase):
    def test_exact_erf_gaussian_identity_zero_mean_and_zero_variance(self):
        mu = np.array([0., 0., -2., 1.])
        var = np.array([1., 3., 0., 0.])
        actual = study.erf_gaussian_mean(mu, var, np)
        expected = [1/(2*math.sqrt(math.pi)), 3/(2*math.sqrt(2*math.pi)),
                    -math.erfc(math.sqrt(2)), .5*math.erfc(-1/math.sqrt(2))]
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
        with self.assertRaises(ValueError):
            study.erf_gaussian_mean(np.zeros(1), np.array([-1.]), np)

    def test_tanh_odd_even_identity_and_gaussian_rule(self):
        x = np.array([.01, .5, 2., 5.])
        np.testing.assert_allclose(study.tanh_gelu(x, np)-study.tanh_gelu(-x, np), x, atol=1e-15)
        # Symmetric paired derivative has exactly half mean, yet GELU mean varies.
        eps = 1e-5
        deriv = lambda v: (study.tanh_gelu(v+eps, np)-study.tanh_gelu(v-eps, np))/(2*eps)
        np.testing.assert_allclose((deriv(x)+deriv(-x))/2, .5, atol=4e-11)
        self.assertGreater(study.tanh_gelu(np.array([2., -2.]), np).mean(),
                           study.tanh_gelu(np.array([.1, -.1]), np).mean())
        self.assertAlmostEqual(sum(study.GH_WEIGHTS), 1., places=14)
        self.assertAlmostEqual(sum(w*x*x for w,x in zip(study.GH_WEIGHTS, study.GH_NODES)), .5, places=14)
        fixed = study.tanh_gaussian_mean(np.array([1., -2.]), np.zeros(2), np)
        np.testing.assert_allclose(fixed, study.tanh_gelu(np.array([1., -2.]), np), atol=1e-14)

    def test_ignored_covariance_can_create_variance_from_an_exact_zero_projection(self):
        q = np.array([[-1., 1.], [1., -1.]])
        mean, var = study.moments(q, np)
        work = study.Work(maximum=4)
        mu, diagonal = study.input_moments(mean[None], var[None], np.array([[1., 1.]]), np.zeros((1,1)), work, np)
        actual = q @ np.array([[1.], [1.]])
        self.assertEqual(float(actual.var()), 0.)
        self.assertEqual(float(diagonal[0,0]), 2.)
        self.assertGreater(float(study.erf_gaussian_mean(mu, diagonal, np)[0,0]), 0.)
        self.assertEqual(work.mac, 4)

    def test_same_mean_and_variance_different_nonGaussian_gelu_mean(self):
        # Exact finite support distribution; no random draw or model data.
        epsilon = 1e-4
        amplitude = 1/math.sqrt(epsilon)
        rare_mean = math.sqrt(epsilon) * (.5*math.erfc(-amplitude/math.sqrt(2))-.5)
        gaussian_mean = study.erf_gaussian_mean(np.zeros(1), np.ones(1), np)[0]
        self.assertAlmostEqual(epsilon*amplitude*amplitude, 1.)
        self.assertLess(rare_mean, gaussian_mean/50)

    def test_Smooth2_outside_GELU_and_mean_Q2_noncommutativity(self):
        z = np.array([[-2., 1.], [2., 3.]])
        scale = np.array([2., .5])
        correct = (study.tanh_gelu(z, np)/scale).mean(0)
        moved_inside = study.tanh_gelu(z/scale, np).mean(0)
        self.assertGreater(float(np.linalg.norm(correct-moved_inside)), .01)
        # Stand-in nonlinear quantizer counterexample, explicitly not native HiF4.
        q = lambda x: np.where(x > .5, 1., 0.)
        values = np.array([.1, .6])
        self.assertNotEqual(float(q(values).mean()), float(q(values.mean())))

    def test_signed_W2_and_gate_keep_cancellation_and_different_channel_consequences(self):
        coefficients = np.array([[1., 1.]])
        w2 = np.array([[1., -1.], [2., 3.]])
        y = study.Work(maximum=4).dot(coefficients, w2.T)[0]
        self.assertEqual(y.tolist(), [0., 5.])
        metric = study.mean_error(y, np.zeros(2), np.array([1., 0.]), 128, np)
        self.assertEqual(metric['raw']['mean_vector_error_L2_squared'], 25.)
        self.assertEqual(metric['FP64_current_gate']['mean_vector_error_L2_squared'], 0.)
        self.assertIsNone(metric['raw']['relative_mean_vector_SSE'])

    def test_mean_success_can_hide_all_tokenwise_error(self):
        target = np.array([[-1., 2.], [1., -2.]])
        prediction = np.zeros_like(target)
        self.assertEqual(float(np.linalg.norm(target.mean(0)-prediction.mean(0))), 0.)
        self.assertEqual(float(((target-prediction)**2).sum()), 10.)

    def test_readout_bridge_keeps_bias_once_and_saved_preQ2_optional(self):
        q = np.array([[-1., 1.], [1., -1.]])
        arrays = {'z': q, 'gelu': study.tanh_gelu(q, np), 'q2_post': q,
                  'q1_post': q, 'gate': np.ones((1,2)), 'b1': np.ones(2), 'b2': np.array([3., -2.]),
                  'ffn_output': q + np.array([3., -2.])}
        names, means, mu, var = study.coefficient_means(arrays, np.ones(2), np.zeros(2), np.ones(2), np)
        report = study.summarize_panel(names, means, means, arrays, mu, var, mu, var, np)
        self.assertEqual(len(names), 6)
        self.assertNotIn('saved_postQ2_minus_saved_preQ2_Q2_ONLY', report['stage_bridge_mean_differences'])
        self.assertEqual(report['means_vs_actual_saved_FFN_mean_b2_added_once']['decoded_postQ2_plus_b2']['raw']['mean_vector_error_L2_squared'], 0.)
        arrays['q2_pre'] = q.copy()
        names, means, mu, var = study.coefficient_means(arrays, np.ones(2), np.zeros(2), np.ones(2), np)
        report = study.summarize_panel(names, means, means, arrays, mu, var, mu, var, np)
        self.assertEqual(len(names), 7)
        self.assertIn('saved_postQ2_minus_saved_preQ2_Q2_ONLY', report['stage_bridge_mean_differences'])

    def test_MAC_precharge_includes_failed_attempt_and_excludes_unlaunched_overbudget(self):
        class Fault:
            ndim, shape = 2, (1, 2)
            def __matmul__(self, other):
                raise ArithmeticError('synthetic multiplication failure')
        work = study.Work(maximum=2)
        with self.assertRaises(ArithmeticError):
            work.dot(Fault(), np.ones((2,1)))
        self.assertEqual((work.mac, work.completed_mac, work.products, work.completed_products), (2, 0, 1, 0))
        with self.assertRaises(RuntimeError):
            work.dot(np.ones((1,2)), np.ones((2,1)))
        self.assertEqual(work.mac, 2)
        self.assertEqual(study.MAX_MAC, 2_406_481_920)

    def test_header_selected_row_byte_order_and_bad_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mock.safetensors'
            descriptor = {'x': {'dtype': 'BF16', 'shape': [3,2], 'data_offsets': [0,12]}}
            header = json.dumps(descriptor).encode()
            path.write_bytes(struct.pack('<Q', len(header))+header+bytes(range(12)))
            self.assertEqual(study.field_bytes(path, 'x', [2,0]), bytes(range(8,12))+bytes(range(4)))
            with self.assertRaises(ValueError):
                study.field_bytes(path, 'x', [3])
            descriptor['x']['dtype'] = 'F16'
            header = json.dumps(descriptor).encode()
            path.write_bytes(struct.pack('<Q', len(header))+header+bytes(range(12)))
            with self.assertRaises(ValueError):
                study.field_bytes(path, 'x')
            with self.assertRaises(ValueError):
                study.relative_path(directory, '../escape')

    def test_failed_import_preserves_single_attempt_ledger_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory)/'plan.json'
            plan_path.write_text('{}')
            out = Path(directory)/'attempt1'
            plan = {'canonical_output': str(out), 'max_wall_seconds': 180,
                    'max_output_bytes': 1 << 19, 'weight_shard_sha256': 'synthetic',
                    'max_RSS_bytes': 4 << 30}
            real_import = __import__
            def fail_numpy(name, *args, **kwargs):
                if name == 'numpy':
                    raise ImportError('synthetic import failure')
                return real_import(name, *args, **kwargs)
            with patch.object(study, 'require_registration'), patch.object(study, 'sys', SimpleNamespace(modules={})), patch('builtins.__import__', side_effect=fail_numpy), patch.dict('os.environ'):
                with self.assertRaises(ImportError):
                    study.execute(plan_path, plan, [], Path(directory)/'unused.pt')
                ledger = json.loads((out/'RUN.json').read_text())
                self.assertEqual(ledger['status'], 'FAILED_PRESERVED_NO_RETRY')
                self.assertEqual(ledger['CPU_matmul_MAC'], 0)
                self.assertEqual(ledger['GPU_calls'], 0)
                self.assertFalse(ledger['automatic_retry'])
                with self.assertRaises(FileExistsError):
                    study.execute(plan_path, plan, [], Path(directory)/'unused.pt')

    def test_aggregate_output_cap_and_default_import_are_numerically_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            study.save_result(out, {'status': 'small'}, 100)
            with self.assertRaises(RuntimeError):
                study.save_result(out, {'status': 'x'*90}, 100)
            self.assertEqual(json.loads((out/'RUN.json').read_text())['status'], 'small')
            self.assertEqual([p.name for p in out.iterdir()], ['RUN.json'])
        snippet = ('import sys;sys.path.insert(0,' + repr(str(ROOT/'tools')) + ');'
                   'import e0_even_residual_mean_cpu_discriminator;'
                   'assert "numpy" not in sys.modules and "torch" not in sys.modules')
        completed = subprocess.run([sys.executable, '-c', snippet], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
