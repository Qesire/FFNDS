"""Q1 fixed-iterate status contract without importing Torch."""
import ast
import math
import unittest
from pathlib import Path

path = Path(__file__).with_name('q1_v4_fixed_iterate_candidate.py')
tree = ast.parse(path.read_text(encoding='utf-8'))
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'validate_returned_iterate_info')
namespace = {'math': math}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)


class TestV4Status(unittest.TestCase):
    def test_finite_and_converged_are_separate(self):
        validate = namespace['validate_returned_iterate_info']
        base = {'status': 'CONVERGED', 'true_relative_residual': 9e-6, 'initial_residual': 1.0,
                'true_normal_residual': 9e-6, 'iterations': 80, 'restart_interval': 20,
                'certificate': 'CERTIFICATE_UNRESOLVED',
                'true_residual_history': [{'iteration': 0, 'true_residual': 1.0}]}
        validate(base)
        finite = dict(base, status='FINITE_ITERATE_NOT_OPTIMUM', true_relative_residual=3e-5,
                      true_normal_residual=3e-5, iterations=240)
        validate(finite)
        for invalid in (dict(finite, true_relative_residual=float('nan')), dict(finite, status='CONVERGED'),
                        dict(finite, iterations=241), dict(finite, iterations=239),
                        dict(finite, initial_residual=float('inf')), dict(finite, restart_interval=10),
                        dict(finite, true_residual_history=[{'iteration': 0, 'true_residual': float('nan')}])):
            with self.assertRaises(RuntimeError):
                validate(invalid)


if __name__ == '__main__':
    unittest.main()
