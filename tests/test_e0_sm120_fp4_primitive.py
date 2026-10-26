from contextlib import ExitStack
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_e0_sm120_fp4_primitive as primitive
from run_e0_sm120_fp4_primitive import native_pass, foreign_pids


class NativePrimitiveGuards(unittest.TestCase):
    def test_upstream_unsupported_architecture_exit_zero_is_not_pass(self):
        self.assertFalse(native_pass(0, 'This example requires a GPU of Blackwell architecture.'))
        self.assertFalse(native_pass(0, ''))

    def test_explicit_correctness_is_required_even_with_timing(self):
        self.assertFalse(native_pass(0, 'Avg runtime: .1 ms\nDisposition: Failed'))
        self.assertFalse(native_pass(1, 'Disposition: Passed'))
        self.assertTrue(native_pass(0, 'Disposition: Passed\nAvg runtime: .1 ms'))

    def test_only_actual_child_is_allowed_to_compute(self):
        self.assertEqual(foreign_pids([['17', '500']], {17}), [])
        self.assertEqual(foreign_pids([['17', '500'], ['99', '100']], {17}), [99])
        self.assertEqual(foreign_pids([['17', '500']], set()), [17])

    def exercise_mock_failure(self, kind):
        """Exercise the real ledger/cleanup, with no subprocess or NVML call."""
        child = Mock(pid=17, returncode=None)
        child.poll.side_effect = lambda: child.returncode
        child.terminate.side_effect = lambda: setattr(child, 'returncode', -15)
        child.wait.side_effect = lambda timeout=None: child.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'research').mkdir()
            protocol = root / 'protocol.json'; protocol.write_text('{}')
            private = root / 'private.json'; private.write_text('{}')
            binary = root / 'never_executed'; binary.write_bytes(b'synthetic binary identity only')
            (root / 'research/ACTIVE_RESEARCH.json').write_text(json.dumps({
                'experiment_authority': {'current_user_authorized_scopes': [{
                    'id': primitive.ID, 'protocol_sha256': primitive.sha(protocol),
                    'execution_status': 'AUTHORIZED_NOT_STARTED'}]}}))
            output = root / 'output'
            plan = {'private_worker_config_sha256': primitive.sha(private),
                    'canonical_output': str(output), 'max_HF_output_bytes': 1048576}

            def query(index, argument):
                self.assertEqual(index, '0')
                if argument == '--query-gpu=name,compute_cap,memory.total,driver_version':
                    return [['NVIDIA Synthetic Blackwell', '12.0', '1000', 'synthetic']]
                if argument == '--query-compute-apps=pid':
                    return []
                if argument == '--query-compute-apps=pid,used_memory':
                    return [['17', '64'], ['99', '32']] if kind == 'foreign' else [['17', '64']]
                if argument == '--query-gpu=memory.used':
                    return [['96']]
                raise AssertionError('Unexpected synthetic query: ' + argument)

            def launch(command, **kwargs):
                if kind == 'launch':
                    raise OSError('synthetic launch failure')
                kwargs['stdout'].write(b'synthetic child log preserved\n')
                kwargs['stdout'].flush()
                return child

            clock_calls = [0]
            def clock():
                clock_calls[0] += 1
                return 0. if clock_calls[0] <= 2 else 121.

            with ExitStack() as stack:
                stack.enter_context(patch.object(primitive, 'ROOT', root))
                stack.enter_context(patch.object(primitive, 'private_config_path', lambda: private))
                stack.enter_context(patch.object(primitive, 'require_worker'))
                stack.enter_context(patch.object(primitive, 'gpu_uuid', return_value='GPU-synthetic'))
                stack.enter_context(patch.object(primitive, 'query', side_effect=query))
                popen = stack.enter_context(patch.object(primitive.subprocess, 'Popen', side_effect=launch))
                stack.enter_context(patch.dict(os.environ, {
                    'FFNDS_PHYSICAL_GPU': '0', 'CUDA_VISIBLE_DEVICES': 'GPU-synthetic'}))
                if kind == 'timeout':
                    stack.enter_context(patch.object(primitive.time, 'monotonic', side_effect=clock))
                error = {'foreign': RuntimeError, 'timeout': TimeoutError, 'launch': OSError}[kind]
                with self.assertRaises(error):
                    primitive.execute(protocol, plan, binary)

            run = json.loads((output / 'RUN.json').read_text())
            self.assertEqual(run['status'], 'FAILED_PRESERVED_NO_RETRY')
            self.assertFalse(run['automatic_retry'])
            self.assertEqual(len(run['cases']), 1)
            self.assertEqual(popen.call_count, 1)
            case = run['cases'][0]
            self.assertTrue(case['launch_attempted'])
            self.assertEqual(case['GEMMs_reserved'], 1)
            self.assertEqual(case['GEMMs_completed'], 'UNKNOWN_UNTIL_NATIVE_PASS')
            self.assertEqual(case['command'], popen.call_args.args[0])
            self.assertEqual(run['actual_logs'], [{
                'path': 'case0.log', 'sha256': primitive.sha(output / 'case0.log'),
                'bytes': (output / 'case0.log').stat().st_size}])
            self.assertFalse((output / 'case1.log').exists())
            return run, child

    def test_foreign_gpu_pid_terminates_actual_child_and_preserves_terminal_ledger(self):
        run, child = self.exercise_mock_failure('foreign')
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=10)
        child.kill.assert_not_called()
        case = run['cases'][0]
        self.assertTrue(case['child_started'])
        self.assertEqual(case['child_PID'], 17)
        self.assertEqual(case['status'], 'FAILED_CHILD_TERMINAL')
        self.assertEqual(case['returncode'], -15)
        self.assertGreaterEqual(case['wall_seconds'], 0.)
        self.assertEqual(run['observed_resources'][0]['foreign_count'], 1)

    def test_timeout_terminates_child_without_starting_second_case(self):
        run, child = self.exercise_mock_failure('timeout')
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=10)
        case = run['cases'][0]
        self.assertEqual(case['status'], 'FAILED_CHILD_TERMINAL')
        self.assertEqual(case['returncode'], -15)
        self.assertEqual(case['error_type'], 'TimeoutError')
        self.assertEqual(case['wall_seconds'], 121.)
        self.assertEqual(run['observed_resources'][0]['foreign_count'], 0)

    def test_launch_failure_is_distinct_from_a_started_native_child(self):
        run, child = self.exercise_mock_failure('launch')
        child.terminate.assert_not_called()
        child.wait.assert_not_called()
        case = run['cases'][0]
        self.assertFalse(case['child_started'])
        self.assertIsNone(case['child_PID'])
        self.assertIsNone(case['returncode'])
        self.assertEqual(case['status'], 'FAILED_BEFORE_CHILD_START')
        self.assertEqual(case['error_type'], 'OSError')


if __name__ == '__main__':
    unittest.main()
