"""Different targets at identical deployed capacity; full original flow layout."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from native_head_replay import replay, unpatchify_flow
from reduced_rank_residual import predict_y
from run_native_window_objective import objective, native_run
from ar2_checkpoint import digest


class NativeWindowObjectiveTests(unittest.TestCase):
    def test_native_scheduler_keyword_contract_and_fullstep_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image'; image.write_bytes(b'image')
            sample = {'seed': 42, 'image_path': str(image), 'prompt': 'fixed',
                      'input_provenance': {'image_sha256': digest(image)}}
            payloads = [{'input': torch.tensor([float(i)]), 'flow': torch.tensor([float(i + 10)]),
                         'post': torch.tensor([float(i + 1)])} for i in range(4)]
            refs = {arm: {0: {'case': {'input': sample}, 'payloads': payloads}} for arm in ('DQ', 'UB1')}
            scheduler = SimpleNamespace()

            def pre(step_index):
                scheduler.step_index = step_index
                scheduler.latents = payloads[step_index]['input'].bfloat16()

            def post():
                scheduler.latents = payloads[scheduler.step_index]['post'].bfloat16()

            scheduler.step_pre, scheduler.step_post = pre, post
            runtime = SimpleNamespace(counters={'linear_forward_calls': 0, 'activation_qdq_calls': 0, 'cache_misses': 0}, begin_request=lambda: None)

            def generate(**kwargs):
                for step in range(4):
                    scheduler.step_pre(step_index=step)
                    scheduler.noise_pred = payloads[step]['flow'].clone()
                    runtime.counters['linear_forward_calls'] += 400
                    runtime.counters['activation_qdq_calls'] += 400
                    scheduler.step_post()

            pipe = SimpleNamespace(runner=SimpleNamespace(scheduler=scheduler), generate=generate)
            with mock.patch('run_native_window_objective.verify_runtime') as verify:
                receipt = native_run(pipe, runtime, refs, Path(tmp) / 'native', 0)
                verify.assert_called_once_with(runtime, 'UB1')
            self.assertEqual(len(receipt['steps']), 4)
            self.assertIs(scheduler.step_pre, pre)
            self.assertIs(scheduler.step_post, post)
            self.assertTrue(all(s['flow_byte_equal'] for s in receipt['steps']))

    def test_local_and_flow_have_distinct_targets_at_same_parameters(self):
        torch.manual_seed(503)
        d = {k: torch.randn(31668, 8).bfloat16() for k in ('u', 'main', 'base')}
        d.update({k: torch.randn(8).bfloat16() for k in ('gate', 'shift', 'scale')})
        d.update(head_weight=torch.randn(8, 4).bfloat16(), head_bias=torch.randn(4).bfloat16())
        p = {k: torch.randn(*shape).requires_grad_() for k, shape in
             [('first', (2, 8)), ('bias_first', (2,)), ('second', (8, 2)), ('bias_second', (8,))]}
        with torch.no_grad():
            d['teacher_y'] = d['main'] + predict_y(d['u'], p)
            d['teacher_flow'] = unpatchify_flow(replay(d, p), (21, 29, 52)).float() + .1
        self.assertEqual(float(objective(d, p, 'local').detach()), 0.)
        self.assertGreater(float(objective(d, p, 'flow').detach()), .009)
        objective(d, p, 'flow').backward()
        self.assertGreater(sum(float(v.grad.abs().sum()) for v in p.values()), 0.)
        with self.assertRaises(ValueError):
            objective(d, p, 'larger_rank')


if __name__ == '__main__':
    unittest.main()
