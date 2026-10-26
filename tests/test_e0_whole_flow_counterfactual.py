"""Synthetic math/call-order/authorization checks, no model or GPU assets."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_whole_flow_counterfactual import (LatentOnlyMain, WholeFlowCounterfactualController,
                                         flow_split_geometry, native_map_split_geometry)
from e0_paired_wan_controller import PairedWanController
from run_e0_whole_flow_counterfactual import PASS, close_attempt, execute, reserve_attempt
from test_e0_paired_wan_controller import fixture


def state(*values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, 1, -1)


def counterfactual_fixture(fail_dense=False):
    runner, observer, policy, adapter, factory, cuda, log = fixture()
    observer.runtime = NS(apply=lambda *a, **k: 'original_dense', counters={'linear_forward_calls': 0})
    observer.base_apply = observer.runtime.apply
    observer.active = observer.pending = None
    observer.site = {}
    observer.current_kind = None
    old_branch = observer.branch

    @contextmanager
    def branch(infer, **kwargs):
        observer.current_kind = kwargs['kind']
        try:
            with old_branch(infer, **kwargs):
                yield
        finally:
            observer.current_kind = None
    observer.branch = branch
    for model in runner.model.model:
        old_block = model.transformer_infer.infer_block
        def block(*args, old_block=old_block):
            value = old_block(*args)
            if observer.current_kind == 'candidate':
                value.add_(.03125)
            return value
        model.transformer_infer.infer_block = block
        if fail_dense:
            old_up = model.pre_weight.to_cuda
            def up(old_up=old_up):
                if observer.runtime.apply == observer.base_apply:
                    raise RuntimeError('synthetic extra dense allocation failure')
                old_up()
            model.pre_weight.to_cuda = up
    # A distinct installed observer hook, restored after each whole-dense path.
    observer.runtime.apply = lambda *a, **k: 'paired_hook'
    return runner, observer, policy, adapter, factory, cuda, log


class WholeFlowTest(unittest.TestCase):
    def test_finite_split_distinguishes_stable_dense_from_biased_candidate(self):
        out = flow_split_geometry(state(0), state(1), state(0), state(-1), state(0), h=.5)
        self.assertEqual(out['model_error_SSE'], 1)
        self.assertEqual(out['dense_state_response_SSE'], 0)
        self.assertEqual(out['dense_discrete_map_energy_gain'], 1)
        self.assertEqual(out['ideal_next_delta_SSE'], 2.25)
        self.assertEqual(out['candidate_dense_map_cross_term'], 1)

    def test_positive_delta_response_dot_still_allows_dense_discrete_expansion(self):
        out = flow_split_geometry(state(0), state(1), state(0), state(5), state(5), h=1)
        self.assertGreater(out['delta_state_response_dot'], 0)
        self.assertEqual(out['dense_discrete_map_energy_gain'], 16)
        self.assertEqual(out['model_error_SSE'], 0)

    def test_native_bf16_map_is_separate_from_ideal_flow_update(self):
        out = native_map_split_geometry(state(1).bfloat16(), state(3).bfloat16(), state(2).bfloat16())
        self.assertEqual(out['dense_map_response_SSE'], 1)
        self.assertEqual(out['candidate_effect_SSE'], 1)
        self.assertEqual(out['dense_map_candidate_effect_dot'], 1)
        self.assertEqual(out['actual_next_delta_SSE'], 4)
        with self.assertRaises(ValueError):
            flow_split_geometry(*([state(float('nan'))] * 5), h=1)
        with self.assertRaises(ValueError):
            flow_split_geometry(*([torch.zeros(2)] * 5), h=1)

    def test_tripled_full_block_path_preserves_original_latents_and_restores_hooks(self):
        runner, observer, policy, adapter, factory, cuda, log = counterfactual_fixture()
        with PairedWanController(runner, observer, policy, adapter, factory,
                backend=torch, cuda=cuda) as original:
            expected = runner.run_segment().clone()
            hashes = original.trajectory
        runner, observer, policy, adapter, factory, cuda, log = counterfactual_fixture()
        previous_apply = observer.runtime.apply
        saves = []
        with WholeFlowCounterfactualController(runner, observer, policy, adapter, factory,
                backend=torch, cuda=cuda, save_counterfactual=lambda step, values, record:
                (saves.append((step, list(values))) if values is not None else None) or {'synthetic': True}) as controller:
            actual = runner.run_segment()
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(controller.trajectory, hashes)
            self.assertEqual(len(controller.counterfactual_records), 4)
            self.assertIsNone(controller.counterfactual_records[0]['geometry']['dense_discrete_map_energy_gain'])
            self.assertTrue(all(len(keys) == 8 for _, keys in saves))
            self.assertEqual([m.transformer_infer.offload_manager.swaps for m in runner.model.model], [160, 160])
            self.assertIs(observer.runtime.apply, previous_apply)
        self.assertEqual(sum(x[0] == 'begin' for x in log), 320)
        self.assertEqual(sum(x[0] == 'end' for x in log), 320)
        self.assertIsNone(adapter._expert_ctx.get())

    def test_extra_dense_failure_restores_original_hook_and_scheduler(self):
        runner, observer, policy, adapter, factory, cuda, log = counterfactual_fixture(fail_dense=True)
        previous_apply, scheduler = observer.runtime.apply, runner.scheduler
        with self.assertRaisesRegex(RuntimeError, 'extra dense allocation failure'):
            with WholeFlowCounterfactualController(runner, observer, policy, adapter, factory,
                                                  backend=torch, cuda=cuda):
                runner.run_segment()
        self.assertIs(observer.runtime.apply, previous_apply)
        self.assertIs(runner.model.scheduler, scheduler)
        self.assertIn(('pre_down', 'high_noise_model'), log)
        self.assertIn(('head_down', 'high_noise_model'), log)

    def test_nonfinite_extra_flow_preserves_raw_capture_before_geometry_rejection(self):
        runner, observer, policy, adapter, factory, cuda, log = counterfactual_fixture()
        saved = []
        with self.assertRaisesRegex(ValueError, 'Nonfinite whole-flow'):
            with WholeFlowCounterfactualController(runner, observer, policy, adapter, factory,
                    backend=torch, cuda=cuda, save_counterfactual=lambda step, values, record:
                    saved.append((step, values)) or {'synthetic': True}) as controller:
                controller._dense_flow = lambda step, scheduler: torch.full_like(scheduler.latents.float(), float('nan'))
                runner.run_segment()
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(saved[0][1]), 8)
        self.assertEqual(controller.counterfactual_records[0]['status'], 'RAW_SAVED_GEOMETRY_PENDING')

    def test_latent_only_lifecycle_never_decodes_and_restores_on_exception(self):
        calls = []
        runner = NS(video_segment_num=1, run_main=lambda: 'old')
        for name in ['init_run', 'check_stop', 'end_run']:
            setattr(runner, name, lambda name=name: calls.append(name))
        for name in ['init_run_segment', 'end_run_segment']:
            setattr(runner, name, lambda segment, name=name: calls.append((name, segment)))
        def actual_end_segment(segment):
            runner.gen_video_final = runner.gen_video
            calls.append(('end_run_segment', segment))
        runner.end_run_segment = actual_end_segment
        runner.run_segment = lambda segment: calls.append(('run_segment', segment)) or 'latent'
        old = runner.run_main
        with LatentOnlyMain(runner):
            self.assertEqual(runner.run_main(), 'latent')
        self.assertIs(runner.run_main, old)
        self.assertIsNone(runner.gen_video_final)
        self.assertEqual(calls, ['init_run', 'check_stop', ('init_run_segment', 0),
            ('run_segment', 0), ('end_run_segment', 0), 'end_run'])

    def test_no_authority_or_changed_output_root_starts_no_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'separate explicit authorization'):
                execute(None, None, None, {'model_scope_authorized': False})
            with self.assertRaisesRegex(ValueError, 'canonical ledger'):
                execute(NS(study_root=root / 'other'), {'canonical_study_root': str(root / 'canonical')}, [],
                        {'model_scope_authorized': True})
            self.assertEqual(list(root.iterdir()), [])

    def test_failed_attempt_blocks_later_pair_and_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'id': 'fixture', 'pairs': [{'id': 'a'}, {'id': 'b'}]}
            reserve_attempt(root, plan, 'sha', {'id': 'a'})
            receipt = root / 'RUN.json'
            receipt.write_text(json.dumps({'status': PASS}))
            close_attempt(root, 'a', receipt)
            reserve_attempt(root, plan, 'sha', {'id': 'b'})
            receipt.write_text(json.dumps({'status': 'FAILED_PRESERVED_NO_RETRY'}))
            close_attempt(root, 'b', receipt)
            for pair in ['a', 'b']:
                with self.assertRaises(RuntimeError):
                    reserve_attempt(root, plan, 'sha', {'id': pair})
            self.assertEqual(len(json.loads((root / 'STUDY.json').read_text())['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
