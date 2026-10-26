"""Synthetic CPU call-order integration. No native Wan/HiF4 or model assets."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_e0_full_depth_interface import REQUIRED_CONFIG
from e0_full_depth_trace_contract import FullDepthTrace
from e0_paired_wan_controller import PairedWanController, tensor_identity


class Wan22StepDistillScheduler:
    def __init__(self, config):
        self.config = config

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed))
        self.sigmas = torch.tensor([1., .75, .5, .25])

    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.bfloat16()

    def step_post(self):
        value = self.latents.float() - self.noise_pred.float() * self.sigmas[self.step_index]
        if self.step_index < 3:
            value = value + self.noise_pred.float() * self.sigmas[self.step_index + 1]
        self.latents = value.to(self.latents.dtype)


class Stream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class Manager:
    def __init__(self):
        self.compute_stream = Stream(); self.need_init_first_buffer = True
        self.cuda_buffers = [None]; self.swaps = 0

    def init_first_buffer(self, blocks):
        self.blocks = blocks; self.cuda_buffers[0] = blocks[0]; self.need_init_first_buffer = False

    def prefetch_weights(self, index, blocks):
        self.next = blocks[index]

    def swap_blocks(self):
        self.cuda_buffers[0] = self.next; self.swaps += 1


class FixtureObserver:
    @contextmanager
    def branch(self, infer, **kw):
        self.branch_calls.append(kw)
        yield

    def finish_site(self, step, block, expert, a, b, grid):
        arms = dict(dense_teacher=a, approx_teacher=a, dense_student=b, approx_student=b)
        self.trace.add_site(step, block, expert, teacher_q1=a, student_q1=b,
            ffn_outputs=arms, native_writebacks=arms, teacher_block=a, student_block=b,
            numerical_check=dict(status='MEASURED_SAME_INPUT', coefficient_provenance_sha256='SYNTHETIC',
                                 row_ids=[0], teacher_native_vs_continuous={'difference_SSE': 0},
                                 student_native_vs_continuous={'difference_SSE': 0}))


def fixture(fail_at=None):
    config = {k: v for k, v in REQUIRED_CONFIG.items() if k not in ['scheduler', 'attention']}
    config.update({key: 'torch_sdpa' for key in ['self_attn_1_type', 'cross_attn_1_type', 'cross_attn_2_type']})
    adapter = NS(**{name: ContextVar(name, default=None) for name in ['_expert_ctx', '_step_ctx', '_block_ctx']})
    log = []
    models = []
    for expert in ['high_noise_model', 'low_noise_model']:
        blocks = [NS(index=b, compute_phases=[NS(), NS(), NS()]) for b in range(40)]
        manager = Manager()
        model = NS()
        def infer_block(current, x, pre, model=model, expert=expert):
            step = model.scheduler.step_index; block = model.transformer_infer.block_idx
            assert current.index == block
            assert (adapter._expert_ctx.get(), adapter._step_ctx.get(), adapter._block_ctx.get()) == (expert, step, block)
            if fail_at == (step, block):
                raise RuntimeError('injected interruption')
            log.append(('block', expert, step, block))
            return x.add_(.125)
        infer = NS(has_post_adapter=False, seq_p_group=None, use_compile=False, lazy_load=False,
                   infer_dtype=torch.bfloat16, sensitive_layer_dtype=torch.bfloat16,
                   offload_manager=manager, _use_mxfp8_quant_fuse=lambda: False,
                   reset_attention_states=lambda blocks: None, reset_infer_states=lambda x, context: None,
                   infer_block=infer_block, infer_non_blocks=lambda weights, x, embed: x)
        model.transformer_infer = infer
        model.transformer_weights = NS(blocks=blocks,
            non_block_weights_to_cuda=lambda e=expert: log.append(('head_up', e)),
            non_block_weights_to_cpu=lambda e=expert: log.append(('head_down', e)))
        model.pre_weight = NS(to_cuda=lambda e=expert: log.append(('pre_up', e)),
                              to_cpu=lambda e=expert: log.append(('pre_down', e)))
        def pre_infer(weights, inputs, model=model):
            return NS(x=model.scheduler.latents.reshape(27, 1).clone(), grid_sizes=NS(tuple=(3, 3, 3)),
                      embed=torch.zeros(1), embed0=torch.zeros(1), context=torch.zeros(1), cos_sin=torch.zeros(1))
        model.pre_infer = NS(infer=pre_infer)
        model.post_infer = NS(infer=lambda x, pre: [x.reshape(1, 3, 3, 3).float()])
        models.append(model)
    wrapper = NS(model=models, cur_model_index=-1)
    def bind(scheduler):
        wrapper.scheduler = scheduler
        for m in models:
            m.scheduler = scheduler
    wrapper.set_scheduler = bind
    def select():
        wrapper.cur_model_index = 0 if wrapper.scheduler.step_index < 2 else 1
    wrapper.get_current_model_index = select
    scheduler = Wan22StepDistillScheduler(config); scheduler.prepare(42, (1, 3, 3, 3))
    bind(scheduler)
    runner = NS(config=config, model=wrapper, scheduler=scheduler, run_segment=lambda i: 'old',
                input_info=NS(seed=42, latent_shape=(1, 3, 3, 3)), inputs={'image_encoder_output': {}},
                video_segment_num=1, check_stop=lambda: None)
    observer = FixtureObserver(); observer.branch_calls = []
    policy = NS(begin_block=lambda e, b: log.append(('begin', e, b)),
                end_block=lambda fence: (fence(), log.append(('end',))))
    cuda = NS(current_stream=lambda: Stream(), stream=lambda stream: nullcontext(), synchronize=lambda: None)
    def trace_factory(shape, state_shape):
        return FullDepthTrace(shape, 0, 42, dict(protocol_sha256='fixture', native_source_sha256='fixture',
                    model_digest='fixture', bank_RUN_sha256='fixture'), state_shape=state_shape, synthetic=True)
    return runner, observer, policy, adapter, trace_factory, cuda, log


class ControllerTest(unittest.TestCase):
    def test_full_four_step_order_and_restore_without_native_assets(self):
        runner, observer, policy, adapter, factory, cuda, log = fixture()
        old_run, old_scheduler = runner.run_segment, runner.scheduler
        with PairedWanController(runner, observer, policy, adapter, factory,
                                 modes=('dense', 'dense'), backend=torch, cuda=cuda) as controller:
            result = runner.run_segment(0)
            self.assertEqual(controller.completed_trace['status'], 'SYNTHETIC_SOFTWARE_VALIDATION_ONLY')
            self.assertEqual(len(controller.completed_trace['sites']), 160)
            self.assertEqual(len(observer.branch_calls), 320)
            self.assertTrue(torch.equal(result, controller.teacher_latents))
            self.assertEqual([m.transformer_infer.offload_manager.swaps for m in runner.model.model], [80, 80])
        self.assertIs(runner.run_segment, old_run)
        self.assertIs(runner.model.scheduler, old_scheduler)
        self.assertEqual(runner.model.cur_model_index, -1)
        self.assertIsNone(adapter._expert_ctx.get())
        self.assertEqual(sum(x[0] == 'begin' for x in log), 160)
        self.assertEqual(sum(x[0] == 'end' for x in log), 160)

    def test_injected_block_failure_restores_binding_context_and_weight_lifecycle(self):
        runner, observer, policy, adapter, factory, cuda, log = fixture(fail_at=(0, 3))
        old = runner.run_segment
        with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
            with PairedWanController(runner, observer, policy, adapter, factory,
                                     modes=('dense', 'dense'), backend=torch, cuda=cuda):
                runner.run_segment(0)
        self.assertIs(runner.run_segment, old)
        self.assertIsNone(adapter._step_ctx.get())
        self.assertEqual(sum(x[0] == 'begin' for x in log), sum(x[0] == 'end' for x in log))
        self.assertIn(('pre_down', 'high_noise_model'), log)
        self.assertIn(('head_down', 'high_noise_model'), log)

    def test_hash_binds_dtype_even_when_values_match(self):
        x = torch.ones(2, 2)
        self.assertNotEqual(tensor_identity(x), tensor_identity(x.bfloat16()))
        self.assertEqual(tensor_identity(x), tensor_identity(x.clone()))

    def test_failure_during_head_load_still_unloads_pre_and_restores_scheduler(self):
        runner, observer, policy, adapter, factory, cuda, log = fixture()
        def fail():
            raise RuntimeError('injected head allocation failure')
        runner.model.model[0].transformer_weights.non_block_weights_to_cuda = fail
        old = runner.run_segment
        with self.assertRaisesRegex(RuntimeError, 'injected head allocation failure'):
            with PairedWanController(runner, observer, policy, adapter, factory,
                                     modes=('dense', 'dense'), backend=torch, cuda=cuda):
                runner.run_segment(0)
        self.assertIs(runner.run_segment, old)
        self.assertIn(('pre_down', 'high_noise_model'), log)
        self.assertIn(('head_down', 'high_noise_model'), log)


if __name__ == '__main__':
    unittest.main()
