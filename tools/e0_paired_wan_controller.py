"""Single-pipeline, per-block interleaving of two real four-step Wan trajectories.

Integration component, not an experiment launcher. An authorized driver must
bind source/config/assets, install native HiF4 hooks and supply output limits.
"""
from contextlib import contextmanager
import hashlib
import json
import sys
import time

from check_e0_full_depth_interface import REQUIRED_CONFIG
from e0_full_depth_trace_contract import scheduler_geometry, step_pre_geometry


def tensor_identity(value):
    """Bind actual dtype, shape and storage bytes, including native BF16 bits."""
    import torch
    cpu = value.detach().contiguous().cpu()
    header = {'dtype': str(cpu.dtype), 'shape': list(cpu.shape), 'storage': 'contiguous_' + sys.byteorder + '_endian'}
    payload = json.dumps(header, sort_keys=True, separators=(',', ':')).encode() + b'\0'
    return hashlib.sha256(payload + cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def target_context(adapter, expert, step, block):
    settings = [(adapter._expert_ctx, expert), (adapter._step_ctx, step), (adapter._block_ctx, block)]
    tokens = [(context, context.set(value)) for context, value in settings]
    try:
        yield
    finally:
        for context, token in reversed(tokens):
            context.reset(token)


class PairedWanController:
    def __init__(self, runner, observer, residency, adapter, trace_factory, *,
                 modes=('dense', 'candidate'), backend=None, cuda=None, progress=None):
        if tuple(modes) not in [('dense', 'dense'), ('candidate', 'candidate'), ('dense', 'candidate')]:
            raise ValueError('Unknown frozen trajectory pair')
        self.runner, self.observer, self.policy, self.adapter = runner, observer, residency, adapter
        self.trace_factory, self.modes, self.progress = trace_factory, tuple(modes), progress
        self.torch, self.cuda = backend, cuda
        self.completed_trace = None
        self.trajectory = {'teacher': [], 'student': []}
        self.step_wall_seconds = []

    def __enter__(self):
        if self.torch is None:
            import torch
            self.torch = torch
        if self.cuda is None:
            self.cuda = self.torch.cuda
        config = self.runner.config
        for key, expected in REQUIRED_CONFIG.items():
            if key in ['scheduler', 'attention']:
                continue  # Contract aliases are not actual LightX2V runtime config keys.
            if key not in config or config[key] != expected or type(config[key]) is not type(expected):
                raise ValueError('Unsupported effective paired config: ' + key)
        if type(self.runner.scheduler).__name__ != 'Wan22StepDistillScheduler':
            raise ValueError('Actual Wan22 distilled scheduler instance required')
        if any(config.get(key) != 'torch_sdpa' for key in
               ['self_attn_1_type', 'cross_attn_1_type', 'cross_attn_2_type']):
            raise ValueError('Actual native attention implementations must be torch_sdpa')
        for key in ['lazy_load', 'unload_modules', 'use_compile', 'compile', 'use_stream_vae',
                    'lora_dynamic_apply', 'lora_configs', 'video_frame_interpolation']:
            if config.get(key):
                raise ValueError('Unsupported paired source branch: ' + key)
        self.wrapper = self.runner.model
        if len(self.wrapper.model) != 2:
            raise ValueError('Exactly two already-loaded experts required')
        for model in self.wrapper.model:
            infer = model.transformer_infer
            if (infer.has_post_adapter or infer.seq_p_group is not None or infer.use_compile
                    or len(model.transformer_weights.blocks) != 40 or infer.lazy_load or
                    infer.infer_dtype != self.torch.bfloat16 or infer.sensitive_layer_dtype != self.torch.bfloat16):
                raise ValueError('Only plain NoCaching block-offload Wan is supported')
        self.old_run_segment = self.runner.run_segment
        self.old_scheduler = self.runner.scheduler
        self.old_current_index = self.wrapper.cur_model_index
        self.runner.run_segment = self.run_segment
        return self

    def __exit__(self, *error):
        self.runner.run_segment = self.old_run_segment
        self.runner.scheduler = self.old_scheduler
        self.wrapper.set_scheduler(self.old_scheduler)
        self.wrapper.cur_model_index = self.old_current_index

    def _bind(self, scheduler):
        self.wrapper.set_scheduler(scheduler)

    def _metadata(self, infer, pre):
        infer.cos_sin = pre.cos_sin
        infer.reset_infer_states(pre.x, pre.context)

    def _step(self, step, schedulers):
        torch, cuda = self.torch, self.cuda
        before = {name: scheduler.latents.detach().clone() for name, scheduler in schedulers.items()}
        hashes = {name + '_before_step_pre': tensor_identity(value) for name, value in before.items()}
        for scheduler in schedulers.values():
            scheduler.step_pre(step_index=step)
            scheduler.infer_condition = True
        after_pre = {name: scheduler.latents.detach().clone() for name, scheduler in schedulers.items()}
        hashes.update({name + '_after_step_pre': tensor_identity(value) for name, value in after_pre.items()})
        entry = step_pre_geometry(before['teacher'], before['student'], after_pre['teacher'], after_pre['student'])
        self._bind(schedulers['teacher'])
        self.wrapper.get_current_model_index()
        index = self.wrapper.cur_model_index  # Native method intentionally returns None.
        if index != (0 if step < 2 else 1):
            raise RuntimeError('Wrong native expert schedule')
        expert = 'high_noise_model' if index == 0 else 'low_noise_model'
        model = self.wrapper.model[index]
        infer, weights = model.transformer_infer, model.transformer_weights
        try:
            model.pre_weight.to_cuda()
            weights.non_block_weights_to_cuda()
            pre = {}
            for name, scheduler in schedulers.items():
                self._bind(scheduler)
                pre[name] = model.pre_infer.infer(model.pre_weight, self.runner.inputs)
            grid = tuple(pre['teacher'].grid_sizes.tuple)
            if grid != tuple(pre['student'].grid_sizes.tuple):
                raise RuntimeError('Paired raster grids differ')
            for field in ['embed', 'embed0', 'context', 'cos_sin']:
                if not torch.equal(getattr(pre['teacher'], field), getattr(pre['student'], field)):
                    raise RuntimeError('Paired time/conditioning/RoPE differs: ' + field)
            if step == 0:
                if not torch.equal(pre['teacher'].x, pre['student'].x):
                    raise RuntimeError('First native pre hidden differs before any approximation')
                self.trace = self.trace_factory(tuple(pre['teacher'].x.shape), tuple(after_pre['teacher'].shape))
                self.observer.trace = self.trace
            if tuple(pre['teacher'].x.shape) != self.trace.shape:
                raise RuntimeError('Full-M shape changed across steps')
            infer.reset_attention_states(weights.blocks)
            hidden = {name: value.x for name, value in pre.items()}
            manager = infer.offload_manager
            manager.compute_stream.wait_stream(cuda.current_stream())
            for block in range(40):
                if self.progress is not None:
                    self.progress(step, block)
                infer.block_idx = block
                if manager.need_init_first_buffer:
                    manager.init_first_buffer(weights.blocks)
                manager.prefetch_weights((block + 1) % 40, weights.blocks)
                self.policy.begin_block(expert, block)
                try:
                    with cuda.stream(manager.compute_stream):
                        current = manager.cuda_buffers[0]
                        if (hasattr(current.compute_phases[0], 'before_proj') or
                                hasattr(current.compute_phases[2], 'after_proj') or
                                infer._use_mxfp8_quant_fuse()):
                            raise ValueError('Unexpected adapter or fused FFN path')
                        for (name, scheduler), kind in zip(schedulers.items(), self.modes):
                            self._bind(scheduler)
                            self._metadata(infer, pre[name])
                            with target_context(self.adapter, expert, step, block), self.observer.branch(
                                    infer, branch=name, kind=kind, expert=expert, step=step, block=block):
                                hidden[name] = infer.infer_block(current, hidden[name], pre[name])
                        # Both native outputs are consumed before either next block mutates them.
                        self.observer.finish_site(step, block, expert, hidden['teacher'], hidden['student'], grid)
                        if self.modes[0] == self.modes[1] and not torch.equal(hidden['teacher'], hidden['student']):
                            raise RuntimeError('Identity-control trajectories diverged inside a block')
                    manager.swap_blocks()  # Exactly once, after two trajectories and shadows.
                finally:
                    self.policy.end_block(manager.compute_stream.synchronize)
            cuda.current_stream().wait_stream(manager.compute_stream)
            flows = {}
            for name, scheduler in schedulers.items():
                self._bind(scheduler)
                head = infer.infer_non_blocks(weights, hidden[name], pre[name].embed)
                flow = model.post_infer.infer(head, pre[name])[0]
                flows[name] = flow
                scheduler.noise_pred_cond = scheduler.noise_pred_guided = scheduler.noise_pred = flow
                scheduler.noise_pred_uncond = None
            cuda.synchronize()
        finally:
            original_error = sys.exc_info()[1]
            cleanup_errors = []
            for operation in [cuda.synchronize, model.pre_weight.to_cpu, weights.non_block_weights_to_cpu]:
                try:
                    operation()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                message = 'Paired step cleanup failed; terminate this attempt: ' + repr(cleanup_errors)
                if original_error is None:
                    raise RuntimeError(message) from cleanup_errors[0]
                if hasattr(original_error, 'add_note'):
                    original_error.add_note(message)
        for scheduler in schedulers.values():
            scheduler.step_post()
        after = {name: scheduler.latents for name, scheduler in schedulers.items()}
        hashes.update({name + '_after_step_post': tensor_identity(value) for name, value in after.items()})
        sigma = float(schedulers['teacher'].sigmas[step])
        sigma_next = float(schedulers['teacher'].sigmas[step + 1]) if step < 3 else 0.
        geometry = scheduler_geometry(after_pre['teacher'], after_pre['student'], flows['teacher'], flows['student'],
                                       after['teacher'], after['student'], sigma=sigma, sigma_next=sigma_next)
        self.trace.add_step(step, geometry, entry_cast=entry, boundary_hashes=hashes,
                            pre_dtype_teacher=str(after_pre['teacher'].dtype), post_dtype_teacher=str(after['teacher'].dtype),
                            pre_dtype_student=str(after_pre['student'].dtype), post_dtype_student=str(after['student'].dtype))
        for name, value in after.items():
            legacy = hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            self.trajectory[name].append({'step': step, 'sha256': legacy, 'shape': list(value.shape),
                                         'native_dtype': str(value.dtype), 'native_identity': hashes[name + '_after_step_post']})

    def run_segment(self, segment_idx=0):
        if segment_idx != 0 or self.runner.video_segment_num != 1:
            raise ValueError('One ordinary offline Wan segment required')
        teacher = self.wrapper.scheduler  # Already prepared by the unchanged runner.init_run.
        student = type(teacher)(self.runner.config)
        self.input_info = self.runner.input_info
        student.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape,
                        image_encoder_output=self.runner.inputs['image_encoder_output'])
        if (teacher.latents.data_ptr() == student.latents.data_ptr() or
                teacher.latents.dtype != student.latents.dtype or not self.torch.equal(teacher.latents, student.latents)
                or not self.torch.equal(teacher.sigmas, student.sigmas)):
            raise RuntimeError('Independent identically seeded native scheduler states required')
        schedulers = {'teacher': teacher, 'student': student}
        try:
            for step in range(4):
                self.runner.check_stop()
                started = time.perf_counter()
                self._step(step, schedulers)
                self.cuda.synchronize()
                self.step_wall_seconds.append(time.perf_counter() - started)
            self.teacher_latents = teacher.latents.detach().clone()
            self.completed_trace = self.trace.finish()
            self.completed_trace['trajectory_pair_modes'] = list(self.modes)
            self.completed_trace['cost_scope'] = 'OFFLINE_PAIRED_SHADOW_NUMERICAL_AND_STRATUM_OBSERVATION_NOT_DEPLOYMENT_LATENCY'
            del self.runner.inputs
            return student.latents
        finally:
            self._bind(self.old_scheduler)
