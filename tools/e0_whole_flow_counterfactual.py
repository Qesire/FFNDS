"""Finite whole-network dense response at actual student states.

Integration component only. The caller must bind a separate authorized scope.
The original paired controller and four-arm computations are reused unchanged.
"""
from copy import copy
import math
import sys
import time

from e0_paired_wan_controller import PairedWanController, target_context, tensor_identity


def flow_split_geometry(z_dense, z_student, v_dense, v_student, v_dense_at_student, *, h):
    """Exact finite decomposition; all arithmetic reductions use FP64.

    P = delta - h*R is the ideal dense discrete map's response at this pair
    of states. Its norm is more informative than the sign of <delta,R>.
    """
    import torch
    values = [z_dense, z_student, v_dense, v_student, v_dense_at_student]
    if (not math.isfinite(h) or h <= 0 or len(z_dense.shape) != 4
            or any(v.shape != z_dense.shape for v in values)):
        raise ValueError('Same full C,F,H,W states/flows and positive sigma gap required')
    totals = dict.fromkeys(['delta_SSE', 'model_error_SSE', 'dense_state_response_SSE',
        'flow_difference_SSE', 'delta_model_error_dot', 'delta_state_response_dot',
        'model_error_state_response_dot', 'dense_map_response_SSE',
        'dense_map_model_error_dot', 'ideal_next_delta_SSE', 'recomposition_SSE'], 0.)
    # Channel chunks bound temporary FP64 storage without sampling coordinates.
    for start in range(len(z_dense)):
        a, b, c, d, f = [v[start:start + 1].detach().to(dtype=torch.float64) for v in values]
        delta, error, response = b - a, d - f, f - c
        difference = d - c
        dense_map = delta - h * response
        ideal = delta - h * difference
        products = {'delta_SSE': delta ** 2, 'model_error_SSE': error ** 2,
            'dense_state_response_SSE': response ** 2, 'flow_difference_SSE': difference ** 2,
            'delta_model_error_dot': delta * error, 'delta_state_response_dot': delta * response,
            'model_error_state_response_dot': error * response,
            'dense_map_response_SSE': dense_map ** 2, 'dense_map_model_error_dot': dense_map * error,
            'ideal_next_delta_SSE': ideal ** 2, 'recomposition_SSE': (difference - error - response) ** 2}
        for key, product in products.items():
            totals[key] += float(product.sum())
    if not all(math.isfinite(v) for v in totals.values()):
        raise ValueError('Nonfinite whole-flow observation')
    energy = totals['model_error_SSE'] + totals['dense_state_response_SSE'] + 2 * totals['model_error_state_response_dot']
    update = totals['dense_map_response_SSE'] + h * h * totals['model_error_SSE'] - 2 * h * totals['dense_map_model_error_dot']
    scale = max(totals['flow_difference_SSE'], totals['model_error_SSE'], totals['dense_state_response_SSE'], 1e-30)
    if (totals['recomposition_SSE'] > 1e-18 * scale
            or not math.isclose(energy, totals['flow_difference_SSE'], rel_tol=1e-9, abs_tol=1e-20)
            or not math.isclose(update, totals['ideal_next_delta_SSE'], rel_tol=1e-9, abs_tol=1e-20)):
        raise ValueError('Whole-flow signed identities do not close')
    return {**totals, 'h': h, 'shape': list(z_dense.shape),
        'dense_discrete_map_energy_gain': totals['dense_map_response_SSE'] / totals['delta_SSE'] if totals['delta_SSE'] else None,
        'candidate_update_energy': h * h * totals['model_error_SSE'],
        'candidate_dense_map_cross_term': -2 * h * totals['dense_map_model_error_dot'],
        'evidence_scope': 'FINITE_ACTUAL_STATE_PAIR_NOT_JACOBIAN_OR_UNIVERSAL_STABILITY',
        'quality_accepted': False}


def native_map_split_geometry(dense_after, student_after, dense_at_student_after):
    """Actual BF16 step_post outputs, distinct from ideal Euler geometry."""
    import torch
    values = [dense_after, student_after, dense_at_student_after]
    if len(dense_after.shape) != 4 or any(v.shape != dense_after.shape for v in values):
        raise ValueError('Same full C,F,H,W native outputs required')
    totals = dict.fromkeys(['dense_map_response_SSE', 'candidate_effect_SSE',
                           'dense_map_candidate_effect_dot', 'actual_next_delta_SSE', 'recomposition_SSE'], 0.)
    for start in range(len(dense_after)):
        a, b, c = [v[start:start + 1].detach().to(dtype=torch.float64) for v in values]
        p, n, total = c - a, b - c, b - a
        for key, value in [('dense_map_response_SSE', p ** 2), ('candidate_effect_SSE', n ** 2),
                           ('dense_map_candidate_effect_dot', p * n), ('actual_next_delta_SSE', total ** 2),
                           ('recomposition_SSE', (total - p - n) ** 2)]:
            totals[key] += float(value.sum())
    if not all(math.isfinite(v) for v in totals.values()):
        raise ValueError('Nonfinite native map outputs')
    expected = totals['dense_map_response_SSE'] + totals['candidate_effect_SSE'] + 2 * totals['dense_map_candidate_effect_dot']
    if not math.isclose(expected, totals['actual_next_delta_SSE'], rel_tol=1e-9, abs_tol=1e-20):
        raise ValueError('Native map identity does not close')
    return {**totals, 'boundary': 'ACTUAL_NATIVE_STEP_POST_BF16_OUTPUTS'}


class WholeFlowCounterfactualController(PairedWanController):
    """Tap native step entries/flows and evaluate a third complete dense path.

    The third path is evaluated after the paired step; its shadow step_post
    never changes a live trajectory. Original latent hashes are driver-checked.
    """
    def __init__(self, *args, counterfactual_steps=(0, 1, 2, 3), save_counterfactual=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(counterfactual_steps) != (0, 1, 2, 3) or self.modes != ('dense', 'candidate'):
            raise ValueError('Fixed four-step dense/candidate counterfactual required')
        self.counterfactual_steps = tuple(counterfactual_steps)
        self.save_counterfactual = save_counterfactual
        self.counterfactual_records = []

    def _dense_flow(self, step, scheduler):
        """Actual pre + all40 native dense blocks + native head/post."""
        torch, cuda = self.torch, self.cuda
        index = 0 if step < 2 else 1
        expert = 'high_noise_model' if index == 0 else 'low_noise_model'
        model = self.wrapper.model[index]
        infer, weights = model.transformer_infer, model.transformer_weights
        previous_apply = self.observer.runtime.apply
        if self.observer.active is not None or self.observer.pending is not None or self.observer.site:
            raise RuntimeError('Whole-flow counterfactual requires completed four-arm observers')
        # Original dense HiF4 runtime, including every non-FFN operation; no
        # candidate interception or local FFN shadows on this complete path.
        self.observer.runtime.apply = self.observer.base_apply
        try:
            self._bind(scheduler)
            model.pre_weight.to_cuda()
            weights.non_block_weights_to_cuda()
            pre = model.pre_infer.infer(model.pre_weight, self.runner.inputs)
            self._metadata(infer, pre)
            infer.reset_attention_states(weights.blocks)
            hidden = pre.x
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
                    with cuda.stream(manager.compute_stream), target_context(self.adapter, expert, step, block):
                        hidden = infer.infer_block(manager.cuda_buffers[0], hidden, pre)
                    manager.swap_blocks()
                finally:
                    self.policy.end_block(manager.compute_stream.synchronize)
            cuda.current_stream().wait_stream(manager.compute_stream)
            head = infer.infer_non_blocks(weights, hidden, pre.embed)
            result = model.post_infer.infer(head, pre)[0]
            cuda.synchronize()
            return result
        finally:
            original_error = sys.exc_info()[1]
            failures = []
            for operation in [cuda.synchronize, model.pre_weight.to_cpu, weights.non_block_weights_to_cpu]:
                try:
                    operation()
                except BaseException as error:
                    failures.append(error)
            self.observer.runtime.apply = previous_apply
            if failures:
                message = 'Whole-flow cleanup failed: ' + repr(failures)
                if original_error is None:
                    raise RuntimeError(message) from failures[0]
                if hasattr(original_error, 'add_note'):
                    original_error.add_note(message)

    def _step(self, step, schedulers):
        model = self.wrapper.model[0 if step < 2 else 1]
        original_post = model.post_infer.infer
        original_pre = {name: scheduler.step_pre for name, scheduler in schedulers.items()}
        entries, flows = {}, []

        def tap_post(*args, **kwargs):
            output = original_post(*args, **kwargs)
            flows.append(output[0].detach().clone())
            return output

        def tap_pre(name, scheduler):
            def call(*args, **kwargs):
                result = original_pre[name](*args, **kwargs)
                entries[name] = copy(scheduler)
                entries[name].latents = scheduler.latents.detach().clone()
                return result
            return call

        model.post_infer.infer = tap_post
        for name, scheduler in schedulers.items():
            scheduler.step_pre = tap_pre(name, scheduler)
        try:
            super()._step(step, schedulers)
        finally:
            model.post_infer.infer = original_post
            for name, scheduler in schedulers.items():
                scheduler.step_pre = original_pre[name]
        if set(entries) != {'teacher', 'student'} or len(flows) != 2:
            raise RuntimeError('Missing actual full-step inputs or ordered teacher/student flows')
        for name in entries:
            expected = self.trace.steps[step]['boundary_hashes'][name + '_after_step_pre']
            if tensor_identity(entries[name].latents) != expected:
                raise RuntimeError('Counterfactual input differs from actual post-step_pre state')
        before = dict(self.observer.runtime.counters)
        started = time.perf_counter()
        try:
            dense_at_student = self._dense_flow(step, entries['student'])
        finally:
            self._bind(schedulers['student'])
        self.cuda.synchronize()
        wall = time.perf_counter() - started
        counters = {k: self.observer.runtime.counters[k] - before[k] for k in before}
        shadow_scheduler = copy(entries['student'])
        shadow_scheduler.latents = entries['student'].latents.detach().clone()
        shadow_scheduler.noise_pred_cond = shadow_scheduler.noise_pred_guided = shadow_scheduler.noise_pred = dense_at_student
        shadow_scheduler.noise_pred_uncond = None
        shadow_scheduler.step_post()
        record = {'step': step, 'status': 'RAW_SAVED_GEOMETRY_PENDING', 'native_runtime_counts': counters,
                  'dense_counterfactual_wall_seconds': wall,
                  'flow_hashes': {name: tensor_identity(value) for name, value in
                    zip(['dense_teacher', 'candidate_student', 'dense_student'], [*flows, dense_at_student])},
                  'input_hashes': {name: tensor_identity(value.latents) for name, value in entries.items()}}
        # Preserve actual tensors before finite/identity reductions can fail.
        if self.save_counterfactual is not None:
            record['saved_tensors'] = self.save_counterfactual(step, {
                'z_dense': entries['teacher'].latents, 'z_student': entries['student'].latents,
                'v_dense': flows[0], 'v_student': flows[1], 'v_dense_at_student': dense_at_student,
                'z_dense_after': schedulers['teacher'].latents, 'z_student_after': schedulers['student'].latents,
                'z_dense_at_student_after': shadow_scheduler.latents}, record)
        self.counterfactual_records.append(record)
        geometry = flow_split_geometry(entries['teacher'].latents, entries['student'].latents,
                                       *flows, dense_at_student, h=self.trace.steps[step]['h'])
        native = native_map_split_geometry(schedulers['teacher'].latents, schedulers['student'].latents,
                                           shadow_scheduler.latents)
        native['dense_discrete_map_energy_gain'] = native['dense_map_response_SSE'] / geometry['delta_SSE'] if geometry['delta_SSE'] else None
        record.update(geometry=geometry, native_map_geometry=native, status='GEOMETRY_COMPLETE_CHECKS_PENDING')
        if step == 0 and not self.torch.equal(flows[0], dense_at_student):
            raise RuntimeError('Whole-model neutral control differs at identical initial state')
        if step == 0 and not self.torch.equal(shadow_scheduler.latents, schedulers['teacher'].latents):
            raise RuntimeError('Native shadow step_post differs at identical initial state')
        if not math.isclose(native['actual_next_delta_SSE'], self.trace.steps[step]['delta_after_SSE'],
                            rel_tol=1e-9, abs_tol=1e-20):
            raise RuntimeError('Native map after-state differs from paired scheduler observation')
        for key in ['delta_SSE', 'flow_difference_SSE', 'ideal_next_delta_SSE']:
            old_key = {'delta_SSE': 'delta_before_SSE', 'ideal_next_delta_SSE': 'ideal_after_SSE'}.get(key, key)
            if not math.isclose(geometry[key], self.trace.steps[step][old_key], rel_tol=1e-9, abs_tol=1e-20):
                raise RuntimeError('Counterfactual main-flow tap differs from original scheduler observation')
        record['status'] = 'PASS_FLOW_AND_NATIVE_MAP_IDENTITIES'
        if self.save_counterfactual is not None:
            self.save_counterfactual(step, None, record)  # Metadata only; raw bytes stay unchanged.


class LatentOnlyMain:
    """Keep native input/step setup, omit VAE decoding and media creation."""
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        self.old_main = self.runner.run_main
        self.runner.run_main = self.run_main
        return self

    def __exit__(self, *error):
        self.runner.run_main = self.old_main

    def run_main(self):
        runner = self.runner
        runner.init_run()
        if runner.video_segment_num != 1:
            raise RuntimeError('One ordinary native segment required')
        runner.check_stop()
        runner.init_run_segment(0)
        result = runner.run_segment(0)
        # Frozen BaseRunner.end_run_segment reads this media attribute even
        # when the VAE decoder is deliberately omitted.
        runner.gen_video = None
        runner.end_run_segment(0)
        runner.end_run()
        return result
