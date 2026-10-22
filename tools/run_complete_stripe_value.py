"""Fixed R14N0 capture/value runner; refuses unregistered execution.

Capture and evaluation are separate processes. No model, torch or CUDA import
occurs before the canonical route, human authority and source checks pass.
"""
import argparse
import json
from pathlib import Path
import shutil

from r14_stripe_contract import (Budget, CALLS, ORDER, ROOT, execution_root,
                                 read, require_registration, sha, write_new)


def resource_check(root, draft):
    used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    if (used > draft['resources']['incremental_output_bytes_max']
            or shutil.disk_usage(root).free < draft['resources']['node_free_reserve_bytes_min']):
        raise RuntimeError('Registered output/node reserve limit exceeded')
    return used


def capture(protocol, draft, root):
    import time
    import torch
    from ar2_checkpoint import atomic_json, digest
    from collect_ub1_native_states import policy_alias, verify_runtime
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from run_ar2_restart import tensor_sha
    from stripe_teacher_capture import StripeTeacherCapture
    from ub1_native_data import load_ub1_collections
    from variable_teacher_capture import TeacherRecords

    if root.exists():
        raise RuntimeError('Capture root already exists; no implicit retries or overwrite')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < (draft['resources']['node_free_reserve_bytes_min']
                                             + draft['resources']['incremental_output_bytes_max']):
        raise RuntimeError('Insufficient disk reserve before capture')
    history = HistoricalSources()
    kwargs, cache, assets = verified_restored_assets(history)
    cases, bindings, identity = load_ub1_collections(
        [ROOT / f'reports/worker_20261020/OP2_UB1_calibration24_shard{s}_RUN.json' for s in (0, 1)], ROOT)
    source = ROOT / '.runtime/inputs/calibration24/canonical.json'
    if identity['assets'] != assets or digest(source) != identity['inputs_sha256']:
        raise RuntimeError('Original OP2 input/model identity differs')
    samples = read(source)
    root.mkdir()
    budget = Budget(root, create=True)
    alias, original_policy_sha = policy_alias(history, root)
    if original_policy_sha != identity['original_UB1_policy_sha256']:
        raise RuntimeError('Original OP2 UB1 policy differs')
    report = {'status': 'RUNNING_ORIGINAL_UB1_OBSERVATIONS',
              'method_id': protocol['id'], 'protocol_sha256': sha(ROOT / protocol['self_path']),
              'original_collection_bindings': bindings, 'assets': assets, 'completed': [],
              'new_candidate_rollouts': 0, 'new_fitted_parameters': 0}
    path = root / 'CAPTURE.json'
    atomic_json(path, report)
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from step_budget_runtime import create_runtime
    try:
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime = create_runtime(cache['path'], alias)
        teacher = TeacherRecords(cache)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for index in ORDER:
                resource_check(root, draft)
                out = root / f'input{index:02d}'
                out.mkdir()
                sample = dict(samples[index])
                sample.update(seed=sample['calibration_seed'], image_path=sample['img_paths'][0])
                if (sample['seed'] != 42 + index
                        or digest(sample['image_path']) != cases[index]['input']['input_provenance']['image_sha256']):
                    raise RuntimeError('Original source/seed changed')
                observer = StripeTeacherCapture(runtime, teacher, out, index, draft, cases[index])
                scheduler = pipe.runner.scheduler
                old_post = scheduler.step_post
                trajectory = []

                def post():
                    value = old_post()
                    latent = scheduler.latents.detach().float().cpu().contiguous()
                    if not bool(torch.isfinite(latent).all()):
                        raise RuntimeError('Nonfinite original native state')
                    trajectory.append({'step': int(scheduler.step_index), 'sha256': tensor_sha(latent),
                                       'shape': list(latent.shape)})
                    return value

                video = out / 'observed_UB1.mp4'
                scheduler.step_post, scheduler.generator = post, None
                runtime.begin_request()
                before = dict(runtime.counters)
                began = time.monotonic()
                try:
                    with budget.call(index, 'capture'), torch.inference_mode(), observer:
                        pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                                      negative_prompt='', save_result_path=str(video))
                        torch.cuda.synchronize()
                finally:
                    scheduler.step_post = old_post
                delta = {key: runtime.counters[key] - before[key] for key in before}
                if (delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600
                        or delta['cache_misses'] or len(observer.scopes) != 1):
                    raise RuntimeError('Original four-step/one-target coverage differs')
                verify_runtime(runtime, 'UB1')
                if (trajectory != cases[index]['trajectories']['observed']
                        or digest(video) != cases[index]['videos']['observed']['sha256']):
                    raise RuntimeError('Observation changed old original UB1 trajectory or video')
                case_path = out / 'INPUT.json'
                atomic_json(case_path, {'index': index, 'scope': observer.scopes[0],
                    'original_trajectory_bitwise': True, 'trajectory': trajectory,
                    'original_video_bitwise': True, 'video_sha256': digest(video),
                    'image_sha256': digest(sample['image_path']), 'seed': sample['seed'],
                    'counter_delta': delta, 'elapsed_seconds': time.monotonic() - began})
                report['completed'].append({'index': index, 'report': str(case_path), 'sha256': digest(case_path)})
                report['output_bytes'] = resource_check(root, draft)
                atomic_json(path, report)
                print(f'R14N0 originalUB1 observation source{index:02d} complete; no candidate rollout', flush=True)
        finally:
            adapter.deactivate_lightx2v_hooks()
        if [case['index'] for case in report['completed']] != list(ORDER):
            raise RuntimeError('Fixed source coverage incomplete')
        report['status'] = 'PASS_FOUR_ORIGINAL_UB1_OBSERVATIONS'
    except BaseException as error:
        report.update(status='FAILED_FIXED_SCOPE_NO_AUTOMATIC_RETRY', error_type=type(error).__name__)
        raise
    finally:
        report['budget'] = budget.state
        atomic_json(path, report)
        budget.close()
    # Return to the CLI and exit the process before any standalone evaluation.


def evaluate(protocol, draft, root):
    import math
    import torch
    from safetensors.torch import load_file
    from ar2_checkpoint import atomic_json, digest
    from analyze_stripe_native_values import analyze
    from historical_source_snapshot import HistoricalSources
    from stripe_native_operator import LocalOperator, check_composed, compose, count_linears
    from ub1_native_replay import UB1Positions
    from run_ar2_restart import tensor_sha
    from HiFloat4.ptq.adapters.lightx2v_wan_distill import native_hif4_activation_qdq

    captured = read(root / 'CAPTURE.json')
    if (captured['status'] != 'PASS_FOUR_ORIGINAL_UB1_OBSERVATIONS'
            or captured['protocol_sha256'] != sha(ROOT / protocol['self_path'])
            or [case['index'] for case in captured['completed']] != list(ORDER)):
        raise RuntimeError('All four unchanged baseline captures required')
    if (root / 'VALUE.json').exists():
        raise RuntimeError('Preserve the first fixed-scope evaluation')
    budget = Budget(root)
    if budget.state['used'] != {'model_generations': 4, 'standalone_FFN_forwards': 0, 'extra_linear_calls': 8}:
        raise RuntimeError('Unexpected prior evaluation debit; no implicit retry')
    history = HistoricalSources()
    cache = read(ROOT / '.runtime/hif4_rebuilt_cache.json')
    old_policy = ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json'
    params = UB1Positions(old_policy, history, expected_policy_sha256=
                         read(root / 'UB1_execution_alias.json')['historical_policy_sha256'])
    position = params.position(2, 20)
    shard = next(s for s in cache['shards'] if s['expert'] == 'low_noise_model' and s['block'] == 20)
    if digest(shard['path']) != shard['sha256']:
        raise RuntimeError('Original full native teacher shard changed')
    records = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)['low_noise_model']
    affine_record = next(a for a in read(ROOT / 'research/20261021/R9A1_native_affine_results.json')['affine_matrices']
                         if 2 in a['shared_steps'])
    affine_path = Path(cache['path']).parents[1] / affine_record['HF_derived_relative']
    if digest(affine_path) != draft['parameters']['affine_file_sha256']:
        raise RuntimeError('Original frozen R9 affine changed')
    affine_data = load_file(str(affine_path), device='cpu')
    if any(tensor_sha(affine_data[key]) != draft['parameters']['affine_tensor_sha256'][name]
           for name, key in (('matrix', 'matrix_BF16'), ('bias', 'bias_total_BF16'),
                             ('slope', 'slope_FP32'), ('intercept', 'intercept_FP32'))):
        raise RuntimeError('Original frozen affine tensor identity differs')
    affine = affine_data['matrix_BF16'].to('cuda')
    entries = {case['index']: case for case in captured['completed']}
    full = {'w0': records['blocks.20.ffn.0']['weight_qdq'].to('cuda'),
            'w2': records['blocks.20.ffn.2']['weight_qdq'].to('cuda'),
            'smooth2': records['blocks.20.ffn.2']['smooth_scale'].to(device='cuda', dtype=torch.float32)}
    progress = {'status': 'RUNNING_FIXED_COMPLETE_STRIPE_VALUE', 'method_id': protocol['id'],
                'protocol_sha256': sha(ROOT / protocol['self_path']), 'cases': [], 'new_candidate_rollouts': 0,
                'new_fitted_parameters': 0, 'native_video_or_speed_acceptance': False}
    state_path = root / 'VALUE_PROGRESS.json'
    atomic_json(state_path, progress)
    operators, complements, value_cases, final_rows = {}, {}, {}, []

    def load(index):
        entry = entries[index]
        if digest(entry['report']) != entry['sha256']:
            raise RuntimeError('New capture source receipt changed')
        case = read(entry['report'])
        scope = case['scope']
        if digest(scope['path']) != scope['sha256'] or scope['teacher_shard_sha256'] != shard['sha256']:
            raise RuntimeError('New complete-stripe payload or teacher identity changed')
        values = load_file(scope['path'], device='cpu')
        if any(tensor_sha(values[key]) != expected for key, expected in scope['tensor_sha256'].items()):
            raise RuntimeError('New stripe tensor identity changed')
        if 'b0' not in full:
            full['b0'], full['b2'] = values['raw_b0'].to('cuda'), values['b2'].to('cuda')
        if tensor_sha(values['raw_b0']) != position['raw_b0_sha256']:
            raise RuntimeError('Original full b0 identity differs from frozen UB1')
        return LocalOperator(values, scope, draft, full, affine, position, native_hif4_activation_qdq)

    def run(index, label, function):
        with budget.call(index, label) as debit, count_linears(CALLS[label]) as calls:
            result = function()
            torch.cuda.synchronize()
            debit['actual_linear_shapes'] = calls
            debit['actual_linear_calls'] = len(calls)
            if torch.cuda.max_memory_allocated() > draft['resources']['standalone_FFN_allocated_bytes_max']:
                raise RuntimeError('Standalone FFN allocation cap exceeded')
        resource_check(root, draft)
        return result

    def initial(index):
        op = load(index)
        run(index, 'teacher_replay', op.teacher)
        ub1 = run(index, 'UB1_replay', op.ub1)
        a = run(index, 'complement_A', lambda: op.candidate(list(range(32))))
        b = run(index, 'complement_B', lambda: op.candidate(list(range(32, 64))))
        one = [b['SSE'][g] if g < 32 else a['SSE'][g] for g in range(64)]
        two = [a['SSE'][g] if g < 32 else b['SSE'][g] for g in range(64)]
        value = {'index': index, 'one_point_SSE': one, 'two_point_SSE': two,
                 'UB1_SSE': ub1['SSE'], 'pre_W1_contrast_norm2': op.norms}
        value_cases[index] = value
        complements[index] = a, b
        operators[index] = op

    def finish(index, fixed):
        op = operators.pop(index)
        a, b = complements.pop(index)
        value = value_cases[index]
        oracle = sorted(sorted(range(64), key=lambda g: (-(value['one_point_SSE'][g] - value['two_point_SSE'][g]), g))[:32])
        actual = {}
        controls = {}
        for label, mask in (('fixed', fixed), ('dynamic', op.dynamic), ('oracle', oracle)):
            actual[label] = run(index, label, lambda mask=mask: op.candidate(mask))
            controls[label] = check_composed(actual[label], compose(a, b, mask))
        repeat = run(index, 'repeat_dynamic', lambda: op.candidate(op.dynamic))
        repeat_control = check_composed(repeat, actual['dynamic'])
        row = {'index': index, 'actual_native_SSE': {key: math.fsum(output['SSE']) for key, output in actual.items()},
               'actual_native_gated_Y_SSE': {key: math.fsum(output['gated_Y_SSE']) for key, output in actual.items()},
               'primary_group_losses': {key: output['SSE'] for key, output in actual.items()},
               'repeat_dynamic_SSE': math.fsum(repeat['SSE']), 'repeat_bitwise_control': repeat_control,
               'actual_composition_controls': controls, 'Q2_controls': op.quant_controls,
               'auxiliary_controls': op.auxiliary_controls, 'norm_max_relative_error': op.score_error,
               'epilogue': op.epilogue}
        final_rows.append(row)
        progress['cases'] = final_rows
        atomic_json(state_path, progress)
        print(f'R14N0 native local source{index:02d} fixed/dynamic/oracle complete', flush=True)

    try:
        with torch.inference_mode():
            initial(1)
            initial(11)
            # No evaluation source payload/loss has been opened before this seal.
            importance = [math.fsum(value_cases[i]['one_point_SSE'][g] - value_cases[i]['two_point_SSE'][g]
                                    for i in (1, 11)) for g in range(64)]
            fixed = sorted(sorted(range(64), key=lambda g: (-importance[g], g))[:32])
            seal_path = root / 'TRAINING_ONLY_FIXED_MASK.json'
            write_new(seal_path, {'training_indices': [1, 11], 'fixed_two_point_stripes': fixed,
                                  'training_loss_binding': {str(i): value_cases[i] for i in (1, 11)},
                                  'evaluation_payloads_opened': 0, 'evaluation_labels_opened': 0})
            seal_sha = digest(seal_path)
            finish(1, fixed)
            finish(11, fixed)
            for index in (0, 10):
                initial(index)
                if digest(seal_path) != seal_sha:
                    raise RuntimeError('Training-only fixed mask changed after evaluation exposure')
                finish(index, fixed)
        values = [value_cases[i] for i in ORDER]
        analysis = analyze(values)
        if analysis['fixed_two_point_stripes'] != fixed:
            raise RuntimeError('Frozen training-only fixed mask does not reproduce')
        for row, analyzed in zip(final_rows, analysis['per_source']):
            for arm in ('fixed', 'dynamic', 'oracle'):
                if not math.isclose(row['actual_native_SSE'][arm], analyzed[arm + '_SSE'], rel_tol=1e-12, abs_tol=1e-8):
                    raise RuntimeError('Executed action SSE disagrees with its local native values')
        evaluation = [row for row in final_rows if row['index'] in (0, 10)]
        eta = math.fsum(abs(row['repeat_dynamic_SSE'] - row['actual_native_SSE']['dynamic']) for row in evaluation)
        total = analysis['evaluation_totals']
        support = total['dynamic_SSE'] < min(total['fixed_SSE'], total['UB1_SSE']) - 3 * eta
        result = {**progress, 'status': 'COMPLETE_FIXED_NATIVE_STRIPE_VALUE',
                  'evidence_level': 'ACTUAL_NATIVE_COMPLETE_STRIPE_VALUES', 'cases': values,
                  'actual_execution': final_rows, 'analysis': analysis, 'eta_absolute_evaluation_SSE': eta,
                  'finite_local_dynamic_support': support, 'training_only_fixed_mask_sha256': seal_sha,
                  'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'max_reserved_bytes': torch.cuda.max_memory_reserved(), 'budget': budget.state,
                  'scope': 'Sampled-domain native local writeback value only. Old exposed sources; no full-scene policy, downstream flow, free-student video quality or native full-model speed acceptance.'}
        if budget.state['used'] != {'model_generations': 4, 'standalone_FFN_forwards': 32, 'extra_linear_calls': 104}:
            raise RuntimeError('Final actual fixed budget differs from registration')
        write_new(root / 'VALUE.json', result)
        print('R14N0 finite local dynamic support:', support, flush=True)
    except BaseException as error:
        progress.update(status='FAILED_FIXED_SCOPE_NO_AUTOMATIC_RETRY', error_type=type(error).__name__)
        raise
    finally:
        progress['budget'] = budget.state
        atomic_json(state_path, progress)
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['capture', 'evaluate'], required=True)
    args = parser.parse_args()
    protocol, draft = require_registration()
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from worker_identity import require_worker
    require_worker()
    import torch
    torch.set_num_threads(8)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One standard locked idle GPU required')
    root = execution_root(protocol)
    {'capture': capture, 'evaluate': evaluate}[args.stage](protocol, draft, root)


if __name__ == '__main__':
    main()
