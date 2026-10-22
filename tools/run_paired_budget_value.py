"""Registered R12 D0: controls -> repeated baseline -> eight fixed actions."""
import argparse
import fcntl
from pathlib import Path
import shutil
import time

from r12d0_contract import (ROOT, METHOD, PROTOCOL, jobs, read, sha, prerequisites)


def source_binding():
    return read(ROOT / 'research/CURRENT_DECISION.json')['registered_diagnostic']['source_sha256']


def storage_guard(raw):
    # The single attempt is capped across all six jobs, including videos/receipts.
    size = sum(p.stat().st_size for p in raw.rglob('*') if p.is_file())
    free = shutil.disk_usage(raw).free
    if size > (1 << 29) or free < (80 << 30) + max(0, (1 << 29) - size):
        raise RuntimeError('D0 HF artifact allowance or80GiB node reserve would be exceeded')
    return {'attempt_artifact_bytes': size, 'free_bytes': free}


def capture(pipe, runtime, teacher, arm, index, refs, folder):
    import torch
    from safetensors.torch import save_file
    from ar2_checkpoint import atomic_json
    from paired_budget_observer import PairObserver
    from paired_budget_runtime import begin, verify
    from run_ar2_restart import tensor_sha
    from run_supplement_teacher_flow import exact_native_latent, error_stats
    if folder.exists():
        raise RuntimeError('Preserve every attempted case; no rerun/overwrite')
    folder.mkdir(parents=True)
    scheduler = pipe.runner.scheduler
    old_pre, old_post = scheduler.step_pre, scheduler.step_post
    dq = refs['DQ'][index]
    baseline = refs['DQ' if arm == 'DQ' else 'UB1'][index]
    sample = dq['case']['input']
    if sha(sample['image_path']) != sample['input_provenance']['image_sha256'] or sample['seed'] != 42:
        raise RuntimeError('Frozen source image/seed changed')
    begin(runtime, arm)
    before, steps, current = dict(runtime.counters), [], {}
    scheduler.generator = None

    def pre(step_index):
        value = old_pre(step_index)
        latent = scheduler.latents.detach().float().cpu().contiguous()
        expected = dq['payloads'][step_index]['input']
        if tensor_sha(latent) != tensor_sha(expected):
            raise RuntimeError('Complete incoming native DQ step state bytes differ')
        current[int(step_index)] = tensor_sha(latent)
        return value

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        value = old_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        target, control = dq['payloads'][step], baseline['payloads'][step]
        required = arm in ('DQ', 'UB1') or step != 3
        flow_sha, post_sha = tensor_sha(flow), tensor_sha(latent)
        if required and (flow_sha != tensor_sha(control['flow']) or post_sha != tensor_sha(control['post'])):
            raise RuntimeError(f'Frozen TF2 {arm} flow/post bytes differ at step{step}')
        if not bool(torch.isfinite(flow).all() and torch.isfinite(latent).all()):
            raise RuntimeError('Nonfinite native flow/post')
        row = {'step': step, 'input_sha256': current[step], 'flow_sha256': flow_sha,
               'post_sha256': post_sha, 'required_original_bitwise_match': required,
               'flow_error': error_stats(flow, target['flow']),
               'sigma': float(scheduler.sigmas[step]), 'timestep': float(scheduler.timesteps[step])}
        if step == 3:
            path = folder / 'step3_flow.safetensors'
            save_file({'flow': flow}, str(path))
            row.update(path=str(path), sha256=sha(path))
        steps.append(row)
        # Only between completed steps. No reset within the30/31 pair.
        scheduler.latents = exact_native_latent(target['post'], scheduler.latents)
        return value

    observer = PairObserver(runtime, teacher)
    scheduler.step_pre, scheduler.step_post = pre, post
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    report = {'status': 'RUNNING', 'arm': arm, 'index': index, 'steps': steps,
              'quality_or_speed_acceptance': False, 'scopes': observer.scopes}
    atomic_json(folder / 'INPUT.json', report)
    try:
        with torch.inference_mode(), observer:
            pipe.generate(seed=42, image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(folder / 'diagnostic_only.mp4'))
        torch.cuda.synchronize()
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']:
            raise RuntimeError('Full four-step1600 model linear/QDQ contract failed')
        if [r['step'] for r in steps] != list(range(4)) or [s['block'] for s in observer.scopes] != [30, 31] or observer.pending_post:
            raise RuntimeError('Incomplete four-step/pair/writeback coverage')
        if arm in ('DQ', 'UB1'):
            previous = read(ROOT / 'research/20261021/R7D1_results.json')
            source = next(r for r in previous['rows'] if r['index'] == index)
            old_scopes = source['scopes']['DQ' if arm == 'DQ' else 'none']
            for actual, old in zip(observer.scopes, old_scopes):
                for key in ('norm_input_sha256', 'b0_sha256', 'b2_sha256', 'gate_sha256', 'returned_Y_sha256'):
                    if actual[key] != old[key]:
                        raise RuntimeError('Original native pair operand/Y identity changed')
            report['original_R7_pair_operands_and_Y_bitwise_equal'] = True
        report.update(status='PASS_R12_D0_NATIVE_CASE', counter_delta=delta, execution=verify(runtime),
                      elapsed_seconds=time.monotonic() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      diagnostic_video_sha256=sha(folder / 'diagnostic_only.mp4'),
                      diagnostic_video_is_DQ_by_construction=True)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        scheduler.step_pre, scheduler.step_post = old_pre, old_post
        atomic_json(folder / 'INPUT.json', report)
    print(f'R12 D0 {arm} input{index:02d} PASS flowSSE={steps[3]["flow_error"]["squared_error"]:.9f}', flush=True)
    return {'index': index, 'arm': arm, 'path': str(folder / 'INPUT.json'), 'sha256': sha(folder / 'INPUT.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['controls', 'baselines', 'interventions'], required=True)
    parser.add_argument('--baseline', choices=['DQ', 'UB1'], default='UB1')
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Fail before importing model/CUDA modules, not after allocating a model.
    from check_research_routes import load_records, validate_records, method_eligibility
    records = load_records(ROOT)
    errors = validate_records(*records)
    if errors or not method_eligibility(records[0], records[1], METHOD)[0]:
        raise RuntimeError(f'D0 is not consistently registered: {errors}')
    name, schedule = jobs(args.phase, args.baseline, args.shard)
    out, protocol = args.out.resolve(), read(PROTOCOL)
    if out != (ROOT / protocol['execution_plan']['execution_root']).resolve():
        raise RuntimeError('Only the single frozen24-case attempt root is authorized')
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve existing job, including failures; no automatic resume')
    import torch
    from ar2_checkpoint import atomic_json
    from historical_source_snapshot import HistoricalSources
    from collect_ub1_native_states import policy_alias
    from restored_assets import verified_restored_assets
    from run_cross_layer_transport import references
    from resident_ffn_runtime import make_resident, unused_raw_placeholders
    from native_teacher_capture import TeacherRecords
    from paired_budget_runtime import install, make_pair_resident
    torch.set_num_threads(8)
    history = HistoricalSources()
    alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    refs, ref_bindings = references(history, assets)
    sources = source_binding()
    identity = {'protocol_sha256': sha(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': policy_sha, 'original_TF2_reports': ref_bindings,
                'operand_audit_sha256': sha(ROOT / 'research/20261022/R12_E0_operand_audit.json'),
                'original_R7_pair_receipts_sha256': sha(ROOT / 'research/20261021/R7D1_results.json')}
    prior = prerequisites(out, args.phase, identity)
    raw_attempt = Path(cache['path']).parents[1] / 'r12d0_native_value' / out.name
    raw = raw_attempt / name
    raw.mkdir(parents=True, exist_ok=True)
    resource = storage_guard(raw_attempt)
    # Worker wrapper owns the GPU lock and rejects foreign processes. Check
    # projected native resident/scratch capacity again before loading weights.
    free, total = torch.cuda.mem_get_info()
    if free < 72 << 30:
        raise RuntimeError('Projected resident model, pair variants and native scratch require72GiB free')
    report = {'status': 'RUNNING', 'job': name, 'scientific_identity': identity,
              'jobs': schedule, 'completed': [], 'control_bindings': prior,
              'resource_preflight': {**resource, 'GPU_free_bytes': free, 'GPU_total_bytes': total},
              'quality_or_speed_acceptance': False}
    atomic_json(folder / 'RUN.json', report)
    try:
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Strict raw placeholder coverage incomplete')
        dense = args.phase == 'controls' and args.baseline == 'DQ'
        if dense:
            runtime = adapter.HIF4CacheRuntime(cache['path'])
        else:
            from step_budget_runtime import create_runtime
            runtime = create_runtime(cache['path'], alias)
        install(runtime, cache, dense=dense)
        report['residency'] = make_resident(runtime)
        report['pair_residency'] = make_pair_resident(runtime)
        report['pair_pack_receipts'] = runtime.r12_pack_receipts
        teacher = TeacherRecords(cache)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for index, arm in schedule:
                if (out / 'STOP.json').exists():
                    raise RuntimeError('Attempt stopped; do not start another case')
                if sha(PROTOCOL) != identity['protocol_sha256'] or any(sha(ROOT / p) != v for p, v in sources.items()):
                    raise RuntimeError('Frozen source or protocol changed during execution')
                storage_guard(raw_attempt)
                row = capture(pipe, runtime, teacher, arm, index, refs, raw / f'input{index:02d}_{arm}')
                if audit['fallback_calls']:
                    raise RuntimeError('Forbidden dense/raw placeholder fallback')
                report['completed'].append(row)
                report['resource_after_case'] = storage_guard(raw_attempt)
                atomic_json(folder / 'RUN.json', report)
        finally:
            adapter.deactivate_lightx2v_hooks()
        report.update(status='PASS_ALL_REGISTERED_RUNS', raw_placeholder_fallback_calls=audit['fallback_calls'],
                      historical_source_receipts=history.receipts)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)


if __name__ == '__main__':
    main()
