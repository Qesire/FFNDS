"""Separately authorized finite native paired traces with mandatory video metrics.

Default is read-only preflight: no torch/native import, model or GPU operation.
The frozen scope and exact protocol SHA must also be registered in ACTIVE_RESEARCH.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
MANDATORY_DRIVER_SOURCES = {
    'tools/' + name for name in [
        'run_e0_paired_full_depth.py', 'e0_paired_wan_controller.py', 'e0_paired_ffn_observer.py',
        'check_e0_full_depth_interface.py',
        'e0_paired_weight_residency.py', 'e0_full_depth_trace_contract.py', 'ffnds_video_metrics.py',
        'ffnds_video_metric_gate.py', 'ffnds_observation_common.py', 'restored_assets.py',
        'historical_source_snapshot.py', 'startup_gpu_reservation.py', 'assigned_gpu_resources.py',
        'worker_identity.py', 'ar2_checkpoint.py', 'with_transformers_source_view.py']}
MANDATORY_NATIVE_SOURCES = {
    '.runtime/code/LightX2V/lightx2v/' + name for name in [
        'pipeline.py', 'models/networks/base_model.py', 'models/networks/wan/model.py',
        'models/networks/wan/distill_model.py', 'models/networks/wan/infer/transformer_infer.py',
        'models/networks/wan/infer/offload/transformer_infer.py', 'models/networks/wan/infer/pre_infer.py',
        'models/networks/wan/infer/post_infer.py', 'models/networks/wan/weights/transformer_weights.py',
        'models/runners/base_runner.py', 'models/runners/default_runner.py', 'models/runners/wan/wan_runner.py',
        'models/runners/wan/wan_distill_runner.py', 'models/schedulers/scheduler.py',
        'models/schedulers/wan/scheduler.py', 'models/schedulers/wan/step_distill/scheduler.py',
        'common/offload/manager.py', 'common/ops/mm/mm_weight.py', 'common/ops/attn/torch_sdpa.py',
        'common/transformer_infer/transformer_infer.py', 'utils/envs.py']}
MANDATORY_NATIVE_SOURCES |= {'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/' + name for name in
                             ['lightx2v_wan_distill.py', 'lightx2v_wan_distill_contract.json']}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def preflight(protocol, bank):
    plan = json.loads(protocol.read_text())
    if (not MANDATORY_DRIVER_SOURCES <= set(plan['driver_sources_sha256']) or
            not MANDATORY_NATIVE_SOURCES <= set(plan['native_source_sha256'])):
        raise ValueError('Missing mandatory driver/native source bindings')
    if (plan['max_full_generations_total'] != 6 or plan['planned_simultaneous_GPU_cards'] != 1
            or len(plan['pairs']) != 3 or plan['automatic_retry'] is not False):
        raise ValueError('Unsupported finite paired execution scope')
    for group in ['driver_sources_sha256', 'native_source_sha256']:
        for name, expected in plan[group].items():
            if sha256(ROOT / name) != expected:
                raise ValueError('Frozen source drift: ' + name)
    for name, expected in [(plan['input_default'], plan['input_default_sha256']),
                           ('.runtime/hif4_rebuilt_cache.json', plan['cache_manifest_sha256']),
                           (plan['previous_result'], plan['previous_result_sha256'])]:
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen provenance drift: ' + name)
    if sha256(bank / 'RUN.json') != plan['bank_RUN_sha256']:
        raise ValueError('Immutable bank receipt changed')
    manifest = json.loads((bank / 'RUN.json').read_text())
    packs = [p for p in manifest['packs'] if p['family'] == 'affine_residual']
    if len(packs) != 80 or len({(p['expert'], p['block']) for p in packs}) != 80:
        raise ValueError('Exactly80 fixed affine packs required')
    for pack in packs:
        if sha256(bank / pack['path']) != pack['sha256']:
            raise ValueError('Immutable pack changed')
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    authorized = any(s['id'] == plan['id'] and s['protocol_sha256'] == sha256(protocol)
                     and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'] for s in scopes)
    return plan, packs, {'status': 'STATIC_BINDINGS_PASS_NOT_NATIVE_VALIDATION',
                        'model_scope_authorized': authorized, 'new_generations': 0, 'GPU_initialized': False}


@contextmanager
def ledger_lock(study):
    study.mkdir(parents=True, exist_ok=True)
    with (study / 'LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def reserve_attempt(study, plan, protocol_sha, pair):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        state = json.loads(path.read_text()) if path.exists() else {'id': plan['id'], 'protocol_sha256': protocol_sha, 'attempts': []}
        order = [p['id'] for p in plan['pairs']]
        if state['protocol_sha256'] != protocol_sha or len(state['attempts']) >= len(order):
            raise RuntimeError('Wrong study binding or finite attempt budget exhausted')
        if any(a['status'] != 'PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY' for a in state['attempts']):
            raise RuntimeError('Earlier attempt running/failed; no automatic retry or followup')
        if order[len(state['attempts'])] != pair['id']:
            raise RuntimeError('Fixed pair order required')
        state['attempts'].append({'pair': pair['id'], 'status': 'STARTED', 'generations_reserved': 2})
        write_json(path, state)


def close_attempt(study, pair_id, status, receipt):
    with ledger_lock(study):
        path = study / 'STUDY.json'; state = json.loads(path.read_text())
        entry = next(a for a in state['attempts'] if a['pair'] == pair_id)
        entry.update(status=status, RUN_sha256=sha256(receipt))
        write_json(path, state)


def evaluate_pair(out, native, pair, sample, original, protocol_sha):
    from ffnds_video_metrics import evaluate, validate
    from ffnds_video_metric_gate import inspect
    receipt = out / 'NATIVE_RUN.json'
    reference_receipt = Path(original['receipt'])
    if sha256(reference_receipt) != original['receipt_sha256'] or sha256(Path(original['video'])) != original['video_sha256']:
        raise ValueError('Historical dense reference changed')
    old = json.loads(reference_receipt.read_text())
    old_phase = next(p for p in old['phases'] if p['source_index'] == pair['source_index'])
    if old_phase['seed'] != native['seed'] or old_phase['video_sha256'] != original['video_sha256']:
        raise ValueError('Historical reference source/seed/video identity differs')
    if any(native['assets'][key] != old['assets'][key] for key in
           ['checkpoint_sha256', 'cache_manifest_sha256', 'contract']):
        raise ValueError('Current and original dense asset identities differ')
    def identity(kind, receipt_hash, assets):
        binding = {'checkpoint_sha256': assets['checkpoint_sha256'],
                   'cache_manifest_sha256': assets['cache_manifest_sha256'],
                   'prototype_family': 'dense' if kind == 'dense' else 'affine_residual',
                   'bank_RUN_sha256': None if kind == 'dense' else native['bank_RUN_sha256']}
        return {'source_id': native['source_id'], 'seed': native['seed'], 'expected_frames': 81,
                'input_prompt_digest': object_sha({'image_sha256': native['source_id'], 'prompt': sample['prompt'],
                                                  'negative_prompt': '', 'contract': assets['contract']}),
                'model_digest': object_sha(binding), 'protocol_sha256': protocol_sha, 'run_sha256': receipt_hash}
    teacher_identity = identity(pair['modes'][0], sha256(receipt), native['assets'])
    old_identity = identity('dense', original['receipt_sha256'], old['assets'])
    old_identity['protocol_sha256'] = original['protocol_sha256']
    bundle = {'schema': 'ffnds.video.bundle.v1', 'expected_video_count': 2, 'videos': []}
    for branch, kind in zip(['teacher', 'student'], pair['modes']):
        video = out / (branch + '.mp4')
        reference = Path(original['video']) if branch == 'teacher' else out / 'teacher.mp4'
        reference_identity = old_identity if branch == 'teacher' else teacher_identity
        metadata = {'identity': identity(kind, sha256(receipt), native['assets']), 'reference_identity': reference_identity,
                    'cost': native['cost'], 'cost_scope': native['cost_scope']}
        report = evaluate(video, reference, metadata)
        report['cost_scope'] = native['cost_scope']
        report['cost_deduplication_key'] = native['cost_deduplication_key']
        failures = validate(report)
        if failures:
            raise ValueError('; '.join(failures))
        metrics = out / (branch + '_metrics.json'); write_json(metrics, report)
        bundle['videos'].append({'source_id': native['source_id'], 'seed': native['seed'],
            'variant': pair['id'] + '_' + branch, 'video_path': video.name, 'video_sha256': sha256(video),
            'run_receipt_path': receipt.name, 'run_sha256': sha256(receipt),
            'metric_report_path': metrics.name, 'metric_report_sha256': sha256(metrics),
            'reference_video_path': str(reference.resolve()), 'reference_video_sha256': sha256(reference),
            'reference_run_receipt_path': str(reference_receipt.resolve()) if branch == 'teacher' else receipt.name,
            'reference_run_sha256': reference_identity['run_sha256']})
    write_json(out / 'BUNDLE.json', bundle)
    failures = inspect(out / 'BUNDLE.json')
    if failures:
        raise ValueError('; '.join(failures))


def collect_failed_video_evidence(out, run, pair, sample, protocol_sha):
    """Failed runs still diagnose every actual clip; missing/invalid evidence stays explicit."""
    from ffnds_video_metrics import evaluate, validate
    receipt = out / 'FAILED_NATIVE_RUN.json'
    write_json(receipt, run)
    evidence = {'native_status': 'FAILED_PRESERVED_NO_RETRY', 'quality_accepted': False,
                'core_gate_pass': False, 'failed_receipt_sha256': sha256(receipt), 'videos': []}
    for branch, kind in zip(['teacher', 'student'], pair['modes']):
        video = out / (branch + '.mp4')
        entry = {'branch': branch, 'path': video.name}
        if not video.exists():
            entry['status'] = 'MISSING_NOT_A_GENERATED_VIDEO'
            evidence['videos'].append(entry)
            continue
        identity = {'source_id': pair['source_id'], 'seed': pair['seed'], 'expected_frames': 81,
                    'input_prompt_digest': object_sha({'source_id': pair['source_id'], 'prompt': sample.get('prompt'),
                                                      'negative_prompt': '', 'contract': run.get('assets', {}).get('contract')}),
                    'model_digest': object_sha({'kind': kind, 'assets': run['assets'], 'bank_RUN_sha256': run.get('bank_RUN_sha256')}) if run.get('assets') else None,
                    'protocol_sha256': protocol_sha, 'run_sha256': sha256(receipt)}
        costs = {key: run.get('cost', {}).get(key) for key in
                 ['generation_wall_seconds', 'ffn_boundary_seconds', 'setup_seconds', 'peak_gpu_memory_mib']}
        metadata = {'identity': identity, 'cost': costs}
        target = out / (branch + '_failed_native_metrics.json')
        try:
            report = evaluate(video, None, metadata)
            failures = validate(report)
            report['failed_native_scope'] = True
            report['metadata_failures'] = failures
            report['paired_reference_status'] = 'UNPAIRED_FORENSIC_ONLY; original native failure retained'
            write_json(target, report)
            entry.update(status='FULL_FRAME_DIAGNOSTICS_OF_FAILED_NATIVE',
                         decoded_frames=report['metrics']['artifact']['decoded_frames'], metadata_failures=failures)
        except BaseException as exc:
            write_json(target, {'status': 'DECODE_OR_EVALUATION_FAILED', 'identity': identity,
                                'video_sha256': sha256(video), 'error_type': type(exc).__name__,
                                'error': str(exc), 'quality_accepted': False})
            entry.update(status='DECODE_OR_EVALUATION_FAILED', error=str(exc))
        entry.update(video_sha256=sha256(video), report=target.name, report_sha256=sha256(target))
        evidence['videos'].append(entry)
    write_json(out / 'FAILED_VIDEO_EVIDENCE.json', evidence)
    return {'path': 'FAILED_VIDEO_EVIDENCE.json', 'sha256': sha256(out / 'FAILED_VIDEO_EVIDENCE.json'),
            'core_gate_pass': False}


def execute(args, plan, packs_manifest, status):
    if not status['model_scope_authorized']:
        raise RuntimeError('Exact fixed model scope requires separate explicit authorization registered in ACTIVE_RESEARCH')
    if args.study_root.resolve() != Path(plan['canonical_study_root']).resolve():
        raise ValueError('One frozen canonical study ledger; changing output root cannot reset attempts')
    from worker_identity import private_config_path, require_worker
    if sha256(private_config_path()) != plan['private_worker_config_sha256']:
        raise ValueError('Frozen privately selected executor changed')
    require_worker()
    pair = next(p for p in plan['pairs'] if p['id'] == args.pair)
    protocol_sha = sha256(args.protocol)
    reserve_attempt(args.study_root, plan, protocol_sha, pair)
    out = args.study_root / pair['id']; out.mkdir(exist_ok=False)
    run = {'status': 'STARTED_NOT_ACCEPTED', 'id': plan['id'], 'protocol_sha256': protocol_sha,
           'pair': pair, 'generations_started': 0, 'training_updates': 0, 'new_weight_fits': 0,
           'official_quality_calls': 0, 'quality_accepted': False, 'automatic_retry': False}
    write_json(out / 'RUN.json', run)
    monitor = None
    try:
        for name in ['HiF4_Sp', 'LightX2V', 'Wan2.2-I2V-A14B-W4A4']:
            sys.path.insert(0, str(ROOT / '.runtime/code' / name))
        for name in ['HiFloat4', 'HiFloat4/hif4_gpu']:
            sys.path.insert(0, str(ROOT / '.runtime/code/HiF4_Sp' / name))
        import torch
        from safetensors.torch import load_file, save_file
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from startup_gpu_reservation import reserve_before_assets
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from e0_paired_weight_residency import PairedWeightResidency
        from e0_paired_ffn_observer import PairedFFNObserver
        from e0_paired_wan_controller import PairedWanController
        from e0_full_depth_trace_contract import FullDepthTrace
        require_worker()
        host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
            raise RuntimeError('Host reserve unavailable; attempt consumed, no retry')
        reservation, monitor, receipt = reserve_before_assets(torch)
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Frozen single idle A100 required')
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
        setup_started = time.perf_counter()
        snapshot = out / 'EXECUTED_SOURCES'; snapshot.mkdir()
        for name in {**plan['driver_sources_sha256'], **plan['native_source_sha256']}:
            target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
        shutil.copyfile(args.protocol, snapshot / args.protocol.name)
        kwargs, cache, assets = verified_restored_assets(HistoricalSources())
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime = adapter.HIF4CacheRuntime(cache['path']); adapter.activate_lightx2v_hooks(runtime=runtime)
        packs = {(p['expert'], p['block']): load_file(str(args.bank / p['path']), device='cuda') for p in packs_manifest}
        coefficients = {(p['expert'], p['block']): p['sha256'] for p in packs_manifest}
        samples = json.loads((ROOT / plan['input_default']).read_text()); sample = samples[pair['source_index']]
        image = Path(sample.get('image_path') or sample['img_paths'][0]); seed = sample.get('seed', sample.get('calibration_seed'))
        if seed != pair['seed'] or sha256(image) != pair['source_id']:
            raise ValueError('Actual source/seed drift')
        run.update(source_id=pair['source_id'], seed=seed, assets=assets, CUDA_reservation=receipt,
                   bank_RUN_sha256=plan['bank_RUN_sha256'])
        def budget(step=None, block=None):
            monitor.check()
            if shutil.disk_usage(out).free < plan['disk_reserve_gib'] * (1 << 30):
                raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > plan['max_HF_output_gib'] * (1 << 30):
                raise RuntimeError('Fixed complete-study output cap reached')
        def save_vectors(step, block, ids, values):
            path = out / 'vectors' / f's{step}_b{block:02d}.safetensors'; path.parent.mkdir(exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, str(path))
            return {'path': str(path.relative_to(out)), 'sha256': sha256(path), 'row_ids': ids,
                    'coverage': 'PREDECLARED8_ROWS_FULL_WIDTH_BF16_NOT_FULL_M_VECTOR_STORAGE'}
        provenance = {'protocol_sha256': protocol_sha, 'native_source_sha256': object_sha(plan['native_source_sha256']),
                      'model_digest': object_sha(assets), 'bank_RUN_sha256': plan['bank_RUN_sha256']}
        factory = lambda shape, state_shape: FullDepthTrace(shape, pair['source_index'], seed, provenance, state_shape=state_shape)
        with torch.inference_mode(), PairedWeightResidency(runtime,
                max_non_ffn_bytes=plan['max_non_ffn_resident_bytes'], max_block_ffn_bytes=plan['max_block_ffn_resident_bytes']) as policy:
            for expert, records in runtime.records.items():
                for name, record in records.items():
                    if not name.endswith(('.ffn.0', '.ffn.2')):
                        runtime._weight(expert=expert, block_index=int(name.split('.')[1]), module_name=name,
                                        record=record, device=torch.device('cuda', 0), dtype=torch.bfloat16)
            torch.cuda.synchronize(); budget()
            setup_seconds = time.perf_counter() - setup_started
            with PairedFFNObserver(runtime, packs, adapter, None, coefficients, timing_events=True,
                    numerical_rows=32, vector_rows=8, save_vectors=save_vectors if pair['save_vectors'] else None) as observer:
                with PairedWanController(pipe.runner, observer, policy, adapter, factory,
                        modes=pair['modes'], progress=budget) as controller:
                    pipe.runner.scheduler.generator = None
                    run['generations_started'] = 2; write_json(out / 'RUN.json', run)
                    before = dict(runtime.counters); started = time.perf_counter()
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample['prompt'], negative_prompt='',
                                  save_result_path=str(out / 'student.mp4'))
                    info = controller.input_info
                    original_info = pipe.runner.input_info
                    try:
                        pipe.runner.input_info = info; info.save_result_path = str(out / 'teacher.mp4')
                        pipe.runner.gen_video_final = pipe.runner.run_vae_decoder(controller.teacher_latents)
                        pipe.runner.process_images_after_vae_decoder()
                    finally:
                        pipe.runner.input_info = original_info
                    torch.cuda.synchronize(); pair_wall = time.perf_counter() - started; budget()
                    timings = observer.timing_report()
                    if len(timings) != 320 or observer.counts['completed_sites'] != 160:
                        raise RuntimeError('Missing native FFN boundaries/sites')
                    expected = {'dense_main_GEMM': 640 if pair['modes'] == ['dense', 'dense'] else 320,
                                'dense_shadow_GEMM': 0 if pair['modes'] == ['dense', 'dense'] else 320,
                                'candidate_main_GEMM': 0 if pair['modes'] == ['dense', 'dense'] else 480,
                                'candidate_shadow_GEMM': 960 if pair['modes'] == ['dense', 'dense'] else 480,
                                'native_QDQ': 1280, 'continuous_GEMM': 960, 'completed_sites': 160}
                    if any(observer.counts[k] != v for k, v in expected.items()):
                        raise RuntimeError('Four-arm actual call ledger mismatch')
                    counts = {k: runtime.counters[k] - before[k] for k in before}
                    if counts['linear_forward_calls'] != 3200 or counts['activation_qdq_calls'] != 3200 or counts['cache_misses']:
                        raise RuntimeError('Dense/native non-FFN call ledger mismatch')
                    if policy.snapshot()['ffn_transfers'] != 320 or counts['weight_transfers'] != 320:
                        raise RuntimeError('Bounded current-block FFN transfers mismatch')
                    previous = json.loads((ROOT / plan['previous_result']).read_text())
                    for branch, kind in zip(['teacher', 'student'], pair['modes']):
                        old = next(r for r in previous['runs'] if r['mode'] == ('dense' if kind == 'dense' else 'affine_residual'))
                        raw_path = args.hf_root / old['raw_binding']['HF_relative']
                        if sha256(raw_path) != old['raw_binding']['sha256']:
                            raise ValueError('Original native receipt changed')
                        old_phase = next(v for v in json.loads(raw_path.read_text())['phases'] if v['source_index'] == pair['source_index'])
                        latent = [{k: v[k] for k in ['step', 'sha256', 'shape']} for v in controller.trajectory[branch]]
                        if latent != old_phase['trajectory'] or sha256(out / (branch + '.mp4')) != old_phase['video_sha256']:
                            raise RuntimeError('Instrumented branch changed original native latent/video identity: ' + branch)
                    write_json(out / 'TRACE.json', controller.completed_trace)
                    run.update(status='PASS_NATIVE_PAIRED_ACCOUNTING_REQUIRES_CORE_METRICS', trajectory=controller.trajectory,
                        trace_sha256=sha256(out / 'TRACE.json'), observer_counts=observer.counts,
                        native_runtime_counts=counts, residency=policy.snapshot(), FFN_primary_plus_shadow_timings=timings,
                        cost={'generation_wall_seconds': pair_wall, 'ffn_boundary_seconds': sum(t['primary_plus_shadow_ms'] for t in timings) / 1000,
                              'setup_seconds': setup_seconds, 'peak_gpu_memory_mib': monitor.report()['peak_physical_used_memory_MiB']},
                        cost_scope='WHOLE_PAIRED_OFFLINE_OBSERVATION_AND_TWO_DECODES_NOT_SINGLE_CANDIDATE_DEPLOYMENT_COST',
                        videos={b: {'path': b + '.mp4', 'sha256': sha256(out / (b + '.mp4'))} for b in ['teacher', 'student']})
        adapter.deactivate_lightx2v_hooks()
        del pipe, packs, runtime, controller, observer, policy
        gc.collect(); torch.cuda.empty_cache()
        run['GPU_allocated_bytes_before_CPU_metrics'] = torch.cuda.memory_allocated()
        monitor.close(); monitor.check(); run['assigned_GPU_resources'] = monitor.report(); monitor = None
        run['cost']['peak_gpu_memory_mib'] = run['assigned_GPU_resources']['peak_physical_used_memory_MiB']
        run['cost_deduplication_key'] = protocol_sha + '/' + pair['id']
        write_json(out / 'NATIVE_RUN.json', run)
        dense_run = next(r for r in previous['runs'] if r['mode'] == 'dense')
        path = args.hf_root / dense_run['raw_binding']['HF_relative']; raw = json.loads(path.read_text())
        phase = next(p for p in raw['phases'] if p['source_index'] == pair['source_index'])
        original = {'receipt': str(path), 'receipt_sha256': dense_run['raw_binding']['sha256'],
                    'video': str(path.parent / phase['video']), 'video_sha256': phase['video_sha256'], 'protocol_sha256': raw['protocol_sha256']}
        evaluate_pair(out, run, pair, sample, original, protocol_sha)
        run['status'] = 'PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY'; run['native_RUN_sha256'] = sha256(out / 'NATIVE_RUN.json')
    except BaseException as exc:
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        try:
            run['failed_video_evidence'] = collect_failed_video_evidence(
                out, run, pair, locals().get('sample', {}), protocol_sha)
        except BaseException as evidence_error:
            run['failed_video_evidence_error'] = repr(evidence_error)
        raise
    finally:
        if 'adapter' in locals():
            adapter.deactivate_lightx2v_hooks()
        if monitor is not None:
            monitor.close(); run['assigned_GPU_resources'] = monitor.report()
        write_json(out / 'RUN.json', run)
        close_attempt(args.study_root, pair['id'], run['status'], out / 'RUN.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'bank', 'hf-root', 'study-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pair', default='dd_source00')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    plan, packs, status = preflight(args.protocol, args.bank)
    print(json.dumps(status), flush=True)
    if args.execute_native:
        execute(args, plan, packs, status)


if __name__ == '__main__':
    main()
