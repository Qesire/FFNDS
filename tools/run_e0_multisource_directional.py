"""Four exposed-source native paired + whole-flow + video replication.

New bounded protocol; old paired and whole-flow drivers remain immutable.
Default verifies metadata/hashes only. No implicit native execution or retry.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from ffnds_observation_common import sha256, write_json
from run_e0_paired_full_depth import (ROOT, ledger_lock, object_sha, persist_observation,
                                    MANDATORY_DRIVER_SOURCES, MANDATORY_NATIVE_SOURCES,
                                    collect_failed_video_evidence)

PASS = 'PASS_NATIVE_MULTISOURCE_AND_CORE_METRICS_NOT_QUALITY'
ID = 'E0_MULTISOURCE_DIRECTIONAL_NATIVE_20261026_V1'
INDICES = [4, 8, 16, 20]


def original_source(plan, hf_root, pair):
    binding = plan['original_dense_sources'][str(pair['source_index'])]
    receipt = (hf_root / binding['HF_relative'] / 'RUN.json').resolve()
    if not receipt.is_relative_to(hf_root.resolve()) or sha256(receipt) != binding['run_sha256']:
        raise ValueError('Original passive receipt changed/escaped HF root')
    raw = json.loads(receipt.read_text())
    phase = raw['phases']['plain']
    video = receipt.parent / 'plain.mp4'
    if (raw['status'] != 'PASS_PASSIVE_CAPTURE_NOOP'
            or raw['source_index'] != pair['source_index'] or raw['source_id'] != pair['source_id']
            or raw['seed'] != pair['seed'] or phase['trajectory'] != binding['trajectory']
            or phase['video_sha256'] != binding['video_sha256']
            or sha256(video) != binding['video_sha256']):
        raise ValueError('Original dense source/seed/video/trajectory changed')
    return receipt, raw, phase


def preflight(protocol, bank, hf_root):
    plan = json.loads(protocol.read_text())
    needed = MANDATORY_DRIVER_SOURCES | {'tools/e0_whole_flow_counterfactual.py',
        'tools/run_e0_multisource_directional.py', 'tools/e0_disjoint_native_bands.py'}
    expected = {'whole_latent_trajectories': 8, 'additional_complete_dense_flows': 16,
                'new_videos': 8, 'new_weight_fits': 0, 'training_updates': 0, 'official_quality_calls': 0}
    if (plan['id'] != ID or not needed <= set(plan['driver_sources_sha256'])
            or not MANDATORY_NATIVE_SOURCES <= set(plan['native_source_sha256'])
            or [(p['id'], p['source_index'], p['seed'], p['modes']) for p in plan['pairs']] !=
               [(f'dc_source{i:02d}', i, 42+i, ['dense', 'candidate']) for i in INDICES]
            or plan['max_full_generations_total'] != 8
            or plan['planned_simultaneous_GPU_cards'] != 1 or plan['max_gpu_cards'] != 4
            or plan.get('counterfactual_steps') != [0, 1, 2, 3]
            or plan.get('render_videos') is not True
            or any(plan.get('flow_discriminator_budget', {}).get(k) != v for k, v in expected.items())
            or any(p['save_vectors'] for p in plan['pairs'])
            or plan.get('max_HF_output_gib') != 1
            or plan.get('automatic_retry') is not False):
        raise ValueError('Unsupported fixed four-exposed-source scope')
    for group in ['driver_sources_sha256', 'native_source_sha256']:
        for name, expected_sha in plan[group].items():
            if sha256(ROOT / name) != expected_sha:
                raise ValueError('Frozen source drift: ' + name)
    for name, expected_sha in [(plan['input_default'], plan['input_default_sha256']),
            ('.runtime/hif4_rebuilt_cache.json', plan['cache_manifest_sha256']),
            ('.runtime/model_view.json', plan['model_view_sha256']),
            ('.runtime/node_local_model_view.json', plan['local_model_view_sha256']),
            (plan['source_manifest'], plan['source_manifest_sha256']),
            (plan['directional_design'], plan['directional_design_sha256'])]:
        if sha256(ROOT / name) != expected_sha:
            raise ValueError('Frozen provenance drift: ' + name)
    local_view = json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
    vae = [r for r in local_view['files'] if r['filename'] == 'Wan2.1_VAE.pth']
    if (len(vae) != 1 or vae[0]['sha256'] != plan['VAE_sha256']
            or sha256(Path(vae[0]['view_path'])) != plan['VAE_sha256']
            or Path(vae[0]['view_path']).stat().st_size != plan['VAE_bytes']):
        raise ValueError('Actual original VAE checkpoint binding changed')
    manifest = json.loads((ROOT / plan['source_manifest']).read_text())
    samples = json.loads((ROOT / plan['input_default']).read_text())
    for pair in plan['pairs']:
        source = next(s for s in manifest['sources'] if s['index'] == pair['source_index'])
        binding = plan['original_dense_sources'][str(pair['source_index'])]
        if any(binding[k] != source[k] for k in ['source_id', 'seed', 'run_sha256', 'HF_relative', 'trajectory', 'video_sha256']):
            raise ValueError('Source manifest join differs')
        sample = samples[pair['source_index']]
        image = Path(sample.get('image_path') or sample['img_paths'][0])
        if (sha256(image) != pair['source_id']
                or sample.get('seed', sample.get('calibration_seed')) != pair['seed']
                or object_sha(sample['prompt']) != pair['prompt_digest']
                or pair['role'] != 'EXPOSED_AFFINE54_CALIBRATION'):
            raise ValueError('Frozen actual source/image/prompt/seed changed')
        original_source(plan, hf_root, pair)
    if sha256(bank / 'RUN.json') != plan['bank_RUN_sha256']:
        raise ValueError('Original bank receipt changed')
    packs = [p for p in json.loads((bank / 'RUN.json').read_text())['packs'] if p['family'] == 'affine_residual']
    if len(packs) != 80 or len({(p['expert'], p['block']) for p in packs}) != 80:
        raise ValueError('Exactly80 original fixed packs required')
    for pack in packs:
        if sha256(bank / pack['path']) != pack['sha256']:
            raise ValueError('Original immutable pack changed')
    scopes = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())['experiment_authority']['current_user_authorized_scopes']
    authorized = any(s['id'] == ID and s['protocol_sha256'] == sha256(protocol)
        and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'] for s in scopes)
    return plan, packs, {'status': 'STATIC_BINDINGS_PASS_NOT_NATIVE_VALIDATION',
        'model_scope_authorized': authorized, 'source_and_original_dense_verified': True,
        'new_videos': 0, 'GPU_initialized': False}


def reserve_attempt(study, plan, protocol_sha, pair):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        state = json.loads(path.read_text()) if path.exists() else {'id': plan['id'], 'protocol_sha256': protocol_sha, 'attempts': []}
        order = [p['id'] for p in plan['pairs']]
        if (state['id'] != plan['id'] or state['protocol_sha256'] != protocol_sha
                or len(state['attempts']) >= len(order)
                or [a.get('pair') for a in state['attempts']] != order[:len(state['attempts'])]
                or any(a['status'] != PASS for a in state['attempts'])
                or order[len(state['attempts'])] != pair['id']):
            raise RuntimeError('Fixed scope consumed, prior attempt failed/running or wrong pair order')
        state['attempts'].append({'pair': pair['id'], 'status': 'STARTED',
                                  'latent_trajectories_reserved': 2, 'extra_dense_flows_reserved': 4,
                                  'new_videos_reserved': 2})
        write_json(path, state)


def close_attempt(study, pair_id, receipt):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        state = json.loads(path.read_text())
        matches = [a for a in state['attempts'] if a['pair'] == pair_id]
        if len(matches) != 1:
            raise RuntimeError('Unique actually started source attempt required')
        entry = matches[0]
        run = json.loads(receipt.read_text())
        entry.update(status=run['status'], RUN_sha256=sha256(receipt))
        write_json(path, state)


def evaluate_current_pair(out, native, pair, sample, protocol_sha):
    """Both clips reference the current byte-verified dense branch inventory."""
    from ffnds_video_metrics import evaluate, validate
    from ffnds_video_metric_gate import inspect
    from ffnds_paired_video_strict_gate import inspect as strict_inspect, model_digest
    receipt = out / 'NATIVE_RUN.json'
    receipt_sha = sha256(receipt)

    def identity(kind):
        return {'source_id': pair['source_id'], 'seed': pair['seed'], 'expected_frames': 81,
            'input_prompt_digest': object_sha({'image_sha256': pair['source_id'], 'prompt': sample['prompt'],
                                              'negative_prompt': '', 'contract': native['assets']['contract']}),
            'model_digest': model_digest(native, kind), 'protocol_sha256': protocol_sha,
            'run_sha256': receipt_sha}

    dense = out / 'teacher.mp4'
    bundle = {'schema': 'ffnds.video.bundle.v1', 'expected_video_count': 2, 'videos': []}
    for branch, kind in zip(['teacher', 'student'], pair['modes']):
        video = out / (branch + '.mp4')
        metadata = {'identity': identity(kind), 'reference_identity': identity('dense'),
                    'cost': native['cost'], 'cost_scope': native['cost_scope']}
        report = evaluate(video, dense, metadata)
        report['cost_scope'] = native['cost_scope']
        report['cost_deduplication_key'] = native['cost_deduplication_key']
        failures = validate(report)
        if failures:
            raise ValueError('; '.join(failures))
        metric = out / (branch + '_metrics.json'); write_json(metric, report)
        bundle['videos'].append({'source_id': pair['source_id'], 'seed': pair['seed'],
            'variant': pair['id'] + '_' + branch, 'video_path': video.name, 'video_sha256': sha256(video),
            'run_receipt_path': receipt.name, 'run_sha256': receipt_sha,
            'metric_report_path': metric.name, 'metric_report_sha256': sha256(metric),
            'reference_video_path': dense.name, 'reference_video_sha256': sha256(dense),
            'reference_run_receipt_path': receipt.name, 'reference_run_sha256': receipt_sha})
    write_json(out / 'BUNDLE.json', bundle)
    if inspect(out / 'BUNDLE.json'):
        raise ValueError('Core full-video bundle failed')
    write_json(out / 'STRICT_BUNDLE.json', {'schema': 'ffnds.paired.video.strict.v1',
        'core_bundle_path': 'BUNDLE.json', 'core_bundle_sha256': sha256(out / 'BUNDLE.json'),
        'native_generation_receipts': [{'path': receipt.name, 'sha256': receipt_sha}]})
    failures = strict_inspect(out / 'STRICT_BUNDLE.json')
    if failures:
        raise ValueError('; '.join(failures))


def execute(args, plan, manifests, status):
    if not status['model_scope_authorized']:
        raise RuntimeError('Exact fixed model scope requires separate explicit authorization registered in ACTIVE_RESEARCH')
    if args.study_root.resolve() != Path(plan['canonical_study_root']).resolve():
        raise ValueError('One canonical ledger; output root cannot reset attempts')
    if not status.get('source_and_original_dense_verified'):
        raise ValueError('Original actual source and dense artifact preflight required')
    from worker_identity import private_config_path, require_worker
    if sha256(private_config_path()) != plan['private_worker_config_sha256']:
        raise ValueError('Privately selected executor binding changed')
    require_worker()
    pair = next(p for p in plan['pairs'] if p['id'] == args.pair)
    protocol_sha = sha256(args.protocol)
    reserve_attempt(args.study_root, plan, protocol_sha, pair)
    out = args.study_root / pair['id']
    out.mkdir(exist_ok=False)
    run = {'status': 'STARTED_NOT_ACCEPTED', 'id': plan['id'], 'protocol_sha256': protocol_sha,
           'pair': pair, 'latent_trajectories_started': 0, 'new_videos': 0,
           'new_weight_fits': 0, 'training_updates': 0, 'official_quality_calls': 0,
           'quality_accepted': False, 'automatic_retry': False}
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
        from e0_whole_flow_counterfactual import WholeFlowCounterfactualController
        from e0_full_depth_trace_contract import FullDepthTrace
        host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
            raise RuntimeError('Host reserve unavailable; consumed attempt, no retry')
        reservation, monitor, receipt = reserve_before_assets(torch)
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('One idle A100 required')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = plan['cudnn_allow_tf32']
        run['environment'] = {'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
            'GPU': torch.cuda.get_device_name(0), 'python': sys.version, 'TF32': False,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        setup_started = time.perf_counter()
        snapshot = out / 'EXECUTED_SOURCES'
        snapshot.mkdir()
        for name in {**plan['driver_sources_sha256'], **plan['native_source_sha256']}:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(args.protocol, snapshot / args.protocol.name)
        kwargs, cache, assets = verified_restored_assets(HistoricalSources())
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        run['VAE_sha256'] = plan['VAE_sha256']
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        packs = {(p['expert'], p['block']): load_file(str(args.bank / p['path']), device='cuda') for p in manifests}
        coefficients = {(p['expert'], p['block']): p['sha256'] for p in manifests}
        sample = json.loads((ROOT / plan['input_default']).read_text())[pair['source_index']]
        image = Path(sample.get('image_path') or sample['img_paths'][0])
        seed = sample.get('seed', sample.get('calibration_seed'))
        if seed != pair['seed'] or sha256(image) != pair['source_id']:
            raise ValueError('Actual source/seed drift')
        original_path, original_raw, original_phase = original_source(plan, args.hf_root, pair)
        if any(assets[k] != original_raw['assets'][k] for k in ['checkpoint_sha256', 'cache_manifest_sha256', 'contract']):
            raise ValueError('Current and historical dense asset identities differ')
        run.update(source_id=pair['source_id'], seed=seed, assets=assets, CUDA_reservation=receipt,
            source_role=pair['role'], bank_RUN_sha256=plan['bank_RUN_sha256'],
            original_dense_receipt={'path': str(original_path), 'sha256': sha256(original_path)})

        def budget(step=None, block=None):
            monitor.check()
            if shutil.disk_usage(out).free < plan['disk_reserve_gib'] * (1 << 30):
                raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > plan['max_HF_output_gib'] * (1 << 30):
                raise RuntimeError('Fixed complete-study output cap reached')

        def save_counterfactual(step, tensors, record):
            path = out / f'flow_s{step}.safetensors'
            if tensors is not None:
                save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))
                binding = {'path': path.name, 'sha256': sha256(path),
                           'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in tensors.items()}}
            else:
                binding = record['saved_tensors']
            record['saved_tensors'] = binding
            write_json(out / f'flow_s{step}.json', record)
            budget()
            return binding

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
            torch.cuda.synchronize()
            budget()
            run['setup_seconds'] = time.perf_counter() - setup_started
            with PairedFFNObserver(runtime, packs, adapter, None, coefficients, timing_events=True,
                                   numerical_rows=32, vector_rows=8, save_vectors=None) as observer:
                with WholeFlowCounterfactualController(pipe.runner, observer, policy, adapter, factory,
                        modes=pair['modes'], progress=budget, save_counterfactual=save_counterfactual,
                        counterfactual_steps=plan['counterfactual_steps']) as controller:
                    pipe.runner.scheduler.generator = None
                    run['latent_trajectories_started'] = 2
                    write_json(out / 'RUN.json', run)
                    before = dict(runtime.counters)
                    started = time.perf_counter()
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample['prompt'], negative_prompt='',
                                  save_result_path=str(out / 'student.mp4'))
                    torch.cuda.synchronize()
                    run['observation'] = persist_observation(out, controller, observer, runtime, before, policy)
                    # Preserve completed trace before the second VAE/media step.
                    write_json(out / 'RUN.json', run)
                    info, original_info = controller.input_info, pipe.runner.input_info
                    try:
                        pipe.runner.input_info = info; info.save_result_path = str(out / 'teacher.mp4')
                        pipe.runner.gen_video_final = pipe.runner.run_vae_decoder(controller.teacher_latents)
                        pipe.runner.process_images_after_vae_decoder()
                    finally:
                        pipe.runner.input_info = original_info
                    torch.cuda.synchronize()
                    run['observation_wall_seconds'] = time.perf_counter() - started
                    timings = observer.timing_report()
                    run.update(trajectory=controller.trajectory, counterfactual_records=controller.counterfactual_records,
                        observer_counts=dict(observer.counts), native_runtime_counts={k: runtime.counters[k] - before[k] for k in before},
                        residency=policy.snapshot(), FFN_primary_plus_shadow_timings=timings,
                        trace_sha256=sha256(out / 'TRACE.json'),
                        cost={'generation_wall_seconds': run['observation_wall_seconds'],
                              'ffn_boundary_seconds': sum(t['primary_plus_shadow_ms'] for t in timings)/1000,
                              'setup_seconds': run['setup_seconds'],
                              'peak_gpu_memory_mib': monitor.report()['peak_physical_used_memory_MiB']},
                        cost_scope='OFFLINE_PAIRED_2_TRAJECTORIES_4_EXTRA_DENSE_FLOWS_LOCAL_SHADOWS_2_DECODES_NOT_DEPLOYMENT_LATENCY',
                        cost_deduplication_key=protocol_sha + '/' + pair['id'],
                        videos={b: {'path': b+'.mp4', 'sha256': sha256(out/(b+'.mp4'))} for b in ['teacher','student']})
                    write_json(out / 'RUN.json', run)
                    counts = run['native_runtime_counts']
                    if (len(controller.counterfactual_records) != 4 or observer.counts['completed_sites'] != 160
                            or counts['linear_forward_calls'] != 4800 or counts['activation_qdq_calls'] != 4800
                            or counts['cache_misses'] or counts['weight_transfers'] != 640
                            or policy.snapshot()['ffn_transfers'] != 640
                            or any(r['native_runtime_counts']['linear_forward_calls'] != 400
                                   or r['native_runtime_counts']['activation_qdq_calls'] != 400
                                   for r in controller.counterfactual_records)):
                        raise RuntimeError('Whole-flow native coverage/call/residency ledger mismatch')
                    expected = {'dense_main_GEMM': 320, 'dense_shadow_GEMM': 320, 'candidate_main_GEMM': 480,
                                'candidate_shadow_GEMM': 480, 'native_QDQ': 1280, 'continuous_GEMM': 960}
                    if any(observer.counts[k] != v for k, v in expected.items()):
                        raise RuntimeError('Original paired observer ledger changed')
                    identity = [{k: v[k] for k in ['step', 'sha256', 'shape']} for v in controller.trajectory['teacher']]
                    if identity != original_phase['trajectory'] or sha256(out/'teacher.mp4') != original_phase['video_sha256']:
                        raise RuntimeError('Instrumented dense trajectory/video differs from original')
                    budget()
                    run['status'] = 'PASS_NATIVE_ACCOUNTING_REQUIRES_CPU_BANDS_AND_VIDEO_METRICS'
        adapter.deactivate_lightx2v_hooks()
        del pipe, packs, runtime, controller, observer, policy, reservation
        gc.collect(); torch.cuda.empty_cache()
        run['GPU_allocated_bytes_before_CPU_metrics'] = torch.cuda.memory_allocated()
        monitor.close(); monitor.check(); run['assigned_GPU_resources'] = monitor.report(); monitor = None
        run['cost']['peak_gpu_memory_mib'] = run['assigned_GPU_resources']['peak_physical_used_memory_MiB']
        write_json(out / 'NATIVE_RUN.json', run)
        evaluate_current_pair(out, run, pair, sample, protocol_sha)
        from e0_disjoint_native_bands import describe_saved_pair
        run['disjoint_bands'] = describe_saved_pair(out)
        if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > plan['max_HF_output_gib']*(1<<30):
            raise RuntimeError('Fixed output cap exceeded after metrics')
        run.update(status=PASS, new_videos=2, native_RUN_sha256=sha256(out/'NATIVE_RUN.json'))
    except BaseException as error:
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc())
        costs = run.setdefault('cost', {})
        if 'started' in locals():
            costs.setdefault('generation_wall_seconds', time.perf_counter()-started)
        if 'setup_seconds' in run:
            costs.setdefault('setup_seconds', run['setup_seconds'])
        elif 'setup_started' in locals():
            run['partial_setup_elapsed_seconds'] = time.perf_counter()-setup_started
        if monitor is not None:
            costs.setdefault('peak_gpu_memory_mib', monitor.report().get('peak_physical_used_memory_MiB'))
        run.setdefault('cost_scope', 'FAILED_PARTIAL_OFFLINE_OBSERVATION; MISSING_BOUNDARIES_REMAIN_UNMEASURED')
        if 'controller' in locals():
            run['counterfactual_records'] = controller.counterfactual_records
            try:
                run['observation'] = persist_observation(out, controller, locals().get('observer'),
                    locals().get('runtime'), locals().get('before'), locals().get('policy'))
            except BaseException as preservation_error:
                run['observation_preservation_error'] = repr(preservation_error)
        try:
            run['failed_video_evidence'] = collect_failed_video_evidence(out, run, pair, locals().get('sample', {}), protocol_sha)
        except BaseException as evidence_error:
            run['failed_video_evidence_error'] = repr(evidence_error)
        raise
    finally:
        original_error = sys.exc_info()[1]
        finalization_errors = []
        if 'adapter' in locals():
            try:
                adapter.deactivate_lightx2v_hooks()
            except BaseException as error:
                finalization_errors.append(error)
        if monitor is not None:
            try:
                monitor.close()
                run['assigned_GPU_resources'] = monitor.report()
                monitor.check()
            except BaseException as error:
                finalization_errors.append(error)
        if finalization_errors:
            run.update(status='FAILED_PRESERVED_NO_RETRY', finalization_errors=[repr(e) for e in finalization_errors])
        write_json(out / 'RUN.json', run)
        close_attempt(args.study_root, pair['id'], out / 'RUN.json')
        if finalization_errors and original_error is None:
            raise RuntimeError('Native finalization/isolation failed; attempt preserved') from finalization_errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'bank', 'hf-root', 'study-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pair', choices=[f'dc_source{i:02d}' for i in INDICES], required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    plan, packs, status = preflight(args.protocol, args.bank, args.hf_root)
    print(json.dumps(status), flush=True)
    if args.execute_native:
        execute(args, plan, packs, status)


if __name__ == '__main__':
    main()
