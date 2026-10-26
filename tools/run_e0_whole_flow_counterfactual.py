"""Frozen whole-flow discriminator; default is CPU-only provenance preflight.

Replays complete dense/candidate latent paths, with eight extra complete dense
flows across two sources. No VAE decode, new videos, fitting or correction.
Native execution requires separate exact-scope registration in ACTIVE_RESEARCH.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from ffnds_observation_common import sha256, write_json
from run_e0_paired_full_depth import (ROOT, historical_branch, ledger_lock, object_sha,
                                    persist_observation, preflight as paired_preflight)

PASS = 'PASS_NATIVE_LATENT_IDENTITY_AND_FLOW_SPLIT_NOT_QUALITY'


def preflight(protocol, bank, hf_root):
    plan, packs, status = paired_preflight(protocol, bank, hf_root)
    needed = {'tools/e0_whole_flow_counterfactual.py', 'tools/run_e0_whole_flow_counterfactual.py'}
    expected = {'whole_latent_trajectories': 4, 'additional_complete_dense_flows': 8,
                'new_videos': 0, 'new_weight_fits': 0, 'training_updates': 0, 'official_quality_calls': 0}
    if (not needed <= set(plan['driver_sources_sha256'])
            or plan.get('counterfactual_steps') != [0, 1, 2, 3]
            or plan.get('render_videos') is not False
            or any(plan.get('flow_discriminator_budget', {}).get(k) != v for k, v in expected.items())
            or any(p['save_vectors'] for p in plan['pairs'])
            or plan.get('max_HF_output_gib') != 0.5
            or plan.get('no_automatic_followup') is not True):
        raise ValueError('Unsupported fixed whole-flow scope')
    for pair in plan['pairs']:
        binding = plan['retained_pair_receipts'][pair['id']]
        root = (hf_root / 'ffnds-derived').resolve()
        receipt = (root / binding['RUN_HF_relative']).resolve()
        if not receipt.is_relative_to(root) or sha256(receipt) != binding['RUN_sha256']:
            raise ValueError('Retained native pair receipt changed')
        old = json.loads(receipt.read_text())
        if (old['status'] != 'PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY'
                or old['pair']['source_index'] != pair['source_index'] or old['seed'] != pair['seed']
                or old['trace_sha256'] != binding['TRACE_sha256']
                or sha256(receipt.parent / 'TRACE.json') != binding['TRACE_sha256']
                or sha256(receipt.parent / 'STRICT_BUNDLE.json') != binding['STRICT_BUNDLE_sha256']):
            raise ValueError('Retained source/seed/trace identity mismatch')
        from ffnds_paired_video_strict_gate import inspect
        failures = inspect(receipt.parent / 'STRICT_BUNDLE.json')
        if failures:
            raise ValueError('Retained video evidence failed: ' + '; '.join(failures))
    return plan, packs, {**status, 'retained_native_and_video_evidence_verified': True,
                         'new_videos': 0, 'additional_complete_dense_flows_started': 0}


def reserve_attempt(study, plan, protocol_sha, pair):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        state = json.loads(path.read_text()) if path.exists() else {'id': plan['id'], 'protocol_sha256': protocol_sha, 'attempts': []}
        order = [p['id'] for p in plan['pairs']]
        if (state['id'] != plan['id'] or state['protocol_sha256'] != protocol_sha
                or len(state['attempts']) >= len(order)
                or any(a['status'] != PASS for a in state['attempts'])
                or order[len(state['attempts'])] != pair['id']):
            raise RuntimeError('Fixed scope consumed, prior attempt failed/running or wrong pair order')
        state['attempts'].append({'pair': pair['id'], 'status': 'STARTED',
                                  'latent_trajectories_reserved': 2, 'extra_dense_flows_reserved': 4})
        write_json(path, state)


def close_attempt(study, pair_id, receipt):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        state = json.loads(path.read_text())
        entry = next(a for a in state['attempts'] if a['pair'] == pair_id)
        run = json.loads(receipt.read_text())
        entry.update(status=run['status'], RUN_sha256=sha256(receipt))
        write_json(path, state)


def execute(args, plan, manifests, status):
    if not status['model_scope_authorized']:
        raise RuntimeError('Exact fixed model scope requires separate explicit authorization registered in ACTIVE_RESEARCH')
    if args.study_root.resolve() != Path(plan['canonical_study_root']).resolve():
        raise ValueError('One canonical ledger; output root cannot reset attempts')
    if not status.get('retained_native_and_video_evidence_verified'):
        raise ValueError('Retained artifact preflight required')
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
        from e0_whole_flow_counterfactual import LatentOnlyMain, WholeFlowCounterfactualController
        from e0_full_depth_trace_contract import FullDepthTrace
        host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
            raise RuntimeError('Host reserve unavailable; consumed attempt, no retry')
        reservation, monitor, receipt = reserve_before_assets(torch)
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('One idle A100 required')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        run['environment'] = {'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
            'GPU': torch.cuda.get_device_name(0), 'python': sys.version, 'TF32': False,
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
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        packs = {(p['expert'], p['block']): load_file(str(args.bank / p['path']), device='cuda') for p in manifests}
        coefficients = {(p['expert'], p['block']): p['sha256'] for p in manifests}
        sample = json.loads((ROOT / plan['input_default']).read_text())[pair['source_index']]
        image = Path(sample.get('image_path') or sample['img_paths'][0])
        seed = sample.get('seed', sample.get('calibration_seed'))
        if seed != pair['seed'] or sha256(image) != pair['source_id']:
            raise ValueError('Actual source/seed drift')
        run.update(source_id=pair['source_id'], seed=seed, assets=assets, CUDA_reservation=receipt,
                   bank_RUN_sha256=plan['bank_RUN_sha256'], retained_evidence=plan['retained_pair_receipts'][pair['id']])

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
                        counterfactual_steps=plan['counterfactual_steps']) as controller, LatentOnlyMain(pipe.runner):
                    pipe.runner.scheduler.generator = None
                    run['latent_trajectories_started'] = 2
                    write_json(out / 'RUN.json', run)
                    before = dict(runtime.counters)
                    started = time.perf_counter()
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample['prompt'], negative_prompt='',
                                  save_result_path=str(out / 'NO_VIDEO_CREATED'))
                    torch.cuda.synchronize()
                    run['observation_wall_seconds'] = time.perf_counter() - started
                    run['observation'] = persist_observation(out, controller, observer, runtime, before, policy)
                    run.update(trajectory=controller.trajectory, counterfactual_records=controller.counterfactual_records,
                        observer_counts=dict(observer.counts), native_runtime_counts={k: runtime.counters[k] - before[k] for k in before},
                        residency=policy.snapshot(), FFN_primary_plus_shadow_timings=observer.timing_report(),
                        trace_sha256=sha256(out / 'TRACE.json'),
                        cost_scope='LATENT_REPLAY_PLUS_FULL_DENSE_COUNTERFACTUAL_AND_LOCAL_SHADOWS_NOT_DEPLOYMENT_LATENCY')
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
                    for branch, kind in zip(['teacher', 'student'], pair['modes']):
                        _, _, old_phase = historical_branch(plan, args.hf_root, pair, kind)
                        identity = [{k: v[k] for k in ['step', 'sha256', 'shape']} for v in controller.trajectory[branch]]
                        if identity != old_phase['trajectory']:
                            raise RuntimeError('Counterfactual changed original latent trajectory: ' + branch)
                    if list(out.glob('*.mp4')):
                        raise RuntimeError('Latent-only protocol unexpectedly created a video')
                    budget()
                    run['status'] = PASS
    except BaseException as error:
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc())
        if 'controller' in locals():
            run['counterfactual_records'] = controller.counterfactual_records
            try:
                run['observation'] = persist_observation(out, controller, locals().get('observer'),
                    locals().get('runtime'), locals().get('before'), locals().get('policy'))
            except BaseException as preservation_error:
                run['observation_preservation_error'] = repr(preservation_error)
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
    parser.add_argument('--pair', default='dc_source00')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    plan, packs, status = preflight(args.protocol, args.bank, args.hf_root)
    print(json.dumps(status), flush=True)
    if args.execute_native:
        execute(args, plan, packs, status)


if __name__ == '__main__':
    main()
