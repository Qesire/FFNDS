"""Fixed dense/whole94/affine54 full-video prototype comparison, no acceptance."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from analyze_ffnds_observations import load_arrays
from e0_reconstructed_runtime import ReconstructedRuntime, neutral_check
from ffnds_observation_common import sha256, write_json
from ffnds_all_weight_residency import AllWeightResidency
from prepare_ffnds_observations import preflight
from run_e0_all_weight_residency import FFNStageTimer

ROOT = Path(__file__).resolve().parents[1]


def tensor_sha(value):
    # Scheduler latents are explicitly converted to CPU float32 by the caller.
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def execute(args):
    plan = json.loads(args.protocol.read_text())
    if not args.execute_native or args.mode not in plan['modes']:
        raise ValueError('Fixed registered native mode required')
    host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
        raise RuntimeError('Host memory reserve unavailable before model loading')
    for name, expected in plan['driver_sources_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen driver changed')
    if (sha256(ROOT / plan['input_default']) != plan['input_default_sha256'] or
            sha256(ROOT / '.runtime/hif4_rebuilt_cache.json') != plan['cache_manifest_sha256']):
        raise ValueError('Canonical input/prompt or cache manifest changed')
    if preflight(ROOT, plan)['status'] != 'PREPARED_STATIC_PREFLIGHT':
        raise RuntimeError('Native static preflight failed')
    if sha256(args.bank / 'RUN.json') != plan['bank_RUN_sha256']:
        raise ValueError('Frozen prototype bank changed')
    bank = json.loads((args.bank / 'RUN.json').read_text())
    if bank['status'] != 'PASS_FIXED_RECONSTRUCTION_BANKS':
        raise ValueError('Incomplete prototype bank')
    for name in ['HiF4_Sp', 'LightX2V', 'Wan2.2-I2V-A14B-W4A4']:
        sys.path.insert(0, str(ROOT / '.runtime/code' / name))
    for name in ['HiFloat4', 'HiFloat4/hif4_gpu']:
        sys.path.insert(0, str(ROOT / '.runtime/code/HiF4_Sp' / name))
    import torch
    from safetensors.torch import load_file
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    require_worker()
    references = json.loads((ROOT / plan['accepted_observations']).read_text())
    if sha256(ROOT / plan['accepted_observations']) != plan['accepted_observations_sha256']:
        raise ValueError('Dense references changed')
    inputs = json.loads((ROOT / plan['input_default']).read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / 'EXECUTED_SOURCES'; snapshot.mkdir()
    for name in plan['driver_sources_sha256']:
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    shutil.copyfile(args.protocol, snapshot / args.protocol.name)
    record = {'status': 'STARTED_NOT_ACCEPTED', 'protocol_sha256': sha256(args.protocol),
              'mode': args.mode, 'bank_RUN_sha256': plan['bank_RUN_sha256'],
              'generations_started': 0, 'phases': [], 'neutral_controls': [],
              'optimizer_updates': 0, 'new_weight_fits': 0, 'official_quality_calls': 0,
              'quality_or_speed_acceptance': False, 'automatic_retry': False}
    write_json(args.out / 'RUN.json', record)
    monitor = runtime = policy = None
    try:
        reservation, monitor, receipt = reserve_before_assets(torch); record['CUDA_reservation'] = receipt
        if 'A100' not in torch.cuda.get_device_name(0): raise ValueError('Registered A100 required')
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
        record['environment'] = {'torch': torch.__version__, 'CUDA': torch.version.cuda,
                                 'GPU': torch.cuda.get_device_name(0), 'TF32': False,
                                 'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        setup_started = time.perf_counter()
        kwargs, cache, assets = verified_restored_assets(HistoricalSources()); record['assets'] = assets
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime = adapter.HIF4CacheRuntime(cache['path']); adapter.activate_lightx2v_hooks(runtime=runtime)
        packs = {}
        pack_started = time.perf_counter()
        if args.mode != 'dense':
            selected = [p for p in bank['packs'] if p['family'] == args.mode]
            if len(selected) != 80: raise ValueError('Incomplete mode bank')
            for p in selected:
                path = args.bank / p['path']
                if sha256(path) != p['sha256']: raise ValueError('Fitted weight file changed')
                packs[p['expert'], p['block']] = load_file(str(path), device='cuda')
            torch.cuda.synchronize()
            record['fitted_pack_load_seconds'] = time.perf_counter() - pack_started
            oldsource = next(v for v in references['sources'] if v['index'] == 0)
            rawpath = args.calibration_root / 'source00_attempt1/RUN.json'
            if sha256(rawpath) != oldsource['run_sha256']: raise ValueError('Neutral operand origin changed')
            raw = json.loads(rawpath.read_text())
            for step in [0, 2]:
                for block in [0, 30]:
                    scope = next(v for v in raw['scopes'] if (v['step'], v['block']) == (step, block))
                    path = rawpath.parent / 'observed' / scope['path']
                    if sha256(path) != scope['sha256']: raise ValueError('Neutral operand changed')
                    arrays = load_arrays(path)
                    values = {name: torch.as_tensor(arrays[name]).to('cuda') for name in ['modulated_input', 'b1', 'b2']}
                    record['neutral_controls'].append(neutral_check(runtime, adapter, values,
                        'high_noise_model' if step == 0 else 'low_noise_model', step, block, torch))
            runtime._resident_weights.clear(); runtime._resident_key = None
        retained_bytes = sum(rec['weight_qdq'].numel() * 2 for records in runtime.records.values()
                             for name, rec in records.items() if torch.is_tensor(rec.get('weight_qdq')) and
                             (args.mode == 'dense' or not name.endswith(('.ffn.0', '.ffn.2'))))
        pack_bytes = sum(v.numel() * v.element_size() for pack in packs.values() for v in pack.values())
        estimate = retained_bytes + pack_bytes + plan['original_sampled_peak_MiB'] * (1 << 20) + (2 << 30)
        if estimate > torch.cuda.get_device_properties(0).total_memory: raise RuntimeError('Prototype memory precheck failed')
        record['memory_precheck'] = {'original_resident_bytes': retained_bytes, 'fitted_pack_bytes': pack_bytes,
                                    'estimated_peak_plus2GiB': estimate}
        scheduler = pipe.runner.scheduler; oldpost = scheduler.step_post; trajectory = []
        def post():
            result = oldpost(); latent = scheduler.latents.detach().float().cpu().contiguous()
            if not bool(torch.isfinite(latent).all()): raise RuntimeError('Nonfinite free trajectory')
            trajectory.append({'step': int(scheduler.step_index), 'sha256': tensor_sha(latent), 'shape': list(latent.shape)})
            return result
        scheduler.step_post = post
        try:
            context = ReconstructedRuntime(runtime, packs, adapter) if packs else nullcontext()
            with torch.inference_mode(), AllWeightResidency(runtime, 'all_resident_no_flush', plan['max_resident_original_bytes']) as policy, context as hook, FFNStageTimer(adapter) as timer:
                # Equalize weight residency before either source is timed.
                # Candidate packs were also loaded above outside generation.
                preload_started = time.perf_counter()
                for expert, records in runtime.records.items():
                    for name, rec in records.items():
                        if not torch.is_tensor(rec.get('weight_qdq')) or (packs and name.endswith(('.ffn.0', '.ffn.2'))):
                            continue
                        runtime._weight(expert=expert, block_index=int(name.split('.')[1]), module_name=name,
                                        record=rec, device=torch.device('cuda', torch.cuda.current_device()), dtype=torch.bfloat16)
                torch.cuda.synchronize(); monitor.check()
                record['original_weight_preload_seconds'] = time.perf_counter() - preload_started
                record['setup_seconds_including_assets_pipeline_packs_neutral_and_weight_preload'] = time.perf_counter() - setup_started
                record['residency_before_first_generation'] = policy.snapshot()
                for order, index in enumerate(plan['source_sequence']):
                    if shutil.disk_usage(args.out).free < plan['disk_reserve_gib'] * (1 << 30): raise RuntimeError('Disk reserve reached')
                    sample = inputs[index]; reference = next(v for v in references['sources'] if v['index'] == index)
                    image = Path(sample.get('image_path') or sample['img_paths'][0]); seed = sample.get('seed', sample.get('calibration_seed'))
                    if sha256(image) != reference['source_id'] or seed != reference['seed']: raise ValueError('Input changed')
                    trajectory = []; timer.events.clear(); scheduler.generator = None
                    before = dict(runtime.counters); before_hook = dict(hook.counts) if packs else {}
                    video = args.out / f'source{index:02d}.mp4'
                    record['generations_started'] += 1; write_json(args.out / 'RUN.json', record)
                    started = time.perf_counter()
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample['prompt'], negative_prompt='', save_result_path=str(video))
                    torch.cuda.synchronize(); wall = time.perf_counter() - started; monitor.check()
                    counts = {k: runtime.counters[k] - before[k] for k in before}
                    expected = 1600 if args.mode == 'dense' else 1280
                    if counts['linear_forward_calls'] != expected or counts['activation_qdq_calls'] != expected or counts['cache_misses']:
                        raise RuntimeError('Native runtime call ledger mismatch')
                    if counts['weight_transfers'] != 0:
                        raise RuntimeError('Generation did not use the preregistered resident original weights')
                    delta = {k: hook.counts[k] - before_hook[k] for k in before_hook} if packs else {}
                    if packs and delta != {'up':160, 'down':160, 'Q1':160, 'Q2':160, 'affine':160 if args.mode == 'affine_residual' else 0}:
                        raise RuntimeError('Packed FFN ledger mismatch')
                    timings = timer.report()
                    if len(timings) != 160 or {(v['step'], v['block']) for v in timings} != {(s,b) for s in range(4) for b in range(40)}:
                        raise RuntimeError('Incomplete FFN timing boundary')
                    digest = sha256(video)
                    if args.mode == 'dense' and (trajectory != reference['trajectory'] or digest != reference['video_sha256']):
                        raise RuntimeError('Dense reference identity failed')
                    if [v['step'] for v in trajectory] != [0,1,2,3]: raise RuntimeError('Incomplete free trajectory')
                    record['phases'].append({'order':order, 'source_index':index, 'seed':seed,
                        'wall_seconds_including_video_save_and_common_latent_hash':wall,
                        'FFN_boundary_total_ms':sum(v['elapsed_ms'] for v in timings), 'FFN_boundary_timings':timings,
                        'runtime_native_counts':counts, 'prototype_native_counts':delta,
                        'trajectory':trajectory, 'video':video.name, 'video_sha256':digest,
                        'identity':'EXACT_HISTORICAL_DENSE' if not packs else 'APPROXIMATE_FREE_RUNNING_PROTOTYPE_NOT_ACCEPTED'})
                    write_json(args.out / 'RUN.json', record)
                    print(f'prototype {args.mode} source{index:02d}: wall={wall:.3f}s FFN={sum(v["elapsed_ms"] for v in timings)/1000:.3f}s', flush=True)
        finally: scheduler.step_post = oldpost
        if len(record['phases']) != 2: raise RuntimeError('Incomplete fixed video scope')
        if sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()) > plan['max_HF_output_gib_per_mode'] * (1 << 30):
            raise RuntimeError('Frozen prototype output cap exceeded')
        record['status'] = 'PASS_FIXED_PROTOTYPE_VIDEO_EXECUTION_NOT_QUALITY_ACCEPTANCE'
    except BaseException as exc:
        record.update(status='FAILED_PRESERVED', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        adapter.deactivate_lightx2v_hooks()
        if policy is not None: record['final_residency'] = policy.snapshot()
        if monitor is not None:
            monitor.close(); record['assigned_GPU_resources'] = monitor.report()
        write_json(args.out / 'RUN.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','bank','calibration-root','out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', required=True, choices=['dense','whole','affine_residual'])
    parser.add_argument('--execute-native', action='store_true')
    execute(parser.parse_args())
