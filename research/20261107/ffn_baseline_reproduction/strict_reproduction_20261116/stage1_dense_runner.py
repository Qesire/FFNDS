"""Candidate Stage-1 native plain/full-U runner; requires separately reviewed scope.

Imports the unchanged V2 reference from the explicitly bound reference directory.
No jobs are submitted by this program. ACTIVE/independent review and resource
allocation are the responsibility of the external launcher.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

from full_u_capture import FullUSink, FullUTap, SOURCES, GRID, ARCHIVE_LIMIT, digest


def append_event(path, event):
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(event, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def run(scope_path):
    scope_path = Path(scope_path).resolve()
    scope = json.loads(scope_path.read_text(encoding='utf8'))
    if scope['stage'] != 'STAGE1_DENSE_FULL_U' or scope['sources'] != list(SOURCES):
        raise RuntimeError('wrong scope or source order')
    if scope['max_d4'] != 4 or scope['max_nfe'] != 16 or scope['seed'] != 42:
        raise RuntimeError('budget/seed mismatch')
    ref = Path(scope['v2_reference_dir']).resolve()
    for name, expected in scope['reference_sha256'].items():
        path = (ref / name).resolve()
        if not path.is_relative_to(ref) or digest(path) != expected:
            raise RuntimeError('V2 reference identity mismatch ' + name)
    for name in ('runner_v2.py', 'capture_producer.py', 'panel512.py', 'runtime_guard.py', 'scope.json', 'v2_conditions.json'):
        if name not in scope['reference_sha256']:
            raise RuntimeError('unbound V2 reference ' + name)
    required_impl = {str(Path(__file__).resolve()), str(Path(__file__).with_name('full_u_capture.py').resolve())}
    if {str(Path(p).resolve()) for p in scope['implementation_sha256']} != required_impl:
        raise RuntimeError('implementation files not exactly bound')
    for path, expected in scope['implementation_sha256'].items():
        if digest(path) != expected: raise RuntimeError('implementation identity mismatch ' + path)
    sys.path.insert(0, str(ref))
    import runner_v2 as v2
    from capture_producer import NativeTap512, PANEL_COUNT
    import torch
    import numpy as np
    import imageio_ffmpeg
    prior = json.loads((ref / 'scope.json').read_text(encoding='utf8'))
    conds = json.loads((ref / 'v2_conditions.json').read_text(encoding='utf8'))
    v2.verify_conditions(conds)
    v2.verify_pipeline_binding(prior)
    for path in reversed(prior['python_import_paths']): sys.path.insert(0, path)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('missing Slurm allocation or GPU visibility')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('requires a single allocated A100 80GB')
    if torch.__version__ != prior['runtime_versions']['torch'] or np.__version__ != prior['runtime_versions']['numpy'] or imageio_ffmpeg.__version__ != prior['runtime_versions']['imageio_ffmpeg']:
        raise RuntimeError('runtime mismatch')
    if digest(imageio_ffmpeg.get_ffmpeg_exe()) != prior['ffmpeg_sha256']:
        raise RuntimeError('ffmpeg identity mismatch')
    if digest(scope['scheduler_file']) != scope['scheduler_sha256']:
        raise RuntimeError('scheduler source mismatch')
    if scope['scheduler_sha256'] != '2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db':
        raise RuntimeError('unreviewed scheduler identity')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    out = Path(scope['output']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    journal = out / 'RUN_EVENTS.jsonl'
    append_event(journal, {'event': 'START', 'scope_sha256': digest(scope_path), 'time': time.time()})
    sink = FullUSink(torch, out / 'capture', ARCHIVE_LIMIT)
    started = time.monotonic()
    budget = scope['resources']
    def check_resources():
        import resource
        if time.monotonic() - started > budget['soft_seconds']: raise TimeoutError('time limit')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss > budget['host_memory_gib'] * 1024**3: raise MemoryError('host RSS limit')
        if torch.cuda.max_memory_reserved() > budget['gpu_memory_gib'] * 1024**3: raise MemoryError('GPU limit')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > budget['output_bytes']:
            raise RuntimeError('total output limit')
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    if MMWeight.apply.__module__ != 'lightx2v.common.ops.mm.mm_weight' or MMWeight.apply.__qualname__ != 'MMWeight.apply':
        raise RuntimeError('dense MMWeight patched')
    pipe = v2.build_checked_pipeline(prior, out)
    scheduler_type = type(pipe.runner.scheduler)
    if scheduler_type.__name__ != 'Wan22StepDistillScheduler':
        raise RuntimeError('unexpected scheduler class')
    class_file = Path(inspect.getfile(scheduler_type)).resolve()
    method_file = Path(inspect.getfile(scheduler_type.step_post)).resolve()
    if class_file != Path(scope['scheduler_file']).resolve() or method_file != class_file or digest(class_file) != scope['scheduler_sha256']:
        raise RuntimeError('active scheduler class/method source mismatch')
    v2.exclusive_json(out / 'ALLOCATION.json', {'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'device': torch.cuda.get_device_name(),
        'scheduler_class': scheduler_type.__module__ + '.' + scheduler_type.__qualname__, 'scheduler_source': str(class_file),
        'scheduler_sha256': digest(class_file)})
    byid = {c['record_id']: c for c in conds}
    fixed64 = torch.randperm(31668, generator=torch.Generator(device='cpu').manual_seed(42))[:64].tolist()
    if fixed64 != scope['fixed64_token_ids']:
        raise RuntimeError('historical fixed64 identity mismatch')
    voxel_path = Path(scope['author_voxel_file']).resolve()
    voxel_sha = digest(voxel_path)
    if voxel_sha != scope['author_voxel_sha256'] or voxel_sha != '9a72dc5007e2eab4772d108a7973f9058028a2d17b88db38aa15e54a5f82408c':
        raise RuntimeError('author voxel source identity mismatch')
    from einops import rearrange
    tree = ast.parse(voxel_path.read_bytes(), filename=str(voxel_path))
    names = {'voxel_chunk_no_padding', 'reverse_voxel_chunk_no_padding'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names: raise RuntimeError('missing author voxel function')
    ns = {'torch': torch, 'rearrange': rearrange}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(voxel_path), 'exec'), ns)
    raster = torch.arange(31668, dtype=torch.long).reshape(1, 1, *GRID, 1)
    perm = ns['voxel_chunk_no_padding'](raster, voxel_shape=(4, 6, 8)).reshape(-1)
    restored = ns['reverse_voxel_chunk_no_padding'](perm.reshape(1, 1, -1, 1), tuple(raster.shape), voxel_shape=(4, 6, 8))
    if not torch.equal(restored, raster): raise RuntimeError('author voxel roundtrip mismatch')
    inverse = torch.argsort(perm)
    sink.write_voxel(perm, inverse, voxel_sha)
    completed, attempts = {}, 0
    def arm(cond, capture):
        nonlocal attempts
        check_resources()
        tag = 'full_u' if capture else 'plain'
        video = out / f"{cond['record_id']}__{tag}.mp4"
        tap = None
        if capture:
            def panel(event, payload):
                event.update(image_sha256=cond['image_sha256'], prompt_sha256=cond['prompt_sha256'],
                             source_cluster_id=cond['source_cluster_id'], scope_sha256=digest(scope_path),
                             checkpoint_sha256=prior['checkpoints'][event['expert']]['sha256'])
                sink.write_panel(event, payload); check_resources()
            base = NativeTap512(torch, panel, PANEL_COUNT, MMWeight, GRID)
            base.begin_source(cond['record_id'], 42)
            tap = FullUTap(torch, base, sink)
        scheduler = pipe.runner.scheduler
        old_dispatch, old_post = MultiDistillModelStruct.get_current_model_index, scheduler.step_post
        dispatch, flow, latents, noise = [], [], [], []
        def dispatched(model):
            nonlocal attempts
            step = int(model.scheduler.step_index)
            if attempts >= 16: raise RuntimeError('NFE limit')
            attempts += 1
            append_event(journal, {'event': 'NFE_ATTEMPT', 'source': cond['record_id'], 'arm': tag, 'step': step, 'attempt': attempts})
            value = old_dispatch(model)
            expert = int(model.cur_model_index)
            dispatch.append({'step': step, 'expert': expert})
            if step == 0 and not noise: noise.append(v2.finite_tensor_sha(scheduler.latents, torch))
            if tap: tap.v2.dispatch = expert
            return value
        def observed_post():
            step = int(scheduler.step_index)
            pred = v2.finite_tensor_sha(scheduler.noise_pred, torch)
            row = {'step': step, 'flow': pred, 'timestep': float(scheduler.timesteps[step].item()),
                   'sigma': float(scheduler.sigmas[step].item())}
            expected = scope['schedule'][step]
            if row['timestep'] != expected['timestep'] or row['sigma'] != expected['sigma']:
                raise RuntimeError('actual timestep/sigma differs from frozen schedule')
            result = old_post()
            row['latent'] = v2.finite_tensor_sha(scheduler.latents, torch)
            if step == 3:
                latent_path = out / f"{cond['record_id']}__{tag}__final_latent.pt"
                with latent_path.open('xb') as f:
                    torch.save(scheduler.latents.detach().to('cpu', copy=True), f)
                    f.flush(); os.fsync(f.fileno())
                row['final_latent_artifact'] = {'path': latent_path.name, 'bytes': latent_path.stat().st_size, 'sha256': digest(latent_path)}
            flow.append(row); latents.append({'step': step, **row['latent']})
            append_event(journal, {'event': 'NFE_STEP_POST', 'source': cond['record_id'], 'arm': tag, **row})
            return result
        try:
            if tap: tap.install(WanTransformerInfer)
            MultiDistillModelStruct.get_current_model_index, scheduler.step_post = dispatched, observed_post
            v2.reset_scheduler_generator(scheduler, pipe.runner.model)
            append_event(journal, {'event': 'D4_START', 'source': cond['record_id'], 'arm': tag})
            with torch.no_grad():
                pipe.generate(seed=42, image_path=cond['image_path'], prompt=cond['prompt'], negative_prompt='', save_result_path=str(video))
            if tap: tap.v2.finish_source(); sink.finish_source(cond['record_id'])
        finally:
            scheduler.step_post = old_post
            if tap: tap.uninstall()
            MultiDistillModelStruct.get_current_model_index = old_dispatch
        if dispatch != [{'step': i, 'expert': 0 if i < 2 else 1} for i in range(4)] or [r['step'] for r in flow] != list(range(4)) or len(noise) != 1:
            raise RuntimeError('incomplete trajectory')
        result = {'source': cond['record_id'], 'arm': tag, 'image_sha256': cond['image_sha256'],
                  'prompt_sha256': cond['prompt_sha256'], 'initial_noise': noise[0], 'steps': flow,
                  'dispatch': dispatch, 'video': v2.video_digest(video)}
        v2.exclusive_json(out / f"{cond['record_id']}__{tag}.json", result)
        append_event(journal, {'event': 'D4_COMPLETE', 'source': cond['record_id'], 'arm': tag})
        check_resources()
        return result
    plains = {source: arm(byid[source], False) for source in SOURCES}
    for source in SOURCES:
        plain = plains[source]
        captured = arm(byid[source], True)
        comparison = {'source': source, 'initial_noise_equal': plain['initial_noise'] == captured['initial_noise'],
                      'flows_equal': [a['flow'] == b['flow'] for a, b in zip(plain['steps'], captured['steps'])],
                      'latents_equal': [a['latent'] == b['latent'] for a, b in zip(plain['steps'], captured['steps'])],
                      'time_sigma_equal': [(a['timestep'], a['sigma']) == (b['timestep'], b['sigma']) for a, b in zip(plain['steps'], captured['steps'])],
                      'decoded_equal': plain['video']['decoded_rgb24_sha256'] == captured['video']['decoded_rgb24_sha256']}
        v2.exclusive_json(out / f'NEUTRAL_{source}.json', comparison)
        if not (comparison['initial_noise_equal'] and all(comparison['flows_equal']) and all(comparison['latents_equal']) and all(comparison['time_sigma_equal']) and comparison['decoded_equal']):
            raise RuntimeError('plain/capture neutral mismatch')
        completed[source] = comparison
    manifest = sink.finish(fixed64)
    import resource
    summary = {'status': 'COMPLETE', 'nfe_attempts': attempts, 'd4_completed': 4, 'main_dense_slots': 2,
               'plain_extra_slots': 2, 'per_source': completed, 'archive_bytes': manifest['archive_bytes'],
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'peak_gpu_allocated': torch.cuda.max_memory_allocated(), 'peak_gpu_reserved': torch.cuda.max_memory_reserved()}
    v2.exclusive_json(out / 'SUMMARY.json', summary)
    append_event(journal, {'event': 'COMPLETE', 'nfe_attempts': attempts, 'd4_completed': 4})
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('scope')
    args = parser.parse_args(); run(args.scope)





