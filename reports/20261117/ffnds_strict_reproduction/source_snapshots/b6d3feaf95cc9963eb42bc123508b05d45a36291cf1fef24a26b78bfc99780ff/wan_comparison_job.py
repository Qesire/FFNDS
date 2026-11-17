"""Reviewed Wan free-generation and matching forced-Dense runner for FFN methods."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

from icc_native_job import authorized_stage, build_pipeline, read_json, sha, write_json


DEV11 = ('vbi_1013', 'vbi_1040', 'vbi_0886', 'vbi_0947', 'vbi_1109', 'vbi_1084',
         'vbi_0950', 'vbi_0176', 'vbi_0800', 'vbi_0392', 'vbi_0056')
DEV3 = ('vbi_1013', 'vbi_0886', 'vbi_0800')


def bound_json(spec):
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('changed manifest or receipt')
    return read_json(spec['path'])


def bind_parameter_loader(manifest, weights, *, method, factor_class=None):
    import torch

    if method == 'Q1r2':
        source = bound_json(manifest)
        if source['status'] != 'ALL_80_CONVERGED':
            raise RuntimeError('Q1 parameters incomplete')
        entries = {(row['expert'], row['layer']): row['artifact'] for row in source['sites']}
    else:
        source = bound_json(manifest)
        if source['status'] != 'ALL_80_FROZEN' or source['family'] not in ('SVD', 'CA', 'CD'):
            raise RuntimeError('ICC selected full-depth factors incomplete')
        entries = {(row['identity']['expert'], row['identity']['layer']): row for row in source['factor_pairs']}
    if set(entries) != {(expert, layer) for expert in ('high', 'low') for layer in range(40)}:
        raise RuntimeError('parameter manifest requires exactly all 80 sites')

    def load(expert, layer, device):
        spec = entries[expert, layer]
        path = Path(spec['path'])
        if sha(path) != spec['sha256']:
            raise RuntimeError(f'changed parameter file at {expert}/{layer}')
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        identity = data['identity']
        if identity['checkpoint_sha256'] != weights[expert]['sha256'] or identity['expert'] != expert or identity['layer'] != layer:
            raise RuntimeError('parameter provenance or site differs from runtime')
        if method == 'Q1r2':
            matrix = data['A_q1_f32']
            if matrix.shape != (5120, 5120) or matrix.dtype != torch.float32:
                raise RuntimeError('invalid Q1 F32 parameter matrix')
            return matrix.to(device=device, dtype=torch.bfloat16)
        if identity['family'] != source['family'] or identity['rank'] != 1408:
            raise RuntimeError('wrong frozen ICC factor family/rank')
        factors = tuple(factor_class(data[k + '_left'].to(device), data[k + '_right'].to(device)) for k in ('first', 'second'))
        if any(pair.left.dtype != torch.bfloat16 or pair.right.dtype != torch.bfloat16 for pair in factors):
            raise RuntimeError('ICC runtime factor dtype changed')
        return factors

    return load


def prepare_method(scope, prior, output):
    import sys
    import torch
    from wan_ffn_port import NativeICCPort
    from wan_ffn_comparator_ports import ComparatorPort
    from wan_chip_author_port import NativeAuthorChipPort

    method, forced = scope['arm'], scope['force_dense']
    cache_mode = scope.get('cache_mode', '4S')
    if method == 'Chipmunk Author Safe':
        provider, voxel = None, None
        if not forced:
            from chip_provider import load_author_provider
            provider = load_author_provider(scope['chip_provider'])
            path = scope['voxel']['path']
            if sha(path) != scope['voxel']['sha256']:
                raise RuntimeError('changed voxel map')
            coords = torch.load(path, map_location='cpu', weights_only=True)
            voxel = coords['voxel_to_raster'].cuda(), coords['raster_to_voxel'].cuda()
        return lambda _: NativeAuthorChipPort(force_dense=forced, provider=provider, voxel=voxel)

    loader, gp64 = None, None
    if not forced:
        reference = Path(scope['bws_source_dir']).resolve()
        for file, expected in scope['bws_sources_sha256'].items():
            if sha(reference / file) != expected:
                raise RuntimeError('changed reference arithmetic')
        sys.path.insert(0, str(reference))
        if method in ('ICC', 'Q1r2'):
            from icc_ffn import Factors
            loader = bind_parameter_loader(scope['parameters'], prior['checkpoints'], method=method, factor_class=Factors)
        elif method == 'Global-P64':
            from chipmunk_fullm_proxy_math import rms_score, selection
            from chipmunk_wan_packed_pregate import packed_pregate
            fixed_rows = torch.tensor(scope['fixed64_token_ids'], device='cuda', dtype=torch.long)
            gp64 = fixed_rows, rms_score, selection, packed_pregate

    if method == 'ICC':
        return lambda source: NativeICCPort(output / f'{source}_cache', force_dense=forced, factor_loader=loader, cache_mode=cache_mode)
    if method not in ('Q1r2', 'Global-P64'):
        raise RuntimeError('unknown comparison arm')
    return lambda source: ComparatorPort(output / f'{source}_cache', method=method, force_dense=forced, q1_loader=loader, gp64=gp64)


def render(pipe, scope, output, conditions, make_port):
    import resource
    import torch
    import runner_v2 as v2
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from wan_ffn_port import bind_native_ffn

    attempted = 0
    receipts = []
    forced = scope['force_dense']

    def log(record):
        with (output / 'NFE.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, allow_nan=False) + '\n')
            file.flush()
            os.fsync(file.fileno())

    for source in scope['sources']:
        condition = conditions[source]
        port = make_port(source)
        scheduler = pipe.runner.scheduler
        calls, flow_trace, final, initial, timings = [], [], [], [], []
        old_index = MultiDistillModelStruct.get_current_model_index
        old_post = scheduler.step_post

        def dispatch(model):
            nonlocal attempted
            step = int(scheduler.step_index)
            if step != len(calls) or step >= 4 or attempted >= scope['max_nfe']:
                raise RuntimeError('unexpected execution beyond frozen D4 slots')
            attempted += 1
            log({'event': 'NFE_ATTEMPT', 'source': source, 'step': step, 'attempt': attempted})
            value = old_index(model)
            expert = int(model.cur_model_index)
            if expert != step // 2:
                raise RuntimeError('actual expert dispatch differs from HHLL')
            calls.append({'step': step, 'expert': expert})
            if forced and step == 0:
                initial.append(v2.finite_tensor_sha(scheduler.latents, torch))
            return value

        def post():
            step = int(scheduler.step_index)
            if {'timestep': float(scheduler.timesteps[step]), 'sigma': float(scheduler.sigmas[step])} != scope['schedule'][step]:
                raise RuntimeError('live scheduler values changed')
            if forced:
                flow = v2.finite_tensor_sha(scheduler.noise_pred, torch)
            result = old_post()
            if forced:
                flow_trace.append({'flow': flow, 'latent': v2.finite_tensor_sha(scheduler.latents, torch)})
            if step == 3:
                final.append(scheduler.latents.detach().to('cpu', copy=True))
            return result

        def measured(model, phase, u):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            host_started = monotonic()
            out = port(model, phase, u)
            elapsed_host = monotonic() - host_started
            end.record()
            timings.append((start, end, elapsed_host))
            return out

        path = output / f'{source}.mp4'
        try:
            MultiDistillModelStruct.get_current_model_index, scheduler.step_post = dispatch, post
            v2.reset_scheduler_generator(scheduler, pipe.runner.model)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = monotonic()
            with bind_native_ffn(WanTransformerInfer, measured, exact_source_sha=scope['native_transformer_sha256']):
                pipe.generate(seed=42, image_path=condition['image_path'], prompt=condition['prompt'],
                              negative_prompt='', save_result_path=str(path))
            torch.cuda.synchronize()
            wall = monotonic() - start
        finally:
            scheduler.step_post = old_post
            MultiDistillModelStruct.get_current_model_index = old_index
        sites = port.finish()
        if len(calls) != 4 or len(final) != 1:
            raise RuntimeError('incomplete D4 trajectory')
        with (output / f'{source}_latent.pt').open('xb') as file:
            torch.save(final[0], file)
        video = v2.video_digest(path)
        receipt = {'source': source, 'arm': scope['arm'], 'sites': sites, 'wall_seconds': wall, 'video': video,
                   'ffn_stream_elapsed_ms': sum(a.elapsed_time(b) for a, b, _ in timings),
                   'ffn_host_call_seconds': sum(c for _, _, c in timings),
                   'peak_gpu_reserved': torch.cuda.max_memory_reserved(), 'peak_gpu_allocated': torch.cuda.max_memory_allocated(),
                   'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   'final_latent': v2.finite_tensor_sha(final[0], torch), 'dispatch': calls}
        if forced:
            baseline = bound_json(scope['plain_baselines'][source])
            prior = [{k: step[k] for k in ('flow', 'latent')} for step in baseline['steps']]
            matched = (prior == flow_trace and initial == [baseline['initial_noise']]
                       and video['decoded_rgb24_sha256'] == baseline['video']['decoded_rgb24_sha256'])
            receipt['neutral'] = matched
            write_json(output / f'{source}.json', receipt)
            if not matched:
                raise RuntimeError('forced Dense case differs from exact plain')
        else:
            write_json(output / f'{source}.json', receipt)
        receipts.append({'source': source, 'receipt_sha256': sha(output / f'{source}.json')})
    if attempted != scope['max_nfe']:
        raise RuntimeError('actual forward calls differ from registered slots')
    write_json(output / 'SUMMARY.json', {'status': 'COMPLETE', 'arm': scope['arm'], 'nfe': attempted, 'records': receipts})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('scope_sha')
    scope_sha = cli.parse_args().scope_sha
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, scope_sha)
    ids = scope['sources']
    forced_arms = scope.get('forced_arms')
    expected_arms = ['Q1r2', 'Global-P64', 'Chipmunk Author Safe']
    if forced_arms is not None and (forced_arms != expected_arms or not scope['force_dense']):
        raise RuntimeError('unexpected grouped control arms')
    expected = 4 * len(ids) * (len(forced_arms) if forced_arms is not None else 1)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(DEV11) or scope['max_nfe'] != expected:
        raise RuntimeError('source, count or budget differs from reviewed DEV cohort')
    if scope['force_dense']:
        if ids != ['vbi_1013', 'vbi_0800'] or (forced_arms is None and scope['arm'] not in expected_arms):
            raise RuntimeError('forced Dense scope does not fit approved controls')
    else:
        frozen = bound_json(scope['frozen_comparison'])
        if not frozen['main_family_selected'] or frozen['optional_cal22_enabled'] or scope['arm'] not in frozen['arms']:
            raise RuntimeError('approximate video cohort started before eligible freeze')
        if scope.get('cache_mode') == '2R' and (scope['arm'] != 'ICC' or tuple(ids) != DEV3 or not frozen['e4_same_factor_approved']):
            raise RuntimeError('invalid E4 source or factor contract')
    import torch
    import runner_v2 as v2

    with torch.inference_mode():
        condition_rows = read_json(reference / 'v2_conditions.json')
        v2.verify_conditions(condition_rows)
        conditions = {c['record_id']: c for c in condition_rows}
        pipe, output, prior = build_pipeline(stage, scope, reference)
        if forced_arms is None:
            maker = prepare_method(scope, prior, output)
            render(pipe, scope, output, conditions, maker)
        else:
            grouped = []
            for arm in forced_arms:
                child = output / {'Q1r2': 'q1', 'Global-P64': 'gp64', 'Chipmunk Author Safe': 'chip'}[arm]
                child.mkdir()
                arm_scope = dict(scope, arm=arm, max_nfe=8)
                maker = prepare_method(arm_scope, prior, child)
                render(pipe, arm_scope, child, conditions, maker)
                grouped.append({'arm': arm, 'receipt_sha256': sha(child / 'SUMMARY.json')})
            write_json(output / 'SUMMARY.json', {'status': 'PASS', 'arms': grouped, 'nfe': expected, 'extra_d4': 6})


if __name__ == '__main__':
    main()
