"""Guarded native jobs for the frozen ICC host neutral and calibration scopes."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def authorized_stage(stage, expected_sha):
    scope_path = stage / 'scope.json'
    if sha(scope_path) != expected_sha:
        raise RuntimeError('changed execution scope')
    scope = read_json(scope_path)
    review = read_json(stage / 'REVIEW.json')
    active = read_json(stage / 'ACTIVE_RESEARCH.json')[scope['scope_id'].lower()]
    if (not active['runnable'] or active['scope_sha256'] != expected_sha or review['scope_sha256'] != expected_sha
            or review['recheck_disposition'] != 'APPROVE' or active['review_sha256'] != sha(stage / 'REVIEW.json')):
        raise RuntimeError('native launch needs the matching independent scope and ACTIVE record')
    if not Path(sys.executable).samefile(scope['python_executable']) or sha(sys.executable) != scope['python_executable_sha256']:
        raise RuntimeError('interpreter identity changed')
    if scope['method'] == 'ICC_AUTHOR_CALIBRATION':
        prerequisite = scope['icc_forced_dense_prerequisite']
        prior_stage = Path(prerequisite['stage'])
        if (sha(prior_stage / 'scope.json') != prerequisite['scope_sha256']
                or read_json(prior_stage / 'receipt' / 'EXIT.json')['exit_code'] != 0
                or read_json(prior_stage / 'output' / 'SUMMARY.json')['status'] != 'PASS'):
            raise RuntimeError('ICC host forced-dense prerequisite did not pass')
    for name, digest in scope['bundle_sha256'].items():
        bound = (stage / name).resolve()
        if not bound.is_relative_to(stage) or sha(bound) != digest:
            raise RuntimeError(f'changed stage file {name}')
    reference = Path(scope['v2_reference_dir'])
    for name, digest in scope['reference_sha256'].items():
        if sha(reference / name) != digest:
            raise RuntimeError(f'changed V2 reference file {name}')
    sys.path.insert(0, str(reference))
    from runtime_guard import require_compute_allocation

    allocation = require_compute_allocation()
    ceilings = scope['resources']
    if 'Partition=P107-A100 ' not in allocation['scheduler_job_record'] or allocation['memory_mib'] > ceilings['host_memory_gib'] * 1024:
        raise RuntimeError('unexpected partition or memory')
    limit = re.search(r'\bTimeLimit=(\d+):(\d+):(\d+)\b', allocation['scheduler_job_record'])
    if not limit or sum(int(v) * k for v, k in zip(limit.groups(), (3600, 60, 1))) > ceilings['wall_seconds']:
        raise RuntimeError('unexpected wall limit')
    if shutil.disk_usage(stage).free < ceilings['min_free_disk_gib'] * 1024 ** 3:
        raise RuntimeError('insufficient scratch')
    write_json(stage / 'receipt' / 'RUN.json', {'scope_sha256': expected_sha, 'allocation': allocation,
                                              'start_utc': datetime.now(timezone.utc).isoformat()})
    return scope, reference


def build_pipeline(stage, scope, reference):
    import imageio_ffmpeg
    import numpy as np
    import torch
    import runner_v2 as v2

    prior = read_json(reference / 'scope.json')
    v2.verify_pipeline_binding(prior)
    for path in reversed(prior['python_import_paths']):
        sys.path.insert(0, path)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('one A100 80 GB required')
    if (torch.__version__, np.__version__, imageio_ffmpeg.__version__) != tuple(prior['runtime_versions'][k] for k in ('torch', 'numpy', 'imageio_ffmpeg')):
        raise RuntimeError('native numeric runtime changed')
    if sha(imageio_ffmpeg.get_ffmpeg_exe()) != prior['ffmpeg_sha256']:
        raise RuntimeError('decoder changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, scope['resources']['gpu_memory_gib'] * 1024 ** 3 / props.total_memory))
    output = stage / 'output'
    output.mkdir()
    pipe = v2.build_checked_pipeline(prior, output)
    sched = type(pipe.runner.scheduler)
    if sched.__name__ != 'Wan22StepDistillScheduler' or sha(inspect.getfile(sched.step_post)) != scope['scheduler_sha256']:
        raise RuntimeError('active Wan scheduler identity changed')
    return pipe, output, prior


def forced_dense(pipe, stage, scope, reference, output):
    import torch
    import runner_v2 as v2
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from wan_ffn_port import bind_native_ffn, NativeICCPort

    sources = scope['sources']
    if sources != ['vbi_1013', 'vbi_0800'] or scope['max_d4'] != 2 or scope['max_nfe'] != 8:
        raise RuntimeError('forced-dense scope was extended')
    conditions = read_json(reference / 'v2_conditions.json')
    v2.verify_conditions(conditions)
    by_id = {row['record_id']: row for row in conditions}
    baselines = {source: read_json(stage / f'{source}__plain.json') for source in sources}
    attempts = 0

    def event(record):
        with (output / 'NFE.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    for source in sources:
        row = by_id[source]
        plain = baselines[source]
        if row['image_sha256'] != plain['image_sha256'] or row['prompt_sha256'] != plain['prompt_sha256'] or row['seed'] != 42:
            raise RuntimeError('plain source conditions differ')
        port = NativeICCPort(output / f'{source}_cache', force_dense=True)
        scheduler = pipe.runner.scheduler
        old_index = MultiDistillModelStruct.get_current_model_index
        old_post = scheduler.step_post
        dispatch, trace, noise = [], [], []

        def index(model):
            nonlocal attempts
            step = int(scheduler.step_index)
            if attempts >= 8 or step != len(dispatch):
                raise RuntimeError('NFE ceiling or dispatch ordering')
            attempts += 1
            event({'event': 'NFE_ATTEMPT', 'source': source, 'step': step, 'number': attempts})
            if step == 0:
                noise.append(v2.finite_tensor_sha(scheduler.latents, torch))
            answer = old_index(model)
            dispatch.append({'step': step, 'expert': int(model.cur_model_index)})
            return answer

        def post():
            step = int(scheduler.step_index)
            before = v2.finite_tensor_sha(scheduler.noise_pred, torch)
            result = old_post()
            observed = {'step': step, 'flow': before, 'latent': v2.finite_tensor_sha(scheduler.latents, torch),
                        'timestep': float(scheduler.timesteps[step]), 'sigma': float(scheduler.sigmas[step])}
            trace.append(observed)
            event({'event': 'NFE_COMPLETE', 'source': source, **observed})
            return result

        path = output / f'{source}__icc_forced_dense.mp4'
        try:
            MultiDistillModelStruct.get_current_model_index = index
            scheduler.step_post = post
            v2.reset_scheduler_generator(scheduler, pipe.runner.model)
            with bind_native_ffn(WanTransformerInfer, port, exact_source_sha=scope['native_transformer_sha256']):
                pipe.generate(seed=42, image_path=row['image_path'], prompt=row['prompt'], negative_prompt='', save_result_path=str(path))
        finally:
            scheduler.step_post = old_post
            MultiDistillModelStruct.get_current_model_index = old_index
        live_video = v2.video_digest(path)
        plain_trace = [{k: step[k] for k in ('step', 'flow', 'latent', 'timestep', 'sigma')} for step in plain['steps']]
        passed = (trace == plain_trace and noise == [plain['initial_noise']] and dispatch == plain['dispatch']
                  and live_video['decoded_rgb24_sha256'] == plain['video']['decoded_rgb24_sha256'])
        write_json(output / f'{source}_NEUTRAL.json', {'source': source, 'passed': passed, 'trace': trace,
                                                     'dispatch': dispatch, 'sites': port.finish(), 'video': live_video})
        if not passed:
            raise RuntimeError('ICC forced Dense failed exact host equivalence')
    if attempts != 8:
        raise RuntimeError('incomplete ICC host-neutral budget')
    write_json(output / 'SUMMARY.json', {'status': 'PASS', 'nfe': attempts, 'd4': 2, 'main_slots': 0})


def calibrate(pipe, stage, scope, output, prior, retry_audit=None):
    from icc_calibration_driver import execute, NFEJournal
    from icc_wan_calibration_adapter import WanCalibrationAdapter
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from lightx2v.utils.input_info import I2VInputInfo

    manifest = read_json(stage / 'ICC_CLEAN_CAL11_BOUND.json')
    oracle = read_json(stage / 'ORACLE.json')
    if oracle['status'] != 'PASS' or oracle['collector_sha256'] != sha(stage / 'icc_four_boundary_collector.py'):
        raise RuntimeError('missing verified author statistics oracle')
    if sha(inspect.getfile(WanTransformerInfer.infer_ffn)) != scope['native_transformer_sha256']:
        raise RuntimeError('active Wan FFN identity changed')
    for name, method in {'native_i2v': pipe.runner.run_input_encoder, 'native_vae': type(pipe.runner.vae_encoder).encode,
                         'native_text': pipe.runner.run_text_encoder, 'input_info': I2VInputInfo}.items():
        if sha(inspect.getfile(inspect.unwrap(method))) != scope['runtime_sha256'][name]:
            raise RuntimeError(f'changed live {name}')
    adapter = WanCalibrationAdapter(pipe.runner, output / 'cal', manifest, scope['schedule'],
                                    prior['checkpoints'], (I2VInputInfo, MultiDistillModelStruct, WanTransformerInfer, MMWeight))
    retry = scope.get('cd_retry')
    if retry_audit is not None:
        from icc_calibration_recovery import MatchingRetry
        adapter = MatchingRetry(adapter, retry_audit)

    def authorize(_, jobs, budget):
        observed_count = budget['shard_main_nfe'] + budget['shard_diagnostic_nfe']
        if observed_count != scope['max_nfe'] or {job.source for job in jobs} != set(scope['sources']):
            raise RuntimeError('native calibration shard differs from ACTIVE range')

    summary = execute(manifest, adapter, NFEJournal(output / 'NFE.jsonl'), authorize=authorize,
                      source_ids=scope['sources'], retry_pair=retry['transaction'] if retry else None)
    write_json(output / 'SUMMARY.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope_sha')
    args = parser.parse_args()
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, args.scope_sha)
    retry_audit = None
    if 'cd_retry' in scope:
        from icc_calibration_recovery import audit_interrupted
        retry_audit = audit_interrupted(scope['cd_retry'], read_json(stage / 'ICC_CLEAN_CAL11_BOUND.json'))
    import torch

    with torch.inference_mode():
        pipe, output, prior = build_pipeline(stage, scope, reference)
        if scope['method'] == 'ICC_FORCED_DENSE':
            forced_dense(pipe, stage, scope, reference, output)
        elif scope['method'] == 'ICC_AUTHOR_CALIBRATION':
            calibrate(pipe, stage, scope, output, prior, retry_audit=retry_audit)
        else:
            raise RuntimeError('unknown experiment mode')


if __name__ == '__main__':
    main()
