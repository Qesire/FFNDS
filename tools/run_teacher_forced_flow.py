"""Capture native flow error on identical verified D-Q states (diagnostic only)."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from distill_assets import ROOT, verified_assets
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from run_ar2_restart import tensor_sha


def exact_native_latent(saved, like):
    """Reject lossy state injection; saved FP32 must encode native values exactly."""
    if saved.dtype != torch.float32 or saved.shape != like.shape or not torch.isfinite(saved).all():
        raise RuntimeError('Invalid saved latent geometry/precision/values')
    restored = saved.to(device=like.device, dtype=like.dtype).contiguous()
    if tensor_sha(restored.float().cpu()) != tensor_sha(saved.cpu()):
        raise RuntimeError('Saved latent cannot be restored to the native dtype exactly')
    return restored


def error_stats(candidate, reference):
    x, y = candidate.double(), reference.double()
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError('Invalid paired diagnostic tensors')
    den = float(y.square().sum())
    if den <= 0:
        raise RuntimeError('Degenerate reference energy')
    error = float((x - y).square().sum())
    return {'squared_error': error, 'reference_squared_norm': den, 'relative_RMSE': (error / den) ** .5}


def capture_one(pipe, runtime, sample, index, reference, teacher, out, payload_root):
    attempt = 1
    while (out / f'input{index:02d}_attempt{attempt}.json').exists() or (payload_root / f'input{index:02d}_attempt{attempt}').exists():
        attempt += 1
    folder = payload_root / f'input{index:02d}_attempt{attempt}'
    folder.mkdir(parents=True)
    scheduler = pipe.runner.scheduler
    original_pre, original_post = scheduler.step_pre, scheduler.step_post
    trajectory, observed_inputs = [], {}
    if hasattr(runtime, 'begin_request'):
        runtime.begin_request()
    before = dict(runtime.counters)
    scheduler.generator = None

    def pre(step_index):
        result = original_pre(step_index)
        observed_inputs[int(step_index)] = scheduler.latents.detach().float().cpu().contiguous()
        return result

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        result = original_post()
        actual = scheduler.latents.detach().float().cpu().contiguous()
        if step not in observed_inputs or not torch.isfinite(flow).all() or not torch.isfinite(actual).all():
            raise RuntimeError('Incomplete/nonfinite native diagnostic step')
        metadata = {'step': step, 'sigma': float(scheduler.sigmas[step]),
                    'actual_model_timestep': float(scheduler.timesteps[step]),
                    'native_latent_dtype': str(scheduler.latents.dtype),
                    'native_flow_dtype': str(scheduler.noise_pred.dtype),
                    'input_sha256': tensor_sha(observed_inputs[step]), 'post_sha256': tensor_sha(actual),
                    'flow_sha256': tensor_sha(flow)}
        original_latent = reference['trajectory'][step]
        if teacher is None:
            if metadata['post_sha256'] != original_latent['raw_tensor_sha256']:
                raise RuntimeError('Native dense control changed an existing formal latent')
            metadata['original_formal_latent_bitwise_equal'] = True
        else:
            control = teacher['steps'][step]
            if digest(control['path']) != control['sha256']:
                raise RuntimeError('Dense flow control payload changed')
            target = load_file(control['path'], device='cpu')
            if (metadata['input_sha256'] != control['input_sha256']
                    or tensor_sha(target['input']) != control['input_sha256']
                    or tensor_sha(target['flow']) != control['flow_sha256']
                    or tensor_sha(target['post']) != control['post_sha256']):
                raise RuntimeError('Candidate did not receive the exact native D-Q state')
            if step == 0 and metadata['post_sha256'] != original_latent['raw_tensor_sha256']:
                raise RuntimeError('Unforced first step changed the frozen candidate')
            metadata.update(input_bitwise_equals_DQ=True,
                            flow_error=error_stats(flow, target['flow']),
                            one_step_post_error=error_stats(actual, target['post']))
            scheduler.latents = exact_native_latent(target['post'], scheduler.latents)
        path = folder / f'step{step}.safetensors'
        save_file({'input': observed_inputs[step], 'flow': flow, 'post': actual}, str(path))
        metadata.update(path=str(path), sha256=digest(path))
        trajectory.append(metadata)
        return result

    scheduler.step_pre, scheduler.step_post = pre, post
    video = folder / 'diagnostic_only.mp4'
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            pipe.generate(seed=42, image_path=sample['image_path'], prompt=sample['prompt'],
                          negative_prompt='', save_result_path=str(video))
        torch.cuda.synchronize()
    finally:
        scheduler.step_pre, scheduler.step_post = original_pre, original_post
    delta = {k: runtime.counters[k] - v for k, v in before.items()}
    if ([v['step'] for v in trajectory] != list(range(4)) or delta['linear_forward_calls'] != 1600
            or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']):
        raise RuntimeError('All four native steps and bridge operations required')
    if teacher is None and digest(video) != reference['video']['sha256']:
        raise RuntimeError('Dense control video differs from frozen formal source')
    record = {'status': 'PASS_TF1_NATIVE_DENSE_CONTROL' if teacher is None else 'PASS_TF1_SAME_DENSE_INPUT_STATES',
              'index': index, 'input': sample, 'steps': trajectory, 'hif4_counter_delta': delta,
              'runtime': runtime.report(), 'elapsed_seconds_including_observation': time.perf_counter() - started,
              'video': {'path': str(video), 'sha256': digest(video), 'diagnostic_only': teacher is not None},
              'quality_or_speed_acceptance': False}
    path = out / f'input{index:02d}_attempt{attempt}.json'
    atomic_json(path, record)
    return {'index': index, 'path': str(path), 'sha256': digest(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-suite', type=Path, required=True)
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--teacher-report', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--payload-root', type=Path, required=True)
    args = ap.parse_args()
    if bool(args.policy) != bool(args.teacher_report):
        ap.error('Candidate requires both --policy and --teacher-report')
    root = ROOT / 'runs/worker_restart_20261020'
    quality = json.loads((root / 'formal20_UB1_comparison/COMPARISON.json').read_text())
    resident = json.loads((root / 'formal20_UB1_resident_equivalence/RUN.json').read_text())
    if (quality['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
            or quality['formal20_quality_pass']
            or resident['status'] != 'PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT'):
        raise RuntimeError('TF1 runs only after frozen UB1 quality rejection and completed resident equivalence')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = ROOT / 'config/teacher_forced_flow_tf1.json'
    indices = json.loads(protocol.read_text())['indices']
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82 * 1024 ** 3):
        raise RuntimeError('Diagnostic tensors require node HF cache and80GiB reserve plus2GiB allowance')
    source = json.loads(args.reference_suite.read_text())
    samples = json.loads((ROOT / '.runtime/inputs/formal20/canonical.json').read_text())
    kwargs, cache, assets = verified_assets()
    if (source['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY'
            or source['identity']['indices'] != list(range(20))
            or source['identity']['assets'] != assets
            or source['identity']['inputs_sha256'] != digest(ROOT / '.runtime/inputs/formal20/canonical.json')
            or source['identity']['packed_policy_sha256'] != (digest(args.policy) if args.policy else None)):
        raise RuntimeError('Frozen complete formal source/policy/assets differ')
    teacher, controls = None, {}
    if args.teacher_report:
        teacher = json.loads(args.teacher_report.read_text())
        if (teacher['status'] != 'PASS_TF1_FOUR_SOURCE_DENSE_CONTROLS'
                or teacher['identity']['assets'] != assets
                or teacher['identity']['protocol_sha256'] != digest(protocol)
                or teacher['identity']['source_sha256'] != digest(Path(__file__))):
            raise RuntimeError('Complete matching native dense controls required')
        for row in teacher['completed']:
            if digest(row['path']) != row['sha256']:
                raise RuntimeError('Dense flow control case changed')
            controls[row['index']] = json.loads(Path(row['path']).read_text())
        if set(controls) != set(indices):
            raise RuntimeError('All four fixed dense controls required')
    identity = {'assets': assets, 'protocol_sha256': digest(protocol), 'source_sha256': digest(Path(__file__)),
                'source_suite_sha256': digest(args.reference_suite),
                'source_identity': source['identity'], 'teacher_report_sha256': digest(args.teacher_report) if teacher else None,
                'payload_root': str(args.payload_root.resolve()),
                'condition_quality_sha256': digest(root / 'formal20_UB1_comparison/COMPARISON.json'),
                'condition_resident_sha256': digest(root / 'formal20_UB1_resident_equivalence/RUN.json')}
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = args.out / 'RUN.json'
    completed = []
    if report.exists():
        previous = json.loads(report.read_text())
        if previous['identity'] != identity:
            raise RuntimeError('TF1 diagnostic identity changed')
        completed = previous['completed']
    for row in completed:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Completed TF1 case changed')
    meta = {'status': 'RUNNING', 'identity': identity, 'completed': completed, 'quality_or_speed_acceptance': False}
    atomic_json(report, meta)
    try:
        pending = [i for i in indices if i not in {v['index'] for v in completed}]
        if pending:
            from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
            with unused_raw_placeholders(adapter) as audit:
                pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
            if len(audit['names']) != 400 or audit['calls'] < 800:
                raise RuntimeError('All covered raw matrices must be replaced by strict placeholders')
            if args.policy:
                from packed_reference_runtime import create_runtime
                runtime = create_runtime(cache['path'], args.policy)
            else:
                runtime = adapter.HIF4CacheRuntime(cache['path'])
            meta['residency'] = make_resident(runtime)
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    source_row = next(v for v in source['completed'] if v['index'] == index)
                    if digest(source_row['case_report']) != source_row['case_report_sha256']:
                        raise RuntimeError('Frozen formal case changed')
                    ref = json.loads(Path(source_row['case_report']).read_text())
                    if ref['input'] != samples[index]:
                        raise RuntimeError('Conditioning source differs')
                    completed.append(capture_one(pipe, runtime, samples[index], index, ref,
                                                controls.get(index), args.out, args.payload_root.resolve()))
                    if audit['fallback_calls']:
                        raise RuntimeError('Covered raw placeholder fallback occurred')
                    atomic_json(report, meta)
                    print(f'TF1 committed input{index:02d}, candidate={bool(args.policy)}', flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if {v['index'] for v in completed} != set(indices) or len(completed) != 4:
            raise RuntimeError('Exactly four fixed sources required')
        meta['status'] = 'PASS_TF1_FOUR_SOURCE_CANDIDATE_SAME_DENSE_STATES' if teacher else 'PASS_TF1_FOUR_SOURCE_DENSE_CONTROLS'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(report, meta)


if __name__ == '__main__':
    main()
