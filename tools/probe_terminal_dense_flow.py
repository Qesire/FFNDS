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
    record = {'status': 'PASS_UB2_TF_NATIVE_DENSE_CONTROL' if teacher is None else 'PASS_UB2_TF_SAME_DENSE_INPUT_STATES',
              'index': index, 'input': sample, 'steps': trajectory, 'hif4_counter_delta': delta,
              'runtime': runtime.report(), 'elapsed_seconds_including_observation': time.perf_counter() - started,
              'video': {'path': str(video), 'sha256': digest(video), 'diagnostic_only': teacher is not None},
              'quality_or_speed_acceptance': False}
    path = out / f'input{index:02d}_attempt{attempt}.json'
    atomic_json(path, record)
    return {'index': index, 'path': str(path), 'sha256': digest(path)}


def main():
    ap = argparse.ArgumentParser(description='Verify UB2 exact dense terminal flow on one fixed D-Q state trajectory')
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--identity-proof',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--payload-root',type=Path,required=True)
    args = ap.parse_args()
    policy = json.loads(args.policy.read_text())
    if policy['identity_control'] or policy['kind'] != 'TERMINAL_DENSE_STEP_REFERENCE':
        raise RuntimeError('The active frozen UB2 policy is required')
    proof = json.loads(args.identity_proof.read_text())
    if proof['status'] != 'PASS_UB2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES':
        raise RuntimeError('Finish the four fixed UB1 identity controls first')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82*1024**3):
        raise RuntimeError('Node HF outputs and80GiB reserve plus2GiB allowance required')
    r = ROOT/'runs/worker_restart_20261020'
    sample = json.loads((ROOT/'.runtime/inputs/disjoint20/canonical.json').read_text())[0]
    kwargs,cache,assets = verified_assets()
    controls = {}
    bindings = {}
    for arm,status in [('DQ','PASS_TF2_FOUR_SOURCE_DENSE_CONTROLS'),('UB1','PASS_TF2_FOUR_SOURCE_CANDIDATE_SAME_DENSE_STATES')]:
        path = r/f'tf2_flow/{arm}/RUN.json'
        suite = json.loads(path.read_text())
        if suite['status'] != status or suite['identity']['assets'] != assets:
            raise RuntimeError('The verified completed TF2 controls must remain unchanged')
        row = next(v for v in suite['completed'] if v['index'] == 0)
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Fixed source0 TF2 control changed')
        controls[arm] = json.loads(Path(row['path']).read_text())
        if controls[arm]['input'] != sample:
            raise RuntimeError('Fixed source0 conditioning changed')
        bindings[arm] = {'suite_sha256':digest(path),'case_sha256':row['sha256']}
    sourcepath = r/'disjoint20_UB1/RUN.json'
    source = json.loads(sourcepath.read_text())
    if source['identity']['packed_policy_sha256'] != policy['parent_policy']['sha256']:
        raise RuntimeError('Exact frozen UB1 first-step reference required')
    row = next(v for v in source['completed'] if v['index'] == 0)
    if digest(row['case_report']) != row['case_report_sha256']:
        raise RuntimeError('Original UB1 reference changed')
    reference = json.loads(Path(row['case_report']).read_text())
    if reference['input'] != sample or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
        raise RuntimeError('Fixed source0 reference/image differs')
    args.out.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = {'assets':assets,'policy_sha256':digest(args.policy),'identity_control_proof_sha256':digest(args.identity_proof),
        'protocol_sha256':policy['protocol']['sha256'],'source_sha256':digest(Path(__file__)),
        'TF2_bindings':bindings,'source_suite_sha256':digest(sourcepath),'index':0,
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/terminal_dense_runtime.py',ROOT/'tools/terminal_dense_validation.py',
            ROOT/'tools/resident_ffn_runtime.py',ROOT/'tools/run_supplement_teacher_flow.py']}}
    meta = {'status':'RUNNING','identity':identity,'quality_or_speed_acceptance':False}
    atomic_json(args.out/'RUN.json',meta)
    try:
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from terminal_dense_runtime import create_runtime
        from terminal_dense_validation import validate_runtime
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('All400 original covered weights must use strict placeholders')
        runtime = create_runtime(cache['path'],args.policy)
        meta['residency'] = make_resident(runtime)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            record = capture_one(pipe,runtime,sample,0,reference,controls['DQ'],args.out,args.payload_root.resolve())
        finally:
            adapter.deactivate_lightx2v_hooks()
        case = json.loads(Path(record['path']).read_text())
        validate_runtime(case['runtime']['packed_reference'],digest(args.policy),require_timing=False)
        unchanged = []
        for step in [0,2]:
            actual,old = case['steps'][step],controls['UB1']['steps'][step]
            if actual['flow_sha256'] != old['flow_sha256'] or actual['post_sha256'] != old['post_sha256']:
                raise RuntimeError('The unchanged step0/2 functions differ from frozen UB1 at the same D-Q states')
            unchanged.append(step)
        final,dense = case['steps'][3],controls['DQ']['steps'][3]
        if (final['flow_sha256'] != dense['flow_sha256'] or final['post_sha256'] != dense['post_sha256']
                or final['flow_error']['squared_error'] != 0 or final['one_step_post_error']['squared_error'] != 0
                or audit['fallback_calls']):
            raise RuntimeError('Terminal full-model flow/post update must equal native D-Q exactly')
        meta.update(status='PASS_UB2_DENSE_TERMINAL_AND_UNCHANGED_STEPS_NATIVE_FLOW',case=record,
                    unchanged_steps0_2_bitwise=True,terminal_flow_post_bitwise_equals_DQ=True,
                    all_four_input_states_bitwise_equals_DQ=True,steps=case['steps'],
                    scope='one fixed supplemental0 source on D-Q states; forced final video excluded from quality, no terminal attribution')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.out/'RUN.json',meta)


if __name__ == '__main__':
    main()

