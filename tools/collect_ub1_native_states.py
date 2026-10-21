"""OP2: versioned native teacher controls and calibration on actual UB1 states."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import imageio.v2 as imageio
import torch

from ar2_checkpoint import atomic_json, canonical_json, digest
from historical_source_snapshot import HistoricalSources
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from variable_teacher_capture import TeacherRecords, VariableTeacherCapture

RUNS = ROOT / 'runs/worker_restart_20261020'


def policy_alias(history, out):
    original = RUNS / 'ub1_step_budget_fit/UB1_policy.json'
    expected = json.loads((RUNS / 'disjoint20_UB1_comparison/COMPARISON.json').read_text())['candidate_policy_sha256']
    if digest(original) != expected:
        raise RuntimeError('Original observed UB1 policy changed')
    policy = json.loads(original.read_text())
    binding = policy['protocol']
    protocol = history.original(binding['path'], binding['sha256'])
    policy['protocol'] = {**binding, 'path': str(protocol)}
    policy['historical_policy_sha256'] = expected
    policy['execution_alias_scope'] = 'only protocol path resolution; unchanged numeric parameters, separately proven against old native outputs'
    path = out / 'UB1_execution_alias.json'
    if path.exists() and json.loads(path.read_text()) != policy:
        raise RuntimeError('Execution alias changed; choose a new collection directory')
    atomic_json(path, policy)
    return path, expected


def reference_case(arm, index, assets):
    path = RUNS / ('formal20_dense/RUN.json' if arm == 'DQ' else 'formal20_UB1/FULL_RUN.json')
    report = json.loads(path.read_text())
    if report['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY' or report['identity']['assets'] != assets:
        raise RuntimeError('Old control generation is incomplete or has different assets')
    for source, sha in report['identity']['native_sources'].items():
        if digest(source) != sha:
            raise RuntimeError('Original model operator or scheduler source changed')
    entry = next(v for v in report['completed'] if v['index'] == index)
    if digest(entry['case_report']) != entry['case_report_sha256']:
        raise RuntimeError('Original control case report changed')
    case = json.loads(Path(entry['case_report']).read_text())
    if digest(case['video']['path']) != case['video']['sha256']:
        raise RuntimeError('Original control video changed')
    return case, entry


def verify_runtime(runtime, arm):
    if arm == 'DQ':
        return
    report = runtime.report()['packed_reference']
    residual = report['residual_correction']
    if (report['actual_FFN_linear_calls'] != 320 or residual['actual_positions'] != 160
            or residual['pending_predictions']):
        raise RuntimeError('UB1 capture requires every main and residual position')
    for step, width in enumerate([3456, 6912, 6912, 10368]):
        for block in range(40):
            for slot, shape in [(0, [width, 5120]), (2, [5120, width])]:
                if report['coverage'][f'step{step}_block{block}_ffn{slot}']['actual_weight_shape'] != shape:
                    raise RuntimeError('Observed main shape differs from frozen UB1 budget')


def collect_one(pipe, runtime, teacher, sample, index, arm, raw_root, fingerprint, reference=None):
    parent = raw_root / fingerprint / f'input{index:02d}'
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent / f'attempt{attempt}').exists():
        attempt += 1
    out = parent / f'attempt{attempt}'
    out.mkdir()
    observer = VariableTeacherCapture(runtime, teacher, out, index, dense_control=arm == 'DQ')
    trajectories, videos, counters = {}, {}, {}
    for phase in ['plain', 'observed']:
        if shutil.disk_usage('/var/tmp').free < 80 << 30:
            raise RuntimeError('Preserve80GiB node reserve')
        scheduler = pipe.runner.scheduler
        original_post = scheduler.step_post
        trajectory = []

        def post():
            result = original_post()
            latent = scheduler.latents.detach().float().cpu().contiguous()
            if not bool(torch.isfinite(latent).all()):
                raise RuntimeError('Nonfinite native state')
            trajectory.append({'step': int(scheduler.step_index), 'sha256': tensor_sha(latent), 'shape': list(latent.shape)})
            return result

        scheduler.step_post = post
        scheduler.generator = None
        if hasattr(runtime, 'begin_request'):
            runtime.begin_request()
        before = dict(runtime.counters)
        video = out / f'{phase}.mp4'
        started = time.perf_counter()
        try:
            with torch.inference_mode(), observer if phase == 'observed' else nullcontext():
                pipe.generate(seed=sample['seed'], image_path=sample['image_path'], prompt=sample['prompt'],
                              negative_prompt='', save_result_path=str(video))
            torch.cuda.synchronize()
        finally:
            scheduler.step_post = original_post
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if (delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600
                or delta['cache_misses'] or [v['step'] for v in trajectory] != list(range(4))):
            raise RuntimeError('Native four-step/all-linear coverage differs')
        verify_runtime(runtime, arm)
        trajectories[phase] = trajectory
        counters[phase] = delta
        videos[phase] = {'path': str(video), 'sha256': digest(video), 'elapsed_including_observation': time.perf_counter() - started}
        if reference:
            expected = [{'step': v['step'], 'sha256': v['raw_tensor_sha256'], 'shape': v['shape']} for v in reference['trajectory']]
            if trajectory != expected or videos[phase]['sha256'] != reference['video']['sha256']:
                raise RuntimeError('Post-redaction execution differs from original native latent/video bytes')
        atomic_json(out / f'PHASE_{phase}.json', {'trajectory': trajectory, 'video': videos[phase], 'counter_delta': delta})
        print(f'OP2 {arm} input{index:02d} {phase} complete', flush=True)
    if trajectories['plain'] != trajectories['observed'] or videos['plain']['sha256'] != videos['observed']['sha256']:
        raise RuntimeError('Teacher observer changed native trajectory/video bytes')
    if len(observer.scopes) != 160 or {(s['step'], s['block']) for s in observer.scopes} != {(s, b) for s in range(4) for b in range(40)}:
        raise RuntimeError('Incomplete/duplicate native teacher scopes')
    if not all(s['Q1_teacher_equals_actual'] for s in observer.scopes):
        raise RuntimeError('Teacher and actual native Q1 differ')
    if arm == 'DQ' and not all(s['Q2_teacher_equals_actual'] and s['Y_teacher_equals_actual_dense'] for s in observer.scopes):
        raise RuntimeError('Dense teacher full-native Q2/Y control failed')
    with imageio.get_reader(videos['observed']['path']) as reader:
        frames, info = reader.count_frames(), reader.get_meta_data()
    if frames != 81 or abs(float(info['fps']) - 16) > .01:
        raise RuntimeError('Native frames/fps changed')
    path = out / 'INPUT.json'
    atomic_json(path, {'status': 'PASS_OP2_NATIVE_INPUT_AND_NOOP', 'index': index, 'arm': arm,
        'fingerprint': fingerprint, 'input': sample, 'trajectories': trajectories, 'videos': videos,
        'scopes': observer.scopes, 'native_counter_deltas': counters,
        'original_reference_bitwise': reference is not None, 'scope': 'teacher collection only, no fit, speed or quality acceptance',
        'offline_extra_teacher_GEMMs': sum(s['extra_full_teacher_GEMMs'] for s in observer.scopes)})
    teacher.release()
    return {'index': index, 'input_report': str(path), 'input_report_sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['control', 'calibration'], required=True)
    parser.add_argument('--arm', choices=['DQ', 'UB1'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'calibration':
        if args.arm != 'UB1':
            raise RuntimeError('Only actual UB1 calibration is registered')
        for arm in ['DQ', 'UB1']:
            control = json.loads((RUNS / f'op2_native_controls/{arm}/RUN.json').read_text())
            if control['status'] != 'PASS_OP2_BOTH_FIXED_NATIVE_CONTROLS' or {c['index'] for c in control['completed']} != {0, 10}:
                raise RuntimeError('Both native control arms must finish before bulk collection')
    torch.set_num_threads(8)
    history = HistoricalSources()
    alias, original_policy_sha = policy_alias(history, args.out)
    kwargs, cache, assets = verified_restored_assets(history)
    source = ROOT / '.runtime/inputs' / ('formal20/canonical.json' if args.stage == 'control' else 'calibration24/canonical.json')
    recovery = source.with_name('recovery_report.json')
    samples = json.loads(source.read_text())
    recover = json.loads(recovery.read_text())
    if recover['canonical_sha256'] != digest(source) or len(samples) != (20 if args.stage == 'control' else 24):
        raise RuntimeError('Input recovery binding differs')
    indices = [0, 10] if args.stage == 'control' else [i for i in range(24) if i % 2 == args.shard]
    raw_root = Path.home() / '.cache/huggingface/ffnds-derived/op2_ub1_native_states_20261021' / args.stage / args.arm
    raw_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw_root).free < 200 << 30:
        raise RuntimeError('Preserve200GiB shared home reserve')
    identity = {'protocol_sha256': digest(ROOT / 'config/onpolicy_ub1_op2.json'), 'stage': args.stage,
        'arm': args.arm, 'indices': indices, 'assets': assets, 'inputs_sha256': digest(source),
        'recovery_sha256': digest(recovery), 'original_UB1_policy_sha256': original_policy_sha,
        'UB1_execution_alias_sha256': digest(alias), 'historical_source_receipts': history.receipts,
        'sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in [
            Path(__file__).name, 'variable_teacher_capture.py', 'native_teacher_capture.py', 'historical_source_snapshot.py',
            'restored_assets.py', 'step_budget_runtime.py', 'trained_packed_runtime.py', 'reduced_rank_residual.py']}}
    fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    report = args.out / 'RUN.json'
    completed = []
    if report.exists():
        prior = json.loads(report.read_text())
        if prior['identity'] != identity:
            raise RuntimeError('Collection identity changed; preserve previous output')
        completed = prior['completed']
    for entry in completed:
        if digest(entry['input_report']) != entry['input_report_sha256']:
            raise RuntimeError('Completed native input report changed')
    meta = {'status': 'RUNNING', 'identity': identity, 'fingerprint': fingerprint,
            'completed': completed, 'quality_or_speed_acceptance': False}
    atomic_json(report, meta)
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    try:
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        if args.arm == 'UB1':
            from step_budget_runtime import create_runtime
            runtime = create_runtime(cache['path'], alias)
        else:
            runtime = adapter.HIF4CacheRuntime(cache['path'])
        teacher = TeacherRecords(cache)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for index in indices:
                if index in {r['index'] for r in completed}:
                    continue
                sample = dict(samples[index])
                if args.stage == 'calibration':
                    sample.update(seed=sample['calibration_seed'], image_path=sample['img_paths'][0])
                    if sample['seed'] != 42 + index:
                        raise RuntimeError('Original calibration seed differs')
                if digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
                    raise RuntimeError('Native input image changed')
                reference, binding = reference_case(args.arm, index, assets) if args.stage == 'control' else (None, None)
                entry = collect_one(pipe, runtime, teacher, sample, index, args.arm, raw_root, fingerprint, reference)
                if binding:
                    entry['original_reference_binding'] = binding
                completed.append(entry)
                atomic_json(report, meta)
        finally:
            adapter.deactivate_lightx2v_hooks()
        if {e['index'] for e in completed} != set(indices):
            raise RuntimeError('Incomplete native collection')
        meta['status'] = 'PASS_OP2_BOTH_FIXED_NATIVE_CONTROLS' if args.stage == 'control' else 'PASS_OP2_UB1_CALIBRATION_SHARD'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(report, meta)


if __name__ == '__main__':
    main()
