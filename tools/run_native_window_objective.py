"""R11T1: fixed-budget local/native objectives in the verified final window."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from collect_ub1_native_states import policy_alias, verify_runtime
from historical_source_snapshot import HistoricalSources
from native_head_replay import PARAMETERS, normalized_update, patchify_flow, replay, unpatchify_flow
from probe_native_head_gradient import FinalBoundary, source_binding as prerequisite_sources
from reduced_rank_residual import predict_y
from resident_ffn_runtime import make_resident, unused_raw_placeholders
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from run_cross_layer_transport import references
from run_supplement_teacher_flow import exact_native_latent
from variable_teacher_capture import FrozenBias, TeacherRecords
from verify_native_head_bytes import same_bytes

METHOD = 'R11T1_MATCHED_NATIVE_HEAD_OBJECTIVE'
PROTOCOL = ROOT / 'research/20261021/R11T1_matched_objective_protocol.json'
TRAIN, DEV = [0, 5, 10], 15


def sources():
    return {**prerequisite_sources(), **{f'tools/{n}': digest(ROOT / 'tools' / n) for n in
            ['run_native_window_objective.py', 'variable_teacher_capture.py', 'verify_native_head_bytes.py']}}


class TeacherBoundary(FinalBoundary):
    """Record exact original same-input teacher; leave native UB1 unchanged."""
    def __init__(self, runtime, teacher):
        super().__init__(runtime)
        self.teacher = teacher
        self.teacher_receipt = None

    def __enter__(self):
        self.original_apply = self.runtime.apply

        def apply(mm, value, **kw):
            selected = (kw['step_index'], kw['block_index']) == (3, 39)
            if selected and kw['module_name'].endswith('.ffn.0'):
                self.norm_input, self.b0 = value.detach(), mm._get_actual_bias().detach().clone()
            result = self.original_apply(mm, value, **kw)
            if selected and kw['module_name'].endswith('.ffn.2'):
                shard = self.teacher.configure('low_noise_model', 39)
                runtime = self.teacher.runtime
                before = dict(runtime.counters)
                cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                common = dict(expert='low_noise_model', step_index=3, block_index=39)
                try:
                    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
                    scale = runtime.records['low_noise_model']['blocks.39.ffn.0']['smooth_scale'].to(value.device)
                    teacher_u = native_hif4_activation_qdq((self.norm_input.float() / scale).to(self.norm_input.dtype), force_fp32=True).to(self.norm_input.dtype)
                    if not same_bytes(teacher_u, self.data['u'].to(value.device)):
                        raise RuntimeError('Full teacher Q1 differs from actual residual input')
                    del teacher_u
                    z = runtime.apply(FrozenBias(self.b0), self.norm_input, module_name='blocks.39.ffn.0', **common)
                    y = runtime.apply(FrozenBias(mm._get_actual_bias()), F.gelu(z, approximate='tanh'), module_name='blocks.39.ffn.2', **common)
                    self.put('teacher_y', y)
                    delta = {k: runtime.counters[k] - before[k] for k in before}
                    if delta['linear_forward_calls'] != 2 or delta['activation_qdq_calls'] != 2 or delta['cache_misses']:
                        raise RuntimeError('Same-input teacher lacks two original native linears/QDQ')
                    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                        raise RuntimeError('Offline teacher changed native RNG')
                    self.teacher_receipt = {'shard_sha256': shard['sha256'], 'norm_input_sha256': tensor_sha(self.norm_input),
                                            'b0_sha256': tensor_sha(self.b0), 'b2_sha256': tensor_sha(mm._get_actual_bias()),
                                            'counter_delta': delta, 'native_RNG_unchanged': True, 'teacher_Q1_byte_equals_actual': True}
                finally:
                    self.teacher.release()
                    del self.norm_input, self.b0
            return result

        self.runtime.apply = apply
        return super().__enter__()

    def __exit__(self, *error):
        super().__exit__(*error)
        self.runtime.apply = self.original_apply
        self.teacher.release()


def initialize(folder, report):
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from step_budget_runtime import create_runtime
    history = HistoricalSources()
    alias, policy_sha = policy_alias(history, folder)
    kwargs, cache, assets = verified_restored_assets(history)
    refs, bindings = references(history, assets)
    with unused_raw_placeholders(adapter) as audit:
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
    adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
    if len(audit['names']) != 400 or audit['calls'] < 800:
        raise RuntimeError('Strict original loader coverage missing')
    runtime = create_runtime(cache['path'], alias)
    report.update(residency=make_resident(runtime), assets=assets, original_policy_sha256=policy_sha,
                  reference_bindings=bindings, historical_source_receipts=history.receipts)
    raw = Path(cache['path']).parents[1] / 'r11t1_native_objective' / folder.parent.name
    raw.mkdir(parents=True, exist_ok=True)
    return adapter, pipe, runtime, refs, audit, raw, cache


def native_run(pipe, runtime, refs, raw, index, observer=None, expected=None):
    if raw.exists():
        raise RuntimeError('Preserve previous native attempt')
    raw.mkdir(parents=True)
    dq, baseline = refs['DQ'][index], refs['UB1'][index]
    scheduler = pipe.runner.scheduler
    old_pre, old_post = scheduler.step_pre, scheduler.step_post
    steps = []
    sample = dq['case']['input']
    if sample['seed'] != 42 or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
        raise RuntimeError('Original source identity changed')
    runtime.begin_request()
    before = dict(runtime.counters)
    scheduler.generator = None

    def pre(step_index):
        value = old_pre(step_index)
        if not same_bytes(scheduler.latents.detach().float().cpu(), dq['payloads'][step_index]['input']):
            raise RuntimeError('Original matched native input changed')
        return value

    def post():
        step = int(scheduler.step_index)
        flow = scheduler.noise_pred.detach().float().cpu().contiguous()
        value = old_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        target = expected if step == 3 and expected is not None else baseline['payloads'][step]['flow']
        if not same_bytes(flow, target):
            raise RuntimeError('Complete native flow differs from frozen original/replay prediction')
        if expected is None or step != 3:
            if not same_bytes(latent, baseline['payloads'][step]['post']):
                raise RuntimeError('Complete unchanged native post differs')
        if observer is not None and step == 3:
            observer.put('native_flow', flow)
            observer.put('teacher_flow', dq['payloads'][step]['flow'])
        steps.append({'step': step, 'input_byte_equal': True, 'flow_byte_equal': True,
                      'original_post_byte_equal': True if expected is None or step != 3 else None,
                      'flow_sha256': tensor_sha(flow)})
        scheduler.latents = exact_native_latent(dq['payloads'][step]['post'], scheduler.latents)
        return value

    scheduler.step_pre, scheduler.step_post = pre, post
    try:
        from contextlib import nullcontext
        with torch.inference_mode(), observer if observer is not None else nullcontext():
            pipe.generate(seed=42, image_path=sample['image_path'], prompt=sample['prompt'], negative_prompt='',
                          save_result_path=str(raw / 'diagnostic_only.mp4'))
        delta = {k: runtime.counters[k] - before[k] for k in before}
        if delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']:
            raise RuntimeError('Complete native runtime counters changed')
        verify_runtime(runtime, 'UB1')
        if len(steps) != 4:
            raise RuntimeError('Missing complete native steps')
        return {'index': index, 'steps': steps, 'counter_delta': delta, 'diagnostic_video_is_DQ_by_construction': True}
    finally:
        scheduler.step_pre, scheduler.step_post = old_pre, old_post


def collect(args, folder, report):
    manifest = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    if shutil.disk_usage(Path(manifest['path']).parent).free < 90 << 30:
        raise RuntimeError('Need80GiB reserve plus10GiB total four-source capture allowance before loading')
    adapter, pipe, runtime, refs, audit, raw, cache = initialize(folder, report)
    teacher = TeacherRecords(cache)
    adapter.activate_lightx2v_hooks(runtime=runtime)
    report['completed'] = []
    prior = json.loads((ROOT / 'research/20261021/R11G1_native_head_results.json').read_text())
    try:
        for index in ([0, 5] if args.shard == 0 else [10, 15]):
            if shutil.disk_usage(raw).free < 83 << 30:
                raise RuntimeError('Preserve80GiB reserve plus3GiB case allowance')
            observer = TeacherBoundary(runtime, teacher)
            row = native_run(pipe, runtime, refs, raw / f'input{index:02d}', index, observer=observer)
            d = {k: v.to('cuda') for k, v in observer.data.items()}
            p = {k: d['p_' + k] for k in PARAMETERS}
            with torch.inference_mode():
                predicted = replay(d, p)
                checks = {'residual': same_bytes(predict_y(d['u'], p), d['correction']),
                          'post': same_bytes(d['base'] + d['native_y'] * d['gate'], d['native_post']),
                          'tokens': same_bytes(predicted, d['native_tokens']),
                          'flow': same_bytes(unpatchify_flow(predicted, (21, 29, 52)).float(), d['native_flow'])}
            if not all(checks.values()) or observer.teacher_receipt is None or audit['fallback_calls']:
                raise RuntimeError('Complete native/teacher prerequisite failed')
            del d, p, predicted
            tensors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_sha(v)} for k, v in observer.data.items()}
            repeat = index == 0
            if repeat and any(tensors[k] != value for k, value in prior['payload']['tensors'].items()):
                raise RuntimeError('Repeated source00 capture differs from R11G1')
            path = raw / f'input{index:02d}' / 'boundary.safetensors'
            save_file(observer.data, str(path))
            row.update(path=str(path), sha256=digest(path), bytes=path.stat().st_size, tensors=tensors,
                       zero_update_byte_controls=checks, teacher=observer.teacher_receipt, source00_R11G1_repeat_equal=repeat)
            report['completed'].append(row)
            atomic_json(folder / 'RUN.json', report)
            print(f'R11T1 native capture input{index:02d} PASS', flush=True)
        report['status'] = 'PASS_FIXED_WINDOW_CAPTURE_SHARD'
    finally:
        adapter.deactivate_lightx2v_hooks()


def captures(args):
    reports = [json.loads((args.out / f'collect{i}/RUN.json').read_text()) for i in (0, 1)]
    rows = {}
    for report in reports:
        if report['status'] != 'PASS_FIXED_WINDOW_CAPTURE_SHARD' or report['sources'] != sources() or report['protocol_sha256'] != digest(PROTOCOL):
            raise RuntimeError('All fixed native controls must pass before training')
        for row in report['completed']:
            if row['index'] in rows or digest(row['path']) != row['sha256']:
                raise RuntimeError('Missing/repeated/changed complete capture')
            rows[row['index']] = row
    if set(rows) != set(TRAIN + [DEV]):
        raise RuntimeError('Original four-source membership changed')
    for row in rows.values():
        for k in PARAMETERS:
            if row['tensors']['p_' + k] != rows[0]['tensors']['p_' + k]:
                raise RuntimeError('Matched arms must have identical initial parameters')
    return rows


def objective(d, p, arm):
    if arm == 'flow':
        return (replay(d, p).float() - patchify_flow(d['teacher_flow'], (21, 29, 52))).square().mean()
    if arm != 'local':
        raise ValueError('Only registered matched objectives supported')
    y = d['main'] + predict_y(d['u'], p)
    return ((y.float() - d['teacher_y'].float()) * d['gate'].float()).square().mean()


def train(args, folder, report):
    rows = captures(args)
    data = [load_file(rows[i]['path'], device='cuda') for i in TRAIN]
    p = {k: data[0]['p_' + k].float().detach().requires_grad_(True) for k in PARAMETERS}
    report.update(training_indices=TRAIN, development_loss_read=False, iterations=[])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(20):
        loss_value = 0.
        for d in data:
            loss = objective(d, p, args.arm) / len(data)
            loss_value += float(loss.detach())
            loss.backward()
        updated, update = normalized_update(p, .001)
        report['iterations'].append({'step': step, 'train_objective_before_update': loss_value, 'update': update})
        p = {k: v.requires_grad_(True) for k, v in updated.items()}
    torch.cuda.synchronize()
    with torch.no_grad():
        final = sum(float(objective(d, p, args.arm)) for d in data) / len(data)
    path = Path(rows[0]['path']).parents[1] / f'fit_{args.arm}.safetensors'
    if path.exists():
        raise RuntimeError('Preserve frozen fitted parameters')
    saved = {k: v.detach().bfloat16().contiguous() for k, v in p.items()}
    save_file(saved, str(path))
    report.update(status='PASS_FIXED20_PARAMETER_FIT_FROZEN_BEFORE_DEVELOPMENT', arm=args.arm,
                  final_train_objective=final, train_objective_decreased=final < report['iterations'][0]['train_objective_before_update'],
                  parameters_path=str(path), parameters_sha256=digest(path),
                  parameter_tensor_sha256={k: tensor_sha(v) for k, v in saved.items()},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_train_seconds=time.perf_counter() - start,
                  capture_receipt_sha256={str(i): digest(args.out / f'collect{i}/RUN.json') for i in (0, 1)})


def fits(args):
    results = {arm: json.loads((args.out / f'train_{arm}/RUN.json').read_text()) for arm in ('local', 'flow')}
    for arm, fit in results.items():
        if fit['status'] != 'PASS_FIXED20_PARAMETER_FIT_FROZEN_BEFORE_DEVELOPMENT' or fit['sources'] != sources() or fit['protocol_sha256'] != digest(PROTOCOL) or fit['development_loss_read']:
            raise RuntimeError('Both objectives must be frozen before any development loss')
        if digest(fit['parameters_path']) != fit['parameters_sha256']:
            raise RuntimeError('Frozen deployment parameters changed')
    return results


def evaluate(args, folder, report):
    rows, fitted = captures(args), fits(args)
    parameters = {arm: load_file(fit['parameters_path'], device='cuda') for arm, fit in fitted.items()}
    report.update(rows=[], fitted_receipt_sha256={arm: digest(args.out / f'train_{arm}/RUN.json') for arm in fitted})
    for index in TRAIN + [DEV]:
        d = load_file(rows[index]['path'], device='cuda')
        target = patchify_flow(d['teacher_flow'], (21, 29, 52))
        baseline = float((d['native_tokens'].double() - target.double()).square().sum())
        r = {'index': index, 'role': 'development_holdout' if index == DEV else 'train', 'UB1_flow_SSE': baseline, 'arms': {}}
        with torch.inference_mode():
            for arm, p in parameters.items():
                tokens = replay(d, p)
                sse = float((tokens.double() - target.double()).square().sum())
                r['arms'][arm] = {'flow_SSE': sse, 'flow_SSE_over_UB1': sse / baseline,
                                  'local_gated_MSE': float(objective(d, p, 'local')), 'tokens_sha256': tensor_sha(tokens)}
        report['rows'].append(r)
        del d, target, tokens
    dev = report['rows'][-1]
    report.update(status='PASS_FIXED_MATCHED_OBJECTIVE_EVALUATION', development_gate=(
        dev['arms']['flow']['flow_SSE'] <= .98 * dev['UB1_flow_SSE']
        and dev['arms']['flow']['flow_SSE'] <= .99 * dev['arms']['local']['flow_SSE']),
        no_quality_or_speed_acceptance=True)


def validate(args, folder, report):
    rows, fitted = captures(args), fits(args)
    evaluation_path = args.out / 'evaluate/RUN.json'
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation['status'] != 'PASS_FIXED_MATCHED_OBJECTIVE_EVALUATION' or evaluation['sources'] != sources():
        raise RuntimeError('Complete frozen evaluation required')
    d = load_file(rows[DEV]['path'], device='cuda')
    p = load_file(fitted[args.arm]['parameters_path'], device='cuda')
    with torch.inference_mode():
        expected = unpatchify_flow(replay(d, p), (21, 29, 52)).float().cpu()
    del d
    adapter, pipe, runtime, refs, audit, raw, cache = initialize(folder, report)
    runtime.residual_parameters[3, 39] = p
    adapter.activate_lightx2v_hooks(runtime=runtime)
    try:
        report['native'] = native_run(pipe, runtime, refs, raw / f'validate_{args.arm}', DEV, expected=expected)
        if audit['fallback_calls']:
            raise RuntimeError('Forbidden placeholder fallback')
        report.update(status='PASS_UPDATED_COMPLETE_NATIVE_FLOW_BYTE_EQUALS_REPLAY', arm=args.arm,
                      parameters_sha256=fitted[args.arm]['parameters_sha256'], evaluation_sha256=digest(evaluation_path),
                      complete_flow_sha256=tensor_sha(expected), deployment_parameter_change_only=[3, 39])
    finally:
        adapter.deactivate_lightx2v_hooks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['collect', 'train', 'evaluate', 'validate'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--arm', choices=['local', 'flow'], default='flow')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(ROOT)
    if validate_records(*records) or not method_eligibility(records[0], records[1], METHOD)[0]:
        raise RuntimeError('Fixed matched diagnostic not consistently registered')
    args.out = args.out.resolve()
    job = f'collect{args.shard}' if args.phase == 'collect' else f'{args.phase}_{args.arm}' if args.phase in ('train', 'validate') else 'evaluate'
    folder = args.out / job
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists():
        raise RuntimeError('Preserve completed/failed matched experiment')
    torch.set_num_threads(8)
    report = {'id': METHOD, 'status': 'RUNNING', 'phase': args.phase, 'protocol_sha256': digest(PROTOCOL),
              'sources': sources(), 'quality_or_speed_acceptance': False}
    atomic_json(folder / 'RUN.json', report)
    try:
        {'collect': collect, 'train': train, 'evaluate': evaluate, 'validate': validate}[args.phase](args, folder, report)
        if report['sources'] != sources() or report['protocol_sha256'] != digest(PROTOCOL):
            raise RuntimeError('Frozen matched experiment source changed')
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)
    print(json.dumps({'phase': args.phase, 'status': report['status']}), flush=True)


if __name__ == '__main__':
    main()
